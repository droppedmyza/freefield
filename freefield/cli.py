from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from . import media, pipelines, runner
from .presets import Preset, PresetError, load_presets

MAX_SEED = 2**31 - 1


class InputError(ValueError):
    pass


def seed_arg(text: str) -> int:
    try:
        seed = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"seed must be a whole number, got {text!r}") from None
    if not 0 <= seed <= MAX_SEED:
        raise argparse.ArgumentTypeError(f"seed must be between 0 and {MAX_SEED}, got {seed}")
    return seed


def emit(data: dict) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def progress_printer():
    last = {"t": 0.0}

    def on_progress(ev: dict) -> None:
        now = time.monotonic()
        if ev["type"] != "progress" or now - last["t"] > 2:
            last["t"] = now
            print("  ", json.dumps(ev), file=sys.stderr, flush=True)

    return on_progress


def print_stage(stage: str) -> None:
    print(f"   {stage}", file=sys.stderr, flush=True)


def find_preset(preset_id: str) -> Preset:
    presets, errors = load_presets()
    if preset_id in presets:
        return presets[preset_id]
    if preset_id in errors:
        raise InputError(f"preset {preset_id!r} does not load: {errors[preset_id]}")
    raise InputError(f"unknown preset {preset_id!r}; run `python -m freefield list` to see them")


def check_source(preset: Preset, source: Path) -> None:
    if not source.is_file():
        raise InputError(f"{source} is not a file")
    try:
        pipelines.check_source(preset, source)
    except media.MediaError as e:
        raise InputError(str(e)) from None


def texts_for(preset: Preset, caption: str | None) -> dict[str, str]:
    if caption is None:
        return {}
    name = pipelines.text_input(preset)
    if name is None:
        raise InputError(f"{preset.id} takes no caption")
    limit = int(preset.inputs[name].get("max_length", 500))
    if len(caption) > limit:
        raise InputError(f"the {name} is {len(caption)} characters; {preset.id} takes at most {limit}")
    return {name: caption}


def result_report(preset: Preset, result: runner.RunResult) -> dict:
    return {"preset": preset.id, "status": result.status, "seconds": round(result.seconds, 1),
            "files": {k: [str(p) for p in v] for k, v in result.files.items()}, "error": result.error,
            "warnings": result.warnings, "checks": result.checks}


def run_dir(base: str, preset: Preset) -> Path:
    return Path(base) / f"{preset.id}-{time.strftime('%Y%m%d-%H%M%S')}"


def cmd_list(_: argparse.Namespace) -> int:
    presets, errors = load_presets()
    for p in presets.values():
        print(f"{p.id:20s} {p.family:10s} {p.name}: {p.description[:70]}")
    for folder, err in errors.items():
        print(f"BROKEN {folder}: {err}", file=sys.stderr)
    return 1 if errors else 0


def cmd_run(args: argparse.Namespace) -> int:
    preset = find_preset(args.preset)
    source = Path(args.source)
    check_source(preset, source)
    texts = texts_for(preset, args.caption)
    seed = args.seed if args.seed is not None else pipelines.default_seed(preset)
    try:
        result = pipelines.run_preset(preset, source, seed, run_dir(args.out, preset), texts=texts,
                                      on_progress=progress_printer(), on_stage=print_stage)
    except PresetError as e:
        raise InputError(str(e)) from None
    emit(result_report(preset, result))
    return 0 if result.ok else 1


def cmd_real(args: argparse.Namespace) -> int:
    preset = find_preset("out-of-the-frame-real")
    clip = Path(args.clip)
    check_source(preset, clip)
    seed = args.seed if args.seed is not None else pipelines.default_seed(preset)
    result = pipelines.run_pipeline(preset, clip, seed, run_dir(args.out, preset), on_stage=print_stage,
                                    shot1="still" if args.still else "wan")
    emit(result_report(preset, result))
    return 0 if result.ok else 1


def cmd_factory(args: argparse.Namespace) -> int:
    from .factory import FactoryError, Spec, build

    photo = Path(args.photo)
    if not photo.is_file():
        raise InputError(f"{photo} is not a file")
    try:
        report = build(Spec.load(Path(args.spec)), photo, render_preview=not args.no_preview)
    except FactoryError as e:
        emit({"status": "failed", "error": str(e)})
        return 1
    emit({k: report[k] for k in ("id", "published", "seconds")} | {"steps": report["steps"]})
    return 0 if report["published"] or args.no_preview else 1


def cmd_build(args: argparse.Namespace) -> int:
    from . import builder, recipe, ui_export

    if args.render and not args.photo:
        raise InputError("--render needs --photo")
    if args.photo and not Path(args.photo).is_file():
        raise InputError(f"{args.photo} is not a file")
    try:
        r = recipe.parse(args.description)
        r.profile = "studio" if args.studio else "8gb"
        if args.reference:
            r.reference = recipe.Asset(Path(args.reference).name, args.reference_license or "", args.reference_source or "",
                                       Path(args.reference))
        d = builder.draft(r, comfy_url=args.comfyui)
        out = {"id": r.id, "folder": str(d.folder), "recipe": r.public(), "schemas": d.schema_source, "errors": d.errors}
        if d.errors:
            emit(out | {"status": "rejected"})
            return 1
        out["layout"] = ui_export.export_layout(d.folder, args.comfyui) or "workflow.json written"
        out["status"] = "drafted"
        if args.render:
            rec = builder.preview(d.folder, Path(args.photo), seed=args.seed)
            out["status"], out["preview"] = rec["status"], rec["preview"]
            if args.publish:
                out["published"] = str(builder.publish(d.folder, approved=args.approve))
                out["status"] = "published"
    except recipe.BuildError as e:
        emit({"status": "failed", "error": str(e)})
        return 1
    emit(out)
    return 0 if out["status"] != "draft (preview failed its checks)" else 1


def cmd_recheck(args: argparse.Namespace) -> int:
    from . import builder, recipe

    try:
        rec = builder.preview(builder.DRAFTS / args.id, Path(args.photo), render=False)
    except (recipe.BuildError, FileNotFoundError) as e:
        emit({"status": "failed", "error": str(e)})
        return 1
    emit({"status": rec["status"], "preview": rec["preview"]})
    return 0 if rec["preview"]["passed"] else 1


def cmd_publish(args: argparse.Namespace) -> int:
    from . import builder, recipe

    try:
        target = builder.publish(builder.DRAFTS / args.id, approved=args.approve)
    except (recipe.BuildError, FileNotFoundError) as e:
        emit({"status": "draft", "error": str(e)})
        return 1
    emit({"status": "published", "preset": target.name, "folder": str(target)})
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    serve(port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="freefield", description="Open, Higgsfield-style viral video presets on Comfy")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list presets").set_defaults(fn=cmd_list)

    r = sub.add_parser("run", help="run a preset on a photo (or a clip, for the presets that take one)")
    r.add_argument("preset")
    source = r.add_mutually_exclusive_group(required=True)
    source.add_argument("--photo", dest="source", help="your photo")
    source.add_argument("--clip", dest="source", help="your clip (earth-zoom, out-of-the-frame-real)")
    r.add_argument("--seed", type=seed_arg)
    r.add_argument("--caption", help="the preset's text: the place name, the offer, the title or the bonk word")
    r.add_argument("--out", default="outputs")
    r.set_defaults(fn=cmd_run)

    q = sub.add_parser("real", help="Out of the Frame with your own selfie clip: you come out of a painting of you")
    q.add_argument("--clip", required=True, help="a 6-10 s selfie video: hold still a second, then lean in toward the phone")
    q.add_argument("--seed", type=seed_arg)
    q.add_argument("--out", default="outputs")
    q.add_argument("--still", action="store_true", help="hold the painting still instead of animating it (no Wan; 8 GB GPU)")
    q.set_defaults(fn=cmd_real)

    f = sub.add_parser("factory", help="make a new breakout preset from a spec (a draft until its preview passes the gate)")
    f.add_argument("spec")
    f.add_argument("--photo", required=True, help="test photo for the gated preview")
    f.add_argument("--no-preview", action="store_true", help="stop after the plate and the preset (no video)")
    f.set_defaults(fn=cmd_factory)

    b = sub.add_parser("build", help="describe an effect in plain words; get a validated draft preset, a preview, then publish")
    b.add_argument("description", help='e.g. "make me a paper cutout, then I slowly turn my head and smile, with a reveal, titled \\"ME\\""')
    b.add_argument("--photo", help="a photo for the preview")
    b.add_argument("--render", action="store_true", help="render the preview on your GPU (about 2 minutes)")
    b.add_argument("--publish", action="store_true", help="publish if the preview passes its checks")
    b.add_argument("--approve", action="store_true", help="you looked at the preview and approve it")
    b.add_argument("--seed", type=seed_arg, default=42)
    b.add_argument("--studio", action="store_true", help="animate with Wan 2.2 I2V-A14B (24 GB GPU) instead of TI2V-5B")
    b.add_argument("--reference", help="a style reference image (needs --reference-license and --reference-source)")
    b.add_argument("--reference-license")
    b.add_argument("--reference-source")
    b.add_argument("--comfyui", default="http://127.0.0.1:8188", help="ComfyUI address for node schemas and the layout")
    b.set_defaults(fn=cmd_build)

    rc = sub.add_parser("recheck", help="run a draft's preview checks again on the render it already has")
    rc.add_argument("id")
    rc.add_argument("--photo", required=True)
    rc.set_defaults(fn=cmd_recheck)

    pb = sub.add_parser("publish", help="publish a built draft after you looked at its preview")
    pb.add_argument("id")
    pb.add_argument("--approve", action="store_true", help="you watched the preview and approve it")
    pb.set_defaults(fn=cmd_publish)

    s = sub.add_parser("serve", help="start the gallery web app")
    s.add_argument("--port", type=int, default=8190)
    s.set_defaults(fn=cmd_serve)
    return ap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except InputError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
