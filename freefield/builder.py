from __future__ import annotations

import datetime
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import output_checks, runner
from .graph_stages import (
    KLEIN,
    KLEIN_TE,
    KLEIN_VAE,
    LIGHTX_HIGH,
    LIGHTX_LOW,
    WAN,
    WAN14_HIGH,
    WAN14_LOW,
    WAN14_TE,
    WAN14_VAE,
    WAN_TE,
    WAN_VAE,
    keyframe_stage,
    node,
    save_stage,
    video_stage,
    wan14_models,
    wan14_segment,
)
from .header_card import write_header
from .node_check import load_schemas, validate
from .presets import PresetError, parse_preset
from .recipe import Asset, BuildError, Recipe, describe, keyframe_prompt, motion_prompt, recipe_from

PKG = Path(__file__).resolve().parent
log = logging.getLogger(__name__)
BUILDER_VERSION = "1"
FRAMES = 73
GUIDE = (36, 64)
VIDEO_W, VIDEO_H = 480, 832
LICENSES = {"cc0", "public-domain", "cc-by-4.0", "cc-by-sa-4.0", "generated"}
MODEL_LICENSES = {KLEIN: "apache-2.0", KLEIN_TE: "apache-2.0", KLEIN_VAE: "apache-2.0", WAN: "apache-2.0",
                  WAN_TE: "apache-2.0", WAN_VAE: "apache-2.0", WAN14_HIGH: "apache-2.0", WAN14_LOW: "apache-2.0",
                  WAN14_TE: "apache-2.0", WAN14_VAE: "apache-2.0", LIGHTX_HIGH: "apache-2.0", LIGHTX_LOW: "apache-2.0"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")


def _add_start(g: dict, r: Recipe) -> list:
    g["photo"] = node("LoadImage", "Your photo (preset input)", image="")
    refs = ["photo"]
    if r.reference:
        g["reference"] = node("LoadImage", f"Style reference ({r.reference.license}, preset asset)", image="")
        refs.append("reference")
    if r.keep:
        g["start"] = node("ImageScale", "Your photo, vertical (kept as it is)", image=["photo", 0], upscale_method="lanczos",
                          width=720, height=1280, crop="center")
        return ["start", 0]
    if not r.keeps_subject():
        g["photo_tiny"] = node("ImageScale", "Your photo shrunk to a colour guide (no animal features survive)", image=["photo", 0],
                               upscale_method="area", width=GUIDE[0], height=GUIDE[1], crop="center")
        g["photo_blur"] = node("ImageScale", "Colour guide at full size", image=["photo_tiny", 0], upscale_method="bilinear",
                               width=720, height=1280, crop="disabled")
        refs = ["photo_blur"] + refs[1:]
    keyframe_stage(g, refs, width=720, height=1280)
    return ["keyframe", 0]


def _add_animation(g: dict, r: Recipe, start: list) -> tuple[list, str]:
    if r.profile == "studio":
        wan14_models(g)
        wan14_segment(g, "motion", "Animate", start, None, VIDEO_W, VIDEO_H, FRAMES)
        return ["motion_frames", 0], "Animate (Wan 2.2 I2V-A14B, studio profile)"
    video_stage(g, width=VIDEO_W, height=VIDEO_H, length=FRAMES, start=start if r.keep else None)
    return ["video_frames", 0], "Animate (Wan 2.2 TI2V-5B)"


def _add_reveal(g: dict, frames: list) -> list:
    g["before"] = node("ImageScale", "Your photo at clip size", image=["photo", 0], upscale_method="lanczos",
                       width=VIDEO_W, height=VIDEO_H, crop="center")
    g["before_frames"] = node("RepeatImageBatch", "Your photo for every frame", image=["before", 0], amount=FRAMES)
    g["wipe"] = node("LoadVideo", "Wipe mask (preset asset, made by the builder)", file="")
    g["wipe_parts"] = node("GetVideoComponents", "Wipe mask frames", video=["wipe", 0])
    g["wipe_mask"] = node("ImageToMask", "Wipe mask as a mask", image=["wipe_parts", 0], channel="red")
    g["reveal"] = node("ImageCompositeMasked", "Reveal: wipe from your photo into the effect", destination=["before_frames", 0],
                       source=frames, x=0, y=0, resize_source=False, mask=["wipe_mask", 0])
    return ["reveal", 0]


def _add_split(g: dict, frames: list) -> list:
    g["original"] = node("ImageScale", "Your photo beside it", image=["photo", 0], upscale_method="lanczos",
                         width=VIDEO_W, height=VIDEO_H, crop="center")
    g["split"] = node("ImageStitch", "Before | after, side by side", image1=frames, image2=["original", 0],
                      direction="left", match_image_size=True, spacing_width=0, spacing_color="white")
    g["split_even"] = node("ImageScale", "Side by side at an even size (the video encoder needs it)", image=["split", 0],
                           upscale_method="lanczos", width=VIDEO_W * 2, height=VIDEO_H, crop="center")
    return ["split_even", 0]


def _add_title_card(g: dict, frames: list, width: int) -> list:
    g["header"] = node("LoadImage", "Title card (preset asset, made by the builder)", image="")
    g["header_small"] = node("ImageScale", "Title card at clip width", image=["header", 0], upscale_method="lanczos",
                             width=width, height=round(width * 192 / 720), crop="disabled")
    g["titled"] = node("ImageStitch", "Title card above every frame", image1=["header_small", 0], image2=frames,
                       direction="down", match_image_size=True, spacing_width=0, spacing_color="black")
    return ["titled", 0]


def _add_caption(g: dict, frames: list) -> list:
    g["caption"] = node("TextOverlay", "Caption (your text, typed per run)", images=frames, text="", font_size=6.0,
                        color="#ffffff", position="bottom", align="center", outline=True)
    return ["caption", 0]


def _fixed_values(r: Recipe) -> dict:
    fixed = {"motion_prompt.text": motion_prompt(r)}
    if not r.keep:
        fixed["keyframe_prompt.text"] = keyframe_prompt(r)
    if r.reveal:
        fixed["wipe.file"] = f"asset:{r.id}-wipe.mp4"
    if r.title:
        fixed["header.image"] = f"asset:{r.id}-header.png"
    if r.reference:
        fixed["reference.image"] = f"asset:{r.reference.name}"
    return fixed


def _numbered(stages: list[dict]) -> list[dict]:
    return [{**st, "title": f"{i}. {st['title']}"} for i, st in enumerate(stages[:-1], 1)] + stages[-1:]


def assemble(r: Recipe) -> tuple[dict, dict, list[dict]]:
    g: dict = {}
    start = _add_start(g, r)
    frames, animate = _add_animation(g, r, start)
    stages = [{"title": "Your photo", "color": "#335", "match": ["photo", "reference", "start"]}]
    if not r.keep:
        stages.append({"title": "Transform (FLUX.2 klein 4B)", "color": "#353", "match": ["klein_", "keyframe", "photo_", "reference_"]})
    stages.append({"title": animate, "color": "#533", "match": ["wan_", "wan14_", "motion_", "motion", "video_"]})
    width = VIDEO_W
    if r.reveal:
        frames = _add_reveal(g, frames)
        stages.append({"title": "Reveal (masked composition)", "color": "#553", "match": ["before", "wipe", "reveal"]})
    if r.split:
        frames, width = _add_split(g, frames), VIDEO_W * 2
        stages.append({"title": "Side by side", "color": "#355", "match": ["original", "split", "split_even"]})
    if r.title:
        frames = _add_title_card(g, frames, width)
        stages.append({"title": "Title card", "color": "#535", "match": ["header", "titled"]})
    if r.caption:
        frames = _add_caption(g, frames)
        stages.append({"title": "Caption", "color": "#553", "match": ["caption"]})
    save_stage(g, frames)
    stages.append({"title": "Save", "color": "#444", "match": ["watermark", "video", "clip_save"]})
    return g, _fixed_values(r), _numbered(stages)


def check_assets(folder: Path, assets: list[Asset]) -> list[str]:
    errors: list[str] = []
    for a in assets:
        if not SAFE_NAME.match(a.name) or ".." in a.name:
            errors.append(f"asset {a.name!r}: unsafe file name (letters, digits, dot, dash, underscore; no folders)")
            continue
        p = (folder / a.name).resolve()
        if folder.resolve() not in p.parents:
            errors.append(f"asset {a.name!r}: outside the preset folder")
        elif not p.is_file():
            errors.append(f"asset {a.name!r}: missing from the preset folder")
        if (a.license or "").lower() not in LICENSES:
            errors.append(f"asset {a.name!r}: license {a.license!r} is not one we can ship (use {sorted(LICENSES)})")
        if not a.source:
            errors.append(f"asset {a.name!r}: say where it comes from (source)")
    return errors


def models_of(graph: dict) -> list[str]:
    names = []
    for n in graph.values():
        for k in ("unet_name", "clip_name", "vae_name", "lora_name"):
            v = n.get("inputs", {}).get(k)
            if isinstance(v, str) and v and v not in names:
                names.append(v)
    return names


PHOTO_LABELS = {"person": "A clear photo of your face", "pet": "A clear photo of your pet",
                "object": "A clear photo of the object (a toy, a product, a mascot)", "scene": "A photo of a place"}


def inputs_of(r: Recipe) -> dict:
    inputs = {"photo": {"type": "image", "label": PHOTO_LABELS[r.subject], "bind": ["photo.image"]}}
    if r.caption:
        inputs["caption"] = {"type": "text", "label": "Caption (for example a city name)", "bind": ["caption.text"],
                             "required": False, "max_length": 40}
    return inputs


def manifest(r: Recipe, fixed: dict, graph: dict, assets: list[Asset], created: str) -> dict:
    return {
        "id": r.id,
        "name": r.name,
        "family": "built",
        "description": describe(r),
        "graph": "graph.json",
        "inputs": inputs_of(r),
        "params": {"seed": {"type": "int", "default": 42, "min": 0, "max": 2147483647,
                            "bind": (["keyframe_noise.noise_seed"] if not r.keep else [])
                            + (["motion_sample_high.noise_seed", "motion_sample.noise_seed"] if r.profile == "studio" else ["video_sample.seed"])}},
        "fixed": fixed,
        "outputs": ["clip_save"] + ([] if r.keep else ["keyframe_save"]),
        "models": models_of(graph),
        "license": "; ".join(f"{m} {MODEL_LICENSES.get(m, 'unknown')}" for m in models_of(graph)),
        "source": {"made_by": f"FreeField effect builder v{BUILDER_VERSION}", "description": r.description,
                   "recipe": r.public(), "created": created},
        "credits": {a.name: f"{a.source} ({a.license})" for a in assets},
        "asset_licenses": {a.name: a.license for a in assets},
        "tags": ["built"] + r.stages()[2:-1] + (["studio-profile"] if r.profile == "studio" else []),
    }


@dataclass
class Draft:
    recipe: Recipe
    folder: Path
    graph: dict
    fixed: dict
    stages: list[dict]
    assets: list[Asset] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    schema_source: str = ""


DRAFTS = PKG / "presets" / "_drafts"
PRESETS = PKG / "presets"
ROOT = PKG.parent


def wipe_asset(out: Path, start: int = 6, end: int = 30) -> Path:
    xs = np.linspace(0, 1, VIDEO_W, dtype=np.float32)[None, :]
    frames = []
    for i in range(FRAMES):
        t = min(1.0, max(0.0, (i - start) / (end - start)))
        edge = t * 1.3 - 0.15
        m = np.clip((edge - xs) / 0.15 + 0.5, 0, 1)
        frames.append(np.repeat((np.repeat(m, VIDEO_H, 0) * 255).astype(np.uint8)[..., None], 3, 2))
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VIDEO_W}x{VIDEO_H}",
           "-r", "24", "-i", "-", "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv444p", str(out)]
    subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), check=True)
    return out


def title_asset(text: str, out: Path) -> Path:
    return write_header(text, out)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def draft(r: Recipe, *, comfy_url: str | None = "http://127.0.0.1:8188", drafts: Path | None = None) -> Draft:
    folder = (drafts or DRAFTS) / r.id
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    graph, fixed, stages = assemble(r)
    assets: list[Asset] = []
    if r.reveal:
        wipe_asset(folder / f"{r.id}-wipe.mp4")
        assets.append(Asset(f"{r.id}-wipe.mp4", "generated", f"FreeField effect builder v{BUILDER_VERSION} (wipe mask)"))
    if r.title:
        title_asset(r.title, folder / f"{r.id}-header.png")
        assets.append(Asset(f"{r.id}-header.png", "generated", f"FreeField effect builder v{BUILDER_VERSION} (scripts/make_header.py, Luckiest Guy font, Apache-2.0)"))
    if r.reference:
        src = r.reference.path
        if src is None or not Path(src).is_file():
            raise BuildError("the style reference image was not found")
        if SAFE_NAME.match(r.reference.name):
            shutil.copy(src, folder / r.reference.name)
        assets.append(r.reference)
    schemas, source = load_schemas(comfy_url)
    d = Draft(r, folder, graph, fixed, stages, assets, schema_source=source)
    d.errors = validate(graph, fixed, schemas, {"photo.image"}) + check_assets(folder, assets)
    (folder / "graph.json").write_text(json.dumps(graph, indent=2) + "\n")
    (folder / "stages.json").write_text(json.dumps(stages, indent=2) + "\n")
    data = manifest(r, fixed, graph, assets, _now())
    (folder / "preset.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    if not d.errors:
        try:
            parse_preset(data, folder)
        except PresetError as e:
            d.errors.append(str(e))
    _record(folder, {"status": "rejected" if d.errors else "drafted", "errors": d.errors, "schemas": source,
                     "recipe": r.public(), "drafted": _now()})
    return d


def _record(folder: Path, update: dict) -> dict:
    path = folder / "build.json"
    rec = json.loads(path.read_text()) if path.exists() else {}
    rec.update(update)
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    return rec


def preview(folder: Path, photo: Path, *, client=None, seed: int = 42, render: bool = True,
            rerolls: int = output_checks.VIDEO_REROLLS) -> dict:
    rec = json.loads((folder / "build.json").read_text())
    if rec.get("errors"):
        raise BuildError("this draft was rejected; fix the description first: " + "; ".join(rec["errors"][:3]))
    preset = parse_preset(json.loads((folder / "preset.json").read_text()), folder)
    r = recipe_from(rec["recipe"])
    rendered = None
    if not render:
        s = rec.get("preview", {}).get("seed", seed)
        where = folder / rec.get("preview", {}).get("render_dir", "render")
        found = {key: sorted(where.glob(f"{key}_*")) for key in ("clip_save", "keyframe_save")}
        rendered = (s, runner.RunResult("", "succeeded" if found["clip_save"] else "failed", 0.0, {key: v for key, v in found.items() if v}))
    dirs = (lambda k: folder / ("render" if k == 0 else f"render-{k}")) if render else (lambda k: where)
    done = output_checks.checked_run(preset, r, photo, dirs, client=client, seed=seed, rerolls=rerolls, rendered=rendered)
    checks = done.checks
    k = checks.pop("attempt")
    if render:
        checks["render_dir"] = dirs(k).name
    clip = checks.pop("clip_path")
    if checks["clip"]:
        shutil.copy(clip, folder / "preview.mp4")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "2", "-i", str(folder / "preview.mp4"), "-frames:v", "1",
                        "-q:v", "3", str(folder / "poster.jpg")], check=False)
    return _record(folder, {"status": "ready for approval" if checks["passed"] else "draft (preview failed its checks)",
                            "preview": checks, "previewed": _now(), "preview_photo": photo.name})


def publish(folder: Path, *, approved: bool, presets: Path | None = None) -> Path:
    rec = json.loads((folder / "build.json").read_text())
    if not approved:
        raise BuildError("publishing needs a person's approval of the preview")
    if not rec.get("preview", {}).get("passed"):
        raise BuildError("the preview has not passed its checks; this stays a draft")
    data = json.loads((folder / "preset.json").read_text())
    target = (presets or PRESETS) / data["id"]
    if target.exists():
        old = json.loads((target / "preset.json").read_text()) if (target / "preset.json").exists() else {}
        if old.get("family") != "built":
            raise BuildError(f"a shipped preset is already called {data['id']!r}; pick another name")
        shutil.rmtree(target)
    shutil.copytree(folder, target, ignore=shutil.ignore_patterns("render", "build.json"))
    data["preview"] = "preview.mp4"
    data["source"]["approved"] = _now()
    (target / "preset.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    _record(folder, {"status": "published", "published": _now(), "published_to": str(target)})
    return target
