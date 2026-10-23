from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from . import breakout, runner
from .plate import Plate, extract, wall_mask
from .presets import GRAPHS_DIR, PRESETS_DIR, PresetError, compile_prompt, fill_for_export, parse_preset
from .runner import CompiledRun

log = logging.getLogger(__name__)

W, H = 720, 1280
DRAFTS = PRESETS_DIR / "_drafts"
REQUIRED = ("id", "name", "description", "scene", "subject", "grip", "out", "shot1", "shot2", "shot3")


class FactoryError(ValueError):
    pass


@dataclass
class Spec:
    id: str
    name: str
    description: str
    scene: str
    subject: str
    grip: str
    out: str
    shot1: str
    shot2: str
    shot3: str
    family: str = "breakout"
    likeness: str = ("Image 1 shows a person. Keep image 1 exactly: the same pose, clothes, background, light and framing. "
                     "Only make the face match the face of the person in image 2 exactly: the same face shape, skin tone, eyes, "
                     "nose, mouth, age and facial hair.")
    border: int = 30
    below: int = 60
    grip_edge: str = "bottom"
    out_from_empty: bool = True
    plate_seeds: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    hole_fill: list[int] = field(default_factory=lambda: [38, 30, 24])
    identity_min: float = 0.30
    painting_identity_min: float = 0.35
    tags: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "Spec":
        data = json.loads(Path(path).read_text())
        missing = [k for k in REQUIRED if not data.get(k)]
        if missing:
            raise FactoryError(f"spec {path} is missing {missing}")
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise FactoryError(f"spec {path} has unknown fields {sorted(unknown)}")
        if "magenta" not in data["scene"]:
            raise FactoryError("spec.scene must ask for the opening to be filled with flat, solid, pure magenta")
        if data.get("grip_edge", "bottom") not in ("bottom", "top"):
            raise FactoryError("spec.grip_edge must be 'bottom' or 'top'")
        return cls(**data)


def round16(v: float) -> int:
    return max(16, int(round(v / 16)) * 16)


def score_plate(p: Plate | None, border: int) -> float:
    if p is None or p.coverage < 0.93:
        return 0.0
    x, y, w, h = p.xywh
    area = w * h / (W * H)
    if not 0.06 <= area <= 0.35 or not 0.35 <= w / h <= 1.6:
        return 0.0
    if x < border + 20 or y < border + 20 or x + w > W - border - 20 or y + h > H - border - 120:
        return 0.0
    return p.coverage * min(area, 0.25)


def view_box(hole: tuple[int, int, int, int], margin: float = 1.65) -> tuple[int, int, int, int]:
    x, y, w, h = hole
    vw = max(w * margin, h * margin * 9 / 16)
    vw = min(vw, W)
    vh = vw * 16 / 9
    if vh > H:
        vh, vw = H, H * 9 / 16
    vw, vh = int(round(vw)), int(round(vw * 16 / 9))
    cx, cy = x + w / 2, y + h / 2
    vx = int(min(max(0, cx - vw / 2), W - vw))
    vy = int(min(max(0, cy - vh / 2), H - vh))
    return vx, vy, vw, vh


def geometry(hole: tuple[int, int, int, int]) -> dict:
    x, y, w, h = hole
    ih = 656
    iw = round16(ih * w / h)
    vx, vy, vw, vh = view_box(hole)
    g = {"painting_fit.width": w, "painting_fit.height": h, "start_frame.x": x, "start_frame.y": y}
    for stage in ("painting", "likeness"):
        g.update({f"{stage}_empty.width": iw, f"{stage}_empty.height": ih, f"{stage}_sigmas.width": iw, f"{stage}_sigmas.height": ih})
    for crop in ("start_view_crop", "grip_view_crop", "shot1_wall_crop"):
        g.update({f"{crop}.x": vx, f"{crop}.y": vy, f"{crop}.width": vw, f"{crop}.height": vh})
    return g


def manifest(spec: Spec, hole: tuple[int, int, int, int]) -> dict:
    fixed = {"plate.image": "asset:plate.png", "plate_hole.image": "asset:plate_hole.png", "wall.image": "asset:wall_mask.png",
             "painting_prompt.text": spec.subject, "likeness_prompt.text": spec.likeness, "grip_prompt.text": spec.grip,
             "breakout_prompt.text": spec.out, "shot1_prompt.text": spec.shot1, "shot2_prompt.text": spec.shot2,
             "shot3_prompt.text": spec.shot3, **geometry(hole)}
    if spec.out_from_empty:
        fixed["breakout_photo_scaled.image"] = ["likeness", 0]
        fixed["breakout_start_frame_latent.pixels"] = ["plate", 0]
    base = json.loads((PRESETS_DIR / "out-of-the-frame" / "preset.json").read_text())
    return {
        "id": spec.id, "name": spec.name, "family": spec.family, "description": spec.description,
        "graph": "selfie_breakout.json",
        "inputs": {"photo": {"type": "image", "label": "A clear photo of your face", "bind": ["photo.image", "artwork_ref.image"]}},
        "params": {"seed": base["params"]["seed"]},
        "fixed": fixed,
        "outputs": base["outputs"], "models": base["models"],
        "license": "apache-2.0 weights; plate generated with FLUX.2 klein (no people) by freefield/factory.py",
        "credits": {"plate.png": f"generated by the preset factory from: {spec.scene}"},
        "tags": spec.tags + ["breakout", "factory", "studio-profile"],
        "gate": {"hole": list(hole), "rerolls": 5, "grip_edge": spec.grip_edge, "frame_border": spec.border, "rail": spec.below,
                 "identity_min": spec.identity_min, "painting_identity_min": spec.painting_identity_min},
    }


def make_plate(spec: Spec, draft: Path, client) -> tuple[Plate, list[dict]]:
    graph = json.loads((GRAPHS_DIR / "portrait.json").read_text())
    graph["keyframe_prompt"]["inputs"]["text"] = spec.scene
    for n in ("keyframe_empty", "keyframe_sigmas"):
        graph[n]["inputs"].update(width=W, height=H)
    tried, best = [], None
    for seed in spec.plate_seeds:
        graph["keyframe_noise"]["inputs"]["noise_seed"] = seed
        graph["keyframe_save"]["inputs"]["filename_prefix"] = f"freefield/plate-{spec.id}-{seed}"
        for _ in range(2):
            res = runner.run(CompiledRun(json.loads(json.dumps(graph)), [], ["keyframe_save"]), draft / "plates" / f"seed-{seed}",
                             client=client, label=f"{spec.id}-plate")
            if res.ok and all(p.stat().st_size > 0 for p in res.files["keyframe_save"]):
                break
            log.warning("plate seed %d came back empty (%s); retrying once", seed, res.error)
        else:
            raise FactoryError(f"plate render failed twice for seed {seed}: {res.error or 'no image returned'}")
        img = Image.open(res.files["keyframe_save"][0])
        p = extract(img, tuple(spec.hole_fill))
        s = score_plate(p, spec.border)
        tried.append({"seed": seed, "score": round(s, 4), "hole": list(p.xywh) if p else None,
                      "coverage": round(p.coverage, 3) if p else None})
        if s > 0 and (best is None or s > best[0]):
            best = (s, p, seed)
    if best is None:
        raise FactoryError(f"no usable plate among seeds {spec.plate_seeds}: {tried}")
    return best[1], tried


STAGE_TITLES = {"1a": "1a. You, transformed (subject)", "1b": "1b. Match the face", "2.": "2. START: you in the opening",
                "3.": "3. GRIP: hands at the opening's edge", "4.": "4. OUT: you, out of it"}


def write_graph_files(spec: Spec, data: dict, folder: Path) -> None:
    stages = json.loads((PRESETS_DIR / "out-of-the-frame" / "stages.json").read_text())
    for st in stages:
        for k, v in STAGE_TITLES.items():
            if st["title"].startswith(k):
                st["title"] = v
    (folder / "stages.json").write_text(json.dumps(stages, indent=1) + "\n")
    graph = fill_for_export(json.loads((GRAPHS_DIR / data["graph"]).read_text()), data["fixed"],
                            {b: "your_photo.png" for b in data["inputs"]["photo"]["bind"]})
    (folder / "workflow_api.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def write_plate_assets(spec: Spec, plate: Plate, draft: Path) -> None:
    plate.plate.save(draft / "plate.png")
    plate.hole.save(draft / "plate_hole.png")
    wall_mask(plate.plate.size, plate.box, spec.border, spec.below).save(draft / "wall_mask.png")


def render_preview_clip(spec: Spec, preset, photo: Path, draft: Path, client) -> tuple[bool, dict]:
    result = breakout.run_gated(compile_prompt(preset, {"photo": str(photo)}), preset.gate, photo, draft / "preview",
                                client=client, label=spec.id)
    gate_record = json.loads((draft / "preview" / "gate.json").read_text())
    video = {k: gate_record[k] for k in ("end_identity", "shot1_intrusion") if k in gate_record}
    passed = bool(result.ok and gate_record["passed"] and breakout.video_ok(video))
    step = {"step": "preview", "ok": result.ok, "gate_passed": gate_record["passed"], "attempts": len(gate_record["attempts"]),
            "video_checks": video, "video_passed": breakout.video_ok(video), "warnings": list(result.warnings),
            "seconds": round(result.seconds, 1)}
    if result.ok:
        shutil.copy(result.files["clip_save"][0], draft / "preview.mp4")
        shutil.copy(result.files["proof_save"][0], draft / "proof.png")
    return passed, step


def publish_draft(spec: Spec, data: dict, draft: Path) -> Path:
    target = PRESETS_DIR / spec.id
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(draft, target, ignore=shutil.ignore_patterns("plates", "preview"))
    write_json(target / "preset.json", data | {"preview": "preview.mp4"})
    return target


def build(spec: Spec, photo: Path, *, client=None, render_preview: bool = True) -> dict:
    started = time.monotonic()
    draft = DRAFTS / spec.id
    if draft.exists():
        shutil.rmtree(draft)
    draft.mkdir(parents=True)
    client = client or runner.make_client()
    report: dict = {"id": spec.id, "spec": spec.__dict__, "steps": []}
    plate, tried = make_plate(spec, draft, client)
    report["steps"].append({"step": "plate", "candidates": tried, "hole": list(plate.xywh)})
    write_plate_assets(spec, plate, draft)
    data = manifest(spec, plate.xywh)
    write_json(draft / "preset.json", data)
    try:
        preset = parse_preset(data, draft)
    except PresetError as e:
        raise FactoryError(f"the generated preset does not validate: {e}") from e
    report["steps"].append({"step": "preset", "valid": True})
    passed = False
    if render_preview:
        passed, step = render_preview_clip(spec, preset, photo, draft, client)
        report["steps"].append(step)
    report["published"] = passed
    report["seconds"] = round(time.monotonic() - started, 1)
    write_json(draft / "factory_report.json", report)
    write_graph_files(spec, data, draft)
    if passed:
        log.info("published %s", publish_draft(spec, data, draft))
    else:
        log.warning("kept %s as a draft: the preview did not pass the keyframe gate or the video checks (see factory_report.json)", draft)
    return report
