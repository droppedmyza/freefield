from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import edge_check, gate, runner, vision
from .presets import compile_prompt
from .recipe import Recipe, recipe_from

log = logging.getLogger(__name__)

VIDEO_REROLLS = 2


def probe_image(clip: str | None, still: str | None, out: Path) -> Path | None:
    if still:
        return Path(still)
    if not clip:
        return None
    probe = out / "last.png"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-sseof", "-0.2", "-i", str(clip), "-frames:v", "1", str(probe)],
                   check=False)
    return probe


def likeness_ok(r: Recipe, photo: Path, probe: Path, out: Path, subject, checks: dict) -> bool:
    if not r.keeps_subject():
        return True
    faces = vision.Faces()
    sim = faces.similarity(photo, probe) if r.subject == "person" else None
    if sim is None and r.subject == "person":
        from PIL import Image

        half = out / "keyframe_half.png"
        im = Image.open(probe)
        im.resize((im.width // 2, im.height // 2)).save(half)
        sim = faces.similarity(photo, half)
    checks["face_match"] = None if sim is None else round(sim, 3)
    if sim is not None:
        return sim >= gate.IDENTITY_MIN
    kept = subject.similarity(photo, probe)
    checks["subject_match"] = round(kept, 3)
    return kept >= vision.SUBJECT_MIN


def shows_description(r: Recipe, photo: Path, probe: Path, subject, checks: dict) -> bool:
    if not r.keep and not r.keeps_subject():
        text = "a photo of a person"
    else:
        text = f"a picture of {r.becomes}" if r.becomes else r.motion
    gain = subject.text_similarity(text, probe) - subject.text_similarity(text, photo)
    checks["description_gain"] = round(gain, 3)
    return gain >= vision.DESCRIPTION_MIN


def kept_and_moving(photo: Path, probe: Path, clip: str, checks: dict) -> bool:
    moved = vision.motion(Path(clip))
    kept = vision.Subject().similarity(photo, probe)
    checks["motion"], checks["subject_match"] = round(moved, 3), round(kept, 3)
    return kept >= vision.SUBJECT_MIN and moved >= vision.MOTION_MIN


def not_a_hybrid(probe: Path, checks: dict) -> bool:
    subject = vision.Subject()
    margin = subject.text_similarity(vision.HYBRID_TEXT, probe) - subject.text_similarity(vision.HUMAN_TEXT, probe)
    checks["hybrid_margin"] = round(margin, 3)
    return margin < vision.HYBRID_MAX


def clip_checks(where: Path, r: Recipe, photo: Path, res, scan) -> dict[str, Any]:
    clip = (res.files.get("clip_save") or [None])[0]
    still = (res.files.get("keyframe_save") or [None])[0]
    out = Path(clip).parent if clip else where
    checks: dict[str, Any] = {"render_ok": res.ok, "error": res.error, "clip": bool(clip and Path(clip).stat().st_size > 10_000)}
    probe = probe_image(clip, still, out)
    seen = probe is not None and probe.exists()
    subject_ok, described_ok = True, False
    if seen:
        subject = vision.Subject()
        subject_ok = likeness_ok(r, photo, probe, out, subject, checks)
        described_ok = shows_description(r, photo, probe, subject, checks)
    if r.keep and clip and seen:
        described_ok = kept_and_moving(photo, probe, clip, checks) or described_ok
    if not r.keep and not r.keeps_subject() and seen:
        described_ok = not_a_hybrid(probe, checks) and described_ok
    entered_ok = True
    if checks["clip"]:
        checks["intruder"] = scan(Path(clip))
        entered_ok = checks["intruder"]["score"] < edge_check.INTRUDER_MIN
    checks["passed"] = bool(checks["render_ok"] and checks["clip"] and subject_ok and described_ok and entered_ok)
    checks["clip_path"] = str(clip) if clip else None
    return checks


def published_recipe(preset) -> Recipe | None:
    if getattr(preset, "family", None) != "built":
        return None
    data = json.loads((preset.folder / "preset.json").read_text())
    recipe = (data.get("source") or {}).get("recipe")
    return recipe_from(recipe) if recipe else None


@dataclass
class Checked:
    checks: dict[str, Any]
    result: Any
    attempts: list[dict[str, Any]]

    def warnings(self) -> list[str]:
        c = self.checks
        if c["passed"] or not c["clip"]:
            return []
        shown = c.get("description_gain", 1.0) >= vision.DESCRIPTION_MIN or (
            c.get("motion", 0.0) >= vision.MOTION_MIN and c.get("subject_match", 0.0) >= vision.SUBJECT_MIN)
        failed = [why for why, bad in (
            ("a hand or object came into the picture", c.get("intruder", {}).get("score", 0.0) >= edge_check.INTRUDER_MIN),
            ("the likeness was not kept", c.get("face_match") is not None and c["face_match"] < gate.IDENTITY_MIN),
            ("it may not show the description", not shown),
            ("it looks like a person-animal hybrid", c.get("hybrid_margin", -1.0) >= vision.HYBRID_MAX)) if bad]
        tries = f" after {len(self.attempts)} tries" if len(self.attempts) > 1 else ""
        return [f"Output check failed{tries}: {'; '.join(failed) or 'the checks did not pass'}. This is the cleanest take; "
                "try another seed or photo."]


def checked_run(preset, r: Recipe, photo: Path, where, *, client=None, seed: int = 42, rerolls: int = VIDEO_REROLLS,
                inputs: dict | None = None, on_progress=None, label: str | None = None, rendered=None) -> Checked:

    words = edge_check.intruder_words(r.subject, " ".join(filter(None, [r.becomes, r.motion])))
    detector = []

    def scan(clip: Path) -> dict:
        if not words:
            return {"score": 0.0, "frame": None, "label": None}
        detector or detector.append(edge_check.Intruders())
        return detector[0].scan(clip, words)

    attempts, results = [], []
    for k in range(1 + (rerolls if rendered is None else 0)):
        if rendered is None:
            s = (seed + 1009 * k) % 2**31
            compiled = compile_prompt(preset, {"photo": str(photo), **(inputs or {})}, {"seed": s})
            res = runner.run(compiled, where(k), client=client, on_progress=on_progress, label=label or f"build-{preset.id}")
        else:
            s, res = rendered
        checks = clip_checks(where(k), r, photo, res, scan) | {"seed": s, "attempt": k}
        attempts.append(checks)
        results.append(res)
        log.info("checked %s attempt %d seed %d: passed=%s intruder=%s", preset.id, k, s, checks["passed"], checks.get("intruder"))
        retry = (checks.get("intruder", {}).get("score", 0.0) >= edge_check.INTRUDER_MIN
                 or checks.get("hybrid_margin", -1.0) >= vision.HYBRID_MAX)
        if checks["passed"] or not checks["clip"] or not retry:
            break
    with_clip = [a for a in attempts if a["clip"]]
    best = next((a for a in attempts if a["passed"]), None) or min(
        with_clip, key=lambda a: a.get("intruder", {}).get("score", 0.0), default=attempts[-1])
    checks = dict(best)
    if len(attempts) > 1:
        checks["attempts"] = [{"seed": a["seed"], "passed": a["passed"], "intruder": a.get("intruder")} for a in attempts]
    return Checked(checks, results[best["attempt"]], attempts)
