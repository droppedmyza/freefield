from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

from . import gate, runner, vision
from .runner import CompiledRun

log = logging.getLogger(__name__)

OUTPUT_TYPES = {"SaveImage", "SaveVideo", "PreviewImage"}
KEYFRAMES = ["start_save", "grip_save", "breakout_save"]
STEP = 7919


def only_outputs(compiled: CompiledRun, keep: list[str]) -> CompiledRun:
    g = copy.deepcopy(compiled.graph)
    for nid in [n for n, v in g.items() if v["class_type"] in OUTPUT_TYPES and n not in keep]:
        del g[nid]
    return CompiledRun(g, [f for f in compiled.files if f[0] in g], list(keep))


def with_keyframe_seeds(compiled: CompiledRun, seed: int) -> CompiledRun:
    g = copy.deepcopy(compiled.graph)
    for n in ("painting_noise", "likeness_noise", "grip_noise", "breakout_noise"):
        g[n]["inputs"]["noise_seed"] = seed
    return CompiledRun(g, compiled.files, compiled.outputs)


SHOT1_SAMPLERS = ("shot1_sample_high", "shot1_sample")


def with_shot1_seed(compiled: CompiledRun, seed: int) -> CompiledRun:
    g = copy.deepcopy(compiled.graph)
    for n in SHOT1_SAMPLERS:
        g[n]["inputs"]["noise_seed"] = seed
    return CompiledRun(g, compiled.files, compiled.outputs)


def video_checks(result: runner.RunResult, compiled: CompiledRun, spec: dict, photo: Path, faces) -> dict:
    clips = (result.files.get("clip_save") or []) if result.ok else []
    if not clips or not Path(clips[0]).exists():
        return {}
    clip, out = Path(clips[0]), {}
    if faces is not None:
        end = faces.end_identity(photo, clip)
        if end is not None:
            out["end_identity"] = round(end, 3)
    if spec.get("shot1_corners"):
        crop = compiled.graph["start_view_crop"]["inputs"]
        frames = int(compiled.graph["shot1_latent"]["inputs"].get("length", 41))
        out["shot1_intrusion"] = round(gate.shot1_intrusion(clip, tuple(spec["hole"]), (crop["x"], crop["y"], crop["width"]),
                                                            frames), 3)
    return out


def video_ok(c: dict) -> bool:
    return c.get("end_identity", 1.0) >= gate.END_ID_MIN and c.get("shot1_intrusion", 0.0) <= gate.SHOT1_MAX


def check_keyframes(spec: dict, photo: Path, files: dict[str, Path], faces, hair) -> gate.GateResult:
    return gate.check(photo, files["start_save"], files["grip_save"], files["breakout_save"], tuple(spec["hole"]), faces,
                      spec.get("grip_edge", "bottom"), spec.get("frame_border", 0), hair,
                      spec.get("identity_min", gate.IDENTITY_MIN), spec.get("painting_identity_min", gate.PAINT_ID_MIN),
                      spec.get("rail"))


def pick_keyframes(compiled: CompiledRun, spec: dict, photo: Path, out_dir: Path, faces, *, client, on_progress, label):
    hair = vision.FacialHair(faces) if spec.get("facial_hair") else None
    base = int(compiled.graph["grip_noise"]["inputs"]["noise_seed"])
    attempts, best = [], None
    for k in range(int(spec.get("rerolls", 3)) + 1):
        seed = base + STEP * k
        trial = with_keyframe_seeds(compiled, seed)
        res = runner.run(only_outputs(trial, KEYFRAMES), out_dir / f"keyframes-{k}", client=client, on_progress=on_progress,
                         label=label)
        if not res.ok:
            return res, None, attempts
        verdict = check_keyframes(spec, photo, {n: res.files[n][0] for n in KEYFRAMES}, faces, hair)
        attempts.append({"attempt": k, "keyframe_seed": seed, **verdict.as_dict()})
        log.info("gate attempt %d (seed %d): %s", k, seed, verdict.as_dict())
        if best is None or verdict.rank() > best[0].rank():
            best = (verdict, trial, seed)
        if verdict.ok:
            break
    return None, best, attempts


def video_rank(checks: dict) -> tuple:
    return video_ok(checks), -checks.get("shot1_intrusion", 0.0), checks.get("end_identity", 1.0)


def render_video(chosen: CompiledRun, spec: dict, photo: Path, out_dir: Path, faces, *, client, on_progress, label):
    video_rerolls = int(spec.get("video_rerolls", 0))
    shot1_base = int(chosen.graph[SHOT1_SAMPLERS[0]]["inputs"]["noise_seed"]) if video_rerolls else 0
    result, checks, tries = None, {}, []
    for j in range(video_rerolls + 1):
        attempt = with_shot1_seed(chosen, shot1_base + STEP * j) if video_rerolls else chosen
        res = runner.run(attempt, out_dir if j == 0 else out_dir / f"video-{j}", client=client, on_progress=on_progress,
                         label=label)
        if not res.ok:
            if result is None:
                return res, {}, tries
            break
        c = video_checks(res, chosen, spec, photo, faces)
        tries.append({"attempt": j, "shot1_seed": shot1_base + STEP * j if video_rerolls else None, **c})
        log.info("video attempt %d: %s", j, c)
        if result is None or video_rank(c) > video_rank(checks):
            result, checks = res, c
        if video_ok(c):
            break
    return result, checks, tries


def unverified_warnings(verdict: gate.GateResult, keyframe_attempts: int, checks: dict, video_attempts: int) -> list[str]:
    warnings = []
    if checks.get("end_identity", 1.0) < gate.END_ID_MIN:
        warnings.append(f"Unverified: your face is not clearly there at the end of the clip (match {checks['end_identity']:.2f}, "
                        f"needs {gate.END_ID_MIN:.2f}); something may cover it or it may have changed. Try another seed.")
    if checks.get("shot1_intrusion", 0.0) > gate.SHOT1_MAX:
        warnings.append(f"Unverified: in the first shot an arm or hand comes in from the top of the painting (score "
                        f"{checks['shot1_intrusion']:.2f}, max {gate.SHOT1_MAX:.2f}) in all {video_attempts} video attempts. "
                        "Try another seed.")
    if not verdict.ok:
        failed = [k for k, v in verdict.checks.items() if not v]
        warnings.append(f"Unverified: none of {keyframe_attempts} keyframe attempts passed the quality gate ({', '.join(failed)}); "
                        "this clip was rendered from the best one and may show it. Try another seed.")
    return warnings


def run_gated(compiled: CompiledRun, spec: dict, photo: Path, out_dir: Path, *, client=None, on_progress=None,
              label: str = "") -> runner.RunResult:
    faces = vision.Faces()
    opts = {"client": client, "on_progress": on_progress, "label": label}
    failed, best, attempts = pick_keyframes(compiled, spec, photo, out_dir, faces, **opts)
    if failed is not None:
        return failed
    verdict, chosen, seed = best
    record = {"hole": list(spec["hole"]), "chosen_seed": seed, "passed": verdict.ok, "attempts": attempts}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate.json").write_text(json.dumps(record, indent=2) + "\n")
    result, checks, tries = render_video(chosen, spec, photo, out_dir, faces, **opts)
    if not checks and not result.ok:
        return result
    record.update(checks)
    if int(spec.get("video_rerolls", 0)):
        record["video_attempts"] = tries
    (out_dir / "gate.json").write_text(json.dumps(record, indent=2) + "\n")
    for msg in unverified_warnings(verdict, len(attempts), checks, len(tries)):
        result.warnings.append(msg)
        log.warning("%s (seed %d)", msg, seed)
    return result
