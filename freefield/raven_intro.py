from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import edge_check, runner
from .errors import EffectError
from .media import FPS, H, W, blend, clean_text, decode_frames, fit_cover, read_rgb, title_layer, whoosh, write_video, write_wav
from .presets import compile_prompt, parse_preset

PKG = Path(__file__).resolve().parent
PRESET = PKG / "presets" / "raven-poof"
REROLLS = 2
SOURCE_SECONDS = 0.5
TAIL_SECONDS = 1.6


class RavenIntroError(EffectError):
    pass


def exposure_match(frames: list[np.ndarray], photo: np.ndarray) -> tuple[list[np.ndarray], dict]:
    import cv2

    ref = cv2.cvtColor(cv2.resize(photo, (frames[0].shape[1], frames[0].shape[0]), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_RGB2LAB).astype(np.float32)
    first = cv2.cvtColor(frames[0], cv2.COLOR_RGB2LAB).astype(np.float32)
    rm, rs = ref.reshape(-1, 3).mean(0), ref.reshape(-1, 3).std(0)
    fm, fs = first.reshape(-1, 3).mean(0), first.reshape(-1, 3).std(0)
    gain = np.clip(rs / np.maximum(fs, 1e-3), 0.8, 1.25)
    out = []
    for f in frames:
        lab = cv2.cvtColor(f, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab = (lab - fm) * gain + rm
        out.append(cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))
    after = cv2.cvtColor(out[0], cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3).mean(0)
    return out, {"lightness_before": round(float(fm[0] - rm[0]), 2), "lightness_after": round(float(after[0] - rm[0]), 2)}


def motion_energy(frames: list[np.ndarray]) -> list[float]:
    import cv2

    small = [cv2.resize(f, (96, 166), interpolation=cv2.INTER_AREA).astype(np.float32) for f in frames]
    return [float(np.abs(b - a).mean()) for a, b in zip(small, small[1:])]


def burst_frame(frames: list[np.ndarray]) -> int:
    energy = motion_energy(frames)
    return int(np.argmax(energy)) + 1 if energy else 0


def settle_frame(frames: list[np.ndarray], start: int) -> int:
    energy = motion_energy(frames)
    peak = max(energy) if energy else 0.0
    for i in range(start, len(energy)):
        if energy[i] < peak * 0.25:
            return i + 1
    return len(frames) - 1


def compose(photo: np.ndarray, clip_frames: list[np.ndarray], title: str = "") -> tuple[list[np.ndarray], dict]:
    import cv2

    matched, exposure = exposure_match(clip_frames, photo)
    big = [cv2.resize(f, (W, H), interpolation=cv2.INTER_LANCZOS4) for f in matched]
    source = fit_cover(photo)
    n_src = round(SOURCE_SECONDS * FPS)
    crossfade = 6
    frames = [source] * n_src
    for i in range(crossfade):
        t = (i + 1) / (crossfade + 1)
        frames.append((source * (1 - t) + big[i] * t).astype(np.uint8))
    frames += big[crossfade:]
    frames += [frames[-1]] * round(TAIL_SECONDS * FPS)
    burst = n_src + burst_frame(matched)
    settle = n_src + settle_frame(matched, burst - n_src)
    title = clean_text(title)
    if title:
        layer = title_layer(title, (W, H), y=0.44, px=72)
        for i in range(settle, len(frames)):
            frames[i] = blend(frames[i], layer, min(1.0, (i - settle + 1) / 10))
    return frames, {"exposure": exposure, "burst_frame": burst, "title_from": settle}


def _render(photo: Path, seed: int, work: Path, client, rec: dict, on_stage=None) -> Path:
    preset = parse_preset(json.loads((PRESET / "preset.json").read_text()), PRESET)
    words = edge_check.intruder_words("object", "it bursts into a flock of black ravens")
    detector = edge_check.Intruders()
    best = None
    for k in range(1 + REROLLS):
        s = seed + 1009 * k
        if on_stage:
            on_stage("Bursting into ravens" + (f" (try {k + 1})" if k else ""))
        res = runner.run(compile_prompt(preset, {"photo": str(photo)}, {"seed": s}), work / f"render-{k}", client=client,
                         label="raven-intro")
        if not res.ok:
            raise RavenIntroError(f"the video render failed: {res.error or res.status}")
        clip = Path(res.files["clip_save"][0])
        found = detector.scan(clip, words)
        rec.setdefault("attempts", []).append({"seed": s, "intruder": found})
        if best is None or found["score"] < best[0]:
            best = (found["score"], clip)
        if found["score"] < edge_check.INTRUDER_MIN:
            break
    if best[0] >= edge_check.INTRUDER_MIN:
        rec["warnings"].append(f"Unverified: something came in at the frame edge on all {1 + REROLLS} tries; check it by eye")
    return best[1]


def run(photo: Path, seed: int, out_dir: Path, *, text: str = "", client=None, on_stage=None, clip: Path | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rec: dict = {"source": photo.name, "seed": seed, "title": text, "warnings": []}
    raw = clip or _render(photo, seed, out_dir, client, rec, on_stage)
    if on_stage:
        on_stage("Matching the exposure to your photo")
    frames, info = compose(read_rgb(photo), decode_frames(raw), text)
    cuts = [info["burst_frame"] / FPS - 0.1]
    wav = write_wav(whoosh(len(frames) / FPS + 0.3, cuts, seed), out_dir / "whoosh.wav")
    write_video(frames, out_dir / "clip.mp4", wav)
    rec |= info | {"raw_clip": str(raw), "frames": len(frames)}
    (out_dir / "ravenintro.json").write_text(json.dumps(rec, indent=2))
    return rec
