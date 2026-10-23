from __future__ import annotations

import json
import logging
import math
import subprocess
from pathlib import Path

import numpy as np

from . import runner, vision
from .errors import EffectError
from .media import FPS, IMAGE_EXT, H, W, blend, clean_text, fit_cover, read_rgb, title_layer, whoosh, write_video, write_wav
from .runner import CompiledRun

log = logging.getLogger(__name__)

PKG = Path(__file__).resolve().parent
GRAPH = PKG / "graphs" / "earth_zoom.json"
STILLS = ("globe", "satellite", "drone")
ZOOMS = ((1.0, 7.0, 1.3), (1.0, 3.0, 1.3), (1.0, 2.2, 1.1), (1.3, 1.0, 0.8))
CUT = 0.22
HOLD = 1.0
MAX_CLIP_SECONDS = 20.0
STILL_REROLLS = 2
SATELLITE_COPY_MAX = 0.88
LABELS = {"globe": "the planet Earth seen from space", "satellite": "a satellite photo of land seen from high above",
          "drone": "an aerial drone photo of rooftops and streets"}
OTHER = ("a street photo taken at eye level", "a close-up photo of an object")


class EarthZoomError(EffectError):
    pass


def clouds(seed: int, size: tuple[int, int] = (W, H)) -> np.ndarray:
    import cv2

    w, h = size
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for octave in range(6):
        g = 3 * 2 ** octave
        noise = rng.random((g * h // w + 2, g + 2)).astype(np.float32)
        acc += amp * cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.55
    return np.clip((acc / total - 0.42) * 3.2, 0, 1)


def _zoom(img: np.ndarray, m: float, blur: float = 0.0) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]

    def warp(mm: float) -> np.ndarray:
        M = np.float32([[mm, 0, w / 2 - mm * w / 2], [0, mm, h / 2 - mm * h / 2]])
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    if blur <= 0.002:
        return warp(m)
    n = 5
    return sum(warp(m * (1 + blur * i / n)) for i in range(n)) / n


def cover(u: float, first: bool, last: bool) -> float:
    a = 0.0
    if not first and u < CUT:
        a = 1 - u / CUT
    if not last and u > 1 - CUT:
        a = max(a, (u - (1 - CUT)) / CUT)
    return a


def segment_frames(stills: list[np.ndarray], seed: int = 0) -> tuple[list[np.ndarray], list[float]]:
    import cv2

    frames, cuts, t = [], [], 0.0
    sky = np.float32([250, 248, 245])
    for k, (img, (m0, m1, seconds)) in enumerate(zip(stills, ZOOMS)):
        img = img.astype(np.float32)
        n = round(seconds * FPS)
        layer = clouds(seed + k, (img.shape[1], img.shape[0]))
        h, w = img.shape[:2]
        for i in range(n):
            u = i / max(1, n - 1)
            m = m0 * (m1 / m0) ** u
            speed = abs(math.log(m1 / m0)) / n
            f = _zoom(img, m, blur=speed * 0.9)
            a = cover(u, k == 0, k == len(stills) - 1)
            if a > 0:
                cm = 1 + 2.5 * (u if u > 0.5 else 1 - u)
                cz = cv2.warpAffine(layer, np.float32([[cm, 0, w / 2 - cm * w / 2], [0, cm, h / 2 - cm * h / 2]]), (w, h),
                                    borderMode=cv2.BORDER_REFLECT)
                c = np.clip(cz * 1.4 + a * 1.2 - 0.5, 0, 1) * min(1.0, a * 1.6)
                c = np.maximum(c, np.clip((a - 0.75) * 4, 0, 1))[..., None]
                f = f * (1 - c) + sky * c
            frames.append(np.clip(f, 0, 255).astype(np.uint8))
        t += n / FPS
        if k < len(stills) - 1:
            cuts.append(t)
    return frames, cuts


def add_title(frames: list[np.ndarray], text: str, start: int, end: int, fade: int = 8) -> list[np.ndarray]:
    if not text.strip() or end <= start:
        return frames
    layer = title_layer(text, (frames[0].shape[1], frames[0].shape[0]))
    out = list(frames)
    for i in range(max(0, start), min(len(frames), end)):
        out[i] = blend(frames[i], layer, min(1.0, (i - start + 1) / fade, (end - i) / fade))
    return out


def read_clip(path: Path, max_seconds: float = MAX_CLIP_SECONDS) -> list[np.ndarray]:
    cmd = ["ffmpeg", "-loglevel", "error", "-i", str(path), "-t", f"{max_seconds:.2f}", "-vf",
           f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, timeout=300).stdout
    n = len(raw) // (W * H * 3)
    if n == 0:
        raise EarthZoomError("could not read any frames from the clip")
    arr = np.frombuffer(raw[: n * W * H * 3], np.uint8).reshape(n, H, W, 3)
    return list(arr)


def read_photo(path: Path) -> np.ndarray:
    return fit_cover(read_rgb(path))


def compose(stills: dict[str, np.ndarray], frame0: np.ndarray, clip: list[np.ndarray] | None, caption: str = "",
            seed: int = 0) -> tuple[list[np.ndarray], list[float], int]:
    seq = [fit_cover(stills[k]) for k in STILLS] + [frame0]
    intro, cuts = segment_frames(seq, seed)
    n_intro = len(intro)
    start = round(sum(z[2] for z in ZOOMS[:1]) * FPS)
    if clip:
        frames = intro + list(clip)
        frames = add_title(frames, caption, start, n_intro + round(0.8 * FPS))
        return frames, cuts, n_intro
    n_hold = round(HOLD * FPS)
    hold = [np.clip(_zoom(frame0.astype(np.float32), 1 + 0.05 * math.sin(math.pi * i / max(1, n_hold - 1))), 0, 255).astype(np.uint8)
            for i in range(n_hold)]
    frames = intro + hold + intro[::-1]
    frames = add_title(frames, caption, start, n_intro + n_hold + round(0.6 * FPS))
    back = (n_intro + n_hold) / FPS
    cuts = cuts + [back + (n_intro / FPS - c) for c in reversed(cuts)]
    return frames, cuts, n_intro


def still_checks(paths: dict[str, Path], subject=None) -> dict[str, dict]:
    subject = subject or vision.Subject()
    out = {}
    for k, p in paths.items():
        own = subject.text_similarity(LABELS[k], p)
        rivals = [LABELS[j] for j in LABELS if j != k and not (k == "satellite" and j == "drone")] + list(OTHER)
        best = max(subject.text_similarity(t, p) for t in rivals)
        out[k] = {"label": round(own, 3), "best_other": round(best, 3), "ok": own > best}
    if "satellite" in paths and "drone" in paths:
        same = subject.similarity(paths["drone"], paths["satellite"])
        out["satellite"]["like_drone"] = round(same, 3)
        out["satellite"]["ok"] = out["satellite"]["ok"] and same < SATELLITE_COPY_MAX
    return out


def _stills(frame0_png: Path, seed: int, work: Path, client, rec: dict, on_stage=None) -> dict[str, Path]:
    best = None
    for k in range(1 + STILL_REROLLS):
        s = seed + 1009 * k
        g = json.loads(GRAPH.read_text())
        for p in STILLS:
            g[f"{p}_noise"]["inputs"]["noise_seed"] = s
        if on_stage:
            on_stage("Painting the views from above" + (f" (try {k + 1})" if k else ""))
        res = runner.run(CompiledRun(g, [("frame0", "image", frame0_png)], [f"{p}_save" for p in STILLS]),
                         work / f"stills-{k}", client=client, label="earth-zoom")
        if not res.ok:
            raise EarthZoomError(f"the image model failed: {res.error or res.status}")
        paths = {p: Path(res.files[f"{p}_save"][0]) for p in STILLS}
        checks = still_checks(paths)
        good = sum(c["ok"] for c in checks.values())
        rec.setdefault("still_attempts", []).append({"seed": s, "checks": checks})
        log.info("earth zoom stills seed %d: %s", s, checks)
        if best is None or good > best[0]:
            best = (good, paths, checks)
        if good == len(STILLS):
            break
    rec["stills"] = {k: str(v) for k, v in best[1].items()}
    rec["still_checks"] = best[2]
    bad = [k for k, c in best[2].items() if not c["ok"]]
    if bad:
        rec["warnings"].append("Unverified: the " + ", ".join(bad) + " view did not read as intended after "
                               f"{1 + STILL_REROLLS} tries; check the clip by eye")
    return best[1]


def run(src: Path, seed: int, out_dir: Path, *, text: str = "", client=None, on_stage=None) -> dict:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    caption = clean_text(text)
    rec: dict = {"source": src.name, "seed": seed, "caption": caption, "warnings": []}
    is_photo = src.suffix.lower() in IMAGE_EXT
    if on_stage:
        on_stage("Reading your " + ("photo" if is_photo else "clip"))
    clip = None if is_photo else read_clip(src)
    frame0 = read_photo(src) if is_photo else clip[0]
    frame0_png = out_dir / "frame0.png"
    cv2.imwrite(str(frame0_png), cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR))
    paths = _stills(frame0_png, seed, out_dir, client, rec, on_stage)
    stills = {k: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for k, p in paths.items()}
    if on_stage:
        on_stage("Flying down through the clouds")
    frames, cuts, n_intro = compose(stills, frame0, clip, caption, seed)
    wav = write_wav(whoosh(len(frames) / FPS + 0.3, cuts, seed), out_dir / "whoosh.wav")
    if on_stage:
        on_stage("Saving")
    write_video(frames, out_dir / "clip.mp4", wav, None if is_photo else src, n_intro / FPS, MAX_CLIP_SECONDS)
    proof = np.hstack([cv2.resize(fit_cover(stills[k]), (180, 320)) for k in STILLS] + [cv2.resize(frame0, (180, 320))])
    cv2.imwrite(str(out_dir / "proof.png"), cv2.cvtColor(proof, cv2.COLOR_RGB2BGR))
    rec |= {"frames": len(frames), "intro_frames": n_intro, "cuts": [round(c, 3) for c in cuts], "mode": "photo loop" if is_photo else "clip"}
    (out_dir / "earthzoom.json").write_text(json.dumps(rec, indent=2))
    return rec
