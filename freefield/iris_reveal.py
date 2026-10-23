from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .errors import EffectError
from .media import FPS, H, W, blend, clean_text, read_rgb, title_layer, whoosh, write_video, write_wav
from .vision import YUNET

EYE_SPAN = 0.42
FILL = 0.62
MAX_UPSCALE = 2.6
MIN_SHARPNESS = 120.0
HOLD_START, ZOOM_IN, HOLD_EYE, ZOOM_OUT, HOLD_END = 0.5, 1.7, 1.2, 1.5, 0.5


class IrisRevealError(EffectError):
    pass


@dataclass
class Eye:
    x: float
    y: float
    width: float
    sharpness: float
    side: str
    iris_found: bool


def sharpness(gray: np.ndarray) -> float:
    import cv2

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def find_eyes(bgr: np.ndarray) -> list[Eye]:
    import cv2

    det = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), 0.6)
    h, w = bgr.shape[:2]
    k = min(1.0, 1024 / max(h, w))
    small = cv2.resize(bgr, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else bgr
    det.setInputSize((small.shape[1], small.shape[0]))
    _, faces = det.detect(small)
    if faces is None or len(faces) == 0:
        raise IrisRevealError("no face found; use a sharp, front-facing selfie")
    face = max(faces, key=lambda f: f[2] * f[3])
    a, b = face[4:6] / k, face[6:8] / k
    span = float(np.linalg.norm(a - b))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    eyes = []
    for c, side in ((a, "right"), (b, "left")):
        s = max(4, int(span * 0.22))
        x0, y0 = int(c[0]) - s, int(c[1]) - s
        patch = gray[max(0, y0):y0 + 2 * s, max(0, x0):x0 + 2 * s]
        if patch.size == 0:
            continue
        cx, cy, found = float(c[0]), float(c[1]), False
        rx, ry = int(span * 0.16), int(span * 0.09)
        ix0, iy0 = max(0, int(cx) - rx), max(0, int(cy) - ry)
        inner = gray[iy0:int(cy) + ry, ix0:int(cx) + rx]
        if inner.size:
            r = max(3, int(span * 0.09))
            disc = np.zeros((4 * r + 1, 4 * r + 1), np.float32)
            cv2.circle(disc, (2 * r, 2 * r), 2 * r, -1.0, -1)
            cv2.circle(disc, (2 * r, 2 * r), r, 0.0, -1)
            disc[disc < 0] = -1.0 / max(1.0, float((disc < 0).sum()))
            core = np.zeros_like(disc)
            cv2.circle(core, (2 * r, 2 * r), r, 1.0, -1)
            kernel = disc + core / core.sum()
            score = cv2.filter2D(inner.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REPLICATE)
            lo, _, (px, py), _ = cv2.minMaxLoc(score)
            if lo < -8:
                cx, cy, found = ix0 + float(px), iy0 + float(py), True
        eyes.append(Eye(cx, cy, span * EYE_SPAN, sharpness(patch), side, found))
    if not eyes:
        raise IrisRevealError("the eyes are outside the photo")
    return eyes


def pick(eyes: list[Eye]) -> Eye:
    eye = max(eyes, key=lambda e: e.sharpness)
    final_w = eye.width / FILL
    if W / final_w > MAX_UPSCALE:
        need = math.ceil(W * FILL / MAX_UPSCALE / EYE_SPAN)
        raise IrisRevealError(f"the eye is only {eye.width:.0f} px wide; a close-up needs about {W * FILL / MAX_UPSCALE:.0f} px "
                        f"(eyes about {need} px apart). Use the full-resolution photo from your phone")
    if eye.sharpness < MIN_SHARPNESS:
        raise IrisRevealError(f"the eye is too soft for a close-up (sharpness {eye.sharpness:.0f} < {MIN_SHARPNESS:.0f}); "
                        "use a sharper selfie in good light")
    return eye


def ease(u: float) -> float:
    return u * u * (3 - 2 * u)


def path(eye: Eye, size: tuple[int, int]) -> list[tuple[float, float, float]]:
    w, h = size
    full = min(w, h * W / H)
    final = eye.width / FILL
    start = (w / 2, h / 2)
    keys = []
    for seconds, kind in ((HOLD_START, "start"), (ZOOM_IN, "in"), (HOLD_EYE, "eye"), (ZOOM_OUT, "out"), (HOLD_END, "end")):
        n = round(seconds * FPS)
        for i in range(n):
            u = i / max(1, n - 1)
            if kind == "start" or kind == "end":
                t = 0.0
            elif kind == "in":
                t = ease(u)
            elif kind == "eye":
                t = 1.0
            else:
                t = ease(1 - u)
            cw = full * (final / full) ** t
            if kind == "eye":
                cw *= 1 - 0.04 * math.sin(math.pi * u)
            cx = start[0] + (eye.x - start[0]) * min(1.0, t * 1.15)
            cy = start[1] + (eye.y - start[1]) * min(1.0, t * 1.15)
            keys.append((cx, cy, cw))
    return keys


def frame(img: np.ndarray, cx: float, cy: float, cw: float) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    ch = cw * H / W
    cx = min(max(cx, cw / 2), w - cw / 2)
    cy = min(max(cy, ch / 2), h - ch / 2)
    s = W / cw
    M = np.float32([[s, 0, W / 2 - s * cx], [0, s, H / 2 - s * cy]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)


def compose(rgb: np.ndarray, eye: Eye, caption: str = "") -> tuple[list[np.ndarray], list[float]]:
    keys = path(eye, (rgb.shape[1], rgb.shape[0]))
    frames = [frame(rgb, *k) for k in keys]
    n0 = round(HOLD_START * FPS)
    n1 = n0 + round(ZOOM_IN * FPS)
    n2 = n1 + round(HOLD_EYE * FPS)
    caption = clean_text(caption)
    if caption:
        layer = title_layer(caption, (W, H), y=0.82, px=60)
        for i in range(n1 - 6, n2 + 6):
            frames[i] = blend(frames[i], layer, min(1.0, (i - (n1 - 6)) / 8, (n2 + 6 - i) / 8))
    return frames, [(n1 - 4) / FPS, (n2 + 10) / FPS]


def run(photo: Path, seed: int, out_dir: Path, *, text: str = "", on_stage=None) -> dict:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = read_rgb(photo)
    if on_stage:
        on_stage("Finding the clearer eye")
    eyes = find_eyes(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    eye = pick(eyes)
    if on_stage:
        on_stage("Zooming into your eye")
    frames, cuts = compose(rgb, eye, text)
    wav = write_wav(whoosh(len(frames) / FPS + 0.3, cuts, seed) * 0.7, out_dir / "whoosh.wav")
    write_video(frames, out_dir / "clip.mp4", wav)
    rec = {"source": photo.name, "photo_size": [rgb.shape[1], rgb.shape[0]], "eye": asdict(eye),
           "eyes": [asdict(e) for e in eyes], "upscale_at_eye": round(W / (eye.width / FILL), 2), "frames": len(frames),
           "warnings": [] if eye.iris_found else ["The iris was not located exactly; the zoom centres on the eye landmark"]}
    (out_dir / "irisreveal.json").write_text(json.dumps(rec, indent=2))
    return rec
