from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import vision
from .media import video_frames_bgr

OUTSIDE_MAX = 0.001
IDENTITY_MIN = 0.30
PAINT_ID_MIN = 0.35
HAIR_PHOTO = 0.90
HAIR_MIN = 0.50
END_ID_MIN = 0.20
SHOT1_MAX = 0.25
FRAME_MAX = 0.035
DRIFT_MAX = 12.0


@dataclass
class GateResult:
    outside: float
    grip_drift: float
    out_drift: float
    identity_out: float | None
    identity_start: float | None = None
    frame: float | None = None
    hair_photo: float | None = None
    hair_start: float | None = None
    id_min: float = IDENTITY_MIN
    paint_min: float = PAINT_ID_MIN

    @property
    def checks(self) -> dict[str, bool]:
        return {"hands_inside": self.outside <= OUTSIDE_MAX, "grip_in_place": self.grip_drift <= DRIFT_MAX,
                "identity": self.identity_out is not None and self.identity_out >= self.id_min,
                "painting_identity": self.identity_start is not None and self.identity_start >= self.paint_min,
                "frame_untouched": self.frame is None or self.frame <= FRAME_MAX,
                "facial_hair": self.hair_photo is None or self.hair_photo < HAIR_PHOTO
                               or (self.hair_start is not None and self.hair_start >= HAIR_MIN)}

    @property
    def ok(self) -> bool:
        return all(self.checks.values())

    def rank(self) -> tuple:
        c = self.checks
        face = c["facial_hair"] and c["painting_identity"] and c["identity"]
        return (face, sum(c.values()), c["frame_untouched"] and c["hands_inside"], c["facial_hair"],
                (self.identity_start or 0.0) + (self.identity_out or 0.0), -(self.frame or 0.0) - self.outside)

    def as_dict(self) -> dict:
        return {"outside": round(self.outside, 4), "grip_drift_px": round(self.grip_drift, 1), "out_drift_px": round(self.out_drift, 1),
                "identity_out": None if self.identity_out is None else round(self.identity_out, 3),
                "identity_start": None if self.identity_start is None else round(self.identity_start, 3),
                "frame": None if self.frame is None else round(self.frame, 4),
                "hair_photo": None if self.hair_photo is None else round(self.hair_photo, 3),
                "hair_start": None if self.hair_start is None else round(self.hair_start, 3), **self.checks, "ok": self.ok}


def _rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB")).astype(np.int16)


def allowed_zone(shape: tuple[int, int], hole: tuple[int, int, int, int], rail: int = 40, side: int = 0,
                 edge: str = "bottom") -> np.ndarray:
    h, w = shape
    x, y, bw, bh = hole
    z = np.zeros((h, w), bool)
    z[y:y + bh, x:x + bw] = True
    if edge == "top":
        z[max(0, y - rail):y + 10, max(0, x - side):min(w, x + bw + side)] = True
    else:
        z[y + bh - 10:min(h, y + bh + rail), max(0, x - side):min(w, x + bw + side)] = True
    return z


def skin(rgb: np.ndarray) -> np.ndarray:
    import cv2

    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.int16)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((hue <= 22) | (hue >= 170)) & (sat >= 40) & (sat <= 200) & (val >= 60)


def align(start: np.ndarray, other: np.ndarray, hole: tuple[int, int, int, int]) -> np.ndarray:
    import cv2

    q = 4
    g1 = cv2.cvtColor(start.astype(np.uint8), cv2.COLOR_RGB2GRAY)[::q, ::q].astype(np.float32)
    g2 = cv2.cvtColor(other.astype(np.uint8), cv2.COLOR_RGB2GRAY)[::q, ::q].astype(np.float32)
    x, y, bw, bh = hole
    mask = np.full(g1.shape, 255, np.uint8)
    mask[max(0, (y - 20) // q):(y + bh + 300) // q, max(0, (x - 20) // q):(x + bw + 20) // q] = 0
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(g1, g2, warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), mask, 5)
    except cv2.error:
        return np.array([[np.nan] * 3] * 2)
    warp[:, 2] *= q
    return warp


def drift(warp: np.ndarray) -> tuple[float, float]:
    scale = float(np.sqrt(abs(np.linalg.det(warp[:, :2]))))
    return abs(scale - 1.0), float(np.hypot(*warp[:, 2]))


def outside_change(start: Path, grip: Path, hole: tuple[int, int, int, int], edge: str = "bottom",
                   rail: int = 40) -> tuple[float, float, float]:
    import cv2

    a, b = _rgb(start), _rgb(grip)
    if a.shape != b.shape:
        b = cv2.resize(b.astype(np.uint8), (a.shape[1], a.shape[0])).astype(np.int16)
    warp = align(a, b, hole)
    ds, dt = drift(warp) if np.isfinite(warp).all() else (1.0, 999.0)
    if np.isfinite(warp).all():
        b = cv2.warpAffine(b.astype(np.uint8), warp, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_REPLICATE).astype(np.int16)
    new = skin(b) & ~cv2.dilate(skin(a).astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    new = cv2.morphologyEx(new.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
    return float(new[~allowed_zone(new.shape, hole, rail=rail, edge=edge)].mean()), ds, dt


def shot1_intrusion(clip: Path, hole: tuple[int, int, int, int], view: tuple[int, int, int], frames: int = 41) -> float:
    import cv2

    grey = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.int16) for f in video_frames_bgr(clip, frames)]
    if len(grey) < 2:
        return 0.0
    k = grey[0].shape[1] / view[2]
    x, y, w, h = ((hole[0] - view[0]) * k, (hole[1] - view[1]) * k, hole[2] * k, hole[3] * k)
    cw, ch = int(0.3 * w), int(0.45 * h)
    boxes = [(int(x), int(y)), (int(x + w) - cw, int(y))]
    worst = 0.0
    for f in grey[1:]:
        for bx, by in boxes:
            a, b = grey[0][by:by + ch, bx:bx + cw], f[by:by + ch, bx:bx + cw]
            dark = a < 70
            worst = max(worst, float(((b - a > 40) & dark).sum() / max(1, int(dark.sum()))))
    return worst


def frame_change(start: Path, grip: Path, hole: tuple[int, int, int, int], border: int, rail: int = 60, edge: str = "bottom") -> float:
    import cv2

    a, b = _rgb(start), _rgb(grip)
    if a.shape != b.shape:
        b = cv2.resize(b.astype(np.uint8), (a.shape[1], a.shape[0])).astype(np.int16)
    warp = align(a, b, hole)
    if np.isfinite(warp).all():
        b = cv2.warpAffine(b.astype(np.uint8), warp, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_REPLICATE).astype(np.int16)
    blur = lambda im: cv2.GaussianBlur(im.astype(np.uint8), (7, 7), 0).astype(np.int16)
    d = np.abs(blur(a) - blur(b)).mean(axis=2)
    x, y, w, h = hole
    ring = np.zeros(d.shape, bool)
    ring[max(0, y - border):y + h + border, max(0, x - border):x + w + border] = True
    ring &= ~allowed_zone(d.shape, hole, rail=rail, edge=edge)
    return float((d[ring] > 30).mean())


def scene_drift(start: Path, other: Path, hole: tuple[int, int, int, int]) -> float:
    a, b = _rgb(start), _rgb(other)
    warp = align(a, b, hole)
    if not np.isfinite(warp).all():
        return 999.0
    ds, dt = drift(warp)
    x, y, w, h = hole
    return dt + ds * float(np.hypot(w, h))


def check(photo: Path, start: Path, grip: Path, out: Path, hole: tuple[int, int, int, int], faces: vision.Faces | None = None,
          edge: str = "bottom", frame_border: int = 0, hair: vision.FacialHair | None = None,
          id_min: float = IDENTITY_MIN, paint_min: float = PAINT_ID_MIN, rail: int | None = None) -> GateResult:
    faces = faces or vision.Faces()
    outside, _, _ = outside_change(start, grip, hole, edge, rail if rail is not None else 40)
    frame = frame_change(start, grip, hole, frame_border, rail=rail if rail is not None else 60, edge=edge) if frame_border else None
    hp, hs = hair.matches(photo, start, hole) if hair else (None, None)
    return GateResult(outside, scene_drift(start, grip, hole), scene_drift(start, out, hole), faces.similarity(photo, out),
                      faces.similarity(photo, start), frame, hp, hs, id_min, paint_min)


