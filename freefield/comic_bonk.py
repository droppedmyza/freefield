from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import runner, vision
from .errors import EffectError
from .media import AUDIO_RATE, FONTS, FPS, H, W, clean_text, fit_cover, read_rgb, write_video, write_wav
from .runner import CompiledRun

PKG = Path(__file__).resolve().parent
GRAPH = PKG / "graphs" / "comic_bonk.json"
ASSETS = PKG / "presets" / "comic-bonk"
FONT = FONTS / "LuckiestGuy-Regular.ttf"
FRAMES = 72
WIND, HIT, LIFT, CALM = 14, 24, 40, 62
SQUASH_Y, SQUASH_X = 0.62, 1.28
ALLOWED = ("a plush toy", "a mascot costume or a figurine", "a cartoon drawing", "a toy character")
REFUSED = ("a photo of a real person", "a photo of a real pet animal")


class ComicBonkError(EffectError):
    pass


def guard(photo: Path, subject=None) -> dict:
    subject = subject or vision.Subject()
    scores = {t: subject.text_similarity(t, photo) for t in ALLOWED + REFUSED}
    top = max(scores, key=scores.get)
    if top in REFUSED:
        raise ComicBonkError("Comic Bonk is for mascots, toys and cartoons only, not real people or pets")
    return {k: round(v, 3) for k, v in scores.items()}


def fit_box(shape: tuple[int, int], box: tuple[int, int, int, int]) -> tuple[float, float, float]:
    h, w = shape
    s = H / h
    cx = (box[0] + box[2]) / 2 * s
    ox = min(max(cx - W / 2, 0), max(0, w * s - W))
    return s, ox, 0.0


def to_canvas(img: np.ndarray, s: float, ox: float, interp=None) -> np.ndarray:
    import cv2

    M = np.float32([[s, 0, -ox], [0, s, 0]])
    return cv2.warpAffine(img, M, (W, H), flags=interp or cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def spring(i: int) -> tuple[float, float]:
    if i < HIT:
        return 1.0, 1.0
    t = (i - HIT) / FPS
    if i < HIT + 2:
        k = (i - HIT + 1) / 2
        return 1 + (SQUASH_X - 1) * k, 1 - (1 - SQUASH_Y) * k
    if i < HIT + 6:
        return SQUASH_X, SQUASH_Y
    t -= 6 / FPS
    a = math.exp(-5.5 * t) * math.cos(2 * math.pi * 2.6 * t)
    if i >= CALM:
        a *= max(0.0, 1 - (i - CALM) / (FRAMES - CALM))
    return 1 + (SQUASH_X - 1) * a, 1 - (1 - SQUASH_Y) * a


def mallet_angle(i: int) -> float | None:
    if i < WIND or i > LIFT + 14:
        return None
    if i < HIT:
        u = (i - WIND) / (HIT - WIND)
        return -70 + 70 * u ** 2.2
    if i < LIFT:
        return 0.0
    u = (i - LIFT) / 14
    return -80 * u ** 1.5


def starburst(size: int, points: int = 12, seed: int = 0) -> np.ndarray:
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = size / 2
    pts = []
    for k in range(points * 2):
        r = c * (0.98 if k % 2 == 0 else 0.62 + 0.08 * rng.random())
        a = math.pi * k / points
        pts.append((c + r * math.cos(a), c + r * math.sin(a)))
    d.polygon(pts, fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=max(3, size // 60))
    return np.asarray(img, np.float32) / 255.0


def lettering(text: str, size: int) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    px = size // 3
    while px > 10:
        f = ImageFont.truetype(str(FONT), px)
        b = d.textbbox((0, 0), text, font=f, stroke_width=px // 8)
        if b[2] - b[0] < size * 0.78:
            break
        px -= 4
    d.text(((size - (b[2] - b[0])) / 2 - b[0], (size - (b[3] - b[1])) / 2 - b[1]), text, font=f, fill=(255, 214, 0, 255),
           stroke_width=px // 8, stroke_fill=(0, 0, 0, 255))
    return np.asarray(img.rotate(-12, resample=Image.BICUBIC), np.float32) / 255.0


def star(size: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    c, pts = size / 2, []
    for k in range(10):
        r = c * (0.95 if k % 2 == 0 else 0.42)
        a = -math.pi / 2 + math.pi * k / 5
        pts.append((c + r * math.cos(a), c + r * math.sin(a)))
    ImageDraw.Draw(img).polygon(pts, fill=(255, 222, 40, 255), outline=(0, 0, 0, 255), width=max(2, size // 12))
    return np.asarray(img, np.float32) / 255.0


def paste(frame: np.ndarray, rgba: np.ndarray, cx: float, cy: float, scale: float = 1.0, alpha: float = 1.0) -> np.ndarray:
    import cv2

    if scale <= 0 or alpha <= 0:
        return frame
    h, w = rgba.shape[:2]
    M = np.float32([[scale, 0, cx - scale * w / 2], [0, scale, cy - scale * h / 2]])
    layer = cv2.warpAffine(rgba, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    a = layer[..., 3:] * alpha
    return frame * (1 - a) + layer[..., :3] * 255 * a


def mallet_sprite() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import cv2

    img = cv2.cvtColor(cv2.imread(str(ASSETS / "mallet.png")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    m = cv2.imread(str(ASSETS / "mallet-matte.png"), 0).astype(np.float32) / 255
    rgba = np.dstack([img, m])
    meta = json.loads((ASSETS / "mallet.json").read_text())
    return rgba, np.array(meta["head"], np.float32), np.array(meta["end"], np.float32)


def place_mallet(frame: np.ndarray, sprite, angle: float, impact: tuple[float, float], length: float) -> np.ndarray:
    import cv2

    rgba, head, end = sprite
    v = head - end
    s = length / float(np.linalg.norm(v))
    base = math.degrees(math.atan2(v[1], v[0]))
    target = 180.0 - angle
    rot = math.radians(target - base)
    pivot = np.array([impact[0] + length, impact[1]], np.float32)
    c, sn = math.cos(rot) * s, math.sin(rot) * s
    M = np.float32([[c, -sn, 0], [sn, c, 0]])
    M[:, 2] = pivot - M[:, :2] @ end
    layer = cv2.warpAffine(rgba, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    a = layer[..., 3:]
    return frame * (1 - a) + layer[..., :3] * 255 * a


@dataclass
class Stage:
    background: np.ndarray
    mascot: np.ndarray
    alpha: np.ndarray
    anchor: tuple[float, float]
    left: float
    height: float
    width: float


def clean_plate_like(photo: np.ndarray, plate: np.ndarray, hard: np.ndarray) -> np.ndarray:
    import cv2

    band = (cv2.dilate(hard, np.ones((61, 61), np.uint8)) - cv2.dilate(hard, np.ones((9, 9), np.uint8))).astype(bool)
    if band.any():
        for ch in range(3):
            ps, qs = photo[..., ch][band], plate[..., ch][band]
            plate[..., ch] = (plate[..., ch] - qs.mean()) * (ps.std() / max(1e-3, qs.std())) + ps.mean()
    return plate


def stage(photo: np.ndarray, matte: np.ndarray, plate: np.ndarray) -> Stage:
    import cv2

    hh, ww = photo.shape[:2]
    ys, xs = np.nonzero(matte > 0.5)
    if len(xs) == 0:
        raise ComicBonkError("no mascot was found in the photo")
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    s, ox, _ = fit_box((hh, ww), box)
    plate = cv2.resize(plate, (ww, hh), interpolation=cv2.INTER_AREA) if plate.shape[:2] != (hh, ww) else plate
    hard = (matte > 0.3).astype(np.uint8)
    plate = clean_plate_like(photo, plate, hard)
    grown = cv2.GaussianBlur(cv2.dilate(hard.astype(np.float32), np.ones((9, 9), np.uint8)), (0, 0), 3)
    feet = int(np.nonzero(hard)[0].max())
    grown[max(0, feet - int(0.04 * hh)):] *= np.linspace(1, 0, hh - max(0, feet - int(0.04 * hh)))[:, None] ** 4
    grown = grown[..., None]
    x0, y0, x1, y1 = box[0] * s - ox, box[1] * s, box[2] * s - ox, box[3] * s
    return Stage(background=to_canvas((photo * (1 - grown) + plate * grown).astype(np.float32), s, ox),
                 mascot=to_canvas(photo.astype(np.float32), s, ox), alpha=to_canvas(matte.astype(np.float32), s, ox)[..., None],
                 anchor=((x0 + x1) / 2, y1), left=x0, height=y1 - y0, width=x1 - x0)


def squashed(st: Stage, sx: float, sy: float) -> np.ndarray:
    import cv2

    if abs(sx - 1) < 1e-3 and abs(sy - 1) < 1e-3:
        return st.mascot.copy()
    M = np.float32([[sx, 0, st.anchor[0] * (1 - sx)], [0, sy, st.anchor[1] * (1 - sy)]])
    f = cv2.warpAffine(st.mascot, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    a = cv2.warpAffine(st.alpha, M, (W, H), flags=cv2.INTER_LINEAR)[..., None]
    return st.background * (1 - a) + f * a


def draw_frame(i: int, st: Stage, sprite, burst: np.ndarray, word: np.ndarray, little: np.ndarray) -> np.ndarray:
    sx, sy = spring(i)
    frame = squashed(st, sx, sy)
    width, top = st.width, st.anchor[1] - st.height * sy
    ang = mallet_angle(i)
    if ang is not None:
        frame = place_mallet(frame, sprite, ang, (st.anchor[0], top - width * 0.13), width * 1.05)
    if HIT <= i < HIT + 2:
        frame = frame * 0.7 + 255 * 0.3
    if HIT <= i < LIFT + 12:
        k = i - HIT
        pop = min(1.15, 0.5 + 0.22 * k) if k < 4 else 1.0 + 0.15 * math.exp(-(k - 3) / 2)
        fade = 1.0 if i < LIFT else max(0.0, 1 - (i - LIFT) / 12)
        cx, cy = max(width * 0.4, st.left + width * 0.05), max(width * 0.45, top - width * 0.3)
        frame = paste(frame, burst, cx, cy, pop * 0.72, fade)
        frame = paste(frame, word, cx, cy, pop * 0.68, fade)
    if HIT + 6 <= i < CALM + 6:
        fade = min(1.0, (i - HIT - 6) / 4, (CALM + 6 - i) / 6)
        for k in range(3):
            t = (i - HIT) / FPS * 2.4 + k * 2 * math.pi / 3
            frame = paste(frame, little, st.anchor[0] + math.cos(t) * width * 0.38,
                          top - width * 0.12 + math.sin(t) * width * 0.08, 1.0, fade)
    return np.clip(frame, 0, 255).astype(np.uint8)


def compose(photo: np.ndarray, matte: np.ndarray, plate: np.ndarray, text: str = "BONK!") -> list[np.ndarray]:
    st = stage(photo, matte, plate)
    sprite = mallet_sprite()
    burst, word, little = starburst(int(st.width * 0.9)), lettering(text, int(st.width * 0.9)), star(int(st.width * 0.12))
    return [draw_frame(i, st, sprite, burst, word, little) for i in range(FRAMES)]


def sounds(seconds: float) -> np.ndarray:
    n = int(seconds * AUDIO_RATE)
    out = np.zeros(n, np.float32)
    rng = np.random.default_rng(0)
    t0 = HIT / FPS
    k = np.arange(int(0.35 * AUDIO_RATE)) / AUDIO_RATE
    bonk = (np.sin(2 * np.pi * (190 * k - 120 * k ** 2)) * np.exp(-k * 14) + 0.5 * rng.standard_normal(len(k)) * np.exp(-k * 90))
    i = int(t0 * AUDIO_RATE)
    out[i:i + len(bonk)] += bonk[: n - i]
    k = np.arange(int(0.9 * AUDIO_RATE)) / AUDIO_RATE
    f = 330 + 140 * np.sin(2 * np.pi * 7 * k) * np.exp(-k * 2.5)
    boing = np.sin(2 * np.pi * np.cumsum(f) / AUDIO_RATE) * np.exp(-k * 3.2) * 0.45
    i = int((t0 + 0.08) * AUDIO_RATE)
    out[i:i + len(boing)] += boing[: n - i]
    k = np.arange(int(0.3 * AUDIO_RATE)) / AUDIO_RATE
    swish = rng.standard_normal(len(k)) * np.sin(np.pi * k / 0.3) ** 2 * 0.25
    i = int((WIND + 2) / FPS * AUDIO_RATE)
    out[i:i + len(swish)] += swish[: n - i]
    return out / max(1e-6, float(np.abs(out).max())) * 0.8


def cache_key(photo: Path, seed: int) -> str:
    import hashlib

    return f"{hashlib.sha256(photo.read_bytes()).hexdigest()[:16]}-{seed}"


def _assets(photo: Path, seed: int, work: Path, client, on_stage=None) -> tuple[Path, Path]:
    g = json.loads(GRAPH.read_text())
    g["plate_noise"]["inputs"]["noise_seed"] = seed
    if on_stage:
        on_stage("Cutting out your mascot and clearing the desk behind it")
    res = runner.run(CompiledRun(g, [("photo", "image", photo)], ["matte_save", "plate_save"]), work / "assets", client=client,
                     label="comic-bonk")
    if not res.ok:
        raise ComicBonkError(f"the matte or plate render failed: {res.error or res.status}")
    return Path(res.files["matte_save"][0]), Path(res.files["plate_save"][0])


def run(photo: Path, seed: int, out_dir: Path, *, text: str = "BONK!", client=None, on_stage=None, cache: Path | None = None) -> dict:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    rec: dict = {"source": photo.name, "seed": seed, "warnings": []}
    src = read_rgb(photo)
    if src.shape[:2] != (1024, 768):
        fitted = out_dir / "photo-3x4.png"
        cv2.imwrite(str(fitted), cv2.cvtColor(fit_cover(src, (768, 1024)), cv2.COLOR_RGB2BGR))
        photo = fitted
    if on_stage:
        on_stage("Checking it is a mascot, toy or cartoon")
    rec["guard"] = guard(photo)
    folder = (cache or PKG.parent / "runs" / "_cache" / "comic-bonk") / cache_key(photo, seed)
    if not ((folder / "matte.png").exists() and (folder / "plate.png").exists()):
        folder.mkdir(parents=True, exist_ok=True)
        m, p = _assets(photo, seed, out_dir, client, on_stage)
        shutil.copy(m, folder / "matte.png")
        shutil.copy(p, folder / "plate.png")
    rgb = read_rgb(photo).astype(np.float32)
    matte = cv2.imread(str(folder / "matte.png"), 0).astype(np.float32) / 255
    matte = cv2.resize(matte, (rgb.shape[1], rgb.shape[0])) if matte.shape != rgb.shape[:2] else matte
    plate = cv2.cvtColor(cv2.imread(str(folder / "plate.png")), cv2.COLOR_BGR2RGB).astype(np.float32)
    if on_stage:
        on_stage("Bonk!")
    text = clean_text(text or "BONK!", 12) or "BONK!"
    frames = compose(rgb, matte, plate, text.upper())
    wav = write_wav(sounds(FRAMES / FPS + 0.05), out_dir / "bonk.wav")
    write_video(frames, out_dir / "clip.mp4", wav)
    rec |= {"frames": len(frames), "text": text.upper()}
    (out_dir / "comicbonk.json").write_text(json.dumps(rec, indent=2))
    return rec
