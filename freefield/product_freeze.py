from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
from pathlib import Path

import numpy as np

from . import runner, vision
from .errors import EffectError
from .media import FPS, H, W, blend, clean_text, fit_cover, read_rgb, title_layer, whoosh, write_video, write_wav
from .runner import CompiledRun

log = logging.getLogger(__name__)

PKG = Path(__file__).resolve().parent
GRAPH = PKG / "graphs" / "product_freeze.json"
DEPTH_MODEL = vision.MODELS / "depth_anything_v2_small"
CACHE = PKG.parent / "runs" / "_cache" / "product-freeze"
ORBIT_SECONDS = 4.4
CARD_SECONDS = 1.6
BLEND = 10
AMP = 0.09
ROLL = 1.6
PUSH = 0.10
REROLLS = 2
DESCRIBED = "a product frozen mid-air in a splash of water"


class ProductFreezeError(EffectError):
    pass


def depth(img: np.ndarray) -> np.ndarray:
    import cv2
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).eval()
    with torch.no_grad():
        out = model(**proc(images=Image.fromarray(img), return_tensors="pt")).predicted_depth
    d = torch.nn.functional.interpolate(out[None], size=img.shape[:2], mode="bicubic")[0, 0].numpy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    return cv2.GaussianBlur(d.astype(np.float32), (0, 0), 2)


def orbit(img: np.ndarray, d: np.ndarray, n: int) -> list[np.ndarray]:
    import cv2

    h, w = d.shape
    focus = float(np.median(d[h // 3: 2 * h // 3, w // 3: 2 * w // 3]))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    par = (d - focus) * AMP * w
    frames = []
    for i in range(n):
        u = i / max(1, n - 1)
        a = math.sin((u - 0.5) * math.pi)
        z = 1.0 + PUSH * (0.5 - 0.5 * math.cos(math.pi * u)) + 0.04
        r = math.radians(ROLL * a)
        cx, cy = (xs - w / 2) / z, (ys - h / 2) / z
        mx = cx * math.cos(r) - cy * math.sin(r) + w / 2 - a * par
        my = cx * math.sin(r) + cy * math.cos(r) + h / 2 - 0.08 * par * (1 - abs(a))
        frames.append(cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
    return frames


def end_card(photo: np.ndarray) -> np.ndarray:
    import cv2

    bg = cv2.GaussianBlur(fit_cover(photo), (0, 0), 28).astype(np.float32) * 0.55
    ph, pw = photo.shape[:2]
    s = min(W * 0.86 / pw, H * 0.54 / ph)
    small = cv2.resize(photo, (round(pw * s), round(ph * s)), interpolation=cv2.INTER_AREA)
    y, x = int(H * 0.34), (W - small.shape[1]) // 2
    bg[y:y + small.shape[0], x:x + small.shape[1]] = small
    return bg.astype(np.uint8)


def compose(still: np.ndarray, d: np.ndarray, photo: np.ndarray, text: str = "") -> tuple[list[np.ndarray], list[float]]:
    n_orbit, n_card = round(ORBIT_SECONDS * FPS), round(CARD_SECONDS * FPS)
    import cv2

    d = d if d.shape == (H, W) else cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
    frames = orbit(fit_cover(still), d, n_orbit)
    card = end_card(photo)
    for i in range(BLEND):
        t = (i + 1) / (BLEND + 1)
        frames[-BLEND + i] = (frames[-BLEND + i] * (1 - t) + card * t).astype(np.uint8)
    frames += [card] * n_card
    text = clean_text(text)
    if text:
        top = title_layer(text, (W, H), y=0.08, px=64)
        big = title_layer(text, (W, H), y=0.20, px=80)
        for i in range(len(frames)):
            if i < n_orbit - BLEND:
                frames[i] = blend(frames[i], top, min(1.0, max(0.0, (i - 12) / 10)))
            else:
                frames[i] = blend(frames[i], big, min(1.0, (i - (n_orbit - BLEND)) / 8))
    return frames, [0.35, (n_orbit - BLEND / 2) / FPS]


def cache_key(photo: Path, seed: int) -> str:
    h = hashlib.sha256(photo.read_bytes() + GRAPH.read_bytes()).hexdigest()[:16]
    return f"{h}-{seed}"


def _still(photo: Path, seed: int, work: Path, client, rec: dict, on_stage=None) -> Path:
    subject = vision.Subject()
    best = None
    for k in range(1 + REROLLS):
        s = seed + 1009 * k
        g = json.loads(GRAPH.read_text())
        g["freeze_noise"]["inputs"]["noise_seed"] = s
        if on_stage:
            on_stage("Freezing the splash" + (f" (try {k + 1})" if k else ""))
        res = runner.run(CompiledRun(g, [("product", "image", photo)], ["freeze_save"]), work / f"still-{k}", client=client,
                         label="product-freeze")
        if not res.ok:
            raise ProductFreezeError(f"the image model failed: {res.error or res.status}")
        p = Path(res.files["freeze_save"][0])
        kept = subject.similarity(photo, p)
        gain = subject.text_similarity(DESCRIBED, p) - subject.text_similarity(DESCRIBED, photo)
        ok = kept >= vision.SUBJECT_MIN and gain >= vision.DESCRIPTION_MIN
        rec.setdefault("still_attempts", []).append({"seed": s, "product_kept": round(kept, 3), "splash_gain": round(gain, 3), "ok": ok})
        log.info("product freeze still seed %d: kept %.3f gain %.3f", s, kept, gain)
        if best is None or (ok, kept) > best[0]:
            best = ((ok, kept), p)
        if ok:
            break
    if not best[0][0]:
        rec["warnings"].append(f"Unverified: the frozen splash did not keep the product clearly after {1 + REROLLS} tries; check it by eye")
    return best[1]


def run(photo: Path, seed: int, out_dir: Path, *, text: str = "", client=None, on_stage=None, cache: Path | None = None) -> dict:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    rec: dict = {"source": photo.name, "seed": seed, "text": clean_text(text), "warnings": []}
    folder = (cache or CACHE) / cache_key(photo, seed)
    hit = (folder / "still.png").exists() and (folder / "depth.npy").exists()
    if hit:
        rec |= json.loads((folder / "still.json").read_text())
        rec["warnings"] = list(rec.get("warnings", []))
    else:
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(_still(photo, seed, out_dir, client, rec, on_stage), folder / "still.png")
        if on_stage:
            on_stage("Measuring depth")
        img = cv2.cvtColor(cv2.imread(str(folder / "still.png")), cv2.COLOR_BGR2RGB)
        np.save(folder / "depth.npy", depth(fit_cover(img)))
        (folder / "still.json").write_text(json.dumps({k: rec[k] for k in ("still_attempts", "warnings") if k in rec}))
    rec["cached"] = hit
    still = cv2.cvtColor(cv2.imread(str(folder / "still.png")), cv2.COLOR_BGR2RGB)
    d = np.load(folder / "depth.npy")
    if on_stage:
        on_stage("Moving the camera around the frozen moment")
    frames, cuts = compose(still, d, fit_cover(read_rgb(photo)), rec["text"])
    wav = write_wav(whoosh(len(frames) / FPS + 0.3, cuts, seed), out_dir / "whoosh.wav")
    write_video(frames, out_dir / "clip.mp4", wav)
    shutil.copy(folder / "still.png", out_dir / "still.png")
    rec["frames"] = len(frames)
    (out_dir / "productfreeze.json").write_text(json.dumps(rec, indent=2))
    return rec
