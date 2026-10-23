from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter


@dataclass
class Plate:
    plate: Image.Image
    hole: Image.Image
    box: tuple[int, int, int, int]
    coverage: float

    @property
    def xywh(self) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = self.box
        return x0, y0, x1 - x0, y1 - y0


def fill_small_gaps(m: np.ndarray, max_gap: int) -> np.ndarray:
    out = m.copy()
    for arr in (out, out.T):
        for line in arr:
            idx = np.flatnonzero(line)
            if len(idx) < 2:
                continue
            for gi in np.flatnonzero(np.diff(idx) > 1):
                a, b = idx[gi], idx[gi + 1]
                if b - a - 1 < max_gap:
                    line[a + 1:b] = True
    return out


def extract(scene: Image.Image, fill: tuple[int, int, int] = (38, 30, 24)) -> Plate | None:
    im = np.asarray(scene.convert("RGB")).astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    magenta = (r > 90) & (b > 60) & (g < 60) & (r - g > 70) & (b - g > 50)
    if magenta.sum() < 0.02 * magenta.size:
        return None
    rows = np.where(magenta.mean(axis=1) > 0.10)[0]
    cols = np.where(magenta.mean(axis=0) > 0.10)[0]
    if not len(rows) or not len(cols):
        return None
    y0, y1, x0, x1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    magenta = fill_small_gaps(magenta, 40)
    mask = Image.fromarray((magenta * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))
    box = (int(x0), int(y0), int(x1), int(y1))
    hole = mask.crop(box)
    plate = Image.composite(Image.new("RGB", scene.size, tuple(fill)), scene.convert("RGB"), mask)
    return Plate(plate, hole, box, float(np.asarray(hole).mean() / 255))


def wall_mask(size: tuple[int, int], box: tuple[int, int, int, int], border: int, below: int, feather: int = 6) -> Image.Image:
    w, h = size
    x0, y0, x1, y1 = box
    m = Image.new("L", (w, h), 255)
    m.paste(0, (max(0, x0 - border), max(0, y0 - border), min(w, x1 + border), min(h, y1 + below)))
    return m.filter(ImageFilter.GaussianBlur(feather))
