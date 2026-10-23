#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from freefield.plate import extract, wall_mask  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--border", type=int)
    ap.add_argument("--below", type=int)
    a = ap.parse_args()
    p = extract(Image.open(a.src))
    if p is None:
        sys.exit("no magenta opening found")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p.plate.save(out / "plate.png")
    p.hole.save(out / "plate_hole.png")
    if a.border is not None:
        wall_mask(p.plate.size, p.box, a.border, a.below if a.below is not None else a.border).save(out / "wall_mask.png")
    x, y, w, h = p.xywh
    print(json.dumps({"x": x, "y": y, "w": w, "h": h, "plate": list(p.plate.size), "coverage": round(p.coverage, 3)}))


if __name__ == "__main__":
    main()
