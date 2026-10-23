#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
YUNET = ROOT / ".runtime" / "ComfyUI" / "models" / "freefield_eval" / "face_detection_yunet_2023mar.onnx"
GREY = (128, 128, 128)


def blank_face(img: np.ndarray, box: tuple[float, float, float, float] | None = None) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    if box:
        x, y, fw, fh = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
        cv2.ellipse(mask, (x + fw // 2, y + fh // 2), (fw // 2, fh // 2), 0, 0, 360, 1.0, -1)
    else:
        det = cv2.FaceDetectorYN.create(str(YUNET), "", (w, h), 0.5)
        _, faces = det.detect(img)
        if faces is None or len(faces) == 0:
            raise ValueError("no face found in the artwork (pass --box x,y,w,h to blank a region by hand)")
        x, y, fw, fh = (int(v) for v in max(faces, key=lambda f: f[2] * f[3])[:4])
        cv2.ellipse(mask, (x + fw // 2, y + int(fh * 0.6)), (int(fw * 0.5), int(fh * 0.52)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, fw / 37))[..., None]
    return (img * (1 - mask) + np.full_like(img, GREY) * mask).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description="Make the keyframe reference for a museum preset: the artwork with its face painted out in flat grey.")
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--box", default="", help="face box as fractions x,y,w,h; skips face detection")
    a = ap.parse_args()
    box = None
    if a.box:
        try:
            box = tuple(float(v) for v in a.box.split(","))
        except ValueError:
            box = ()
        if len(box) != 4 or not all(0.0 <= v <= 1.0 for v in box) or box[0] + box[2] > 1 or box[1] + box[3] > 1:
            ap.error("--box must be four fractions x,y,w,h inside the image")
    img = cv2.imread(str(a.src))
    if img is None:
        print(f"cannot read image {a.src}", file=sys.stderr)
        return 2
    if box is None and not YUNET.is_file():
        print(f"missing {YUNET}; run scripts/setup.sh", file=sys.stderr)
        return 2
    try:
        out = blank_face(img, box)
    except ValueError as e:
        print(f"{a.src}: {e}", file=sys.stderr)
        return 1
    cv2.imwrite(str(a.dst), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("wrote", a.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
