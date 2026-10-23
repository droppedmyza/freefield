#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from freefield import vision

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm"}


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:stream_side_data=rotation:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, timeout=30)
    data = json.loads(out.stdout or "{}")
    st = (data.get("streams") or [{}])[0]
    rot = 0
    for sd in st.get("side_data_list", []):
        rot = int(sd.get("rotation", 0))
    w, h = st.get("width", 0), st.get("height", 0)
    if abs(rot) in (90, 270):
        w, h = h, w
    return {"seconds": float(data.get("format", {}).get("duration", 0) or 0), "width": w, "height": h}


def faces_over_time(path: Path, faces: vision.Faces, step_s: float = 0.25) -> list[tuple[float, tuple | None]]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    every = max(1, int(round(fps * step_s)))
    out, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % every == 0:
            h, w = f.shape[:2]
            faces.det.setInputSize((w, h))
            _, det = faces.det.detect(f)
            box = None if det is None or len(det) == 0 else tuple(float(v) for v in max(det, key=lambda d: d[2] * d[3])[:4])
            out.append((i / fps, box, h))
        i += 1
    cap.release()
    return out


def check(path: Path, faces: vision.Faces) -> list[str]:
    fixes = []
    p = probe(path)
    if not 6 <= p["seconds"] <= 12:
        fixes.append(f"length {p['seconds']:.1f} s: record 6-10 s")
    if p["height"] <= p["width"]:
        fixes.append("hold the phone upright (vertical video)")
    track = faces_over_time(path, faces)
    seen = [(t, b, h) for t, b, h in track if b is not None]
    if not track or track[0][1] is None:
        fixes.append("no face in the very first frame: be facing the phone, in the picture, when recording starts")
        return fixes
    t0, b0, h0 = seen[0]
    share0 = b0[3] / h0
    if share0 > 0.15:
        fixes.append(f"too close at the start (face {share0:.0%} of the picture height): prop the phone further away, about 2 m")
    if share0 < 0.06:
        fixes.append(f"too far at the start (face {share0:.0%} of the picture height): come a little closer")
    peak = max(b[3] for _, b, _ in seen)
    if peak / b0[3] < 1.8:
        fixes.append(f"lean in further: the face only grows {peak / b0[3]:.1f}x (aim for 2x or more)")
    still = [b for t, b, _ in seen if t <= 1.0]
    if len(still) >= 2:
        cx = [b[0] + b[2] / 2 for b in still]
        size = [b[3] for b in still]
        if (max(cx) - min(cx)) / b0[2] > 0.25 or (max(size) - min(size)) / b0[3] > 0.15:
            fixes.append("hold still for the first second before leaning in")
    return fixes


def main() -> int:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "eval" / "real"
    takes = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXT) if folder.exists() else []
    if not takes:
        print(f"no takes in {folder}")
        return 1
    faces = vision.Faces()
    ok = 0
    for take in takes:
        fixes = check(take, faces)
        ok += not fixes
        print(f"{take.name}: " + ("ready" if not fixes else "redo: " + "; ".join(fixes)))
    print(f"{ok} of {len(takes)} ready")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
