from __future__ import annotations

from pathlib import Path

import numpy as np

from .media import video_frames_bgr
from .vision import MODELS

INTRUDER_MIN = 0.20
INTRUDER_AREA = 0.015
INTRUDER_STRONG = 0.35
INTRUDER_STRONG_AREA = 0.05
INTRUDER_CHANGED = 0.4
INTRUDER_EDGE = 0.03
INTRUDER_STEP = 6
INTRUDER_HOLD = 2
INTRUDER_SEEN = 0.6
HAND_WORDS = ("a human hand", "fingers")
TOOL_WORDS = ("a pen", "a pencil", "a knife", "scissors", "a tool", "a paintbrush")


def _iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def changed_share(first: np.ndarray, frame: np.ndarray, box: tuple) -> float:
    import cv2

    x0, y0, x1, y1 = (int(round(v)) for v in box)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return 0.0
    a = cv2.GaussianBlur(first[y0:y1, x0:x1].astype(np.float32), (5, 5), 0)
    b = cv2.GaussianBlur(frame[y0:y1, x0:x1].astype(np.float32), (5, 5), 0)
    return float((np.abs(a - b).max(axis=2) > 40).mean())


def candidates(dets: list, base: list, width: int, height: int, seen=None, frame: int = 0, changed=None) -> list:
    edge_x, edge_y = INTRUDER_EDGE * width, INTRUDER_EDGE * height
    out = []
    for label, score, box in dets:
        if score < INTRUDER_MIN or any(_iou(box, b[2]) > 0.3 for b in base):
            continue
        x0, y0, x1, y1 = max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3])
        area = (x1 - x0) * (y1 - y0) / (width * height)
        if area < INTRUDER_AREA:
            continue
        if not (x0 <= edge_x or y0 <= edge_y or x1 >= width - edge_x or y1 >= height - edge_y):
            continue
        if seen is not None and seen(frame, (x0, y0, x1, y1)):
            continue
        strong = (score >= INTRUDER_STRONG and area >= INTRUDER_STRONG_AREA and changed is not None
                  and changed(frame, (x0, y0, x1, y1)) >= INTRUDER_CHANGED)
        out.append((float(score), label, strong))
    return out


def intrusion(dets: dict[int, list], width: int, height: int, seen=None, changed=None) -> dict:
    frames = sorted(dets)
    if len(frames) < 2:
        return {"score": 0.0, "frame": None, "label": None}
    base = dets[frames[0]]
    hits, strong = [], []
    for f in frames[1:]:
        found = candidates(dets[f], base, width, height, seen, f, changed)
        best = max(found, default=(0.0, None, False))
        hits.append((f, best[0], best[1]))
        strong += [(f, sc, lab) for sc, lab, st in found if st]
    run = max((min(hits[i:i + INTRUDER_HOLD], key=lambda h: h[1]) for i in range(len(hits) - INTRUDER_HOLD + 1)),
              key=lambda h: h[1], default=(None, 0.0, None))
    run = max([run] + strong, key=lambda h: h[1])
    return {"score": round(run[1], 3), "frame": run[0] if run[1] > 0 else None, "label": run[2]}


def already_there(first: np.ndarray, frame: np.ndarray, box: tuple, search: float = 0.12, min_ncc: float = INTRUDER_SEEN) -> bool:
    import cv2

    h, w = first.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return False
    patch = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    m = int(search * max(w, h))
    sx0, sy0, sx1, sy1 = max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m)
    area = cv2.cvtColor(first[sy0:sy1, sx0:sx1], cv2.COLOR_RGB2GRAY)
    if area.shape[0] < patch.shape[0] or area.shape[1] < patch.shape[1] or float(patch.std()) < 2:
        return False
    return float(cv2.matchTemplate(area, patch, cv2.TM_CCOEFF_NORMED).max()) >= min_ncc


def intruder_words(subject: str, description: str) -> list[str]:
    words = list(TOOL_WORDS) + ([] if subject == "person" else list(HAND_WORDS))
    text = description.lower()
    return [w for w in words if w.split()[-1] not in text and w.split()[-1].rstrip("s") not in text]


class Intruders:
    def __init__(self) -> None:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        self.proc = Owlv2Processor.from_pretrained(MODELS / "owlv2")
        self.model = Owlv2ForObjectDetection.from_pretrained(MODELS / "owlv2").eval()

    def detect(self, image, words: list[str]) -> list:
        side = max(image.size)
        inputs = self.proc(text=[words], images=image, return_tensors="pt")
        with self.torch.no_grad():
            out = self.model(**inputs)
        r = self.proc.post_process_grounded_object_detection(out, threshold=0.08, target_sizes=self.torch.tensor([[side, side]]))[0]
        return [(words[int(l)], float(s), [float(v) for v in b]) for s, l, b in zip(r["scores"], r["labels"], r["boxes"])]

    def scan(self, clip: Path, words: list[str]) -> dict:
        import cv2
        from PIL import Image

        if not words:
            return {"score": 0.0, "frame": None, "label": None}
        frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in video_frames_bgr(clip)]
        if not frames:
            return {"score": 0.0, "frame": None, "label": None}
        height, width = frames[0].shape[:2]
        seen = lambda i, box: already_there(frames[0], frames[i], box)
        changed = lambda i, box: changed_share(frames[0], frames[i], box)
        detect = lambda i: self.detect(Image.fromarray(frames[i]), words)
        dets = {i: detect(i) for i in range(0, len(frames), INTRUDER_STEP)}
        for i in sorted(dets)[1:]:
            if candidates(dets[i], dets[0], width, height, seen, i, changed):
                for j in (i - INTRUDER_STEP // 2, i + INTRUDER_STEP // 2):
                    if 0 < j < len(frames) and j not in dets:
                        dets[j] = detect(j)
        return intrusion(dets, width, height, seen, changed)
