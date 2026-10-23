import importlib.util
from pathlib import Path

from freefield import vision

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_takes", ROOT / "scripts" / "check_takes.py")
check_takes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_takes)


def test_a_short_take_is_told_to_record_longer():

    fixes = check_takes.check(ROOT / "tests" / "fixtures" / "video" / "standin-fresh-1.mp4", vision.Faces())
    assert any("record 6-10 s" in x for x in fixes)
    assert not any("lean in further" in x for x in fixes)


def test_a_take_shot_too_close_is_told_to_step_back(tmp_path):
    import cv2

    src = ROOT / "tests" / "fixtures" / "video" / "standin-fresh-1.mp4"
    cap = cv2.VideoCapture(str(src))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        frames.append(cv2.resize(f[int(h * 0.15):int(h * 0.55), int(w * 0.2):int(w * 0.8)], (w, h)))
    out = tmp_path / "close.mp4"
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 16, (frames[0].shape[1], frames[0].shape[0]))
    for f in frames:
        vw.write(f)
    vw.release()

    fixes = check_takes.check(out, vision.Faces())
    assert any("too close" in x for x in fixes)


def test_a_take_whose_first_frame_has_no_face_is_rejected_like_the_renderer_does(tmp_path):
    import cv2
    import numpy as np


    src = ROOT / "tests" / "fixtures" / "video" / "standin-fresh-1.mp4"
    cap = cv2.VideoCapture(str(src))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    out = tmp_path / "late.mp4"
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 16, (w, h))
    for f in [np.zeros_like(frames[0])] * 4 + frames + frames[:40]:
        vw.write(f)
    vw.release()
    fixes = check_takes.check(out, vision.Faces())
    assert any("very first frame" in x for x in fixes)


def test_the_renderer_rejects_a_first_frame_without_a_face(tmp_path):
    import numpy as np
    import pytest

    from freefield import clip_breakout

    black = np.zeros((clip_breakout.CLIP_SIZE[1], clip_breakout.CLIP_SIZE[0], 3), np.uint8)
    matte = np.zeros(black.shape[:2], np.float32)
    with pytest.raises(clip_breakout.ClipBreakoutError, match="no face"):
        clip_breakout._place([black], [matte], tmp_path, {"warnings": []}, vision.Faces())
