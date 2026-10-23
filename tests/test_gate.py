from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from freefield import gate, vision

HOLE = (210, 347, 300, 455)
SKIN = (205, 150, 120)


def scene(tmp_path: Path, name: str, blobs=(), shift=(0, 0)) -> Path:
    rng = np.random.default_rng(0)
    img = np.zeros((1280, 720, 3), np.uint8)
    img[:] = (30, 70, 45)
    img += rng.integers(0, 12, img.shape, dtype=np.uint8)
    x, y, w, h = HOLE
    img[y - 40:y + h + 40, x - 40:x + w + 40] = (190, 150, 60)
    img[y - 40:y + h + 40:6, x - 40:x + w + 40] = (120, 90, 30)
    img[y:y + h, x:x + w] = (35, 28, 22)
    img[1050:] = (150, 95, 60)
    for bx, by, bw, bh in blobs:
        img[by:by + bh, bx:bx + bw] = SKIN
    img = np.roll(img, shift, axis=(0, 1))
    p = tmp_path / f"{name}.png"
    Image.fromarray(img).save(p)
    return p


def test_hands_inside_the_opening_pass(tmp_path):
    start = scene(tmp_path, "start")
    grip = scene(tmp_path, "grip", blobs=[(260, 700, 60, 50), (400, 700, 60, 50)])
    outside, _, _ = gate.outside_change(start, grip, HOLE)
    assert outside <= gate.OUTSIDE_MAX


def test_hands_on_the_wall_fail(tmp_path):
    start = scene(tmp_path, "start")
    grip = scene(tmp_path, "grip", blobs=[(60, 600, 80, 70), (580, 620, 80, 70)])
    outside, _, _ = gate.outside_change(start, grip, HOLE)
    assert outside > gate.OUTSIDE_MAX


@pytest.mark.parametrize("shift, moved", [((0, 0), False), ((24, 18), True)])
def test_a_moved_frame_is_caught(tmp_path, shift, moved):
    start = scene(tmp_path, "start")
    other = scene(tmp_path, "other", shift=shift)
    assert (gate.scene_drift(start, other, HOLE) > gate.DRIFT_MAX) is moved


@pytest.mark.parametrize("name, hole, bad", [
    ("r9-heldout-2-s2-bad", (210, 347, 300, 455), True),
    ("r8-heldout-2-s2-bad", (215, 406, 290, 375), True),
    ("r9-heldout-1-s1-good", (210, 347, 300, 455), False),
])
def test_real_grip_stills(name, hole, bad):
    d = Path(__file__).parent / "fixtures" / "gate"
    outside, _, _ = gate.outside_change(d / f"{name}-start.jpg", d / f"{name}-grip.jpg", hole)
    assert (outside > gate.OUTSIDE_MAX) is bad


@pytest.mark.parametrize("name, bad", [
    ("r10-heldout-2-s3-gilt-bad", True),
    ("r10-heldout-1-s2-gilt-bad", True),
    ("r10-heldout-1-s1-good", False),
])
def test_hands_on_the_gilt_are_caught(name, bad):
    d = Path(__file__).parent / "fixtures" / "gate"
    frame = gate.frame_change(d / f"{name}-start.jpg", d / f"{name}-grip.jpg", HOLE, border=78)
    assert (frame > gate.FRAME_MAX) is bad


@pytest.mark.parametrize("name, ok", [
    ("r11-heldout-2-s1-noms", False),
    ("r11-heldout-2-s3-ms", True),
])
def test_a_missing_moustache_is_caught(name, ok):
    faces = vision.Faces()
    hair = vision.FacialHair(faces)
    fixtures = Path(__file__).resolve().parents[1] / "eval" / "fixtures"
    hp, hs = hair.matches(fixtures / "heldout-2.png", Path(__file__).parent / "fixtures" / "gate" / f"{name}-start.jpg", HOLE)
    assert hp >= gate.HAIR_PHOTO
    assert (hs >= gate.HAIR_MIN) is ok


def test_fallback_prefers_the_right_face_over_clean_hands():
    clean_shaven = gate.GateResult(0.0009, 0.8, 1.0, 0.558, identity_start=0.242, frame=0.0144, hair_photo=0.996, hair_start=0.09)
    moustached = gate.GateResult(0.0041, 2.0, 1.0, 0.542, identity_start=0.553, frame=0.059, hair_photo=0.996, hair_start=0.98)
    assert not clean_shaven.ok and not moustached.ok
    assert moustached.rank() > clean_shaven.rank()


def test_the_rail_depth_comes_from_the_preset(tmp_path):
    start = scene(tmp_path, "start")
    x, y, w, h = HOLE
    grip = scene(tmp_path, "grip", blobs=[(x + 40, y + h + 5, 60, 55), (x + w - 100, y + h + 5, 60, 55)])
    assert gate.outside_change(start, grip, HOLE, rail=40)[0] > gate.OUTSIDE_MAX
    assert gate.outside_change(start, grip, HOLE, rail=70)[0] <= gate.OUTSIDE_MAX
