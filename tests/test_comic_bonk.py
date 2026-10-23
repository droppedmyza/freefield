import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from freefield import comic_bonk as cb
from freefield import node_check, runner, vision

ROOT = Path(__file__).resolve().parents[1]


def scene():
    photo = np.full((1024, 768, 3), 200, np.float32)
    photo[700:, :] = (150, 110, 70)
    matte = np.zeros((1024, 768), np.float32)
    cv2.ellipse(matte, (384, 640), (170, 220), 0, 0, 360, 1.0, -1)
    photo[matte > 0.5] = (150, 90, 200)
    plate = np.full((1024, 768, 3), 200, np.float32)
    plate[700:, :] = (150, 110, 70)
    return photo, matte, plate


def test_the_loop_rests_on_the_untouched_photo_and_squashes_on_impact():
    photo, matte, plate = scene()
    frames = cb.compose(photo, matte, plate)
    assert len(frames) == cb.FRAMES and frames[0].shape == (cb.H, cb.W, 3)
    assert np.abs(frames[0].astype(int) - frames[-1]).mean() < 1.0
    purple = lambda f: ((f[..., 0] > 130) & (f[..., 2] > 180) & (f[..., 1] < 120)).sum()
    rest, hit = purple(frames[2]), purple(frames[cb.HIT + 3])
    rows = lambda f: np.nonzero(((f[..., 0] > 130) & (f[..., 2] > 180) & (f[..., 1] < 120)).any(axis=1))[0]
    assert len(rows(frames[cb.HIT + 3])) < 0.75 * len(rows(frames[2])) and hit > 0 and rest > 0


def test_the_bonk_lands_on_the_impact_frame():
    s = cb.sounds(cb.FRAMES / cb.FPS)
    hit = int(cb.HIT / cb.FPS * cb.AUDIO_RATE)
    assert np.abs(s[hit:hit + 2000]).max() > 3 * np.abs(s[hit - 3000:hit - 1000]).max()


class Clip:
    def __init__(self, top):
        self.top = top

    def text_similarity(self, text, path):
        return 0.3 if text == self.top else 0.2


@pytest.mark.parametrize("top, ok", [("a plush toy", True), ("a cartoon drawing", True), ("a photo of a real person", False),
                                     ("a photo of a real pet animal", False)])
def test_people_and_pets_are_refused(top, ok, tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(b"x")
    if ok:
        assert cb.guard(p, Clip(top))
    else:
        with pytest.raises(cb.ComicBonkError, match="not real people or pets"):
            cb.guard(p, Clip(top))


def test_a_run_renders_once_then_reuses_the_matte_and_plate(tmp_path, monkeypatch):

    photo, matte, plate = scene()
    src = tmp_path / "toy.png"
    cv2.imwrite(str(src), cv2.cvtColor(photo.astype(np.uint8), cv2.COLOR_RGB2BGR))
    calls = []

    def run(compiled, out_dir, **kw):
        calls.append(1)
        out_dir.mkdir(parents=True, exist_ok=True)
        m, p = out_dir / "bonk_matte_00001_.png", out_dir / "plate_00001_.png"
        cv2.imwrite(str(m), (matte * 255).astype(np.uint8))
        cv2.imwrite(str(p), plate.astype(np.uint8))
        return runner.RunResult("j", "succeeded", 1.0, {"matte_save": [m], "plate_save": [p]})

    monkeypatch.setattr(runner, "run", run)
    monkeypatch.setattr(vision, "Subject", lambda: Clip("a plush toy"))
    rec = cb.run(src, 5, tmp_path / "a", cache=tmp_path / "c")
    cb.run(src, 5, tmp_path / "b", text="pow", cache=tmp_path / "c")
    assert len(calls) == 1 and rec["text"] == "BONK!" and (tmp_path / "b" / "clip.mp4").stat().st_size > 10_000


def test_the_preset_and_graph_are_valid():
    from freefield.presets import load_presets

    presets, _ = load_presets()
    assert presets["comic-bonk"].pipeline == "comic_bonk"
    g = json.loads(cb.GRAPH.read_text())
    assert node_check.validate(g, {}, json.loads(node_check.SCHEMAS.read_text()), {"photo.image"}) == []
