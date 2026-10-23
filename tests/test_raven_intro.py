from pathlib import Path

import numpy as np

from freefield import edge_check, runner
from freefield import raven_intro as ri

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "freefield" / "presets" / "raven-poof" / "preview.mp4"
PHOTO = ROOT / "eval" / "objects" / "figurine-a.png"


def test_the_clip_is_matched_to_the_photos_exposure():
    photo = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(PHOTO).convert("RGB"))
    frames = ri.decode_frames(CLIP)
    dark = [np.clip(f.astype(int) - 30, 0, 255).astype(np.uint8) for f in frames]
    _, info = ri.exposure_match(dark, photo)
    assert abs(info["lightness_before"]) > 8 and abs(info["lightness_after"]) < 1.5


def test_the_intro_starts_on_the_photo_and_titles_the_empty_shelf(tmp_path):
    rec = ri.run(PHOTO, 1, tmp_path / "o", text="NIGHT SHIFT", clip=CLIP)
    assert rec["title_from"] > rec["burst_frame"] > round(ri.SOURCE_SECONDS * ri.FPS)
    first = ri.decode_frames(tmp_path / "o" / "clip.mp4")[0]
    from freefield.media import fit_cover as fit
    photo = fit(np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(PHOTO).convert("RGB")))
    assert np.abs(first.astype(int) - photo).mean() < 4


def test_an_intruding_hand_is_re_rendered(tmp_path, monkeypatch):

    scores = iter([0.5, 0.0])
    seeds = []

    def run(compiled, out_dir, **kw):
        seeds.append(compiled.graph["video_sample"]["inputs"]["seed"])
        out_dir.mkdir(parents=True, exist_ok=True)
        clip = out_dir / "clip_00001_.mp4"
        clip.write_bytes(CLIP.read_bytes())
        return runner.RunResult("j", "succeeded", 1.0, {"clip_save": [clip]})

    class Det:
        def scan(self, clip, words):
            s = next(scores)
            return {"score": s, "frame": 30 if s else None, "label": "a human hand" if s else None}

    monkeypatch.setattr(runner, "run", run)
    monkeypatch.setattr(edge_check, "Intruders", Det)
    rec = ri.run(PHOTO, 3, tmp_path / "o")
    assert seeds == [3, 3 + 1009] and rec["warnings"] == [] and len(rec["attempts"]) == 2


def test_the_preset_loads():
    from freefield.presets import load_presets

    presets, _ = load_presets()
    assert presets["raven-intro"].pipeline == "raven_intro" and "title" in presets["raven-intro"].inputs
