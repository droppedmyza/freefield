import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from freefield import breakout, gate, runner, vision

ROOT = Path(__file__).resolve().parents[1]
VIDEO = Path(__file__).parent / "fixtures" / "video"


@pytest.mark.parametrize("clip, photo, ok", [
    ("r16-out-of-the-frame--fresh-1-s1-end.mp4", "eval/fresh/fresh-1.png", True),
    ("r16-out-of-the-photo--heldout-1-s1-end.mp4", "eval/fixtures/heldout-1.png", True),
    ("r16-out-of-the-photo--fresh-2-s1-end.mp4", "eval/fresh/fresh-2.png", False),
])
def test_real_clip_endings(clip, photo, ok):
    end = vision.Faces().end_identity(ROOT / photo, VIDEO / clip)
    assert (end >= gate.END_ID_MIN) is ok


def gated(tmp_path, monkeypatch, end):
    from freefield.runner import CompiledRun

    g = {n: {"class_type": "RandomNoise", "inputs": {"noise_seed": 1}}
         for n in ("painting_noise", "likeness_noise", "grip_noise", "breakout_noise")}
    for n in breakout.KEYFRAMES:
        g[n] = {"class_type": "SaveImage", "inputs": {}}
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")

    def fake_run(c, out_dir, **kw):
        out_dir.mkdir(parents=True, exist_ok=True)
        return runner.RunResult("j", "succeeded", 1.0, {n: [out_dir / f"{n}.png"] for n in breakout.KEYFRAMES} | {"clip_save": [clip]})

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(vision, "Faces", lambda: SimpleNamespace(end_identity=lambda photo, c: end))
    monkeypatch.setattr(gate, "check", lambda *a, **k: gate.GateResult(0.0, 1.0, 1.0, 0.5, identity_start=0.5))
    return breakout.run_gated(CompiledRun(g, [], list(breakout.KEYFRAMES)), {"hole": [0, 0, 10, 10], "rerolls": 0},
                              tmp_path / "p.png", tmp_path / "out")


def test_a_lost_face_at_the_end_is_flagged(tmp_path, monkeypatch):
    result = gated(tmp_path, monkeypatch, 0.0)
    assert any("Unverified" in w and "end of the clip" in w for w in result.warnings)
    assert json.loads((tmp_path / "out" / "gate.json").read_text())["end_identity"] == 0.0


def test_a_clean_ending_adds_no_warning(tmp_path, monkeypatch):
    assert gated(tmp_path, monkeypatch, 0.45).warnings == []


def test_the_web_run_carries_the_warning(tmp_path, monkeypatch):
    from freefield import presets, server

    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    graph_preset = SimpleNamespace(id="p", pipeline=None, params={}, inputs={}, gate=None, family="graph")
    monkeypatch.setattr(server, "load_presets", lambda: ({"p": graph_preset}, []))
    monkeypatch.setattr(presets, "compile_prompt", lambda *a: None)
    monkeypatch.setattr(runner, "make_client", lambda: object())

    def fake_run(c, out_dir, **kw):
        out_dir.mkdir(parents=True, exist_ok=True)
        return runner.RunResult("j", "succeeded", 1.0, {"clip_save": [out_dir / "c.mp4"]}, warnings=["Unverified: test"])

    monkeypatch.setattr(runner, "run", fake_run)
    run = server.Run(id="r1", preset="p", seed=1)
    server.Worker().submit(run, tmp_path / "photo.png")
    for _ in range(100):
        if run.status in ("succeeded", "failed"):
            break
        time.sleep(0.05)
    assert run.status == "succeeded" and run.warnings == ["Unverified: test"]
    assert json.loads((tmp_path / "r1" / "run.json").read_text())["warnings"] == ["Unverified: test"]


@pytest.mark.parametrize("clip, bad", [
    ("r17-out-of-the-frame--heldout-2-s1-shot1.mp4", True),
    ("r17-out-of-the-frame--heldout-2-s2-shot1.mp4", False),
])
def test_arms_from_the_top_of_the_canvas_are_caught(clip, bad):
    score = gate.shot1_intrusion(VIDEO / clip, (210, 347, 300, 455), (110, 118, 500))
    assert (score > gate.SHOT1_MAX) is bad


def video_gated(tmp_path, monkeypatch, shot1_scores):
    from freefield.runner import CompiledRun

    g = {n: {"class_type": "RandomNoise", "inputs": {"noise_seed": 1}}
         for n in ("painting_noise", "likeness_noise", "grip_noise", "breakout_noise")}
    g |= {n: {"class_type": "KSamplerAdvanced", "inputs": {"noise_seed": 1}} for n in breakout.SHOT1_SAMPLERS}
    for n in breakout.KEYFRAMES:
        g[n] = {"class_type": "SaveImage", "inputs": {}}
    seeds = []

    def fake_run(c, out_dir, **kw):
        out_dir.mkdir(parents=True, exist_ok=True)
        seeds.append(c.graph["shot1_sample"]["inputs"]["noise_seed"])
        c.graph["photo"] = {"inputs": {"image": threading.RLock()}}
        clip = out_dir / "c.mp4"
        clip.write_bytes(b"x")
        return runner.RunResult("j", "succeeded", 1.0, {n: [out_dir / f"{n}.png"] for n in breakout.KEYFRAMES} | {"clip_save": [clip]})

    scores = iter(shot1_scores)
    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(vision, "Faces", lambda: None)
    monkeypatch.setattr(gate, "check", lambda *a, **k: gate.GateResult(0.0, 1.0, 1.0, 0.5, identity_start=0.5))
    monkeypatch.setattr(breakout, "video_checks", lambda *a: {"shot1_intrusion": next(scores)})
    spec = {"hole": [0, 0, 10, 10], "rerolls": 0, "shot1_corners": True, "video_rerolls": 2}
    result = breakout.run_gated(CompiledRun(g, [], list(breakout.KEYFRAMES)), spec, tmp_path / "p.png", tmp_path / "out")
    return result, seeds, json.loads((tmp_path / "out" / "gate.json").read_text())


def test_a_bad_first_shot_is_re_rendered_with_new_noise(tmp_path, monkeypatch):
    result, seeds, record = video_gated(tmp_path, monkeypatch, [0.6, 0.05])
    assert seeds[1] != seeds[2] and len(seeds) == 3
    assert result.warnings == [] and "video-1" in str(result.files["clip_save"][0])
    assert [a["shot1_intrusion"] for a in record["video_attempts"]] == [0.6, 0.05] and record["shot1_intrusion"] == 0.05


def test_when_every_first_shot_is_bad_the_cleanest_ships_flagged(tmp_path, monkeypatch):
    result, seeds, record = video_gated(tmp_path, monkeypatch, [0.6, 0.4, 0.5])
    assert len(record["video_attempts"]) == 3 and record["shot1_intrusion"] == 0.4
    assert any("first shot" in w and "Unverified" in w for w in result.warnings)
