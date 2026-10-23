import asyncio
import json
import shutil
from pathlib import Path

import pytest

from freefield import cli, edge_check, runner, server, vision

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "tests" / "fixtures" / "video" / "plush-squish--dog-a.mp4"
PHOTO = ROOT / "eval" / "objects" / "plush-pair-b.png"


def fake_render(monkeypatch, intruders):
    scores, renders, scans = list(intruders), [], []

    class Intruders:
        def scan(self, clip, words):
            scans.append((Path(clip), words))
            score = scores.pop(0)
            return {"score": score, "frame": 12 if score else None, "label": "a human hand" if score else None}

    class Subject:
        def similarity(self, a, b):
            return 0.9

        def text_similarity(self, text, path):
            return 0.0

    def run(compiled, out_dir, **kw):
        out_dir.mkdir(parents=True, exist_ok=True)
        clip = out_dir / f"clip_save_0_clip_{len(renders):05d}_.mp4"
        shutil.copy(CLIP, clip)
        renders.append((out_dir, compiled.graph["motion_sample"]["inputs"]["noise_seed"]))
        return runner.RunResult("j", "succeeded", 1.0, {"clip_save": [clip]})

    monkeypatch.setattr(edge_check, "Intruders", Intruders)
    monkeypatch.setattr(vision, "Subject", Subject)
    monkeypatch.setattr(vision, "motion", lambda clip: 0.2)
    monkeypatch.setattr(runner, "run", run)
    monkeypatch.setattr(runner, "make_client", lambda: object())
    return renders, scans


def gallery_run(tmp_path, monkeypatch, seed=7):
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "runs")
    worker = server.Worker()
    run = server.Run(id="0123456789abcdef", preset="toy-hug-studio", seed=seed)
    (tmp_path / "runs" / run.id).mkdir(parents=True)

    async def wait():
        worker.submit(run, PHOTO)
        for _ in range(500):
            if run.status in ("succeeded", "failed"):
                return
            await asyncio.sleep(0.02)

    asyncio.run(wait())
    return run, json.loads((tmp_path / "runs" / run.id / "run.json").read_text())


def test_a_hand_in_the_first_gallery_render_is_re_rendered_with_new_noise(tmp_path, monkeypatch):
    renders, scans = fake_render(monkeypatch, [0.45, 0.0])
    run, saved = gallery_run(tmp_path, monkeypatch)
    assert run.status == "succeeded", run.error
    assert [s for _, s in renders] == [7, 7 + 1009] and len(scans) == 2 and "a human hand" in scans[0][1]
    assert renders[1][0] == tmp_path / "runs" / run.id / "video-1"
    assert run.outputs == {"clip_save": ["video-1/clip_save_0_clip_00001_.mp4"]} and run.warnings == []
    assert [a["passed"] for a in saved["checks"]["attempts"]] == [False, True] and saved["checks"]["passed"]


def test_when_every_gallery_render_has_a_hand_the_cleanest_comes_back_with_a_warning(tmp_path, monkeypatch):
    renders, scans = fake_render(monkeypatch, [0.6, 0.3, 0.5])
    run, saved = gallery_run(tmp_path, monkeypatch)
    assert run.status == "succeeded" and len(renders) == 3 and len(scans) == 3
    assert run.outputs == {"clip_save": ["video-1/clip_save_0_clip_00001_.mp4"]}
    assert len(run.warnings) == 1 and "after 3 tries" in run.warnings[0] and "a hand or object" in run.warnings[0]
    assert saved["checks"]["passed"] is False and saved["checks"]["intruder"]["score"] == 0.3


def test_a_clean_first_gallery_render_is_scanned_once_and_not_re_rendered(tmp_path, monkeypatch):
    renders, scans = fake_render(monkeypatch, [0.0])
    run, saved = gallery_run(tmp_path, monkeypatch)
    assert run.status == "succeeded" and len(renders) == 1 and len(scans) == 1 and run.warnings == []
    assert run.outputs == {"clip_save": ["clip_save_0_clip_00000_.mp4"]} and saved["checks"]["passed"]


@pytest.mark.parametrize("intruders, renders_expected, warned", [([0.45, 0.0], 2, False), ([0.6, 0.3, 0.5], 3, True)])
def test_the_cli_runs_the_same_checks(tmp_path, monkeypatch, capsys, intruders, renders_expected, warned):
    renders, scans = fake_render(monkeypatch, intruders)
    code = cli.main(["run", "toy-hug-studio", "--photo", str(PHOTO), "--seed", "3", "--out", str(tmp_path)])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["status"] == "succeeded"
    assert len(renders) == renders_expected and len(scans) == renders_expected
    assert renders[1][1] == 3 + 1009 and Path(out["files"]["clip_save"][0]).parent.name == "video-1"
    assert bool(out["warnings"]) is warned and len(out["checks"].get("attempts", [])) == renders_expected


def test_every_published_built_preset_carries_its_recipe():
    from freefield.output_checks import published_recipe
    from freefield.presets import load_presets

    presets, _ = load_presets()
    built = {p.id: published_recipe(p) for p in presets.values() if p.family == "built"}
    assert {"toy-hug-studio", "paper-me", "raven-poof", "mascot-outfit-switch", "marble-statue"} <= set(built)
    assert all(r is not None and r.motion for r in built.values())
