import json
import subprocess
from pathlib import Path

import pytest

from freefield import cli, pipelines, runner

ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / "eval" / "hires" / "iris-standin-1.png"
CLIP = ROOT / "tests" / "fixtures" / "video" / "standin-fresh-1.mp4"


@pytest.fixture
def runs(monkeypatch):
    calls = []

    def run_preset(preset, source, seed, out_dir, **kw):
        calls.append({"preset": preset.id, "source": source, "seed": seed, "out_dir": out_dir, **kw})
        return runner.RunResult("", "succeeded", 1.23, {"clip_save": [out_dir / "clip.mp4"]}, warnings=["w"], checks={"eye": 1})

    monkeypatch.setattr(pipelines, "run_preset", run_preset)
    return calls


@pytest.mark.parametrize("argv, message", [
    (["run", "nope", "--photo", str(PHOTO)], "unknown preset 'nope'"),
    (["run", "iris-reveal", "--photo", "/no/such/photo.png"], "is not a file"),
    (["run", "iris-reveal", "--photo", str(CLIP)], "takes a photo"),
    (["run", "out-of-the-frame-real", "--clip", str(PHOTO)], "takes a video clip"),
    (["run", "iris-reveal", "--photo", str(PHOTO), "--caption", "x" * 41], "takes at most 40"),
    (["run", "raven-poof", "--photo", str(PHOTO), "--caption", "hi"], "takes no caption"),
])
def test_bad_input_is_refused_before_any_work(runs, capsys, argv, message):
    assert cli.main(argv) == 2
    assert message in capsys.readouterr().err
    assert runs == []


SHORT_CLIP = ROOT / "tests" / "fixtures" / "video" / "r16-out-of-the-photo--heldout-1-s1-end.mp4"


def bad_media(kind: str, folder: Path) -> Path:
    from PIL import Image

    if kind == "corrupt.jpg":
        (folder / kind).write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    elif kind == "truncated.jpg":
        Image.open(PHOTO).convert("RGB").save(folder / "whole.jpg", quality=90)
        (folder / kind).write_bytes((folder / "whole.jpg").read_bytes()[:20000])
    elif kind == "text-as.png":
        (folder / kind).write_text("this is a text file with a .png name\n")
    elif kind == "tiny.png":
        Image.new("RGB", (120, 160), (90, 60, 40)).save(folder / kind)
    elif kind == "corrupt.mp4":
        (folder / kind).write_bytes(bytes(range(256)) * 4)
    elif kind == "truncated.mp4":
        (folder / kind).write_bytes(CLIP.read_bytes()[:600000])
    elif kind == "photo-as.mp4":
        (folder / kind).write_bytes(PHOTO.read_bytes())
    elif kind == "audio-only.mp4":
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=5", "-c:a",
                        "aac", str(folder / kind)], check=True)
    return folder / kind


@pytest.fixture
def no_render(runs, monkeypatch):
    calls = []
    monkeypatch.setattr(pipelines, "run_pipeline", lambda *a, **k: calls.append(("run_pipeline", a)))
    monkeypatch.setattr(runner, "run", lambda *a, **k: calls.append(("runner.run", a)))
    return runs, calls


@pytest.mark.parametrize("kind, argv, message", [
    ("corrupt.jpg", ["run", "iris-reveal", "--photo"], "not an image we can read"),
    ("truncated.jpg", ["run", "superstar", "--photo"], "not an image we can read"),
    ("text-as.png", ["run", "iris-reveal", "--photo"], "not an image we can read"),
    ("text-as.png", ["run", "earth-zoom", "--photo"], "not an image we can read"),
    ("tiny.png", ["run", "superstar", "--photo"], "too small"),
    ("corrupt.mp4", ["run", "earth-zoom", "--clip"], "not a video we can read"),
    ("truncated.mp4", ["real", "--clip"], "not a video we can read"),
    ("photo-as.mp4", ["run", "earth-zoom", "--clip"], "not a video we can read"),
    ("audio-only.mp4", ["run", "earth-zoom", "--clip"], "not a video we can read"),
    ("audio-only.mp4", ["real", "--clip"], "not a video we can read"),
    (None, ["real", "--clip"], "use 6-10 s"),
])
def test_unreadable_or_unusable_media_exits_2_and_nothing_renders(no_render, capsys, tmp_path, kind, argv, message):
    runs, calls = no_render
    source = bad_media(kind, tmp_path) if kind else SHORT_CLIP
    assert cli.main([*argv, str(source)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and message in err and "Traceback" not in err
    assert runs == [] and calls == []


def test_a_readable_photo_and_clip_pass_the_same_check(no_render, capsys, tmp_path):
    runs, calls = no_render
    assert cli.main(["run", "earth-zoom", "--clip", str(CLIP), "--out", str(tmp_path)]) == 0
    assert cli.main(["run", "earth-zoom", "--photo", str(PHOTO), "--out", str(tmp_path)]) == 0
    assert [r["source"] for r in runs] == [CLIP, PHOTO]


@pytest.mark.parametrize("seed", ["-1", "2147483648", "seven"])
def test_a_seed_outside_the_range_is_a_usage_error(runs, seed):
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "iris-reveal", "--photo", str(PHOTO), "--seed", seed])
    assert e.value.code == 2 and runs == []


def test_a_run_prints_one_json_report_and_uses_the_preset_default_seed(runs, capsys, tmp_path):
    assert cli.main(["run", "iris-reveal", "--photo", str(PHOTO), "--caption", "Look", "--out", str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {"preset", "status", "seconds", "files", "error", "warnings", "checks"}
    assert report["preset"] == "iris-reveal" and report["status"] == "succeeded" and report["seconds"] == 1.2
    assert report["warnings"] == ["w"] and report["checks"] == {"eye": 1}
    assert runs[0]["seed"] == 0 and runs[0]["texts"] == {"caption": "Look"}
    assert runs[0]["out_dir"].parent == tmp_path and runs[0]["out_dir"].name.startswith("iris-reveal-")


def test_a_clip_goes_to_the_clip_preset_and_a_failed_run_exits_1(monkeypatch, capsys, tmp_path):
    seen = {}

    def run_pipeline(preset, source, seed, out_dir, **kw):
        seen.update(preset=preset.id, seed=seed, shot1=kw["shot1"])
        return runner.RunResult("", "failed", 0.5, error="the clip is shorter than 3 seconds")

    monkeypatch.setattr(pipelines, "run_pipeline", run_pipeline)
    assert cli.main(["real", "--clip", str(CLIP), "--still", "--out", str(tmp_path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "failed" and report["error"] == "the clip is shorter than 3 seconds"
    assert seen == {"preset": "out-of-the-frame-real", "seed": 1, "shot1": "still"}


def test_list_shows_every_preset(capsys):
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "out-of-the-frame " in out and "iris-reveal " in out


def test_build_render_without_a_photo_is_refused(capsys):
    assert cli.main(["build", "make me a robot", "--render"]) == 2
    assert "--render needs --photo" in capsys.readouterr().err
