import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from freefield import factory, vision
from freefield.plate import extract
from freefield.presets import parse_preset

SPEC = Path(__file__).resolve().parents[1] / "specs" / "out-of-the-photo.json"


def scene(box, colour=(255, 0, 255)):
    img = Image.new("RGB", (720, 1280), (150, 110, 70))
    ImageDraw.Draw(img).rectangle(box, fill=colour)
    return img


def test_a_clean_opening_is_found_and_scored():
    p = extract(scene((200, 380, 520, 780)))
    assert p is not None and p.xywh == (200, 380, 321, 401) and p.coverage > 0.99
    assert factory.score_plate(p, border=24) > 0


@pytest.mark.parametrize("box", [(0, 0, 719, 1279), (300, 500, 340, 540), (2, 380, 400, 780)])
def test_unusable_openings_are_rejected(box):
    assert factory.score_plate(extract(scene(box)), border=24) == 0


def test_no_magenta_means_no_plate():
    assert extract(scene((200, 380, 520, 780), colour=(20, 200, 20))) is None


@pytest.mark.parametrize("hole", [(210, 347, 300, 455), (177, 304, 372, 608), (100, 900, 520, 300)])
def test_camera_crops_are_9_by_16_inside_the_plate_and_frame_the_opening(hole):
    vx, vy, vw, vh = factory.view_box(hole)
    x, y, w, h = hole
    assert abs(vw / vh - 9 / 16) < 0.01
    assert 0 <= vx and 0 <= vy and vx + vw <= 720 and vy + vh <= 1280
    assert vx <= x and vy <= y and x + w <= vx + vw and y + h <= vy + vh


def test_the_factory_preset_validates_on_the_shared_graph(tmp_path):
    spec = factory.Spec.load(SPEC)
    data = factory.manifest(spec, (200, 380, 321, 401))
    for name in ("plate.png", "plate_hole.png", "wall_mask.png"):
        Image.new("RGB", (8, 8)).save(tmp_path / name)
    preset = parse_preset(json.loads(json.dumps(data)), tmp_path)
    assert preset.gate["hole"] == [200, 380, 321, 401]
    assert data["fixed"]["breakout_start_frame_latent.pixels"] == ["plate", 0]


def test_bad_specs_say_what_is_wrong(tmp_path):
    good = json.loads(SPEC.read_text())
    for patch, message in (({"scene": "a desk"}, "magenta"), ({"grip": ""}, "missing"), ({"colour": "red"}, "unknown")):
        p = tmp_path / "s.json"
        p.write_text(json.dumps({**good, **patch}))
        with pytest.raises(factory.FactoryError, match=message):
            factory.Spec.load(p)


@pytest.mark.parametrize("gate_passed, end_identity, publish", [
    (True, 0.45, True),
    (False, 0.45, False),
    (True, 0.0, False),
])
def test_the_factory_publishes_only_when_the_preview_passes_the_gate(tmp_path, monkeypatch, gate_passed, end_identity,
                                                                     publish):
    from freefield import breakout, runner
    from freefield.plate import extract

    monkeypatch.setattr(factory, "PRESETS_DIR", tmp_path / "presets")
    monkeypatch.setattr(factory, "DRAFTS", tmp_path / "presets" / "_drafts")
    (tmp_path / "presets" / "out-of-the-frame").mkdir(parents=True)
    real = Path(__file__).resolve().parents[1] / "freefield" / "presets" / "out-of-the-frame" / "preset.json"
    (tmp_path / "presets" / "out-of-the-frame" / "preset.json").write_text(real.read_text())
    (tmp_path / "presets" / "out-of-the-frame" / "stages.json").write_text((real.parent / "stages.json").read_text())
    plate = extract(scene((200, 380, 520, 780)))
    monkeypatch.setattr(factory, "make_plate", lambda spec, draft, client: (plate, [{"seed": 1, "score": 1.0}]))
    monkeypatch.setattr(runner, "make_client", lambda: object())

    def fake_gated(compiled, spec, photo, out_dir, **kw):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "gate.json").write_text(json.dumps({"passed": gate_passed, "attempts": [{}], "end_identity": end_identity}))
        clip, proof = out_dir / "clip.mp4", out_dir / "proof.png"
        clip.write_bytes(b"mp4"), proof.write_bytes(b"png")
        warnings = ["Unverified: your face is not clearly there at the end of the clip"] if end_identity < 0.2 else []
        return runner.RunResult("job", "succeeded", 1.0, {"clip_save": [clip], "proof_save": [proof]}, warnings=warnings)

    monkeypatch.setattr(breakout, "run_gated", fake_gated)
    photo = tmp_path / "me.png"
    Image.new("RGB", (64, 64)).save(photo)
    report = factory.build(factory.Spec.load(SPEC), photo)
    published = tmp_path / "presets" / "out-of-the-photo"
    assert report["published"] is publish
    assert published.exists() is publish
    assert (tmp_path / "presets" / "_drafts" / "out-of-the-photo" / "factory_report.json").exists()
    if publish:
        assert json.loads((published / "preset.json").read_text())["preview"] == "preview.mp4"
        api = json.loads((published / "workflow_api.json").read_text())
        assert api["breakout_prompt"]["inputs"]["text"] == factory.Spec.load(SPEC).out
        assert api["photo"]["inputs"]["image"] == "your_photo.png"
        assert (published / "stages.json").exists()


def test_a_gate_fallback_is_flagged_not_silent(tmp_path, monkeypatch):
    from freefield import breakout, gate, runner
    from freefield.runner import CompiledRun

    g = {"grip_noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": 1}}}
    for n in ("painting_noise", "likeness_noise", "breakout_noise"):
        g[n] = {"class_type": "RandomNoise", "inputs": {"noise_seed": 1}}
    for n in breakout.KEYFRAMES:
        g[n] = {"class_type": "SaveImage", "inputs": {}}
    compiled = CompiledRun(g, [], list(breakout.KEYFRAMES))

    def fake_run(c, out_dir, **kw):
        out_dir.mkdir(parents=True, exist_ok=True)
        return runner.RunResult("j", "succeeded", 1.0, {n: [out_dir / f"{n}.png"] for n in breakout.KEYFRAMES} | {"clip_save": [out_dir / "c.mp4"]})

    bad = gate.GateResult(0.01, 1.0, 1.0, 0.5, identity_start=0.5)
    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(vision, "Faces", lambda: None)
    monkeypatch.setattr(gate, "check", lambda *a, **k: bad)
    result = breakout.run_gated(compiled, {"hole": [0, 0, 10, 10], "rerolls": 1}, tmp_path / "p.png", tmp_path / "out")
    assert result.warnings and "Unverified" in result.warnings[0] and "hands_inside" in result.warnings[0]
    assert json.loads((tmp_path / "out" / "gate.json").read_text())["passed"] is False


def test_the_emptied_opening_takes_the_specs_fill_colour():
    from freefield.plate import extract
    p = extract(scene((200, 380, 520, 780)), fill=(226, 214, 190))
    assert p.plate.getpixel((360, 580)) == (226, 214, 190)
