import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from freefield import breakout, iris_reveal, output_checks, pipelines, presets, runner
from freefield.errors import EffectError
from freefield.presets import load_presets

PRESETS, _ = load_presets()


def fake_effect(monkeypatch, module, record=None, error=None):
    calls = []

    def run(source, seed, out_dir, **kw):
        calls.append({"source": source, "seed": seed, "out_dir": out_dir, **kw})
        if error:
            raise error
        return record or {"warnings": []}

    monkeypatch.setattr(module, "run", run)
    return calls


@pytest.mark.parametrize("preset_id, text_name, needs_client", [
    ("earth-zoom", "caption", True),
    ("product-freeze", "offer", True),
    ("iris-reveal", "caption", False),
    ("comic-bonk", "word", True),
    ("raven-intro", "title", True),
    ("out-of-the-frame-real", None, True),
])
def test_each_pipeline_gets_its_text_and_a_client_only_when_it_renders(monkeypatch, tmp_path, preset_id, text_name, needs_client):
    import importlib

    preset = PRESETS[preset_id]
    module = importlib.import_module(f"freefield.{pipelines.PIPELINES[preset.pipeline].module}")
    calls = fake_effect(monkeypatch, module)
    texts = {text_name: "HELLO"} if text_name else {}
    client = object()
    result = pipelines.run_preset(preset, tmp_path / "in.png", 9, tmp_path / "out", texts=texts, client=client)
    assert result.ok and calls[0]["seed"] == 9
    assert calls[0].get("text") == ("HELLO" if text_name else None)
    assert ("client" in calls[0]) == needs_client and (calls[0].get("client") is client or not needs_client)
    assert set(result.files) == set(pipelines.PIPELINES[preset.pipeline].files)


def test_an_effect_error_becomes_a_failed_result_with_the_message(monkeypatch, tmp_path):
    fake_effect(monkeypatch, iris_reveal, error=EffectError("no face found; use a sharp, front-facing selfie"))
    result = pipelines.run_preset(PRESETS["iris-reveal"], tmp_path / "in.png", 0, tmp_path / "out")
    assert result.status == "failed" and result.error == "no face found; use a sharp, front-facing selfie"
    assert not result.ok and result.files == {}


def test_a_pipeline_reports_its_warnings_and_summary_fields(monkeypatch, tmp_path):
    record = {"warnings": ["The iris was not located exactly"], "eye": {"side": "left"}, "upscale_at_eye": 2.1, "frames": 108}
    fake_effect(monkeypatch, iris_reveal, record=record)
    result = pipelines.run_preset(PRESETS["iris-reveal"], tmp_path / "in.png", 0, tmp_path / "out")
    assert result.warnings == ["The iris was not located exactly"]
    assert result.checks == {"eye": {"side": "left"}, "upscale_at_eye": 2.1}


def test_every_pipeline_preset_has_a_registered_pipeline_and_a_seed_default():
    for preset in PRESETS.values():
        if preset.pipeline:
            assert preset.pipeline in pipelines.PIPELINES
            assert pipelines.default_seed(preset) == preset.params["seed"]["default"]


def test_a_gated_preset_goes_through_the_keyframe_gate(monkeypatch, tmp_path):
    seen = {}

    def gated(compiled, spec, photo, out_dir, **kw):
        seen.update(spec=spec, label=kw["label"], seed=compiled.graph["grip_noise"]["inputs"]["noise_seed"])
        return runner.RunResult("j", "succeeded", 1.0, {"clip_save": [out_dir / "c.mp4"]})

    monkeypatch.setattr(breakout, "run_gated", gated)
    preset = PRESETS["out-of-the-frame"]
    photo = tmp_path / "me.png"
    photo.write_bytes(b"png")
    assert pipelines.run_preset(preset, photo, 5, tmp_path / "out").ok
    assert seen["spec"] is preset.gate and seen["label"] == "out-of-the-frame" and seen["seed"] == 5


def test_a_built_preset_runs_with_its_output_checks(monkeypatch, tmp_path):
    seen = {}

    def checked(preset, recipe, photo, where, **kw):
        seen.update(recipe=recipe, seed=kw["seed"], first=where(0), retry=where(1))
        result = runner.RunResult("j", "succeeded", 1.0, {"clip_save": [where(0) / "c.mp4"]}, warnings=["w"])
        return SimpleNamespace(result=result, warnings=lambda: ["Output check failed"],
                               checks={"passed": False, "clip_path": "x", "intruder": {"score": 0.3}})

    monkeypatch.setattr(output_checks, "checked_run", checked)
    photo = tmp_path / "me.png"
    photo.write_bytes(b"png")
    result = pipelines.run_preset(PRESETS["toy-hug-studio"], photo, 3, tmp_path / "out")
    assert seen["seed"] == 3 and seen["first"] == tmp_path / "out" and seen["retry"] == tmp_path / "out" / "video-1"
    assert result.warnings == ["w", "Output check failed"]
    assert result.checks == {"passed": False, "intruder": {"score": 0.3}}


def test_a_single_graph_preset_is_compiled_with_the_seed_and_the_texts(monkeypatch, tmp_path):
    seen = {}

    def run(compiled, out_dir, **kw):
        seen.update(graph=compiled.graph, label=kw["label"])
        return runner.RunResult("j", "succeeded", 1.0, {"clip_save": [out_dir / "c.mp4"]})

    monkeypatch.setattr(runner, "run", run)
    photo = tmp_path / "me.png"
    photo.write_bytes(b"png")
    preset = PRESETS["superstar"]
    assert pipelines.run_preset(preset, photo, 77, tmp_path / "out").ok
    seed_bind = preset.params["seed"]["bind"][0].split(".")
    assert seen["graph"][seed_bind[0]]["inputs"][seed_bind[1]] == 77 and seen["label"] == "superstar"


def test_compile_errors_reach_the_caller(tmp_path):
    with pytest.raises(presets.PresetError, match="does not exist"):
        pipelines.run_preset(PRESETS["superstar"], tmp_path / "missing.png", 1, tmp_path / "out")


def test_the_preset_format_doc_lists_every_pipeline():
    doc = (Path(__file__).resolve().parents[1] / "docs" / "PRESET_FORMAT.md").read_text()
    assert all(f"`{name}`" in doc for name in pipelines.PIPELINES), json.dumps(sorted(pipelines.PIPELINES))
