import json
from pathlib import Path

import pytest

from freefield.presets import PRESETS_DIR, PresetError, compile_prompt, fill_for_export, load_presets, parse_preset

GRAPH = {"photo": {"class_type": "LoadImage", "inputs": {"image": ""}},
         "sample": {"class_type": "KSampler", "inputs": {"seed": 0, "cfg": 1.0, "sampler_name": "euler"}},
         "caption": {"class_type": "TextOverlay", "inputs": {"text": ""}},
         "clip_save": {"class_type": "SaveVideo", "inputs": {}}}


def manifest(**over):
    data = {"id": "test-preset", "name": "Test", "graph": GRAPH, "outputs": ["clip_save"],
            "inputs": {"photo": {"type": "image", "bind": ["photo.image"]},
                       "caption": {"type": "text", "bind": ["caption.text"], "required": False, "max_length": 10,
                                   "template": "Hello {value}"}},
            "params": {"seed": {"type": "int", "default": 1, "min": 0, "max": 99, "bind": ["sample.seed"]},
                       "cfg": {"type": "float", "default": 1.0, "min": 0.5, "bind": ["sample.cfg"]},
                       "sampler": {"type": "choice", "default": "euler", "options": ["euler", "dpmpp"], "bind": ["sample.sampler_name"]},
                       "note": {"type": "text", "default": "", "max_length": 5}}}
    data.update(over)
    return data


@pytest.mark.parametrize("over, message", [
    ({"id": "Bad Id"}, "lowercase letters"),
    ({"graph": "missing.json"}, "not found"),
    ({"graph": {"x": {"inputs": {}}}}, "API format"),
    ({"inputs": {"photo": {"type": "audio", "bind": ["photo.image"]}}}, "type must be one of"),
    ({"inputs": {"photo": {"type": "image", "bind": []}}}, "at least one binding"),
    ({"inputs": {"photo": {"type": "image", "bind": ["photo"]}}}, "must look like"),
    ({"inputs": {"photo": {"type": "image", "bind": ["ghost.image"]}}}, "not in the graph"),
    ({"params": {"seed": {"type": "int", "bind": ["sample.seed"]}}}, "needs a default"),
    ({"params": {"s": {"type": "choice", "default": "x", "options": ["a"]}}}, "one of its options"),
    ({"fixed": {"photo.image": "asset:nowhere.png"}}, "missing asset"),
    ({"pipeline": "teleport"}, "pipeline must be one of"),
    ({"inputs": {"clip": {"type": "video", "bind": ["photo.image"]}}}, "needs a pipeline"),
    ({"gate": {"hole": [0, 0, 10]}}, "gate.hole"),
    ({"gate": {"hole": [0, 0, 10, 10], "rerolls": 9}}, "gate.rerolls"),
    ({"outputs": ["nowhere"]}, "output node"),
])
def test_a_bad_manifest_says_what_is_wrong(tmp_path, over, message):
    with pytest.raises(PresetError, match=message):
        parse_preset(manifest(**over), tmp_path)


def test_a_manifest_must_be_an_object_with_its_required_fields(tmp_path):
    with pytest.raises(PresetError, match="JSON object"):
        parse_preset([], tmp_path)
    with pytest.raises(PresetError, match="missing required field 'outputs'"):
        parse_preset({"id": "x", "name": "x", "graph": GRAPH}, tmp_path)


@pytest.mark.parametrize("params, message", [
    ({"seed": 100}, "<= 99"),
    ({"seed": None}, "must be a int"),
    ({"cfg": 0.1}, ">= 0.5"),
    ({"cfg": "warm"}, "must be a float"),
    ({"sampler": "ddim"}, "must be one of"),
    ({"note": "toolong"}, "longer than 5"),
    ({"steps": 4}, "unknown params"),
])
def test_bad_params_are_refused(tmp_path, params, message):
    preset = parse_preset(manifest(), tmp_path)
    photo = tmp_path / "me.png"
    photo.write_bytes(b"png")
    with pytest.raises(PresetError, match=message):
        compile_prompt(preset, {"photo": str(photo)}, params)


def test_params_and_texts_are_coerced_and_templated(tmp_path):
    preset = parse_preset(manifest(), tmp_path)
    photo = tmp_path / "me.png"
    photo.write_bytes(b"png")
    run = compile_prompt(preset, {"photo": str(photo), "caption": "Ana"}, {"seed": "7", "cfg": "1.5", "sampler": "dpmpp"})
    assert run.graph["sample"]["inputs"] == {"seed": 7, "cfg": 1.5, "sampler_name": "dpmpp"}
    assert run.graph["caption"]["inputs"]["text"] == "Hello Ana"
    assert run.files == [("photo", "image", photo)]
    skipped = compile_prompt(preset, {"photo": str(photo)})
    assert skipped.graph["caption"]["inputs"]["text"] == ""
    with pytest.raises(PresetError, match="longer than 10"):
        compile_prompt(preset, {"photo": str(photo), "caption": "x" * 11})
    with pytest.raises(PresetError, match="unknown inputs"):
        compile_prompt(preset, {"photo": str(photo), "voice": "x"})
    with pytest.raises(PresetError, match="missing required input 'photo'"):
        compile_prompt(preset, {})


def test_a_duplicate_id_is_reported_and_does_not_hide_the_first(tmp_path):
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "preset.json").write_text(json.dumps(manifest()))
    presets, errors = load_presets(tmp_path)
    assert list(presets) == ["test-preset"] and "duplicate preset id" in errors["b"]


def test_export_filling_names_assets_and_fills_placeholders():
    graph = {"plate": {"class_type": "LoadImage", "inputs": {"image": ""}}, "photo": {"class_type": "LoadImage", "inputs": {"image": ""}},
             "prompt": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}}}
    filled = fill_for_export(graph, {"plate.image": "asset:plate.png", "prompt.text": "a gold frame"}, {"photo.image": "your_photo.png"},
                             lambda name: f"p--{name}")
    assert filled["plate"]["inputs"]["image"] == "p--plate.png" and filled["prompt"]["inputs"]["text"] == "a gold frame"
    assert filled["photo"]["inputs"]["image"] == "your_photo.png" and graph["photo"]["inputs"]["image"] == ""


def test_every_shipped_preset_loads():
    presets, errors = load_presets(PRESETS_DIR)
    assert errors == {} and len(presets) >= 20
    assert Path(PRESETS_DIR).name == "presets"
