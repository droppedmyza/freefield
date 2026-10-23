import json
from pathlib import Path

import pytest

from freefield.presets import PRESETS_DIR, PresetError, compile_prompt, load_presets, parse_preset

GRAPH = {
    "photo": {"class_type": "LoadImage", "inputs": {"image": ""}},
    "prompt": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["te", 0]}},
    "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": 0}},
    "save": {"class_type": "SaveImage", "inputs": {"images": ["x", 0], "filename_prefix": "t"}},
}


def manifest(**over):
    m = {
        "id": "test-preset", "name": "Test", "graph": GRAPH, "outputs": ["save"],
        "inputs": {"photo": {"type": "image", "bind": ["photo.image"]}},
        "params": {"seed": {"type": "int", "default": 7, "min": 0, "max": 100, "bind": ["noise.noise_seed"]}},
        "fixed": {"prompt.text": "hello"},
    }
    m.update(over)
    return m


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    p = tmp_path / "me.png"
    p.write_bytes(b"\x89PNG fake")
    return p


def test_compile_fills_bindings_and_returns_files(tmp_path, photo):
    preset = parse_preset(manifest(), tmp_path)
    run = compile_prompt(preset, {"photo": str(photo)}, {"seed": 9})
    assert run.graph["prompt"]["inputs"]["text"] == "hello"
    assert run.graph["noise"]["inputs"]["noise_seed"] == 9
    assert run.files == [("photo", "image", photo)]
    assert run.outputs == ["save"]
    assert preset.graph["noise"]["inputs"]["noise_seed"] == 0


def test_default_param_used_when_not_given(tmp_path, photo):
    run = compile_prompt(parse_preset(manifest(), tmp_path), {"photo": str(photo)})
    assert run.graph["noise"]["inputs"]["noise_seed"] == 7


def test_asset_values_become_uploads(tmp_path, photo):
    (tmp_path / "art.jpg").write_bytes(b"jpg")
    graph = dict(GRAPH, art={"class_type": "LoadImage", "inputs": {"image": ""}})
    preset = parse_preset(manifest(graph=graph, fixed={"art.image": "asset:art.jpg"}), tmp_path)
    run = compile_prompt(preset, {"photo": str(photo)})
    assert ("art", "image", tmp_path / "art.jpg") in run.files


@pytest.mark.parametrize("bad, message", [
    ({"inputs": {"photo": {"type": "image", "bind": ["nope.image"]}}}, "names node 'nope'"),
    ({"inputs": {"photo": {"type": "image", "bind": ["photo.missing"]}}}, "has no input 'missing'"),
    ({"inputs": {"photo": {"type": "audio", "bind": ["photo.image"]}}}, "type must be one of"),
    ({"inputs": {"photo": {"type": "video", "bind": ["photo.image"]}}}, "needs a pipeline"),
    ({"pipeline": "teleport"}, "pipeline must be one of"),
    ({"id": "Bad Id!"}, "must be lowercase"),
    ({"outputs": ["ghost"]}, "output node 'ghost'"),
    ({"fixed": {"prompt.text": "asset:missing.png"}}, "missing asset"),
    ({"params": {"seed": {"type": "int", "bind": ["noise.noise_seed"]}}}, "needs a default"),
])
def test_invalid_manifests_say_what_is_wrong(tmp_path, bad, message):
    with pytest.raises(PresetError, match=message):
        parse_preset(manifest(**bad), tmp_path)


@pytest.mark.parametrize("params, message", [
    ({"seed": 101}, "must be <= 100"),
    ({"seed": "abc"}, "must be a int"),
    ({"steps": 3}, "unknown params"),
])
def test_bad_params_rejected(tmp_path, photo, params, message):
    with pytest.raises(PresetError, match=message):
        compile_prompt(parse_preset(manifest(), tmp_path), {"photo": str(photo)}, params)


def test_missing_photo_rejected(tmp_path):
    with pytest.raises(PresetError, match="missing required input 'photo'"):
        compile_prompt(parse_preset(manifest(), tmp_path), {})


def test_nonexistent_photo_rejected(tmp_path):
    with pytest.raises(PresetError, match="does not exist"):
        compile_prompt(parse_preset(manifest(), tmp_path), {"photo": str(tmp_path / "nope.png")})


def test_shipped_presets_all_load():
    presets, errors = load_presets()
    assert errors == {}
    assert len(presets) >= 3
    for p in presets.values():
        assert p.outputs, p.id
        assert ("clip" if p.pipeline in ("clip_breakout", "earth_zoom") else "photo") in p.inputs, p.id


def test_broken_preset_does_not_hide_good_ones(tmp_path):
    good = tmp_path / "good"; good.mkdir()
    (good / "preset.json").write_text(json.dumps(manifest(id="good-one")))
    bad = tmp_path / "bad"; bad.mkdir()
    (bad / "preset.json").write_text("{not json")
    presets, errors = load_presets(tmp_path)
    assert list(presets) == ["good-one"]
    assert "bad" in errors


def test_action_figure_title_is_a_header_card_outside_the_animated_area(photo):
    run = compile_prompt(load_presets()[0]["action-figure"], {"photo": str(photo)})
    g = run.graph
    assert ("header", "image", PRESETS_DIR / "action-figure" / "header.png") in run.files
    assert g["with_header"]["inputs"]["image1"] == ["header_small", 0]
    assert g["with_header"]["inputs"]["image2"] == ["video_frames", 0]
    assert g["watermark"]["inputs"]["images"] == ["with_header", 0]
    assert g["video_latent"]["inputs"]["start_image"] == ["keyframe", 0]
    assert "LIMITED" not in g["keyframe_prompt"]["inputs"]["text"]


def test_header_card_matches_the_clip_shape():
    from PIL import Image
    g = load_presets()[0]["action-figure"].graph
    hw, hh = Image.open(PRESETS_DIR / "action-figure" / "header.png").size
    kw, kh = g["keyframe_empty"]["inputs"]["width"], g["keyframe_empty"]["inputs"]["height"]
    vw, vh = g["video_latent"]["inputs"]["width"], g["video_latent"]["inputs"]["height"]
    sw, sh = g["header_small"]["inputs"]["width"], g["header_small"]["inputs"]["height"]
    assert (hw, sw) == (kw, vw)
    assert (vw, vh + sh) == (480, 832)
    assert hh / hw == sh / sw and kh / kw == vh / vw


def test_docs_worked_example_is_a_valid_preset():
    doc = (Path(__file__).resolve().parents[1] / "docs" / "PRESET_FORMAT.md").read_text()
    data = json.loads(doc[doc.index("## Worked example"):].split("```json")[1].split("```")[0])
    preset = parse_preset(data, PRESETS_DIR / data["id"])
    assert preset.id == "monet-muse"
    assert preset.fixed == load_presets()[0]["monet-muse"].fixed


@pytest.mark.parametrize("pid", ["out-of-the-frame", "out-of-the-box", "out-of-the-photo"])
def test_breakout_frames_line_up(photo, pid):
    from PIL import Image
    folder = PRESETS_DIR / pid
    g = compile_prompt(load_presets()[0][pid], {"photo": str(photo)}).graph
    pw, ph = Image.open(folder / "plate.png").size
    mw, mh = Image.open(folder / "plate_hole.png").size
    fit, start = g["painting_fit"]["inputs"], g["start_frame"]["inputs"]
    assert (fit["width"], fit["height"]) == (mw, mh)
    assert start["x"] + mw <= pw and start["y"] + mh <= ph
    for key in ("grip", "breakout"):
        assert (g[f"{key}_empty"]["inputs"]["width"], g[f"{key}_empty"]["inputs"]["height"]) == (pw, ph)
    shots = [v["inputs"] for v in g.values() if v["class_type"] in ("WanFirstLastFrameToVideo", "WanImageToVideo")]
    assert shots and all(abs(pw / ph - s["width"] / s["height"]) < 0.01 for s in shots)
    box = lambda n: tuple(g[n]["inputs"][k] for k in ("x", "y", "width", "height"))
    assert box("start_view_crop") == box("grip_view_crop") == box("shot1_wall_crop")
    for crop in (v["inputs"] for v in g.values() if v["class_type"] == "ImageCrop"):
        assert crop["x"] + crop["width"] <= pw and crop["y"] + crop["height"] <= ph
        assert abs(crop["width"] / crop["height"] - pw / ph) < 0.01
