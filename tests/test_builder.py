import json
import re
from pathlib import Path

import numpy as np
import pytest

from freefield import builder as b
from freefield import edge_check, node_check, recipe, vision
from freefield.recipe import Asset, Recipe

SCHEMAS = json.loads(node_check.SCHEMAS.read_text())


def test_plain_words_become_a_recipe():
    r = recipe.parse('Make me a paper cutout, then I pop up off the page, with a before/after reveal, titled "PAPER ME"')
    assert r.becomes == "paper cutout"
    assert r.motion == "they pop up off the page"
    assert r.reveal and not r.split and r.title == "PAPER ME"
    assert r.id == "paper-me" and r.stages() == ["transform", "animate", "reveal", "title", "save"]


def test_a_bare_description_still_makes_a_recipe():
    r = recipe.parse("a marble statue in a museum")
    assert r.becomes == "a marble statue in a museum" and r.motion == recipe.DEFAULT_MOTION
    assert r.name == "Marble Statue" and not (r.reveal or r.split or r.title)


def test_side_by_side_is_recognised():
    r = recipe.parse("turn me into a watercolor painting, then I wave, side by side with my photo")
    assert r.split and r.becomes == "a watercolor painting" and r.motion == "they wave"


@pytest.mark.parametrize("text", ["", "   ", "x" * 401])
def test_empty_or_huge_descriptions_are_refused(text):
    with pytest.raises(recipe.BuildError):
        recipe.parse(text)


@pytest.mark.parametrize("reveal,split,title,ref", [(r, s, t, f) for r in (0, 1) for s in (0, 1) for t in (0, 1) for f in (0, 1)])
def test_every_recipe_assembles_a_graph_comfyui_accepts(reveal, split, title, ref):
    r = Recipe(name="x", becomes="a robot", reveal=bool(reveal), split=bool(split), title="HI" if title else None,
               reference=Asset("ref.png", "cc0", "test") if ref else None)
    g, fixed, stages = b.assemble(r)
    assert node_check.validate(g, fixed, SCHEMAS, {"photo.image"}) == []
    assert ("reveal" in g) is bool(reveal) and ("split" in g) is bool(split) and ("titled" in g) is bool(title)
    assert g["clip_save"]["inputs"]["video"] == ["video", 0]


def test_different_descriptions_give_structurally_different_graphs():
    plain, _, _ = b.assemble(recipe.parse("a robot"))
    rich, _, _ = b.assemble(recipe.parse('a robot, then it dances, with a reveal, side by side, titled "BOT"'))
    assert len(rich) - len(plain) >= 10 and set(plain) < set(rich)


def graph_and_fixed():
    g, fixed, _ = b.assemble(recipe.parse("a robot, with a reveal"))
    return g, fixed


def test_a_broken_link_is_rejected():
    g, fixed = graph_and_fixed()
    g["reveal"]["inputs"]["source"] = ["nowhere", 0]
    assert any("nowhere" in e for e in node_check.validate(g, fixed, SCHEMAS, {"photo.image"}))


def test_a_link_of_the_wrong_type_is_rejected():
    g, fixed = graph_and_fixed()
    g["reveal"]["inputs"]["mask"] = ["before", 0]
    assert any("expects MASK" in e for e in node_check.validate(g, fixed, SCHEMAS, {"photo.image"}))


def test_an_unknown_node_is_rejected():
    g, fixed = graph_and_fixed()
    g["magic"] = {"class_type": "PartnerApiMagicNode", "inputs": {}}
    assert any("not installed" in e for e in node_check.validate(g, fixed, SCHEMAS, {"photo.image"}))


def test_a_missing_model_is_rejected():
    g, fixed = graph_and_fixed()
    g["klein_model"]["inputs"]["unet_name"] = "flux-2-pro-closed.safetensors"
    assert any("model 'flux-2-pro-closed.safetensors' is not available" in e for e in node_check.validate(g, fixed, SCHEMAS, {"photo.image"}))


def test_a_graph_without_an_output_is_rejected():
    g, fixed = graph_and_fixed()
    del g["clip_save"], g["keyframe_save"]
    assert any("no output node" in e for e in node_check.validate(g, fixed, SCHEMAS, {"photo.image"}))


def test_an_unfilled_input_is_rejected():
    g, fixed = graph_and_fixed()
    assert any("wipe.file" in e for e in node_check.validate(g, {}, SCHEMAS, {"photo.image"}))


@pytest.mark.parametrize("name,license,source,message", [
    ("../../etc/passwd", "cc0", "x", "unsafe"),
    ("sub/ref.png", "cc0", "x", "unsafe"),
    ("ref.png", "all rights reserved", "x", "license"),
    ("ref.png", "cc0", "", "where it comes from"),
    ("missing.png", "cc0", "x", "missing"),
])
def test_unsafe_or_unlicensed_assets_are_rejected(tmp_path, name, license, source, message):
    (tmp_path / "ref.png").write_bytes(b"png")
    errors = b.check_assets(tmp_path, [Asset(name, license, source)])
    assert any(message in e for e in errors)


def test_a_licensed_asset_in_the_folder_passes(tmp_path):
    (tmp_path / "ref.png").write_bytes(b"png")
    assert b.check_assets(tmp_path, [Asset("ref.png", "cc0", "Wikimedia Commons, public domain")]) == []


def fake_runner(monkeypatch, ok=True, face=0.6, gain=0.1, subject=0.7, intruders=(0.0,)):
    from freefield import runner

    scores = list(intruders)
    seen = []

    class Intruders:
        def scan(self, clip, words):
            seen.append((clip, words))
            score = scores.pop(0) if len(scores) > 1 else scores[0]
            return {"score": score, "frame": 12 if score else None, "label": "a human hand" if score else None}

    monkeypatch.setattr(edge_check, "Intruders", Intruders)

    def run(compiled, out_dir, **kw):
        out_dir.mkdir(parents=True, exist_ok=True)
        clip, still = out_dir / "clip.mp4", out_dir / "still.png"
        clip.write_bytes(b"x" * 20000)
        still.write_bytes(b"png")
        return runner.RunResult("j", "succeeded" if ok else "failed", 1.0,
                                {"clip_save": [clip], "keyframe_save": [still]} if ok else {}, error=None if ok else "boom")

    class Faces:
        def similarity(self, a, b):
            return face

    class Subject:
        def similarity(self, a, b):
            return subject

        def text_similarity(self, text, path):
            return gain if "still" in str(path) or "last" in str(path) else 0.0

    monkeypatch.setattr(runner, "run", run)
    monkeypatch.setattr(vision, "Faces", Faces)
    monkeypatch.setattr(vision, "Subject", Subject)
    return seen


def built(tmp_path, text='a robot, then it dances, with a reveal, titled "BOT"'):
    (tmp_path / "me.png").write_bytes(b"png")
    return b.draft(recipe.parse(text), comfy_url=None, drafts=tmp_path / "drafts")


def test_a_draft_folder_holds_the_graph_metadata_and_generated_assets(tmp_path):
    d = built(tmp_path)
    assert d.errors == [] and d.schema_source == "snapshot"
    files = {p.name for p in d.folder.iterdir()}
    assert {"graph.json", "preset.json", "stages.json", "build.json", "bot-wipe.mp4", "bot-header.png"} <= files
    data = json.loads((d.folder / "preset.json").read_text())
    assert data["source"]["description"].startswith("a robot") and data["source"]["recipe"]["stages"][2] == "reveal"
    assert data["asset_licenses"] == {"bot-wipe.mp4": "generated", "bot-header.png": "generated"}
    assert "apache-2.0" in data["license"] and json.loads((d.folder / "build.json").read_text())["status"] == "drafted"


def test_an_unlicensed_reference_makes_a_rejected_draft(tmp_path):
    ref = tmp_path / "art.png"
    ref.write_bytes(b"png")
    r = recipe.parse("a robot")
    r.reference = Asset("art.png", "unknown", "found online", ref)
    d = b.draft(r, comfy_url=None, drafts=tmp_path / "drafts")
    assert any("license" in e for e in d.errors)
    assert json.loads((d.folder / "build.json").read_text())["status"] == "rejected"
    with pytest.raises(recipe.BuildError, match="rejected"):
        b.preview(d.folder, ref)


def test_a_failed_preview_stays_a_draft_and_cannot_be_published(tmp_path, monkeypatch):
    fake_runner(monkeypatch, ok=False)
    d = built(tmp_path)
    rec = b.preview(d.folder, tmp_path / "me.png")
    assert rec["status"].startswith("draft") and not rec["preview"]["passed"]
    with pytest.raises(recipe.BuildError, match="stays a draft"):
        b.publish(d.folder, approved=True, presets=tmp_path / "presets")
    assert not (tmp_path / "presets").exists()


def test_a_preview_that_loses_the_face_stays_a_draft(tmp_path, monkeypatch):
    fake_runner(monkeypatch, face=0.1)
    d = built(tmp_path)
    assert not b.preview(d.folder, tmp_path / "me.png")["preview"]["passed"]


def test_publishing_needs_approval_and_a_passed_preview(tmp_path, monkeypatch):
    fake_runner(monkeypatch)
    d = built(tmp_path)
    assert b.preview(d.folder, tmp_path / "me.png")["status"] == "ready for approval"
    with pytest.raises(recipe.BuildError, match="approval"):
        b.publish(d.folder, approved=False, presets=tmp_path / "presets")
    target = b.publish(d.folder, approved=True, presets=tmp_path / "presets")
    data = json.loads((target / "preset.json").read_text())
    assert data["preview"] == "preview.mp4" and "approved" in data["source"]
    assert not (target / "build.json").exists() and not (target / "render").exists()
    from freefield.presets import load_presets
    presets, errors = load_presets(tmp_path / "presets")
    assert "bot" in presets and errors == {}


def test_a_built_preset_cannot_replace_a_shipped_one(tmp_path, monkeypatch):
    fake_runner(monkeypatch)
    d = built(tmp_path, 'a robot, titled "BOT"')
    b.preview(d.folder, tmp_path / "me.png")
    shipped = tmp_path / "presets" / "bot"
    shipped.mkdir(parents=True)
    (shipped / "preset.json").write_text(json.dumps({"family": "museum"}))
    with pytest.raises(recipe.BuildError, match="already called"):
        b.publish(d.folder, approved=True, presets=tmp_path / "presets")


def test_a_stylised_result_still_shows_the_same_subject_and_not_another():

    kf = Path(__file__).parent / "fixtures" / "builder" / "paper-me-heldout-1-keyframe.jpg"
    fixtures = Path(__file__).resolve().parents[1] / "eval" / "fixtures"
    subject = vision.Subject()
    assert subject.similarity(fixtures / "heldout-1.png", kf) >= vision.SUBJECT_MIN
    assert subject.similarity(fixtures / "heldout-2.png", kf) < vision.SUBJECT_MIN


def test_side_by_side_ends_at_an_even_size_the_encoder_accepts():
    g, _, _ = b.assemble(recipe.parse("a robot, side by side with my photo"))
    even = g["split_even"]["inputs"]
    assert even["width"] % 2 == 0 and even["height"] % 2 == 0 and g["watermark"]["inputs"]["images"] == ["split_even", 0]



def test_a_preview_that_does_not_show_the_description_stays_a_draft(tmp_path, monkeypatch):
    fake_runner(monkeypatch, gain=0.0)
    d = built(tmp_path)
    rec = b.preview(d.folder, tmp_path / "me.png")
    assert not rec["preview"]["passed"] and rec["preview"]["description_gain"] == 0.0


@pytest.mark.parametrize("text,subject,keep,caption,becomes", [
    ("my dog becomes a tiny plush toy, then it gets one gentle squeeze and springs back", "pet", False, False, "a tiny plush toy"),
    ("my sneaker becomes a hero shot frozen in a splash of water, then the camera makes a slow orbit", "object", False, False,
     "a hero shot frozen in a splash of water"),
    ("my photo of a place stays the same, then the camera pulls back through the clouds until the whole Earth is visible, "
     "with a city name", "scene", True, True, ""),
    ("my cat becomes an invented human character with its colors, then they wink", "pet", False, False,
     "an invented human character with its colors"),
])
def test_pets_objects_and_places_are_understood(text, subject, keep, caption, becomes):
    r = recipe.parse(text)
    assert (r.subject, r.keep, r.caption, r.becomes) == (subject, keep, caption, becomes)


def test_a_kept_photo_goes_straight_to_video_with_a_caption_slot():
    r = recipe.parse("my photo of a place stays the same, then the camera pulls back into space, with a city name")
    g, fixed, _ = b.assemble(r)
    assert "klein_model" not in g and g["video_latent"]["inputs"]["start_image"] == ["start", 0]
    assert g["watermark"]["inputs"]["images"] == ["caption", 0] and "keyframe_prompt.text" not in fixed
    assert node_check.validate(g, fixed, SCHEMAS, {"photo.image", "caption.text"}) == []
    data = b.manifest(r, fixed, g, [], "now")
    assert data["inputs"]["caption"]["type"] == "text" and data["outputs"] == ["clip_save"]
    assert data["params"]["seed"]["bind"] == ["video_sample.seed"]


def test_a_pet_that_becomes_a_human_is_not_held_to_its_own_likeness():
    r = recipe.parse("my cat becomes an invented human character, then they wink")
    assert not r.keeps_subject() and "an ordinary invented person with an original face" in recipe.keyframe_prompt(r).lower()
    assert recipe.parse("my dog becomes a plush toy").keeps_subject()


def test_the_caption_is_typed_per_run(tmp_path):
    from freefield.presets import compile_prompt, parse_preset

    r = recipe.parse("my mug stays the same, then the camera orbits it, with a caption")
    d = b.draft(r, comfy_url=None, drafts=tmp_path)
    preset = parse_preset(json.loads((d.folder / "preset.json").read_text()), d.folder)
    photo = tmp_path / "mug.png"
    photo.write_bytes(b"png")
    run = compile_prompt(preset, {"photo": str(photo), "caption": "LISBON"})
    assert run.graph["caption"]["inputs"]["text"] == "LISBON"


def test_a_name_without_a_title_card_and_plural_keep():
    r = recipe.parse('my two plush toys stay the same, then they hug, named "Toy Hug"')
    assert r.name == "Toy Hug" and r.id == "toy-hug" and r.keep and r.title is None and r.motion == "they hug"


def test_a_kept_photo_that_visibly_moves_passes_even_when_clip_cannot_name_the_motion(tmp_path, monkeypatch):

    fake_runner(monkeypatch, gain=0.0, subject=0.9)
    monkeypatch.setattr(vision, "motion", lambda clip: 0.15)
    (tmp_path / "me.png").write_bytes(b"png")
    d = b.draft(recipe.parse('my two plush toys stay the same, then they hug, named "Toy Hug"'), comfy_url=None, drafts=tmp_path / "d")
    assert b.preview(d.folder, tmp_path / "me.png")["preview"]["passed"]
    monkeypatch.setattr(vision, "motion", lambda clip: 0.01)
    assert not b.preview(d.folder, tmp_path / "me.png")["preview"]["passed"]


def boxes(*frames):
    return {i * 6: list(dets) for i, dets in enumerate(frames)}


HAND_AT_EDGE = ("a human hand", 0.4, [380, 300, 480, 420])


def test_a_hand_that_comes_in_from_the_edge_and_stays_is_an_intrusion():

    found = edge_check.intrusion(boxes([], [], [HAND_AT_EDGE], [HAND_AT_EDGE]), 480, 832)
    assert found == {"score": 0.4, "frame": 12, "label": "a human hand"}


@pytest.mark.parametrize("frames", [
    ([HAND_AT_EDGE], [HAND_AT_EDGE], [HAND_AT_EDGE]),
    ([], [HAND_AT_EDGE], [], [HAND_AT_EDGE]),
    ([], [("a pen", 0.6, [200, 300, 300, 420])], [("a pen", 0.6, [200, 300, 300, 420])]),
    ([], [("a pen", 0.6, [0, 300, 20, 360])], [("a pen", 0.6, [0, 300, 20, 360])]),
    ([], [("a knife", 0.15, [380, 300, 480, 420])], [("a knife", 0.15, [380, 300, 480, 420])]),
    ([],),
])
def test_what_is_not_an_intrusion(frames):

    assert edge_check.intrusion(boxes(*frames), 480, 832)["score"] == 0.0


def test_people_may_use_their_own_hands_and_named_tools_are_allowed():

    assert "a human hand" not in edge_check.intruder_words("person", "a paper cutout they pop up")
    assert "a human hand" in edge_check.intruder_words("object", "a plush toy")
    words = edge_check.intruder_words("person", "a painter, then they paint with a paintbrush and scissors")
    assert "a paintbrush" not in words and "scissors" not in words and "a knife" in words


def test_a_hand_coming_into_the_preview_is_re_rendered_with_new_noise(tmp_path, monkeypatch):
    seen = fake_runner(monkeypatch, intruders=(0.45, 0.0))
    d = built(tmp_path, "my mug becomes a frozen splash, then the camera orbits it")
    rec = b.preview(d.folder, tmp_path / "me.png", seed=5)
    p = rec["preview"]
    assert p["passed"] and p["seed"] == 5 + 1009 and p["render_dir"] == "render-1"
    assert [a["passed"] for a in p["attempts"]] == [False, True] and "a human hand" in seen[0][1]
    assert (d.folder / "preview.mp4").exists()


def test_when_every_attempt_has_an_intruder_the_cleanest_stays_a_draft(tmp_path, monkeypatch):
    fake_runner(monkeypatch, intruders=(0.6, 0.3, 0.5))
    d = built(tmp_path, "my mug becomes a frozen splash, then the camera orbits it")
    rec = b.preview(d.folder, tmp_path / "me.png")
    assert rec["status"].startswith("draft") and len(rec["preview"]["attempts"]) == 3
    assert rec["preview"]["intruder"]["score"] == 0.3 and rec["preview"]["render_dir"] == "render-1"
    with pytest.raises(recipe.BuildError, match="stays a draft"):
        b.publish(d.folder, approved=True, presets=tmp_path / "presets")


def test_a_preview_failing_for_another_reason_is_not_re_rendered(tmp_path, monkeypatch):
    fake_runner(monkeypatch, gain=0.0)
    d = built(tmp_path)
    p = b.preview(d.folder, tmp_path / "me.png")["preview"]
    assert not p["passed"] and "attempts" not in p


@pytest.mark.parametrize("clip, subject, bad", [
    ("tests/fixtures/video/plush-squish--second-photo.mp4", "object", True),
    ("tests/fixtures/video/raven-poof--preview.mp4", "object", False),
])
def test_the_detector_on_real_previews(clip, subject, bad):

    root = Path(__file__).resolve().parents[1]
    found = edge_check.Intruders().scan(root / clip, edge_check.intruder_words(subject, ""))
    assert (found["score"] >= edge_check.INTRUDER_MIN) is bad


def test_the_studio_profile_animates_with_the_14b_model():
    r = recipe.parse("my two plush toys stay the same, then they hug")
    r.profile = "studio"
    g, fixed, stages = b.assemble(r)
    assert r.id.endswith("-studio") and "wan_model" not in g and g["motion_latent"]["class_type"] == "WanImageToVideo"
    assert g["motion_latent"]["inputs"]["start_image"] == ["start", 0] and g["watermark"]["inputs"]["images"] == ["motion_frames", 0]
    assert node_check.validate(g, fixed, SCHEMAS, {"photo.image"}) == []
    data = b.manifest(r, fixed, g, [], "now")
    assert data["params"]["seed"]["bind"] == ["motion_sample_high.noise_seed", "motion_sample.noise_seed"]
    assert "studio-profile" in data["tags"] and "unknown" not in data["license"]
    t = recipe.parse("a robot, then it dances")
    t.profile = "studio"
    g, fixed, _ = b.assemble(t)
    assert g["motion_latent"]["inputs"]["start_image"] == ["keyframe", 0] and node_check.validate(g, fixed, SCHEMAS, {"photo.image"}) == []


def test_a_pet_to_human_keyframe_only_sees_a_blurred_colour_guide():
    r = recipe.parse("my cat becomes an invented human character with its colors, then they wink")
    g, fixed, _ = b.assemble(r)
    assert g["photo_tiny"]["inputs"]["width"] <= 40 and g["photo_blur_latent"]["inputs"]["pixels"] == ["photo_blur_scaled", 0]
    assert "photo_photo_latent" not in g and node_check.validate(g, fixed, SCHEMAS, {"photo.image"}) == []
    assert "smooth bare human skin" in recipe.keyframe_prompt(r)


def test_a_cat_human_hybrid_is_re_rendered(tmp_path, monkeypatch):

    fake_runner(monkeypatch)
    margins = iter([0.09, -0.02])

    class Subject:
        def similarity(self, a, b):
            return 0.7

        def text_similarity(self, text, path):
            if text == vision.HYBRID_TEXT:
                return 0.2 + next(margins)
            return 0.2 if text == vision.HUMAN_TEXT else (0.3 if "still" in str(path) else 0.1)

    monkeypatch.setattr(vision, "Subject", Subject)
    d = built(tmp_path, "my cat becomes an invented human character with its colors, then they wink")
    p = b.preview(d.folder, tmp_path / "me.png")["preview"]
    assert p["passed"] and p["hybrid_margin"] == -0.02 and [a["passed"] for a in p["attempts"]] == [False, True]


def test_one_strong_blurred_limb_counts_without_a_second_sample():

    arm = ("a knife", 0.45, [200, 20, 480, 400])
    frames = boxes([], [], [arm], [], [])
    assert edge_check.intrusion(frames, 480, 832)["score"] == 0.0
    assert edge_check.intrusion(frames, 480, 832, changed=lambda i, box: 0.6)["score"] == 0.45
    assert edge_check.intrusion(frames, 480, 832, changed=lambda i, box: 0.1)["score"] == 0.0


def test_a_thin_pencil_at_the_edge_and_things_already_in_frame_0_are_not_intruders():

    pencil = ("a pencil", 0.45, [0, 532, 89, 561])
    assert edge_check.intrusion(boxes([], [pencil], [pencil]), 480, 832)["score"] == 0.0
    big = ("a pen", 0.5, [0, 300, 100, 420])
    assert edge_check.intrusion(boxes([], [big], [big]), 480, 832)["score"] == 0.5
    assert edge_check.intrusion(boxes([], [big], [big]), 480, 832, seen=lambda i, box: True)["score"] == 0.0


def test_the_frame_0_pixel_check():

    rng = np.random.default_rng(0)
    first = (rng.random((200, 120, 3)) * 255).astype(np.uint8)
    moved = np.roll(first, 6, axis=1)
    assert edge_check.already_there(first, moved, (30, 40, 90, 120))
    other = (rng.random((200, 120, 3)) * 255).astype(np.uint8)
    assert not edge_check.already_there(first, other, (30, 40, 90, 120))
    wall, arm = np.full((200, 120, 3), 200, np.uint8), np.full((200, 120, 3), 200, np.uint8)
    arm[:, :30] = 40
    assert edge_check.changed_share(wall, wall, (0, 0, 60, 100)) == 0.0 and 0.4 < edge_check.changed_share(wall, arm, (0, 0, 60, 100)) < 0.6


@pytest.mark.parametrize("folder", sorted(p.parent.name for p in b.PRESETS.glob("*/preset.json")
                                          if json.loads(p.read_text()).get("family") == "built"))
def test_published_built_presets_carry_the_current_builder_prompts(folder):
    data = json.loads((b.PRESETS / folder / "preset.json").read_text())
    rec = data["source"]["recipe"]
    r = recipe.Recipe(**{k: v for k, v in rec.items() if k in recipe.Recipe.__dataclass_fields__ and k != "reference"})
    assert data["fixed"]["motion_prompt.text"] == recipe.motion_prompt(r)
    if "keyframe_prompt.text" in data["fixed"]:
        assert data["fixed"]["keyframe_prompt.text"] == recipe.keyframe_prompt(r)


@pytest.mark.parametrize("folder", sorted(p.parent.name for p in b.PRESETS.glob("*/preset.json")
                                          if json.loads(p.read_text()).get("family") == "built"))
def test_published_built_presets_carry_the_current_description(folder):
    data = json.loads((b.PRESETS / folder / "preset.json").read_text())
    assert data["description"] == recipe.describe(recipe.recipe_from(data["source"]["recipe"]))


def test_a_kept_photo_gets_positive_only_motion_wording():
    r = recipe.parse("my figurine stays the same, then it bursts into a flock of black ravens")
    text = recipe.motion_prompt(r).lower()
    assert r.keep and "no hands" not in text and "no tools" not in text and " no " not in f" {text} "
    assert text.startswith("it bursts into a flock of black ravens.") and "static shot" in text


def test_a_person_description_speaks_to_you():
    r = recipe.parse('Make me a paper cutout, then I slowly turn my head and smile, with a before/after reveal, titled "PAPER ME"')
    assert recipe.describe(r) == "You as a paper cutout; then you slowly turn your head and smile. With a before/after reveal."


@pytest.mark.parametrize("description", [
    "my figurine stays the same, then it bursts into a flock of black ravens",
    "my mascot becomes the same plush mascot wearing a tiny astronaut suit, then it spins around once",
    "my cat becomes an invented human character with the same colors and personality, then they look at the camera and grin",
    "make me a paper cutout, then I slowly turn my head and smile",
])
def test_every_motion_prompt_names_only_what_moves(description):
    text = f" {recipe.motion_prompt(recipe.parse(description)).lower()} "
    assert not any(w in text for w in (" no ", "hand", "tool", "nothing else", "object")), text
    assert "static shot" in text


NEGATIVE = re.compile(r"\b(?:no|not|without|avoid|never|nothing|none)\b|n't", re.I)
POSITIVE_ONLY = ("mascot-outfit-switch", "pet-to-human-studio", "product-freeze")
ROOT = b.PRESETS.parent.parent


def prompt_texts(workflow: dict) -> list[str]:
    nodes = workflow["nodes"] if "nodes" in workflow else [
        {"type": v["class_type"], "widgets_values": [v["inputs"].get("text")]} for v in workflow.values()]
    return [w for n in nodes if n.get("type") == "CLIPTextEncode" for w in n.get("widgets_values") or [] if isinstance(w, str)]


@pytest.mark.parametrize("name", POSITIVE_ONLY)
@pytest.mark.parametrize("kind", ["api.json", "json"])
def test_exported_prompts_of_the_fixed_presets_are_positive_only(name, kind):
    texts = prompt_texts(json.loads((ROOT / "workflows" / f"{name}.{kind}").read_text()))
    assert texts and not [t for t in texts if NEGATIVE.search(t)]


@pytest.mark.parametrize("description", [
    "my mascot becomes the same plush mascot wearing a tiny astronaut suit, then it spins around once",
    "my cat becomes an invented human character with the same colors and personality, then they look at the camera and grin",
])
def test_builder_prompts_for_objects_and_pets_are_positive_only(description):
    r = recipe.parse(description)
    for text in (recipe.keyframe_prompt(r), recipe.motion_prompt(r)):
        assert not NEGATIVE.search(text), text


def test_published_files_of_the_fixed_presets_are_positive_only():
    mascot = b.PRESETS / "mascot-outfit-switch"
    fixed = json.loads((mascot / "preset.json").read_text())["fixed"]
    texts = [fixed["keyframe_prompt.text"], fixed["motion_prompt.text"]]
    texts += prompt_texts(json.loads((mascot / "workflow_api.json").read_text()))
    texts += prompt_texts(json.loads((mascot / "workflow.json").read_text()))
    texts += prompt_texts(json.loads((b.PRESETS.parent / "graphs" / "product_freeze.json").read_text()))
    assert len(texts) >= 5 and not [t for t in texts if NEGATIVE.search(t)]
