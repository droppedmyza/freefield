#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from freefield.graph_stages import keyframe_stage, node, save_stage, video_stage, wan14_models, wan14_segment  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "freefield" / "graphs"

BIREFNET = "birefnet.safetensors"

REAL_PAINTING = ("Repaint image 1 as an oil painting on canvas in the Dutch Golden Age style, with visible brushstrokes "
                 "and fine craquelure. Change only the medium: keep the same person, the same face and expression, the "
                 "same pose, the same clothes, the same light and the same composition, with the dark, warm, softly lit "
                 "background of an old master portrait.")
REAL_BACKDROP = ("Repaint image 1 as the dark, warm, empty painted background of a Dutch Golden Age oil portrait: visible "
                 "brushstrokes and fine craquelure, softly lit, with nothing and nobody in it.")
REAL_SHOT1 = ("The oil painting in the gold frame comes alive: the painted person takes a deep breath, their eyes move "
              "and find the viewer, their eyebrows lift, they tilt their head and break into a big laugh, their shoulders "
              "rise and fall, then they settle back into the same pose and smile. Their hands stay out of view. They "
              "are still a painting, brushstrokes visible. The camera holds perfectly still. The wall and the frame "
              "stay still.")
STANDIN_WIDE = ("Image 1 is a selfie. Make a photo of the exact same person taken by a phone propped on a shelf about two "
                "metres away: they sit at a table, small in the picture, their head only about a tenth of the picture's "
                "height, their whole upper body and their arms resting on the table, with plenty of space above and around "
                "them, looking into the lens with a calm face, the same clothes, in a plain room with natural window "
                "light. Vertical phone photo, photorealistic. The same face, skin tone, age and hair as image 1.")
STANDIN_LEAN = ("Video from a phone propped on a shelf. The person at the table sits still looking into the lens for a "
                "second, then stands up and leans far forward toward the phone, coming much closer, until their head and "
                "shoulders fill most of the picture, breaking into a big warm smile. Their hands stay down out of the "
                "picture. Natural light, real footage. The camera does not move.")


def selfie_to_clip() -> dict:
    g: dict = {}
    g["photo"] = node("LoadImage", "Your photo (preset input)", image="")
    keyframe_stage(g, ["photo"], width=720, height=1280)
    video_stage(g, width=480, height=832, length=73)
    save_stage(g, ["video_frames", 0])
    return g


def selfie_header_clip() -> dict:
    g: dict = {}
    g["photo"] = node("LoadImage", "Your photo (preset input)", image="")
    g["header"] = node("LoadImage", "Header card, 720x192 (preset asset)", image="")
    keyframe_stage(g, ["photo"], width=720, height=1056)
    g["keyframe_with_header"] = node("ImageStitch", "Header above the keyframe (saved still)", image1=["header", 0],
                                     image2=["keyframe", 0], direction="down", match_image_size=True,
                                     spacing_width=0, spacing_color="black")
    g["keyframe_save"]["inputs"]["images"] = ["keyframe_with_header", 0]
    video_stage(g, width=480, height=704, length=73)
    g["header_small"] = node("ImageScale", "Header at video width", image=["header", 0], upscale_method="lanczos",
                             width=480, height=128, crop="disabled")
    g["with_header"] = node("ImageStitch", "Header above every frame", image1=["header_small", 0], image2=["video_frames", 0],
                            direction="down", match_image_size=True, spacing_width=0, spacing_color="black")
    save_stage(g, ["with_header", 0])
    return g


def selfie_art_split() -> dict:
    g: dict = {}
    g["photo"] = node("LoadImage", "Your photo (preset input)", image="")
    g["artwork"] = node("LoadImage", "Artwork (preset asset, public domain)", image="")
    g["artwork_ref"] = node("LoadImage", "Artwork with the face blanked (keyframe reference)", image="")
    keyframe_stage(g, ["photo", "artwork_ref"], width=1088, height=960)
    video_stage(g, width=576, height=512, length=73)
    g["artwork_panel"] = node("ImageScale", "Fit the original artwork to the bottom panel", image=["artwork", 0],
                              upscale_method="lanczos", width=576, height=512, crop="center")
    g["artwork_caption"] = node("TextOverlay", "Caption under the original", images=["artwork_panel", 0], text="",
                                font_size=4.5, color="#ffffff", position="bottom", align="center", outline=True)
    g["split"] = node("ImageStitch", "You on top, the original below", image1=["video_frames", 0], image2=["artwork_caption", 0],
                      direction="down", match_image_size=True, spacing_width=0, spacing_color="black")
    save_stage(g, ["split", 0])
    return g


def view(g: dict, nid: str, title: str, image: list, box: tuple[int, int, int, int], size: tuple[int, int]) -> None:
    x, y, w, h = box
    g[f"{nid}_crop"] = node("ImageCrop", f"{title}: crop (camera position)", image=image, width=w, height=h, x=x, y=y)
    g[nid] = node("ImageScale", f"{title}: at shot size", image=[f"{nid}_crop", 0], upscale_method="lanczos",
                  width=size[0], height=size[1], crop="center")


BREAKOUT_W, BREAKOUT_H = 720, 1280
BREAKOUT_HOLE = (210, 347, 300, 455)
BREAKOUT_START_VIEW = (110, 118, 500, 889)


def breakout_inputs(g: dict) -> None:
    g["photo"] = node("LoadImage", "Your photo (preset input)", image="")
    g["artwork_ref"] = node("LoadImage", "Artwork with the face blanked (keyframe reference)", image="painting_noface.jpg")
    g["plate"] = node("LoadImage", "Gallery wall with an empty frame (preset asset)", image="plate.png")
    g["plate_hole"] = node("LoadImageMask", "Frame opening mask, cropped to the opening (preset asset)", image="plate_hole.png", channel="red")
    g["wall"] = node("LoadImage", "Lock mask: white = wall (and, for Out of the Frame, the gilt) the model may not touch (preset asset, make_plate.py)", image="wall_mask.png")
    g["wall_mask"] = node("ImageToMask", "Wall mask", image=["wall", 0], channel="red")
    g["frame_zone"] = node("InvertMask", "Frame zone (opening, frame, rail)", mask=["wall_mask", 0])


def breakout_keyframes(g: dict) -> None:
    hole, w, h = BREAKOUT_HOLE, BREAKOUT_W, BREAKOUT_H
    keyframe_stage(g, ["photo", "artwork_ref"], width=512, height=656, prefix="painting", label="1a. Costume photo")
    keyframe_stage(g, ["painting", "photo"], width=512, height=656, prefix="likeness", label="1b. Repaint in oil")
    g["painting_fit"] = node("ImageScale", "2. START: fit the painting to the frame opening", image=["likeness", 0],
                             upscale_method="lanczos", width=hole[2], height=hole[3], crop="center")
    g["start_frame"] = node("ImageCompositeMasked", "2. START: hang the painting in the frame", destination=["plate", 0],
                            source=["painting_fit", 0], x=hole[0], y=hole[1], resize_source=False, mask=["plate_hole", 0])
    g["start_save"] = node("SaveImage", "2. START: save", images=["start_frame", 0], filename_prefix="freefield/start")
    keyframe_stage(g, ["start_frame", "photo"], width=w, height=h, prefix="grip", label="3. GRIP", exact=("start_frame",))
    keyframe_stage(g, ["photo", "start_frame"], width=w, height=h, prefix="breakout", label="4. OUT", exact=("start_frame",))
    g["grip_locked"] = node("ImageCompositeMasked", "3. GRIP: keep only the frame zone, the wall stays START's",
                            destination=["start_frame", 0], source=["grip", 0], x=0, y=0, resize_source=False, mask=["frame_zone", 0])
    g["grip_save"]["_meta"]["title"] = "3. GRIP: save (before the wall lock; the gate checks this one)"
    g["breakout_save"]["_meta"]["title"] = "4. OUT: save"


def proof_strip(g: dict) -> None:
    prev = ["photo", 0]
    for i, (img, label) in enumerate((("likeness", "painting"), ("start_frame", "start"), ("grip_locked", "grip"), ("breakout", "out"))):
        g[f"proof_{i + 1}"] = node("ImageStitch", f"Proof strip: + {label}", image1=prev, image2=[img, 0], direction="right",
                                   match_image_size=True, spacing_width=8, spacing_color="white")
        prev = [f"proof_{i + 1}", 0]
    g["proof_save"] = node("SaveImage", "Save the how-it-was-made strip", images=prev, filename_prefix="freefield/proof")


def breakout_shots(g: dict) -> None:
    view_box, w, h = BREAKOUT_START_VIEW, BREAKOUT_W, BREAKOUT_H
    view(g, "start_view", "5. Camera: shot 1 opens tight on the painting", ["start_frame", 0], view_box, (w, h))
    view(g, "grip_view", "5. Camera: shot 1 holds still, shot 2 opens here", ["grip_locked", 0], view_box, (w, h))
    wan14_models(g)
    wan14_segment(g, "shot1", "6. Shot 1, the painting comes alive", ["start_view", 0], ["grip_view", 0], w, h, 41)
    wan14_segment(g, "shot2", "7. Shot 2, out of the frame", ["grip_view", 0], ["breakout", 0], w, h, 41)
    g["shot1_wall_crop"] = node("ImageCrop", "6. Shot 1: wall mask in the camera's crop", image=["wall", 0],
                                width=view_box[2], height=view_box[3], x=view_box[0], y=view_box[1])
    g["shot1_wall_scaled"] = node("ImageScale", "6. Shot 1: wall mask at shot size", image=["shot1_wall_crop", 0],
                                  upscale_method="bilinear", width=w, height=h, crop="disabled")
    g["shot1_wall_mask"] = node("ImageToMask", "6. Shot 1: wall mask", image=["shot1_wall_scaled", 0], channel="red")
    g["shot1_locked"] = node("ImageCompositeMasked", "6. Shot 1: the wall stays the wall in every frame",
                             destination=["shot1_frames", 0], source=["start_view", 0], x=0, y=0, resize_source=False,
                             mask=["shot1_wall_mask", 0])
    g["shot2_last"] = node("ImageFromBatch", "7. Shot 2: last frame", image=["shot2_frames", 0], batch_index=40, length=1)
    wan14_segment(g, "shot3", "8. Shot 3, the second beat", ["shot2_last", 0], None, w, h, 33)


def joined_shots(g: dict) -> None:
    g["shot1_head"] = node("ImageFromBatch", "9. Shot 1 without its pinned last frame", image=["shot1_locked", 0],
                           batch_index=0, length=40)
    g["shot2_tail"] = node("ImageFromBatch", "9. Shot 2 without its pinned first frame", image=["shot2_frames", 0],
                           batch_index=1, length=40)
    g["shot3_tail"] = node("ImageFromBatch", "9. Shot 3 without its first frame (= shot 2's last)", image=["shot3_frames", 0],
                           batch_index=1, length=32)
    g["shots_12"] = node("ImageBatch", "9. Shot 1 + shot 2", image1=["shot1_head", 0], image2=["shot2_tail", 0])
    g["shots_123"] = node("ImageBatch", "9. + shot 3 (112 frames, 7 s)", image1=["shots_12", 0], image2=["shot3_tail", 0])
    save_stage(g, ["shots_123", 0], fps=16.0)


def selfie_breakout(video: bool = True) -> dict:
    g: dict = {}
    breakout_inputs(g)
    breakout_keyframes(g)
    proof_strip(g)
    if video:
        breakout_shots(g)
        joined_shots(g)
    return g


def with_preset_defaults(g: dict, preset: str) -> dict:
    data = json.loads((OUT.parent / "presets" / preset / "preset.json").read_text())
    for bind, value in data["fixed"].items():
        nid, inp = bind.split(".", 1)
        if nid in g:
            g[nid]["inputs"][inp] = value[6:] if isinstance(value, str) and value.startswith("asset:") else value
    return g


def portrait() -> dict:
    g: dict = {}
    keyframe_stage(g, [], width=768, height=1024)
    return g


def real_matte() -> dict:
    g: dict = {}
    g["clip"] = node("LoadVideo", "Your selfie clip (normalised to 24 fps, 540x960 by freefield)", file="")
    g["clip_frames"] = node("GetVideoComponents", "Clip: frames, sound, frame rate", video=["clip", 0])
    g["bg_model"] = node("LoadBackgroundRemovalModel", "BiRefNet background removal (MIT)", bg_removal_name=BIREFNET)
    g["matte"] = node("RemoveBackground", "Matte every frame (you vs the room)", bg_removal_model=["bg_model", 0],
                      image=["clip_frames", 0])
    g["matte_image"] = node("MaskToImage", "Matte as images", mask=["matte", 0])
    g["matte_save"] = node("SaveImage", "Save the mattes (one PNG per frame)", images=["matte_image", 0],
                           filename_prefix="freefield/matte")
    return g


def real_painting() -> dict:
    g: dict = {}
    g["portrait_in"] = node("LoadImage", "Frame 0 in the opening (made by freefield.clip_breakout)", image="")
    g["backdrop_in"] = node("LoadImage", "The empty opening (made by freefield.clip_breakout)", image="")
    keyframe_stage(g, ["portrait_in"], width=592, height=896, prefix="painting", label="Your portrait", exact=("portrait_in",))
    keyframe_stage(g, ["backdrop_in"], width=592, height=896, prefix="backdrop", label="Empty canvas", exact=("backdrop_in",))
    g["painting_prompt"]["inputs"]["text"] = REAL_PAINTING
    g["backdrop_prompt"]["inputs"]["text"] = REAL_BACKDROP
    return g


def real_shot1() -> dict:
    g: dict = {}
    g["start"] = node("LoadImage", "The painting on the wall, at shot size (made by freefield.clip_breakout)", image="")
    wan14_models(g)
    wan14_segment(g, "shot1", "Shot 1, the painting comes alive", ["start", 0], ["start", 0], 720, 1280, 41)
    g["shot1_prompt"]["inputs"]["text"] = REAL_SHOT1
    g["shot1_save"] = node("SaveImage", "Save shot 1 frames (freefield locks the wall and checks them)",
                           images=["shot1_frames", 0], filename_prefix="freefield/shot1")
    return g


def standin() -> dict:
    g: dict = {}
    g["photo"] = node("LoadImage", "Synthetic test face", image="")
    keyframe_stage(g, ["photo"], width=720, height=1280, prefix="wide", label="Propped-phone framing")
    g["wide_prompt"]["inputs"]["text"] = STANDIN_WIDE
    wan14_models(g)
    wan14_segment(g, "lean", "Lean in toward the phone", ["wide", 0], None, 720, 1280, 81)
    g["lean_prompt"]["inputs"]["text"] = STANDIN_LEAN
    g["video"] = node("CreateVideo", "Frames to video", images=["lean_frames", 0], fps=16.0)
    g["clip_save"] = node("SaveVideo", "Save the stand-in clip", video=["video", 0], filename_prefix="freefield/standin",
                          **{"format": "mp4", "format.codec": "h264"})
    return g


EZ_DRONE = ("Show this exact place from high above: a straight-down aerial drone photo of the same place, the same roofs, "
            "streets, trees, land and any water, seen from 400 meters up, natural daylight, photorealistic, no text.")
EZ_SATELLITE = ("Zoom out. The sharp picture in the middle is a small part of a wider straight-down satellite photo taken "
                "from much higher. Replace the flat gray border with the surrounding land, fields, hills and any coast or "
                "water, with scattered white clouds, continuing seamlessly from the middle picture. Keep the middle picture "
                "exactly as it is. Photorealistic satellite imagery, no text, no labels.")
EZ_GLOBE = ("The whole planet Earth seen from space: a round blue globe with white clouds, oceans and continents, floating "
            "in black space with faint stars, centered in the frame. Photorealistic, no text.")


def earth_zoom() -> dict:
    g: dict = {}
    g["frame0"] = node("LoadImage", "The first frame of your clip (or your photo), 720x1280", image="")
    keyframe_stage(g, ["frame0"], width=720, height=1280, prefix="drone", label="Drone view")
    g["drone_prompt"]["inputs"]["text"] = EZ_DRONE
    g["gray"] = node("EmptyImage", "Gray canvas (the zoomed-out view is painted around the drone view)", width=720, height=1280,
                     batch_size=1, color=0x808080)
    g["drone_small"] = node("ImageScale", "Drone view at a third of the size", image=["drone", 0], upscale_method="lanczos",
                            width=240, height=427, crop="disabled")
    g["outpaint"] = node("ImageCompositeMasked", "Drone view in the middle of the gray canvas", destination=["gray", 0],
                         source=["drone_small", 0], x=240, y=427, resize_source=False)
    keyframe_stage(g, ["outpaint"], width=720, height=1280, prefix="satellite", label="Satellite view")
    g["satellite_prompt"]["inputs"]["text"] = EZ_SATELLITE
    keyframe_stage(g, ["satellite"], width=720, height=1280, prefix="globe", label="Earth from space")
    g["globe_prompt"]["inputs"]["text"] = EZ_GLOBE
    return g


FREEZE = ("Image 1 is a product photo. Make a premium advertising hero shot of the exact same product from image 1, "
          "unchanged in shape, colors, material, logo and details, frozen mid-air at the instant it splashes into water: "
          "a crown of clear water and droplets suspended around it, a clean bright studio backdrop, soft reflections, "
          "crisp high-speed photography. The product is the only subject in the picture, with calm empty space at the "
          "top of the frame.")


def product_freeze() -> dict:
    g: dict = {}
    g["product"] = node("LoadImage", "Your product photo", image="")
    keyframe_stage(g, ["product"], width=720, height=1280, prefix="freeze", label="Frozen splash")
    g["freeze_prompt"]["inputs"]["text"] = FREEZE
    return g


def iris_reveal() -> dict:
    g: dict = {}
    g["photo"] = node("LoadImage", "Your selfie (full resolution; the zoom uses only these pixels)", image="")
    g["photo_preview"] = node("PreviewImage", "Your selfie as given", images=["photo", 0])
    return g


BONK_PLATE = ("Remove the toy from image 1 completely and show the empty background where it stood: the same surfaces, the "
              "same objects, the same light and the same camera angle. Nothing else changes. No toy, no people, no text.")


def comic_bonk() -> dict:
    g: dict = {}
    g["photo"] = node("LoadImage", "Your mascot, toy or cartoon (3:4)", image="")
    g["bg_model"] = node("LoadBackgroundRemovalModel", "BiRefNet background removal (MIT)", bg_removal_name=BIREFNET)
    g["matte"] = node("RemoveBackground", "Cut out the mascot", bg_removal_model=["bg_model", 0], image=["photo", 0])
    g["matte_image"] = node("MaskToImage", "Matte as an image", mask=["matte", 0])
    g["matte_save"] = node("SaveImage", "Save the matte", images=["matte_image", 0], filename_prefix="freefield/bonk_matte")
    keyframe_stage(g, ["photo"], width=768, height=1024, prefix="plate", label="Clean plate")
    g["plate_prompt"]["inputs"]["text"] = BONK_PLATE
    return g


GRAPHS = {
    "selfie_to_clip.json": selfie_to_clip,
    "selfie_header_clip.json": selfie_header_clip,
    "selfie_art_split.json": selfie_art_split,
    "selfie_breakout.json": lambda: with_preset_defaults(selfie_breakout(), "out-of-the-frame"),
    "selfie_breakout_stills.json": lambda: with_preset_defaults(selfie_breakout(video=False), "out-of-the-frame"),
    "portrait.json": portrait,
    "real_matte.json": real_matte,
    "real_painting.json": real_painting,
    "real_shot1.json": real_shot1,
    "standin.json": standin,
    "earth_zoom.json": earth_zoom,
    "product_freeze.json": product_freeze,
    "iris_reveal.json": iris_reveal,
    "comic_bonk.json": comic_bonk,
}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, build in GRAPHS.items():
        (OUT / name).write_text(json.dumps(build(), indent=2) + "\n")
        print("wrote", OUT / name)
