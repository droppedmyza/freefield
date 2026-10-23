# The FreeField preset format

A preset is a folder: `freefield/presets/<id>/`. It holds a `preset.json` manifest, any assets it needs (for
example a public-domain painting), and optionally a `preview.mp4` for the gallery card.

The manifest points at a **ComfyUI workflow in API format**: the JSON you get from ComfyUI's
*Workflow -> Export (API)*. It then says which node inputs the user fills in, which the preset author fixes,
and which node outputs are the result. That's the whole idea. Higgsfield's presets ("chains") have this same
shape: a small graph with a few exposed slots. FreeField writes it down as an open file anyone can make.

## Fields
| Field | Required | Meaning |
|---|---|---|
| `id` | yes | lowercase letters, digits, dashes; unique |
| `name`, `description` | yes / no | shown on the gallery card |
| `family` | no | groups cards in the gallery filter (`museum`, `toy`, `spotlight`, ...) |
| `graph` | yes | a file in the preset folder or in `freefield/graphs/`, or an inline API-format object |
| `inputs` | no | what the user provides. `{"photo": {"type": "image", "label": "...", "bind": ["photo.image"]}}`. Types: `image`, `text` (`max_length`, optional `template` with `{value}`), `video` (only with a `pipeline`) |
| `params` | no | user-tunable values with defaults. Types: `int`, `float` (`min`/`max`), `text`, `choice` (`options`) |
| `fixed` | no | the author's values: prompts, captions. `"asset:<file>"` uploads a file from the preset folder |
| `outputs` | yes | node ids whose files are the result (`clip_save`, `keyframe_save`) |
| `models`, `license`, `credits`, `tags` | no | shown in the app and README; list every weight file and asset source |
| `pipeline` | no | the effect runs as several graphs plus CPU compositing, in the named module of `freefield/` (registered in `freefield/pipelines.py`): `earth_zoom`, `product_freeze`, `iris_reveal`, `comic_bonk`, `raven_intro` or `clip_breakout`. The preset's `graph` and bindings still validate. A text input bound to `pipeline:<name>` is passed to the effect as its text (the place name, the offer, the title or the bonk word) |

A **binding** is `"<node_id>.<input_name>"`. FreeField's shared graphs use readable node ids (`photo`,
`keyframe_prompt`, `motion_prompt`), so bindings read like English. Every binding is validated when presets load:
a typo tells you the node's real inputs instead of failing at render time.

**Words in the picture.** FLUX.2 klein 4B garbles any lettering it draws, and Wan pushes in on almost every product
shot, so words drawn into the keyframe either come out garbled or slide out of the frame. Use `selfie_header_clip.json`:
it stacks a static 720x192 header card (a preset asset) above the animated area, so the title stays whole and sharp.
Ask the keyframe prompt for a scene without words, then render the card:
```bash
.runtime/venv/bin/python scripts/make_header.py "LIMITED EDITION" freefield/presets/<id>/header.png --subtitle "COLLECTOR SERIES"
```
and bind it with `"header.image": "asset:header.png"`. The font is Luckiest Guy (Apache-2.0, `freefield/fonts/`);
`--style magazine` gives a serif masthead (Playfair Display, OFL-1.1) instead of the toy card; `--card`, `--fill` and
`--edge` override the colors. See `presets/action-figure/` and `presets/magazine-cover/`.

## Worked example: add a "Monet Muse" preset in 3 minutes
This is how `freefield/presets/monet-muse/` was made. It reuses the shared split-screen graph
(`freefield/graphs/selfie_art_split.json`): your photo plus an artwork make an identity keyframe, Wan animates it,
and the original artwork goes underneath with a caption. The keyframe gets a copy of the artwork with its face blanked
out. If the model can see the painted face, it copies that face instead of yours.

1. Make the folder, add a public-domain painting (Wikimedia Commons, check the license line), and make the blanked
   copy. Crop to the figure first if the face is small in the full canvas (Monet's is):
   ```bash
   mkdir -p freefield/presets/monet-muse
   cp ~/Downloads/monet_woman_with_a_parasol_crop.jpg freefield/presets/monet-muse/painting.jpg
   .runtime/venv/bin/python scripts/blank_face.py freefield/presets/monet-muse/painting.jpg freefield/presets/monet-muse/painting_noface.jpg
   ```
   If it prints "no face found" (stylized faces, e.g. The Scream), pass the face box by hand: `--box x,y,w,h` (fractions).
2. Write `freefield/presets/monet-muse/preset.json`:
   ```json
   {
     "id": "monet-muse",
     "name": "Monet Muse",
     "family": "museum",
     "description": "You, as the woman with the parasol in Monet's 1875 painting, breeze and all. The original sits underneath.",
     "graph": "selfie_art_split.json",
     "inputs": {"photo": {"type": "image", "label": "A clear photo of your face", "bind": ["photo.image"]}},
     "params": {"seed": {"type": "int", "default": 42, "min": 0, "max": 2147483647,
                          "bind": ["keyframe_noise.noise_seed", "video_sample.seed"]}},
     "fixed": {
       "artwork.image": "asset:painting.jpg",
       "artwork_ref.image": "asset:painting_noface.jpg",
       "keyframe_prompt.text": "Image 2 is an impressionist painting whose face has been left blank. Paint the face and head of the person from image 1 into the blank area, so the painting shows that person as the figure holding a green parasol on a windy hilltop: white summer dress, a light veil fluttering, bright blue sky with white clouds, seen from slightly below, the face turned toward the viewer. Loose impressionist brushwork, Claude Monet, 1875. It must be the same person as image 1: same face, skin tone, age and gender.",
       "motion_prompt.text": "An impressionist painting comes to life. A gentle breeze moves the dress and the veil, the person slowly turns their face toward the viewer and smiles, the parasol tilts slightly. White clouds drift slowly across a smooth blue sky. Static camera, soft daylight.",
       "artwork_caption.text": "Monet, 1875"
     },
     "outputs": ["clip_save", "keyframe_save"],
     "license": "apache-2.0 weights; painting CC0 / public domain"
   }
   ```
   (The shipped file also lists `models`, `credits` and `tags`; they are optional.)
3. Validate it: `.runtime/venv/bin/python -m freefield list`. It appears, or prints exactly what is wrong.
4. Try it: `.runtime/venv/bin/python -m freefield run monet-muse --photo eval/fixtures/dev-1.png`.
5. Give it a gallery preview: copy a clip it rendered to `freefield/presets/monet-muse/preview.mp4` (and optionally a
   `poster.jpg`). The gallery card plays it.

## Bring your own workflow
Any ComfyUI workflow works, not only FreeField's shared graphs. Build it in ComfyUI, export it as API format,
save it next to `preset.json`, and bind its inputs. Keep the outputs as `SaveVideo` / `SaveImage` nodes, and
give the important nodes titles (`_meta.title`), so people can read the graph.

## Breakout presets (studio profile) and the preset factory
Out of the Frame, Out of the Box and Out of the Photo share one graph, `graphs/selfie_breakout.json`; each preset
supplies a plate, masks, geometry and prompts. Write a short **spec** and let the factory build the rest:

    python -m freefield factory specs/out-of-the-photo.json --photo me.jpg

Spec fields (`freefield/factory.py: Spec`; unknown fields are rejected):
| Field | What it is |
|---|---|
| `id`, `name`, `description` | as in any preset; the description must say what the clip actually shows |
| `scene` | text for the plate: the scene with its opening "filled with flat, solid, pure magenta" (required wording) |
| `subject` | what you become inside the opening (image 1 = your photo) |
| `grip`, `out` | the GRIP and OUT keyframe edits (OUT is built from the empty plate with your transformed self as its own reference unless `out_from_empty` is false) |
| `shot1`, `shot2`, `shot3` | motion prompts: shot 1 has a still camera, shot 2 pulls back, shot 3 is the second beat |
| `border`, `below` | px around / under the opening the models may change (everything else is locked to the plate); `below` is also the gate's rail |
| `grip_edge` | `bottom` (default) or `top`: which edge the hands grip |
| `identity_min`, `painting_identity_min` | gate thresholds for this preset's faces |
| `plate_seeds` | how many plate candidates to render |
| `hole_fill` | RGB colour the opening is filled with once the subject has left it (default `[38, 30, 24]`, a dark empty canvas; Out of the Photo uses `[226, 214, 190]`, a blank faded print) |

The factory renders the plate candidates and keeps the cleanest opening, writes `plate.png`, `plate_hole.png` and
`wall_mask.png`, derives the geometry and 9:16 camera crops from the opening, writes `preset.json`, then runs the gated
preview. It **publishes** to `presets/<id>/` only if the preview's keyframes pass the gate; otherwise the preset stays in
`presets/_drafts/<id>/` with `factory_report.json` saying which check failed. It never adds nodes or installs anything.

Breakout preset manifests have two extras:
- **`gate`**: `hole` (the opening, x/y/w/h), `rerolls`, `frame_border`, `rail`, `grip_edge`, `facial_hair`,
  `identity_min`, `painting_identity_min`. `freefield/gate.py` checks START/GRIP/OUT on the CPU before any video and
  re-rolls the keyframe seeds while a check fails; every attempt is written to `gate.json` next to the clip. After the
  video renders, an end check compares your photo with the faces in the clip's last 12 frames (`end_identity` in
  `gate.json`); below 0.20 the run carries an Unverified warning, shown in the CLI JSON and the web app.
  `shot1_corners: true` adds the shot-1 check (`shot1_intrusion`: new content over the opening's dark backdrop in its
  upper corners; above 0.25 is an arm that did not come from the sitter) and `video_rerolls` (0-4) re-renders shot 1
  alone with new noise while a video check fails; every attempt is in `gate.json` under `video_attempts`. Only the
  signature turns these on: the check assumes a dark backdrop inside the opening.
- **Links in `fixed`**: a value may be a link (`["node_id", output]`) as well as a literal, to rewire an input. The
  factory uses this to build OUT from the empty plate: `"breakout_start_frame_latent.pixels": ["plate", 0]`.
