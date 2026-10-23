# The graphs and why they are built this way

`scripts/build_graphs.py` writes the shared graphs in `freefield/graphs/` (ComfyUI API format). The stages they share
live in `freefield/graph_stages.py`, so every graph uses the same keyframe, video and save stages. The node ids read
like English (`photo`, `keyframe_prompt`, `motion_prompt`), so a preset's bindings read like English too, and every
node has a title, so the graph explains itself when you open it in ComfyUI. Run
`.runtime/venv/bin/python scripts/build_graphs.py` after changing a stage; a fresh run reproduces the committed files
byte for byte.

The laid-out UI workflows for the gallery presets are in [`workflows/`](../workflows/README.md). Each has numbered
groups and an "About this workflow" note.

## The stages (`graph_stages.py`)
- **`keyframe_stage`**: a FLUX.2 [klein] 4B multi-reference edit. Each reference image is scaled to about 1 MP and
  encoded (or encoded at its own size, `exact`, when it is a scene the edit must not zoom), then attached to both the
  positive and the empty negative conditioning. Four steps, CFG 1, the Flux2 scheduler.
- **`video_stage`**: Wan 2.2 TI2V-5B Turbo (GGUF Q6) animates the keyframe: 73 frames at 24 fps, 4 steps, shift 8.
- **`wan14_models` / `wan14_segment`**: Wan 2.2 I2V-A14B with the lightx2v 4-step LoRAs (the 24 GB studio profile).
  The high-noise expert lays out the motion (steps 0-2), the low-noise expert adds detail (steps 2-4). A segment pins
  its first frame, or its first and last frame.
- **`save_stage`**: the "FreeField · open source" watermark, frames to video, an H.264 MP4.

## The graphs
| Graph | What it makes | Why it is built this way |
|---|---|---|
| `selfie_to_clip` | one photo, one klein keyframe, a 3 s vertical clip | the base chain most presets use |
| `selfie_header_clip` | the same with a static header card above the clip (toy box, magazine) | klein garbles lettering and Wan pushes in on product shots, so a title drawn into the keyframe would come out garbled or leave the frame. The 720x192 card (`freefield/header_card.py`) stays whole above the animated area |
| `selfie_art_split` | you inside an artwork, animated, above the original artwork | the keyframe sees a copy of the artwork with the face blanked (`scripts/blank_face.py`): given a visible face, klein paints the artwork's face instead of yours |
| `selfie_breakout` | Out of the Frame and its companions: painting, START, GRIP, OUT, three Wan 2.2 A14B shots, a proof strip | authored keyframes and a crop-defined camera pin the action; see [HOW-IT-WORKS.md](HOW-IT-WORKS.md#the-breakout-graph) |
| `selfie_breakout_stills` | the same stills without the video | the 8 GB version (klein only) |
| `real_matte`, `real_painting`, `real_shot1` | Out of the Frame with your own clip: a BiRefNet matte per frame; frame 0 repainted in oil plus the empty backdrop; shot 1, where the painting comes alive and settles back into its pose | shot 1 is pinned to the painting at both ends, so your real footage can take over from the same pose |
| `earth_zoom` | three klein stills of your place: globe, satellite, drone | the dive, clouds, title and sound are composited in `freefield/earth_zoom.py` |
| `product_freeze` | your product frozen mid-splash | the parallax camera, text and end card are composited in `freefield/product_freeze.py` |
| `comic_bonk` | a BiRefNet cut-out and a klein clean plate behind it | the bonk animation and sound are in `freefield/comic_bonk.py` |
| `iris_reveal` | loads your selfie and previews it | Iris Reveal uses no generative model, so its graph only satisfies the preset format; the zoom is plain image processing in `freefield/iris_reveal.py` |
| `portrait` | text to a still, no references | made the synthetic test subjects (no real people) and the factory's plate candidates |
| `standin` | a synthetic stand-in for a selfie clip, for testing only | the test face, zoomed out to a propped-phone framing, then leaning in |

## Breakout geometry
A breakout plate is a generated scene with its opening in flat magenta. `scripts/make_plate.py` (or the factory)
writes three files:
- `plate.png`: the scene with the opening emptied.
- `plate_hole.png`: the opening's mask, cropped to its box.
- `wall_mask.png`: white on everything the models may not change. The opening plus a border are open, and a band
  under the bottom rail is open for gripping fingers.

`tests/test_presets.py` checks that each breakout preset's hole, masks and camera crops line up.
