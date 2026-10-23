# How FreeField works

FreeField turns one photo (or a short clip) into a finished vertical video. Every effect is a ComfyUI graph plus a
`preset.json` manifest. The app and the CLI submit the graph through the Comfy SDK. Some effects then finish the
video on the CPU.

```
 browser gallery ──► freefield app (aiohttp) ──► Comfy SDK (comfy-sdk, Comfy API v2)
 python -m freefield ─┘                             │
                     COMFY_BASE_URL=http://127.0.0.1:8189 (default)    unset + COMFY_API_KEY
                                   │                                          │
                          comfy-api-proxy ──► your ComfyUI          Comfy Cloud / Comfy API deployment
```

## Where things live
| Module | What it does |
|---|---|
| `cli.py` | The `freefield` command. It checks the inputs before any work: the preset, that the file is a readable photo or clip of the right kind (and length), the seed and the caption |
| `server.py` | The gallery web app: upload checks, a one-at-a-time run queue, run and draft files |
| `presets.py` | The preset format: load and validate `preset.json`, then fill a graph with a run's inputs |
| `pipelines.py` | The one run path for the CLI and the app: `run_preset()` and the registry of composited effects |
| `runner.py` | Submits a filled graph through the Comfy SDK, streams progress, downloads the outputs |
| `graph_stages.py` | Building blocks for graphs: the FLUX.2 klein keyframe, Wan 2.2 video, the 14B experts, save |
| `media.py` | Shared frame, title, audio and ffmpeg helpers used by the composited effects |
| `earth_zoom.py`, `product_freeze.py`, `iris_reveal.py`, `comic_bonk.py`, `raven_intro.py`, `clip_breakout.py` | One module per composited effect |
| `breakout.py`, `gate.py`, `plate.py`, `factory.py` | The breakout presets (Out of the Frame, Box, Photo): the gated run, the keyframe gate, plate assets and the preset factory |
| `vision.py`, `edge_check.py` | Check models: YuNet + AuraFace faces, OpenCLIP, facial hair, motion; OWLv2 hands and tools at the frame edge |
| `recipe.py`, `builder.py`, `node_check.py`, `output_checks.py`, `ui_export.py`, `header_card.py` | The effect builder: plain words to a recipe, a validated graph, a draft preset, a checked preview and a published card |

## Four kinds of run
`pipelines.run_preset()` picks one from the preset:
1. **A composited effect** (`"pipeline": "earth_zoom"` and the others). The effect's module runs one or more graphs,
   then composes the video on the CPU. Each module has the same entry point, `run(source, seed, out_dir, *, text,
   client, on_stage)`. Each raises an `errors.EffectError` with a message the user can act on.
2. **A gated breakout** (the preset has a `gate`). `breakout.run_gated()` renders only the keyframes and checks them.
   It re-rolls a bad one, then renders the video, checks the clip and re-renders shot 1 if needed.
3. **A built preset** (`family: "built"`). The effect builder made it. `output_checks.checked_run()` renders it and
   checks the clip. It re-rolls when a hand comes in from the edge or a person-animal hybrid appears.
4. **A single graph.** The graph is filled with the photo, the texts and the seed, then rendered.

## The composited effects
| Effect | In the graph | In Python, after the graph |
|---|---|---|
| Earth Zoom | three klein stills: the globe, the satellite view and the drone view of your place | a cloud-cut dive through the stills, the title, a whoosh; your clip plays untouched after the dive (a photo loops back up) |
| Product Freeze | a klein still of your product frozen mid-splash | Depth Anything V2 depth, a parallax camera move, the offer text, an end card. The still and its depth are cached, so a new offer text re-renders in seconds |
| Iris Reveal | none | YuNet finds the eyes, picks the sharper one and refuses low-detail eyes. The zoom uses only your photo's pixels |
| Comic Bonk | a BiRefNet cut-out and a klein clean plate | the mallet swing, the squash-and-recoil spring, the BONK! burst, circling stars and the bonk/boing sound. An OpenCLIP guard refuses real people and pets |
| Raven Intro | the Raven Poof graph (Wan 2.2 5B) | an edge check with re-rolls, then the exposure matched to your photo, your title and a whoosh |
| Out of the Frame, your clip | BiRefNet mattes for every frame, the klein painting, Wan 2.2 A14B shot 1 | the paint lifts off and your own footage plays through the frame: one fixed placement, so leaning in carries you past the frame |

## The breakout graph
The signature graph (`graphs/selfie_breakout.json`) is authored, not improvised. FLUX.2 klein makes three stills and a
camera path made of crops pins the action. Wan 2.2 A14B fills in only the motion between them.
1. **Painting.** Pass 1 makes a photo of you in the artwork's pose and costume. Pass 2 repaints that photo in oil. In
   a single pass klein loses the face.
2. **START.** The painting is composited into the opening of a fixed gallery plate through a fixed mask, the same
   pixels every run.
3. **GRIP.** klein edits START so your painted hands grip the bottom rail. The hands then have an authored path out of
   the painting. GRIP is pasted onto START through the frame zone only, so a hand drawn on the wall disappears.
4. **OUT.** klein edits START into you, real, leaning out of the frame.
5. **Shots.** Shot 1 runs START to GRIP with a still camera. The wall and the gilt are pasted back over every frame,
   so a hand can only come out of the painting. Shot 2 runs GRIP to OUT while the camera pulls back. Shot 3
   continues from OUT. The pinned GRIP frame is sharper than the generated frames around it, so both copies are left
   out of the join.

The gate in `gate.py` checks START, GRIP and OUT on the CPU in about a second, before any video is rendered (see
[CHECKS.md](CHECKS.md)). The factory (`python -m freefield factory specs/out-of-the-photo.json --photo ...`) makes a
new breakout from a scene prompt with a flat magenta opening. It renders plate candidates, keeps the cleanest
opening, derives the masks, the camera crops and the gate geometry, and then runs the gated preview. It publishes
the preset only if the preview passes.

## Running notes
- **Where it runs.** `COMFY_BASE_URL` (default `http://127.0.0.1:8189`, the comfy-api-proxy that
  `scripts/start_comfy.sh` starts in front of ComfyUI) or, with it unset, `COMFY_API_KEY` for Comfy Cloud.
- **Deterministic renders.** `start_comfy.sh` starts ComfyUI with `--disable-async-offload`. With async weight
  offloading, the same graph and seed gave different keyframes from one run to the next when a model didn't fit in
  VRAM. Without it, back-to-back runs were bit-identical and no slower. Don't add `--disable-dynamic-vram`: the pixels
  stay the same, but a clip takes about twice as long and RAM runs out on a 16 GB machine.
- **Memory.** On Linux with systemd, ComfyUI runs in its own cgroup with a RAM cap (`COMFY_MEM_MAX`, default `6G`).
  Model swaps then can't take the desktop or the shell down with them. Set `COMFY_MEM_MAX=` to run without a cap.
- **Preset assets.** `"asset:<file>"` values are uploaded through the SDK before a run. Don't copy files into
  ComfyUI's `input/` folder while renders run. The proxy skips re-uploading a file it thinks is already there.
- **Reruns.** A gated run re-sends the same graph with the chosen seeds, so ComfyUI reuses the cached keyframes. Each
  attempt gets a fresh copy of the graph, because the SDK writes asset handles into a graph it submits, and those
  can't be copied again.
