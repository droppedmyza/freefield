# FreeField workflows (ComfyUI)

These are the exact ComfyUI graphs FreeField runs for each preset in the entry, with the preset's prompts and assets
filled in. Inputs are placeholders: `your_photo.png` or `your_clip.mp4`. Each preset comes as two files:

- `<preset>.json` is the UI workflow. Drag it onto ComfyUI to open it.
- `<preset>.api.json` is the API workflow, the same graph FreeField submits through the Comfy SDK.
- `<preset>.png` is a screenshot of the laid-out canvas.

| Preset | Nodes | Notes |
|---|---|---|
| `out-of-the-frame` | 132 | signature; studio profile (Wan 2.2 I2V-A14B, 24 GB GPU) |
| `out-of-the-photo` | 132 | studio profile |
| `earth-zoom` | 49 | the three klein stills; the dive, clouds, title and sound are composited by `freefield/earth_zoom.py` |
| `product-freeze` | 18 | the klein frozen-splash still; depth and the camera are in `freefield/product_freeze.py` |
| `comic-bonk` | 22 | BiRefNet cut-out and klein clean plate; the motion and sound are in `freefield/comic_bonk.py` |
| `raven-poof` | 14 | Wan 2.2 TI2V-5B (8 GB profile) |
| `raven-intro` | 14 | the Raven Poof graph; the exposure match, title and whoosh are in `freefield/raven_intro.py` |
| `toy-hug-studio` | 19 | studio profile |
| `mascot-outfit-switch` | 36 | 8 GB profile |
| `paper-me` | 39 | 8 GB profile |
| `pet-to-human-studio` | 43 | **draft preset** (not in the gallery; its files are in `freefield/presets/_drafts/pet-to-human-studio/`); studio profile |

Natural Iris Reveal has no generative graph (YuNet plus plain image processing on your own photo), so it has no
workflow file here.

## Before you run one: install the preset files
Five workflows load files that ship with the preset: Out of the Frame and Out of the Photo (the gallery plate, the hole
mask, the wall mask and the painting), and Mascot Outfit Switch, Paper Me and Pet to Human (the before/after wipe and the
header). Those files are in `assets/`. Each is named `<preset>--<file>`, so two presets never overwrite each other's
`plate.png`. `assets/assets.json` lists them with their sha256.

ComfyUI only finds them in its `input` folder. Use one of these:

- **Copy them**, on the machine that runs ComfyUI: `python3 scripts/install_workflow_assets.py --input-dir /path/to/ComfyUI/input`
- **Upload them** to a running ComfyUI, local or remote: `python3 scripts/install_workflow_assets.py --url http://127.0.0.1:8188`.
  For a password-protected one, set `COMFYUI_AUTH=user:password` first.
- **By hand:** copy every file in `assets/` except `assets.json` into `ComfyUI/input/`.

Add a preset id or two at the end to install only those, for example `... --url http://127.0.0.1:8188 paper-me`. The
script checks each file against its sha256 first. Then open the workflow and pick your photo (or clip) in the
**Your photo** node.

## Inside each workflow
Every workflow opens with an **About this workflow** note on the left: what it makes, what to fill in, the stages,
every model it loads, what FreeField adds after the graph (for the composited effects) and how to run it. The nodes
sit in numbered groups, left to right, and every node has a plain-words title.

## Load proof (2026-10-23)
`node scripts/check_workflows.mjs workflows http://127.0.0.1:8188` opens a new headless browser profile for each
file. It loads the file into a fresh graph in ComfyUI 0.41.0 (with the ComfyUI-GGUF custom node installed by
`scripts/setup.sh`) and records the result in `load-proof.json`, with a screenshot in `proof/`.

| Check | Result |
|---|---|
| Workflows that open | **11 of 11** |
| Nodes restored | **every node** (loaded count = file count for each workflow, including its note) |
| Missing node types | **0** |
| Converts back to an API prompt (`app.graphToPrompt`) | **11 of 11**, every node except the note |
| Models not present on the checking machine | only the six 14B studio files in the four studio workflows; `scripts/studio_setup.sh` downloads them on a 24 GB machine |

The screenshots show a red "media input missing" box on the image loader. That is expected: `your_photo.png` is a
placeholder, not a file on this machine.

To regenerate: `.runtime/venv/bin/python scripts/export_workflows.py workflows http://127.0.0.1:8188 <preset ids>`, then
the load proof above.
