# The checks: what each one measures and where its threshold comes from

FreeField checks its own output and re-renders when a check fails. A clip that still fails after the re-rolls
ships with a plain "Unverified: ..." warning saying what to look at. All checks run on the CPU with open models:
- YuNet (MIT) finds and aligns faces.
- AuraFace (Apache-2.0) embeds them.
- OpenCLIP ViT-B-32 (MIT) scores images against text.
- OWLv2 (Apache-2.0) finds hands and tools.

Every threshold was set on stills and clips that people had judged good or bad by eye. The tests run the checks
on real failures kept in `tests/fixtures/gate/` and `tests/fixtures/video/`.

## Breakout keyframes (`gate.py`)
These checks run on START, GRIP and OUT before any video. A failed keyframe set is re-rolled with new seeds: up to
`gate.rerolls` times: 8 for Out of the Frame and Out of the Box, 5 for Out of the Photo.

| Name | Value | What it measures | Calibration |
|---|---|---|---|
| `OUTSIDE_MAX` | 0.001 | share of the area outside the frame zone where GRIP drew new skin (hands on the wall) | GRIP is first aligned onto START, so a slightly moved frame doesn't count as hands |
| `DRIFT_MAX` | 12 px | how far klein moved the scene between START and GRIP (shift plus scale change at the opening's corner) | Wan pins both frames, so a moved frame jumps at the shot seam. OUT drift is reported, not gated: the reaching arm hides the wall the alignment needs |
| `FRAME_MAX` | 0.035 | share of the frame band (gilt) that GRIP visibly changed | clean GRIPs 0.011-0.021; hands on the gilt 0.074-0.194. Needed because about 65 % of gilt passes a skin test |
| `IDENTITY_MIN` | 0.30 | AuraFace cosine between your photo and the best face in OUT | different people score 0.06-0.23 against each other |
| `PAINT_ID_MIN` | 0.35 | the same for the painted face in START | a painting that lost a moustache scored 0.24-0.33, with it 0.47. Toy faces get a per-preset override (0.19-0.40 against their photo) |
| `HAIR_PHOTO`, `HAIR_MIN` | 0.90, 0.50 | CLIP P(moustache): applied only when the photo is clearly moustached, and the painting must then keep it | paintings with the moustache 0.90-1.00, without 0.02-0.13. AuraFace barely moves when a moustache is dropped |

When no attempt passes, the least bad one is used. Ranking puts the right face first, because the locks remove stray
hands but nothing fixes a wrong or clean-shaven face. Then come the most checks passed, the cleanest frame and
hands, and the best likeness.

## Breakout clips (`gate.py`, `vision.py`)
| Name | Value | What it measures | Calibration |
|---|---|---|---|
| `END_ID_MIN` | 0.20 | median photo match over the clip's last 12 frames (0 for a frame without a face) | 22 clean clips 0.30-0.62; a hand with a camera over the face 0.00; a toy judged "loosely him" 0.09 |
| `SHOT1_MAX` | 0.25 | worst share of the painting's dark backdrop, in its two upper corners, that brightened by more than 40 grey levels during shot 1 (arms entering from the top of the canvas) | 6 clips judged as broken limbs 0.36-0.67; 24 others 0.01-0.16 |

A failed shot 1 is re-rendered alone with new noise (`gate.video_rerolls`). Shots 2 and 3 come back from ComfyUI's
cache.

## Built presets and composited effects (`vision.py`, `edge_check.py`)
| Name | Value | What it measures |
|---|---|---|
| `SUBJECT_MIN` | 0.62 | CLIP image similarity: the object, pet or product is still the one in your photo |
| `DESCRIPTION_MIN` | 0.03 | CLIP gain: the result matches the description better than your photo does |
| `MOTION_MIN` | 0.05 | first-to-last-frame change of a kept-photo effect: something actually moved |
| `HYBRID_MAX` | 0.02 | CLIP margin of "a person with animal features" over "an ordinary person" (pet-to-human effects) |
| `INTRUDER_MIN` | 0.20 | OWLv2 score of a hand or a tool that is new at the frame edge. Hands are not searched for when the subject is a person, and nothing the description names is searched for |

The edge check samples every 6th frame. Around any hit it adds the frames 3 before and 3 after. It counts a hit when
a detection:
- covers at least 1.5 % of the frame,
- touches the outer 3 % of the frame,
- is not already in frame 0 (template match),
- holds for 2 sampled frames, or is a single strong detection (score 0.35 or more, at least 5 % of the frame, at least 40 % of its pixels new).

On labelled clips it caught 9 of 10 bad clips and flagged 0 of 37 clean ones. It misses some motion-blurred hands.

## Per-effect checks
- **Earth Zoom.** Each still must read as its label (globe, satellite, drone) better than every other label. The
  satellite view must not just copy the drone view (CLIP similarity under 0.88). It re-rolls the stills up to twice.
- **Product Freeze.** The frozen still must keep the product (`SUBJECT_MIN`) and show the splash (`DESCRIPTION_MIN`).
  It re-rolls up to twice.
- **Iris Reveal.** It refuses an eye too small for a 2.6x close-up, or too soft (Laplacian variance under 120).
- **Comic Bonk.** It refuses a photo that CLIP reads as a real person or a pet rather than a toy, mascot or cartoon.
- **Out of the Frame, your clip.**
  - The painting must look like you (`PAINT_ID_MIN`), shot 1 must keep its corners clean (`SHOT1_MAX`), and your face
    must be there at the end (`END_ID_MIN`).
  - The cut-out must not flicker: the matte area may jump at most 25 % from one frame to the next.
  - The clip must be filmed far enough away to break out of the frame.
