# Examples

Finished clips from FreeField's presets, next to the input each one was made from. Every input is an AI stand-in:
the people, toys, products and places were generated with FLUX.2 [klein] 4B from text (`eval/fixtures/fixtures.json`
lists the prompts for the people). No real person appears. The museum presets use public-domain paintings. The commands run the same effect on the
same input; the clip shown may come from another seed or caption.

| Effect | Input | Result | GPU | Try it |
|---|---|---|---|---|
| Out of the Frame | [heldout-1.jpg](inputs/heldout-1.jpg) | [out-of-the-frame.mp4](out-of-the-frame.mp4) | 24 GB | `python -m freefield run out-of-the-frame --photo heldout-1.jpg` |
| Earth Zoom Into My Clip | [harbor-walk.mp4](inputs/harbor-walk.mp4) | [earth-zoom--harbor-walk.mp4](earth-zoom--harbor-walk.mp4) | 8 GB | `python -m freefield run earth-zoom --clip harbor-walk.mp4 --caption "Fjordhavn"` |
| Earth Zoom, from a photo | [place-a.jpg](inputs/place-a.jpg) | [earth-zoom--place-a.mp4](earth-zoom--place-a.mp4) | 8 GB | `python -m freefield run earth-zoom --photo place-a.jpg --caption "Costa Bella"` |
| Product Freeze Ad | [teapot-b.jpg](inputs/teapot-b.jpg) | [product-freeze--teapot-b.mp4](product-freeze--teapot-b.mp4) | 8 GB | `python -m freefield run product-freeze --photo teapot-b.jpg --caption "SPRING SALE -30%"` |
| Natural Iris Reveal | [iris-standin-1.jpg](inputs/iris-standin-1.jpg) | [iris-reveal--iris-standin-1.mp4](iris-reveal--iris-standin-1.mp4) | none (CPU) | `python -m freefield run iris-reveal --photo eval/hires/iris-standin-1.png` |
| Comic Bonk | [mascot-a.jpg](inputs/mascot-a.jpg) | [comic-bonk--mascot-a.mp4](comic-bonk--mascot-a.mp4) | 8 GB | `python -m freefield run comic-bonk --photo mascot-a.jpg` |
| Mascot Outfit Switch | [mascot-a.jpg](inputs/mascot-a.jpg) | [mascot-outfit-switch--mascot-a.mp4](mascot-outfit-switch--mascot-a.mp4) | 8 GB | `python -m freefield run mascot-outfit-switch --photo mascot-a.jpg` |
| Raven Poof | [figurine-a.jpg](inputs/figurine-a.jpg) | [raven-poof--figurine-a.mp4](raven-poof--figurine-a.mp4) | 8 GB | `python -m freefield run raven-poof --photo figurine-a.jpg` |
| Raven Poof Intro | [vase-b.jpg](inputs/vase-b.jpg) | [raven-intro--vase-b.mp4](raven-intro--vase-b.mp4) | 8 GB | `python -m freefield run raven-intro --photo vase-b.jpg --caption "NIGHT SHIFT"` |
| Toy Hug | [plush-pair-a.jpg](inputs/plush-pair-a.jpg) | [toy-hug--plush-pair-a.mp4](toy-hug--plush-pair-a.mp4) | 24 GB | `python -m freefield run toy-hug-studio --photo plush-pair-a.jpg` |

Iris Reveal needs the full-size photo, `eval/hires/iris-standin-1.png` (1440x2560), run from the repo root. The copy in
`inputs/` is shrunk, so its eye is too small and the effect refuses it.

Every preset also ships its own preview clip in `freefield/presets/<id>/preview.mp4`, which the gallery app shows on
its card. That includes Out of the Photo, Out of the Box, Paper Me and Out of the Frame with your own clip.
