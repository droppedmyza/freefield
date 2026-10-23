from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from . import media, runner
from .errors import EffectError

if TYPE_CHECKING:
    from .presets import Preset

StageFn = Callable[[str], None]


@dataclass(frozen=True)
class Pipeline:
    module: str
    files: dict[str, str]
    summary: tuple[str, ...] = ()
    needs_comfy: bool = True
    clip_seconds: tuple[int, int, str] | None = None


PIPELINES: dict[str, Pipeline] = {
    "earth_zoom": Pipeline("earth_zoom", {"clip_save": "clip.mp4", "proof_save": "proof.png"}, ("still_checks",),
                           clip_seconds=(1, 30, "use 2-10 s")),
    "product_freeze": Pipeline("product_freeze", {"clip_save": "clip.mp4", "still_save": "still.png"},
                               ("cached", "still_attempts")),
    "iris_reveal": Pipeline("iris_reveal", {"clip_save": "clip.mp4"}, ("eye", "upscale_at_eye"), needs_comfy=False),
    "comic_bonk": Pipeline("comic_bonk", {"clip_save": "clip.mp4"}, ("guard",)),
    "raven_intro": Pipeline("raven_intro", {"clip_save": "clip.mp4"}, ("exposure", "attempts")),
    "clip_breakout": Pipeline("clip_breakout", {"clip_save": "clip.mp4", "proof_save": "proof.png"},
                              ("painting_identity", "shot1_intrusion", "end_identity", "matte_jump"),
                              clip_seconds=(3, 30, "use 6-10 s")),
}


def text_input(preset: Preset) -> str | None:
    return next((name for name, spec in preset.inputs.items() if spec.get("type") == "text"), None)


def source_input(preset: Preset) -> dict:
    return next((spec for spec in preset.inputs.values() if spec.get("type") in ("image", "video")), {"type": "image"})


def clip_length_problem(preset: Preset, seconds: float) -> str | None:
    spec = PIPELINES.get(preset.pipeline or "")
    if spec is None or spec.clip_seconds is None:
        return None
    low, high, hint = spec.clip_seconds
    return None if low <= seconds <= high else f"the clip is {seconds:.0f} s long; {hint} ({low}-{high} s works)"


def check_source(preset: Preset, path: Path, name: str | None = None) -> str:
    name = name or path.name
    spec = source_input(preset)
    suffix = path.suffix.lower()
    if spec["type"] == "image" and suffix not in media.IMAGE_EXT:
        raise media.MediaError(f"{preset.id} takes a photo (JPG, PNG or WebP), not {name}")
    if spec["type"] == "video" and suffix not in media.VIDEO_EXT and not (spec.get("photo_ok") and suffix in media.IMAGE_EXT):
        kinds = "a video clip (MP4, MOV or WebM)" + (" or a photo" if spec.get("photo_ok") else "")
        raise media.MediaError(f"{preset.id} takes {kinds}, not {name}")
    try:
        if suffix in media.IMAGE_EXT:
            media.open_photo(path.read_bytes())
            return "image"
        seconds = media.check_video(path)
    except media.MediaError as e:
        raise media.MediaError(f"{name}: {e}") from None
    problem = clip_length_problem(preset, seconds)
    if problem:
        raise media.MediaError(f"{name}: {problem}")
    return "video"


def default_seed(preset: Preset) -> int:
    return int(preset.params.get("seed", {}).get("default", 0))


def run_pipeline(preset: Preset, source: Path, seed: int, out_dir: Path, *, texts: dict[str, str] | None = None,
                 client=None, on_stage: StageFn | None = None, **options) -> runner.RunResult:
    spec = PIPELINES[preset.pipeline]
    module = importlib.import_module(f"{__package__}.{spec.module}")
    kwargs = dict(options)
    name = text_input(preset)
    if name:
        kwargs["text"] = (texts or {}).get(name, "")
    if spec.needs_comfy:
        kwargs["client"] = client
    started = time.monotonic()
    try:
        record = module.run(source, seed, out_dir, on_stage=on_stage, **kwargs)
    except EffectError as e:
        return runner.RunResult("", "failed", time.monotonic() - started, error=str(e))
    return runner.RunResult("", "succeeded", time.monotonic() - started,
                            {key: [out_dir / name] for key, name in spec.files.items()},
                            warnings=list(record.get("warnings", [])),
                            checks={key: record[key] for key in spec.summary if key in record})


def run_preset(preset: Preset, source: Path, seed: int, out_dir: Path, *, texts: dict[str, str] | None = None,
               client=None, on_progress: runner.ProgressFn | None = None,
               on_stage: StageFn | None = None) -> runner.RunResult:
    from . import breakout, output_checks
    from .presets import compile_prompt

    texts = texts or {}
    if preset.pipeline:
        return run_pipeline(preset, source, seed, out_dir, texts=texts, client=client, on_stage=on_stage)
    params = {"seed": seed} if "seed" in preset.params else {}
    compiled = compile_prompt(preset, {"photo": str(source), **texts}, params)
    if preset.gate:
        return breakout.run_gated(compiled, preset.gate, source, out_dir, client=client, on_progress=on_progress,
                                  label=preset.id)
    recipe = output_checks.published_recipe(preset)
    if recipe is not None:
        done = output_checks.checked_run(preset, recipe, source, lambda k: out_dir if k == 0 else out_dir / f"video-{k}",
                                         client=client, seed=seed, inputs=texts, on_progress=on_progress, label=preset.id)
        result = done.result
        result.warnings = [*result.warnings, *done.warnings()]
        result.checks = {k: v for k, v in done.checks.items() if k != "clip_path"}
        return result
    return runner.run(compiled, out_dir, client=client, on_progress=on_progress, label=preset.id)
