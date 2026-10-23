from __future__ import annotations

import io
import json
import subprocess
import wave
from pathlib import Path

import numpy as np

from .errors import EffectError

FONTS = Path(__file__).resolve().parent / "fonts"
TITLE_FONT = FONTS / "PlayfairDisplay-Variable.ttf"
W, H, FPS = 720, 1280, 24
AUDIO_RATE = 44100
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm"}
MIN_PHOTO_SIDE = 256


class MediaError(ValueError):
    pass


def open_photo(data: bytes):
    from PIL import Image, ImageOps

    try:
        Image.open(io.BytesIO(data)).verify()
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        raise MediaError("that file is not an image we can read (use JPG, PNG or WebP)") from None
    if min(img.size) < MIN_PHOTO_SIDE:
        raise MediaError(f"photo is too small; use at least {MIN_PHOTO_SIDE} px on the short side")
    return img


def video_seconds(path: Path) -> float:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_type:format=duration",
                              "-of", "json", str(path)], capture_output=True, text=True, timeout=30)
        probe = json.loads(out.stdout) if out.returncode == 0 else {}
        if not any(st.get("codec_type") == "video" for st in probe.get("streams", [])):
            return float("nan")
        return float(probe.get("format", {}).get("duration", "nan"))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return float("nan")


def last_second_decodes(path: Path) -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-sseof", "-1", "-i", str(path), "-map", "0:v:0",
                              "-f", "framemd5", "-"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    frames = [line for line in out.stdout.splitlines() if line[:1].isdigit()]
    return out.returncode == 0 and not out.stderr.strip() and len(frames) > 0


def check_video(path: Path) -> float:
    seconds = video_seconds(path)
    if not seconds > 0 or not last_second_decodes(path):
        raise MediaError("that file is not a video we can read")
    return seconds


def read_rgb(path: Path) -> np.ndarray:
    from PIL import Image, ImageOps

    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))


def fit_cover(img: np.ndarray, size: tuple[int, int] = (W, H)) -> np.ndarray:
    import cv2

    w, h = size
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    r = cv2.resize(img, (max(w, round(iw * s)), max(h, round(ih * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    y, x = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y:y + h, x:x + w]


def clean_text(text: str | None, limit: int = 40) -> str:
    return " ".join(str(text or "").split())[:limit]


def blend(frame: np.ndarray, layer: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    a = layer[..., 3:] * opacity
    return np.clip(frame * (1 - a) + layer[..., :3] * 255 * a, 0, 255).astype(np.uint8)


def title_layer(text: str, size: tuple[int, int] = (W, H), y: float = 0.36, px: int = 76) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    w, h = size
    text = " ".join(text.upper().split())
    spaced = " ".join(text) if len(text) <= 14 else text
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    while px > 18:
        font = ImageFont.truetype(str(TITLE_FONT), px)
        font.set_variation_by_name("Bold")
        box = draw.textbbox((0, 0), spaced, font=font)
        if box[2] - box[0] <= w * 0.86:
            break
        px -= 4
    x, y = (w - (box[2] - box[0])) // 2 - box[0], int(h * y) - box[1]
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).text((x, y + 3), spaced, font=font, fill=(0, 0, 0, 230), stroke_width=4, stroke_fill=(0, 0, 0, 230))
    layer = Image.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(8)), layer)
    ImageDraw.Draw(layer).text((x, y), spaced, font=font, fill=(255, 255, 255, 255))
    return np.asarray(layer, np.float32) / 255.0


def whoosh(seconds: float, cuts: list[float], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * AUDIO_RATE)
    noise = rng.standard_normal(n).astype(np.float32)
    t = np.arange(n) / AUDIO_RATE
    env = np.zeros(n, np.float32)
    cutoff = np.full(n, 0.02, np.float32)
    for c in cuts:
        rise = np.exp(-((t - c) / 0.28) ** 2) * (t <= c)
        fall = np.exp(-(t - c) / 0.18) * (t > c)
        env += (rise + fall).astype(np.float32)
        cutoff += (0.25 * np.exp(-((t - c) / 0.22) ** 2)).astype(np.float32)
    out = np.empty(n, np.float32)
    y = 0.0
    for i in range(n):
        y += cutoff[i] * (noise[i] - y)
        out[i] = y
    out *= np.clip(env, 0, 1.2)
    peak = float(np.abs(out).max()) or 1.0
    return out / peak * 0.6


def write_wav(samples: np.ndarray, path: Path) -> Path:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(AUDIO_RATE)
        f.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return path


def has_audio(path: Path) -> bool:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True, timeout=30)
    return bool(out.stdout.strip())


def encode(frames: list[np.ndarray], out: Path, inputs: list[str], options: list[str], fps: int = FPS) -> Path:
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", *inputs, *options, str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise EffectError(f"ffmpeg failed writing {out}")
    return out


def write_video(frames: list[np.ndarray], out: Path, audio: Path, clip: Path | None = None, clip_offset: float = 0.0,
                clip_seconds: float = 20.0) -> Path:
    inputs = ["-i", str(audio)]
    if clip is not None and has_audio(clip):
        ms = int(clip_offset * 1000)
        inputs += ["-i", str(clip), "-filter_complex",
                   f"[2:a]atrim=0:{clip_seconds},adelay={ms}|{ms},aformat=channel_layouts=stereo[c];"
                   f"[1:a]aformat=channel_layouts=stereo[s];[s][c]amix=inputs=2:duration=longest:normalize=0[a]",
                   "-map", "0:v", "-map", "[a]"]
    else:
        inputs += ["-map", "0:v", "-map", "1:a"]
    options = ["-c:a", "aac", "-b:a", "160k", "-shortest", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart"]
    return encode(frames, out, inputs, options)


def decode_frames(clip: Path, fps: int = FPS) -> list[np.ndarray]:
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", str(clip)], capture_output=True, text=True, check=True).stdout.strip().split(",")
    w, h = int(probe[0]), int(probe[1])
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(clip), "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt",
                          "rgb24", "-"], capture_output=True, check=True).stdout
    n = len(raw) // (w * h * 3)
    return list(np.frombuffer(raw[: n * w * h * 3], np.uint8).reshape(n, h, w, 3))


def video_frames_bgr(path: Path, limit: int | None = None) -> list[np.ndarray]:
    import cv2

    cap, frames = cv2.VideoCapture(str(path)), []
    while limit is None or len(frames) < limit:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames
