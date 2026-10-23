from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import gate, runner, vision
from .errors import EffectError
from .media import encode, video_frames_bgr
from .runner import CompiledRun
from .vision import YUNET

log = logging.getLogger(__name__)

PKG = Path(__file__).resolve().parent
GRAPHS = PKG / "graphs"
SCENE = PKG / "presets" / "out-of-the-frame"
HOLE = (210, 347, 300, 455)
VIEW = (110, 118, 500, 889)
OUT_SIZE = (720, 1280)
CLIP_SIZE = (1080, 1920)
MATTE_SIZE = (540, 960)
FPS = 24
PAINT_SIZE = (592, 896)
SHOT1_FPS = 24
REVEAL = 30
SHOT1_RETRIES = 2
MATTE_JUMP_MAX = 0.25
RAIL = 78
CUT_FADE = 24
EDGE_FEATHER = 40


class ClipBreakoutError(EffectError):
    pass


class FitError(ClipBreakoutError):
    pass


@dataclass(frozen=True)
class Fit:

    s: float
    ox: float
    oy: float

    def matrix(self) -> np.ndarray:
        return np.array([[self.s, 0.0, self.ox], [0.0, self.s, self.oy]], np.float32)


def fit(alpha0: np.ndarray, face: tuple[float, float, float, float], hole: tuple[int, int, int, int],
        margin: float = 0.10, face_share: float = 0.34) -> Fit:
    H, W = alpha0.shape
    x, y, w, h = hole
    rows = np.nonzero((alpha0 > 0.5).any(axis=1))[0]
    if len(rows) == 0:
        raise FitError("no person found in the first frame")
    top = int(rows[0])
    fx, fy, fw, fh = face
    s = face_share * h / fh
    oy = y + margin * h - s * top
    ox = x + w / 2 - s * (fx + fw / 2)
    if oy + s * H < y + h:
        raise FitError("not enough of you in the first frame: film from a little further away, chest in the picture")
    if s * (fy + fh) > s * H * 0.92:
        raise FitError("the face is too low in the first frame")
    return Fit(s, ox, oy)


def room_to_break_out(f: Fit, clip_wh: tuple[int, int], hole: tuple[int, int, int, int], rail: int = RAIL) -> bool:
    x, y, w, h = hole
    return f.ox < x - rail and f.ox + f.s * clip_wh[0] > x + w + rail


def _warp(img: np.ndarray, f: Fit, size: tuple[int, int], interp=None) -> np.ndarray:
    import cv2

    return cv2.warpAffine(img, f.matrix(), size, flags=cv2.INTER_LINEAR if interp is None else interp,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def edge_fade(clip_hw: tuple[int, int], f: Fit, size: tuple[int, int]) -> np.ndarray:
    H, W = clip_hw
    inside = np.ones((H, W), np.float32)
    feather = max(1, int(round(EDGE_FEATHER / f.s)))
    ramp = np.minimum(1.0, (np.arange(max(H, W)) + 0.5) / feather)
    inside *= ramp[:W][None, :] * ramp[:W][None, ::-1]
    inside *= ramp[:H][:, None] * ramp[:H][::-1][:, None]
    return _warp(inside, f, size)


def color_match(person: np.ndarray, alpha: np.ndarray, ref: np.ndarray, strength: float = 0.35) -> np.ndarray:
    import cv2

    m = alpha > 0.5
    if m.sum() < 100:
        return person
    a = cv2.cvtColor(person, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)
    mu_a, sd_a = a[m].mean(0), a[m].std(0) + 1e-3
    mu_b, sd_b = b.mean(0), b.std(0) + 1e-3
    out = (a - mu_a) * (1 + strength * (sd_b / sd_a - 1)) + mu_a + strength * (mu_b - mu_a)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def composite(scene: np.ndarray, frame: np.ndarray, alpha: np.ndarray, f: Fit, fade: np.ndarray,
              hole: tuple[int, int, int, int], shadow: float = 0.35, ref: np.ndarray | None = None) -> np.ndarray:
    import cv2

    Hs, Ws = scene.shape[:2]
    if ref is not None:
        frame = color_match(frame, alpha, ref)
    person = _warp(frame, f, (Ws, Hs)).astype(np.float32)
    a = _warp(alpha.astype(np.float32), f, (Ws, Hs)) * fade
    x, y, w, h = hole
    cut = (a[y + h - 1] > 0.5).astype(np.float32)
    cut[x:x + w] = 0
    cut = cv2.GaussianBlur(cut[None, :], (0, 0), 6)[0]
    a[y + h:] = 0
    top = max(0, y + h - CUT_FADE)
    ramp = np.clip((y + h - np.arange(top, y + h)) / CUT_FADE, 0, 1).astype(np.float32)
    a[top:y + h] *= 1 - cut[None, :] * (1 - ramp[:, None])
    out = scene.astype(np.float32)
    if shadow:
        sh = cv2.GaussianBlur(a, (0, 0), 14)
        sh = np.roll(sh, (14, 10), axis=(0, 1))
        sh[y:y + h, x:x + w] = 0
        out *= (1 - shadow * sh)[..., None]
    out = out * (1 - a[..., None]) + person * a[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def remove_room(frames: list[np.ndarray], mattes: list[np.ndarray], near: float = 6.0, far: float = 14.0,
                min_seen: int = 3, step: int = 3) -> tuple[list[np.ndarray], float]:
    import cv2

    if not frames or len(frames) != len(mattes):
        return mattes, 0.0
    H, W = mattes[0].shape
    h, w = H // 2, W // 2
    lab = [cv2.cvtColor(cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2LAB).astype(np.float32)
           for f in frames]
    small = [cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA) for m in mattes]
    obs = np.stack([np.where((m < 0.15)[..., None], l, np.nan) for l, m in zip(lab[::step], small[::step])])
    seen = np.sum(~np.isnan(obs[..., 0]), axis=0)
    import warnings

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        room = np.nanmedian(obs, axis=0)
    known = seen >= min_seen

    model = YUNET
    faces = cv2.FaceDetectorYN.create(str(model), "", (W, H), 0.6) if model.exists() else None
    out, removed = [], 0.0
    for f, l, m, sm in zip(frames, lab, mattes, small):
        both = known & (sm < 0.15)
        drift = np.median(l[both] - room[both], axis=0) if both.sum() > 50 else np.zeros(3, np.float32)
        d = np.linalg.norm(l - drift - np.nan_to_num(room), axis=2)
        keep = np.where(known, np.clip((d - near) / (far - near), 0, 1), 1.0).astype(np.float32)
        keep = cv2.resize(keep, (W, H), interpolation=cv2.INTER_LINEAR)
        if faces is not None:
            faces.setInputSize((W, H))
            _, det = faces.detect(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            for b in [] if det is None else det:
                x0, y0, bw, bh = (int(v) for v in b[:4])
                keep[max(0, y0 - bh // 2):y0 + bh + bh // 4, max(0, x0 - bw // 4):x0 + bw + bw // 4] = 1
        new = m * keep
        area = float(m.sum())
        removed = max(removed, float((m - new).sum()) / area if area else 0.0)
        out.append(new)
    return out, removed


def clean_mattes(mattes: list[np.ndarray], window: int = 9, tol: float = 0.25) -> list[np.ndarray]:
    import cv2

    def largest(m: np.ndarray) -> np.ndarray:
        n, lab, st, _ = cv2.connectedComponentsWithStats((m > 0.5).astype(np.uint8))
        if n <= 2:
            return m
        keep = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        return m * cv2.dilate((lab == keep).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)

    ms = [largest(m) for m in mattes]
    areas = np.array([m.mean() for m in ms])
    h = window // 2
    ref = np.array([np.median(areas[max(0, i - h):i + h + 1]) for i in range(len(areas))])
    good = np.abs(areas - ref) <= tol * np.maximum(ref, 1e-4)
    if good.any() and not good.all():
        idx = np.nonzero(good)[0]
        ms = [ms[i] if good[i] else ms[idx[np.argmin(np.abs(idx - i))]] for i in range(len(ms))]
    st = np.stack(ms)
    if len(st) >= 3:
        st = np.concatenate([st[:1], np.median(np.stack([st[:-2], st[1:-1], st[2:]]), axis=0), st[-1:]])
    k = np.ones((5, 5), np.uint8)
    return [cv2.GaussianBlur(cv2.erode(m.astype(np.float32), k), (0, 0), 0.8) for m in st]


def reveal_masks(shape: tuple[int, int], n: int, seed: int = 0,
                 opening: tuple[float, float, float, float] | None = None) -> np.ndarray:
    import cv2

    rng = np.random.default_rng(seed)
    H, W = shape
    streaks = cv2.resize(rng.random((H // 48 + 2, W // 6 + 2)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    streaks = cv2.GaussianBlur(streaks, (0, 0), 3)
    streaks = (streaks - streaks.min()) / (streaks.max() - streaks.min() + 1e-6)
    rows = np.arange(H, dtype=np.float32)[:, None]
    top, height = (opening[1], opening[3]) if opening else (0.0, float(H))
    field = 0.8 * np.clip((rows - top) / max(height, 1.0), 0, 1) + 0.2 * streaks
    inside = np.ones((H, W), np.float32)
    if opening:
        x, y, w, h = (int(round(v)) for v in opening)
        inside[:] = 0
        inside[max(0, y):y + h, max(0, x):x + w] = 1
        inside = cv2.GaussianBlur(inside, (0, 0), 6)
    out = np.empty((n, H, W), np.float32)
    for i in range(n):
        t = (i + 1) / n
        sweep = np.clip(((t * 1.2 - 0.1) - field) / 0.08 + 0.5, 0, 1)
        late = float(np.clip((t - 0.65) / 0.35, 0, 1))
        out[i] = inside * sweep + (1 - inside) * late
    return out


def view_box(box: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    k = OUT_SIZE[0] / VIEW[2]
    return ((box[0] - VIEW[0]) * k, (box[1] - VIEW[1]) * k, box[2] * k, box[3] * k)


def in_order(files: list[Path]) -> list[Path]:
    def key(p: Path) -> int:
        m = re.match(r".*?_(\d+)_", p.name)
        return int(m.group(1)) if m else 0
    return sorted(files, key=key)


def _graph(name: str) -> dict:
    return json.loads((GRAPHS / name).read_text())


def _rgb(path: Path) -> np.ndarray:
    import cv2

    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def _save(path: Path, rgb: np.ndarray) -> Path:
    import cv2

    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return path


def normalise(clip: Path, out: Path, size: tuple[int, int] = CLIP_SIZE) -> Path:
    if shutil.which("ffmpeg") is None:
        raise ClipBreakoutError("ffmpeg is not installed (needed to read and write the clips)")
    w, h = size
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(clip), "-vf",
           f"fps={FPS},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}", "-c:v", "libx264", "-crf", "14",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", str(out)]
    subprocess.run(cmd, check=True)
    return out


def read_frames(path: Path) -> list[np.ndarray]:
    import cv2

    return [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in video_frames_bgr(path)]


def view(plate_img: np.ndarray) -> np.ndarray:
    import cv2

    x, y, w, h = VIEW
    return cv2.resize(plate_img[y:y + h, x:x + w], OUT_SIZE, interpolation=cv2.INTER_AREA)


def paint_inputs(frame0: np.ndarray, alpha0: np.ndarray, f: Fit, plate_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    x, y, w, h = HOLE
    k = PAINT_SIZE[0] / w
    g = Fit(f.s * k, (f.ox - x) * k, (f.oy - y) * k)
    canvas = cv2.resize(plate_img[y:y + h, x:x + w], PAINT_SIZE, interpolation=cv2.INTER_CUBIC).astype(np.float32)
    person = _warp(frame0, g, PAINT_SIZE).astype(np.float32)
    a = _warp(alpha0.astype(np.float32), g, PAINT_SIZE)[..., None]
    return (canvas * (1 - a) + person * a).astype(np.uint8), canvas.astype(np.uint8)


def in_opening(plate_img: np.ndarray, img: np.ndarray) -> np.ndarray:
    import cv2

    x, y, w, h = HOLE
    out = plate_img.copy()
    out[y:y + h, x:x + w] = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return out


def lock_wall(frames: list[np.ndarray], plate_view: np.ndarray, wall_view: np.ndarray) -> list[np.ndarray]:
    m = wall_view[..., None]
    return [(f.astype(np.float32) * (1 - m) + plate_view.astype(np.float32) * m).astype(np.uint8) for f in frames]


def retime(frames: list[np.ndarray], src: float, dst: float) -> list[np.ndarray]:
    n = int(round(len(frames) * dst / src))
    return [frames[min(len(frames) - 1, int(i * src / dst))] for i in range(n)]


def write_mp4(frames: list[np.ndarray], out: Path, audio_from: Path | None = None, audio_delay: float = 0.0) -> Path:
    inputs = []
    if audio_from is not None:
        inputs = ["-itsoffset", f"{audio_delay:.3f}", "-i", str(audio_from), "-map", "0:v", "-map", "1:a?", "-c:a", "aac",
                  "-shortest"]
    return encode(frames, out, inputs, ["-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
                  fps=FPS)


def _mattes(clip: Path, work: Path, n: int, client, rec: dict) -> list[np.ndarray]:
    import cv2


    small = normalise(clip, work / "clip_matte.mp4", MATTE_SIZE)
    cached = work / "mattes.npz"
    if cached.exists():
        raw = list(np.load(cached)["mattes"].astype(np.float32) / 255)
    else:
        g = _graph("real_matte.json")
        res = runner.run(CompiledRun(g, [("clip", "file", small)], ["matte_save"]), work / "matte", client=client,
                         label="real-matte")
        if not res.ok:
            raise ClipBreakoutError(f"matting failed: {res.error}")
        raw = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255 for p in in_order(res.files["matte_save"])]
        np.savez_compressed(cached, mattes=(np.stack(raw) * 255).round().astype(np.uint8))
    if len(raw) != n:
        raise ClipBreakoutError(f"{len(raw)} mattes for {n} frames")
    mattes, removed = remove_room(read_frames(small), raw)
    rec["room_removed"] = round(removed, 3)
    mattes = [cv2.resize(m, CLIP_SIZE, interpolation=cv2.INTER_CUBIC).clip(0, 1) if m.shape[::-1] != CLIP_SIZE else m
              for m in mattes]
    mattes = clean_mattes(mattes)
    areas = np.array([m.mean() for m in mattes])
    jump = float(np.max(np.abs(np.diff(areas)) / np.maximum(areas[1:], 1e-3))) if len(areas) > 1 else 0.0
    rec["matte_jump"] = round(jump, 3)
    if jump > MATTE_JUMP_MAX:
        rec["warnings"].append(f"Unverified: the cut-out of you flickers between frames (area jump {jump:.0%}).")
    shutil.rmtree(work / "matte", ignore_errors=True)
    return mattes


def _place(frames: list[np.ndarray], mattes: list[np.ndarray], work: Path, rec: dict, faces) -> tuple[Fit, Path]:
    import cv2

    photo = _save(work / "frame0.png", frames[0])
    faces.det.setInputSize(CLIP_SIZE)
    _, det = faces.det.detect(cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
    if det is None or len(det) == 0:
        raise ClipBreakoutError("no face found in the first frame")
    face = tuple(float(v) for v in max(det, key=lambda d: d[2] * d[3])[:4])
    f = fit(mattes[0], face, HOLE)
    rec["fit"] = {"s": round(float(f.s), 4), "ox": round(float(f.ox), 1), "oy": round(float(f.oy), 1)}
    if not room_to_break_out(f, CLIP_SIZE, HOLE):
        rec["warnings"].append("Filmed too close to break out of the frame: prop the phone further away (your head about a "
                               "tenth of the picture) and lean all the way in.")
    return f, photo


def _shot1(start_v: np.ndarray, plate_v: np.ndarray, wall_v: np.ndarray, seed: int, work: Path, client, rec: dict) -> list:
    attempts, best = [], None
    for k in range(SHOT1_RETRIES + 1):
        g = _graph("real_shot1.json")
        for n in ("shot1_sample_high", "shot1_sample"):
            g[n]["inputs"]["noise_seed"] = seed + 7919 * k
        res = runner.run(CompiledRun(g, [("start", "image", _save(work / "start_view.png", start_v))], ["shot1_save"]),
                         work / f"shot1-{k}", client=client, label="real-shot1")
        if not res.ok:
            raise ClipBreakoutError(f"shot 1 failed: {res.error}")
        sf = lock_wall([_rgb(p) for p in in_order(res.files["shot1_save"])], plate_v, wall_v)
        tmp = write_mp4(sf, work / f"shot1-{k}.mp4")
        score = gate.shot1_intrusion(tmp, HOLE, VIEW[:3], len(sf))
        attempts.append({"attempt": k, "seed": seed + 7919 * k, "shot1_intrusion": round(score, 3)})
        if best is None or score < best[0]:
            best = (score, sf)
        if score <= gate.SHOT1_MAX:
            break
    rec["shot1_attempts"] = attempts
    rec["shot1_intrusion"] = round(best[0], 3)
    if best[0] > gate.SHOT1_MAX:
        rec["warnings"].append(f"Unverified: an arm or hand comes in from the top of the painting in the first shot "
                               f"in all {len(attempts)} attempts.")
    return best[1]


def run(clip: Path, seed: int, out_dir: Path, *, client=None, shot1: str = "wan", on_stage=None) -> dict:
    import cv2


    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "work"
    work.mkdir(exist_ok=True)
    rec: dict = {"clip": str(clip), "seed": seed, "warnings": []}
    norm = normalise(clip, work / "clip.mp4")
    frames = read_frames(norm)
    if len(frames) < FPS * 3:
        raise ClipBreakoutError("the clip is shorter than 3 seconds")

    if on_stage:
        on_stage("Cutting you out of your clip")
    mattes = _mattes(clip, work, len(frames), client, rec)

    faces = vision.Faces()
    f, photo = _place(frames, mattes, work, rec, faces)
    plate = _rgb(SCENE / "plate.png")
    wall = cv2.imread(str(SCENE / "wall_mask.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255

    if on_stage:
        on_stage("Painting you")
    portrait_in, backdrop_in = paint_inputs(frames[0], mattes[0], f, plate)
    g = _graph("real_painting.json")
    for n in ("painting_noise", "backdrop_noise"):
        g[n]["inputs"]["noise_seed"] = seed
    files = [("portrait_in", "image", _save(work / "portrait_in.png", portrait_in)),
             ("backdrop_in", "image", _save(work / "backdrop_in.png", backdrop_in))]
    res = runner.run(CompiledRun(g, files, ["painting_save", "backdrop_save"]), work / "paint", client=client, label="real-paint")
    if not res.ok:
        raise ClipBreakoutError(f"painting failed: {res.error}")
    painting, backdrop = _rgb(res.files["painting_save"][0]), _rgb(res.files["backdrop_save"][0])
    start, scene = in_opening(plate, painting), in_opening(plate, backdrop)
    paint_id = faces.similarity(photo, _save(work / "painting.png", painting))
    rec["painting_identity"] = None if paint_id is None else round(paint_id, 3)
    if paint_id is None or paint_id < gate.PAINT_ID_MIN:
        rec["warnings"].append(f"Unverified: the painting does not clearly look like you (match {paint_id}).")

    if on_stage:
        on_stage("Bringing the painting to life")
    plate_v, wall_v, start_v = view(plate), view(wall[..., None].repeat(3, 2))[..., 0], view(start)
    shot1_frames = [start_v] * 41 if shot1 == "still" else _shot1(start_v, plate_v, wall_v, seed, work, client, rec)
    if on_stage:
        on_stage("Stepping out of the frame")
    return _finish(out_dir, rec, frames, mattes, f, plate, scene, start, shot1_frames, seed, norm, faces, photo)


def _finish(out_dir: Path, rec: dict, frames, mattes, f: Fit, plate, scene, start, shot1_frames, seed: int, norm: Path,
            faces, photo: Path) -> dict:
    import cv2


    shot1_seq = retime(shot1_frames, SHOT1_FPS, FPS)
    fade = edge_fade(CLIP_SIZE[::-1], f, plate.shape[1::-1])
    ref = plate[VIEW[1]:VIEW[1] + VIEW[3], VIEW[0]:VIEW[0] + VIEW[2]]
    real = [view(composite(scene, fr, a, f, fade, HOLE, ref=ref)) for fr, a in zip(frames, mattes)]
    masks = reveal_masks(OUT_SIZE[::-1], REVEAL, seed, view_box(HOLE))
    last = shot1_seq[-1].astype(np.float32)
    real0 = real[0].astype(np.float32)
    reveal = [(last * (1 - m[..., None]) + real0 * m[..., None]).astype(np.uint8) for m in masks]
    seq = shot1_seq + reveal + real
    final = write_mp4(seq, out_dir / "clip.mp4", audio_from=norm, audio_delay=(len(shot1_seq) + REVEAL) / FPS)

    end = faces.end_identity(photo, final)
    rec["end_identity"] = None if end is None else round(end, 3)
    if end is not None and end < gate.END_ID_MIN:
        rec["warnings"].append(f"Unverified: your face is not clearly there at the end of the clip (match {end:.2f}).")
    tiles = [cv2.resize(frames[0], (360, 640)), cv2.resize(view(start), (360, 640)),
             cv2.resize(shot1_seq[len(shot1_seq) // 2], (360, 640)), cv2.resize(seq[len(shot1_seq) + REVEAL // 2], (360, 640)),
             cv2.resize(seq[-1], (360, 640))]
    _save(out_dir / "proof.png", np.hstack(tiles))
    rec["frames"] = len(seq)
    (out_dir / "gate.json").write_text(json.dumps(rec, indent=2) + "\n")
    return rec
