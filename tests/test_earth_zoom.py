import asyncio
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from freefield import earth_zoom as ez
from freefield import media, node_check, runner, vision

ROOT = Path(__file__).resolve().parents[1]


def solid(rgb, size=(ez.W, ez.H)):
    img = np.zeros((size[1], size[0], 3), np.uint8)
    img[:] = rgb
    img[size[1] // 3: size[1] // 2, size[0] // 3: size[0] // 2] = 255 - np.array(rgb, np.uint8)
    return img


STILLS = {"globe": solid((10, 10, 40)), "satellite": solid((20, 90, 160)), "drone": solid((160, 90, 40))}
FRAME0 = solid((60, 140, 60))


def test_the_dive_ends_exactly_on_the_first_frame_of_your_clip():
    clip = [FRAME0] + [solid((60, 140 + i, 60)) for i in range(1, 12)]
    frames, cuts, n = ez.compose(STILLS, FRAME0, clip)
    assert len(frames) == n + len(clip) and n == sum(round(z[2] * ez.FPS) for z in ez.ZOOMS)
    assert np.abs(frames[n - 1].astype(int) - FRAME0).mean() < 2 and frames[n] is clip[0]
    assert len(cuts) == 3 and all(a < b for a, b in zip(cuts, cuts[1:]))


def test_every_cut_is_hidden_by_clouds():
    frames, cuts, _ = ez.compose(STILLS, FRAME0, [FRAME0])
    for c in cuts:
        i = round(c * ez.FPS)
        assert frames[i - 1].mean() > 235 or frames[i].mean() > 235


def test_a_photo_alone_makes_a_seamless_loop():
    frames, cuts, n = ez.compose(STILLS, FRAME0, None)
    assert len(frames) == 2 * n + round(ez.HOLD * ez.FPS)
    assert np.array_equal(frames[0], frames[-1]) and len(cuts) == 6


def test_the_place_name_is_drawn_and_an_empty_one_is_not():
    with_title, _, _ = ez.compose(STILLS, FRAME0, [FRAME0], caption="Lisbon")
    plain, _, _ = ez.compose(STILLS, FRAME0, [FRAME0])
    mid = round((ez.ZOOMS[0][2] + 0.6) * ez.FPS)
    assert np.abs(with_title[mid].astype(int) - plain[mid]).mean() > 0.5
    assert np.array_equal(with_title[2], plain[2])


def test_the_whoosh_peaks_at_the_cuts():
    cuts = [1.0, 2.5]
    s = ez.whoosh(3.5, cuts)
    assert len(s) == int(3.5 * media.AUDIO_RATE) and abs(np.abs(s).max() - 0.6) < 1e-3
    near = np.abs(s[int(0.9 * media.AUDIO_RATE):int(1.1 * media.AUDIO_RATE)]).mean()
    far = np.abs(s[int(1.7 * media.AUDIO_RATE):int(1.9 * media.AUDIO_RATE)]).mean()
    assert near > 4 * far


class FakeSubject:
    def __init__(self, good):
        self.good = good

    def text_similarity(self, text, path):
        kind = Path(path).stem.split("_")[0]
        return 0.3 if (self.good(path) and text == ez.LABELS.get(kind)) else 0.2

    def similarity(self, a, b):
        return 0.8 if self.good(a) else 0.97


def test_a_satellite_view_that_copies_the_drone_view_fails():
    paths = {k: Path(f"stills-0/{k}_00001_.png") for k in ez.STILLS}
    assert all(c["ok"] for c in ez.still_checks(paths, FakeSubject(lambda p: True)).values())
    checks = ez.still_checks(paths, FakeSubject(lambda p: "drone" not in str(p)))
    assert not checks["satellite"]["ok"] and checks["satellite"]["like_drone"] == 0.97


def fake_stills(monkeypatch, good=lambda path: True):
    import cv2


    calls = []

    def run(compiled, out_dir, **kw):
        out_dir.mkdir(parents=True, exist_ok=True)
        calls.append(compiled.graph["globe_noise"]["inputs"]["noise_seed"])
        files = {}
        for k, img in STILLS.items():
            p = out_dir / f"{k}_00001_.png"
            cv2.imwrite(str(p), img)
            files[f"{k}_save"] = [p]
        return runner.RunResult("j", "succeeded", 1.0, files)

    monkeypatch.setattr(runner, "run", run)
    monkeypatch.setattr(vision, "Subject", lambda: FakeSubject(good))
    return calls


def tiny_clip(path: Path, audio: bool = True) -> Path:
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=24:duration=1.5"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1.5", "-c:a", "aac"]
    subprocess.run(cmd + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(path)], check=True)
    return path


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    return {s["codec_type"]: s for s in json.loads(out.stdout)["streams"]}


def test_your_clip_comes_out_vertical_with_its_own_sound_after_the_dive(tmp_path, monkeypatch):
    fake_stills(monkeypatch)
    rec = ez.run(tiny_clip(tmp_path / "walk.mp4"), 3, tmp_path / "out", text="  Porto  ")
    streams = probe(tmp_path / "out" / "clip.mp4")
    assert (streams["video"]["width"], streams["video"]["height"]) == (720, 1280) and "audio" in streams
    assert rec["mode"] == "clip" and rec["frames"] == rec["intro_frames"] + 36 and rec["caption"] == "Porto"
    assert rec["warnings"] == [] and (tmp_path / "out" / "proof.png").exists()


def test_a_still_that_misses_its_label_is_re_rolled_then_flagged(tmp_path, monkeypatch):
    calls = fake_stills(monkeypatch, good=lambda path: "stills-2" in str(path))
    photo = tmp_path / "place.png"
    import cv2
    cv2.imwrite(str(photo), FRAME0)
    rec = ez.run(photo, 5, tmp_path / "ok")
    assert calls == [5, 5 + 1009, 5 + 2018] and rec["warnings"] == [] and rec["mode"] == "photo loop"
    assert "audio" in probe(tmp_path / "ok" / "clip.mp4")
    calls.clear()
    fake_stills(monkeypatch, good=lambda path: False)
    rec = ez.run(photo, 5, tmp_path / "bad")
    assert len(rec["still_attempts"]) == 3 and any("Unverified" in w for w in rec["warnings"])


def test_the_preset_and_its_graph_are_valid():
    from freefield import builder
    from freefield.presets import load_presets

    presets, errors = load_presets()
    p = presets["earth-zoom"]
    assert p.pipeline == "earth_zoom" and p.inputs["clip"]["photo_ok"] and "earth-zoom" not in (errors or {})
    graph = json.loads(ez.GRAPH.read_text())
    schemas = json.loads(node_check.SCHEMAS.read_text())
    assert node_check.validate(graph, {}, schemas, {"frame0.image"}) == []
    assert builder.models_of(graph) == ["flux-2-klein-4b-fp8.safetensors", "qwen_3_4b_fp8_mixed.safetensors", "flux2-vae.safetensors"]


@pytest.mark.parametrize("name", ["walk.mp4", "place.png"])
def test_the_app_takes_a_clip_or_a_photo_for_earth_zoom(tmp_path, monkeypatch, name):
    from aiohttp import FormData
    from aiohttp.test_utils import TestClient, TestServer

    from freefield import server

    monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(runner, "make_client", lambda: object())
    seen = {}

    def fake_run(src, seed, out, **kw):
        seen.update(src=src, caption=kw["text"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "clip.mp4").write_bytes(b"mp4")
        (out / "proof.png").write_bytes(b"png")
        return {"warnings": []}

    monkeypatch.setattr(ez, "run", fake_run)
    if name.endswith(".mp4"):
        src = tiny_clip(tmp_path / name)
    else:
        import cv2
        src = tmp_path / name
        cv2.imwrite(str(src), FRAME0)

    async def go():
        async with TestClient(TestServer(server.make_app())) as client:
            fd = FormData()
            fd.add_field("preset", "earth-zoom")
            fd.add_field("caption", "Lisbon")
            fd.add_field("photo", src.read_bytes(), filename=name)
            r = await client.post("/api/runs", data=fd)
            assert r.status == 202, await r.text()
            rid = (await r.json())["id"]
            for _ in range(200):
                data = await (await client.get(f"/api/runs/{rid}")).json()
                if data["status"] in ("succeeded", "failed"):
                    break
                await asyncio.sleep(0.02)
            assert data["status"] == "succeeded", data
            assert seen["caption"] == "Lisbon" and seen["src"].suffix == Path(name).suffix
            assert (await client.get(data["files"]["clip_save"][0])).status == 200

    asyncio.run(go())
