import asyncio
import io
import subprocess
from pathlib import Path

import pytest
from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from freefield import pipelines, runner, server

ROOT = Path(__file__).resolve().parents[1]
PHOTO = (ROOT / "eval" / "fresh" / "fresh-1.png").read_bytes()


def png(size: tuple[int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 90, 60)).save(buf, "PNG")
    return buf.getvalue()


def post(fields: dict, photo: bytes | None = PHOTO, name: str = "me.png") -> tuple[int, dict]:
    async def go():
        async with TestClient(TestServer(server.make_app())) as client:
            fd = FormData()
            for k, v in fields.items():
                fd.add_field(k, v)
            if photo is not None:
                fd.add_field("photo", photo, filename=name)
            r = await client.post("/api/runs", data=fd)
            return r.status, await r.json()

    return asyncio.run(go())


@pytest.fixture(autouse=True)
def no_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    submitted = []
    monkeypatch.setattr(server.Worker, "submit", lambda self, run, source: submitted.append((run, source)))
    monkeypatch.setattr(pipelines, "run_preset", lambda *a, **k: submitted.append(("run_preset", a)))
    monkeypatch.setattr(runner, "run", lambda *a, **k: submitted.append(("runner.run", a)))
    return submitted


CLIP = (ROOT / "tests" / "fixtures" / "video" / "standin-fresh-1.mp4").read_bytes()


def audio_only_mp4() -> bytes:
    out = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=5", "-c:a", "aac",
                          "-f", "mp4", "-movflags", "frag_keyframe+empty_moov", "-"], capture_output=True, check=True)
    return out.stdout


@pytest.mark.parametrize("preset, data, name, message", [
    ("iris-reveal", b"\xff\xd8\xff\xe0 not really a jpeg", "corrupt.jpg", "corrupt.jpg: that file is not an image we can read"),
    ("superstar", PHOTO[:5000], "truncated.png", "truncated.png: that file is not an image we can read"),
    ("iris-reveal", b"this is a text file with a .png name\n", "notes.png", "notes.png: that file is not an image we can read"),
    ("earth-zoom", bytes(range(256)) * 4, "corrupt.mp4", "corrupt.mp4: that file is not a video we can read"),
    ("earth-zoom", CLIP[:600000], "truncated.mp4", "truncated.mp4: that file is not a video we can read"),
    ("earth-zoom", PHOTO, "photo.mp4", "photo.mp4: that file is not a video we can read"),
    ("earth-zoom", audio_only_mp4(), "voice-memo.mp4", "voice-memo.mp4: that file is not a video we can read"),
    ("out-of-the-frame-real", audio_only_mp4(), "voice-memo.mp4", "voice-memo.mp4: that file is not a video we can read"),
    ("out-of-the-frame-real", PHOTO, "me.png", "takes a video clip"),
    ("iris-reveal", CLIP, "walk.mp4", "takes a photo"),
])
def test_bad_media_uploads_are_refused_by_the_same_check_as_the_cli(no_queue, tmp_path, preset, data, name, message):
    code, body = post({"preset": preset}, data, name)
    assert code == 400 and message in body["error"], body
    assert no_queue == [] and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fields, photo, name, status, message", [
    ({"preset": "nope"}, PHOTO, "me.png", 400, "unknown preset"),
    ({"preset": "superstar", "seed": "-4"}, PHOTO, "me.png", 400, "seed must be an integer"),
    ({"preset": "superstar", "seed": "abc"}, PHOTO, "me.png", 400, "seed must be an integer"),
    ({"preset": "superstar"}, None, "", 400, "attach a photo"),
    ({"preset": "superstar"}, b"not an image at all", "me.png", 400, "not an image we can read"),
    ({"preset": "superstar"}, png((200, 300)), "me.png", 400, "too small"),
    ({"preset": "iris-reveal", "caption": "x" * 41}, PHOTO, "me.png", 400, "the limit is 40"),
    ({"preset": "earth-zoom"}, b"\x00" * 64, "walk.mp4", 400, "not a video we can read"),
])
def test_bad_requests_get_a_plain_error_and_nothing_is_queued(no_queue, tmp_path, fields, photo, name, status, message):
    code, body = post(fields, photo, name)
    assert code == status and message in body["error"], body
    assert no_queue == [] and list(tmp_path.iterdir()) == []


def test_an_oversize_photo_is_refused_with_413(no_queue, monkeypatch):
    monkeypatch.setattr(server, "MAX_UPLOAD", 1000)
    code, body = post({"preset": "superstar"})
    assert code == 413 and "larger than" in body["error"] and no_queue == []


def test_a_good_photo_is_queued_with_its_seed_and_text(no_queue):
    code, body = post({"preset": "iris-reveal", "seed": "12", "caption": "Look closer"})
    assert code == 202 and len(body["id"]) == 16
    run, source = no_queue[0]
    assert run.seed == 12 and run.texts == {"caption": "Look closer"} and source.name == "photo.png"


def test_unknown_runs_and_drafts_are_404():
    async def go():
        async with TestClient(TestServer(server.make_app())) as client:
            return [(await client.get(p)).status for p in ("/api/runs/0123456789abcdef", "/api/runs/../x",
                                                         "/api/build/nope", "/drafts/nope/graph.json")]

    assert asyncio.run(go()) == [404, 404, 404, 404]
