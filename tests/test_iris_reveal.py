import asyncio
from pathlib import Path

import cv2
import numpy as np
import pytest

from freefield import iris_reveal as ir
from freefield import runner

ROOT = Path(__file__).resolve().parents[1]
FACE = ROOT / "eval" / "fresh" / "fresh-1.png"


def test_a_low_resolution_selfie_is_refused_before_rendering(tmp_path):
    with pytest.raises(ir.IrisRevealError, match="px wide"):
        ir.run(FACE, 0, tmp_path / "o")
    assert not (tmp_path / "o" / "clip.mp4").exists()


def test_an_upscaled_selfie_is_refused_for_lack_of_detail(tmp_path):
    big = tmp_path / "big.png"
    cv2.imwrite(str(big), cv2.resize(cv2.imread(str(FACE)), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
    with pytest.raises(ir.IrisRevealError, match="too soft"):
        ir.run(big, 0, tmp_path / "o")


def test_the_eyes_and_irises_are_found():
    eyes = ir.find_eyes(cv2.imread(str(FACE)))
    assert {e.side for e in eyes} == {"left", "right"} and all(e.iris_found for e in eyes)
    left, right = sorted(eyes, key=lambda e: e.x)
    assert 120 < right.x - left.x < 220 and abs(left.y - right.y) < 30


def test_the_zoom_uses_only_the_photos_pixels_and_returns_to_the_start():
    rgb = np.zeros((2400, 1350, 3), np.uint8)
    rgb[:] = (90, 120, 150)
    eye = ir.Eye(700, 1000, 220, 500, "left", True)
    frames, cuts = ir.compose(rgb, eye)
    assert len(frames) == sum(round(s * ir.FPS) for s in (ir.HOLD_START, ir.ZOOM_IN, ir.HOLD_EYE, ir.ZOOM_OUT, ir.HOLD_END))
    assert np.array_equal(frames[0], frames[-1]) and frames[0].shape == (ir.H, ir.W, 3)
    assert np.abs(frames[len(frames) // 2].astype(int) - [90, 120, 150]).max() <= 1 and len(cuts) == 2


def test_the_preset_loads():
    from freefield.presets import load_presets

    presets, errors = load_presets()
    assert presets["iris-reveal"].pipeline == "iris_reveal" and "iris-reveal" not in (errors or {})


def test_the_app_keeps_the_full_resolution_and_explains_a_refusal(tmp_path, monkeypatch):
    from aiohttp import FormData
    from aiohttp.test_utils import TestClient, TestServer

    from freefield import server

    monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(runner, "make_client", lambda: object())
    seen = {}

    def fake(photo, seed, out, **kw):
        from PIL import Image

        seen["size"] = Image.open(photo).size
        raise ir.IrisRevealError("the eye is only 66 px wide")

    monkeypatch.setattr(ir, "run", fake)
    src = tmp_path / "selfie.png"
    cv2.imwrite(str(src), np.full((3000, 2250, 3), 128, np.uint8))

    async def go():
        async with TestClient(TestServer(server.make_app())) as client:
            fd = FormData()
            fd.add_field("preset", "iris-reveal")
            fd.add_field("photo", src.read_bytes(), filename="selfie.png")
            r = await client.post("/api/runs", data=fd)
            assert r.status == 202, await r.text()
            rid = (await r.json())["id"]
            for _ in range(200):
                data = await (await client.get(f"/api/runs/{rid}")).json()
                if data["status"] in ("succeeded", "failed"):
                    break
                await asyncio.sleep(0.02)
            assert data["status"] == "failed" and "66 px" in data["error"] and seen["size"] == (2250, 3000)

    asyncio.run(go())
