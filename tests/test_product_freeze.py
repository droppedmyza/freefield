import asyncio
import json

import cv2
import numpy as np

from freefield import node_check, runner, vision
from freefield import product_freeze as pf


def product(tmp_path):
    img = np.full((900, 700, 3), 230, np.uint8)
    cv2.circle(img, (350, 450), 180, (30, 30, 200), -1)
    p = tmp_path / "mug.png"
    cv2.imwrite(str(p), img)
    return p


def depth_ramp():
    d = np.tile(np.linspace(0, 1, pf.H, dtype=np.float32)[:, None], (1, pf.W))
    d[400:800, 200:520] = 0.9
    return d


def test_the_ad_is_six_seconds_and_ends_on_your_photo(tmp_path):
    still = cv2.cvtColor(cv2.imread(str(product(tmp_path))), cv2.COLOR_BGR2RGB)
    frames, cuts = pf.compose(still, depth_ramp(), still, "SPRING SALE")
    assert len(frames) == round((pf.ORBIT_SECONDS + pf.CARD_SECONDS) * pf.FPS) and frames[0].shape == (pf.H, pf.W, 3)
    assert np.array_equal(frames[-1], frames[-2]) and len(cuts) == 2
    plain, _ = pf.compose(still, depth_ramp(), still, "")
    assert np.abs(frames[-1].astype(int) - plain[-1]).mean() > 0.3
    assert np.abs(frames[30].astype(int) - frames[60]).mean() > 0.5


def test_a_text_change_reuses_the_cached_render(tmp_path, monkeypatch):

    calls = []

    def run(compiled, out_dir, **kw):
        calls.append(compiled.graph["freeze_noise"]["inputs"]["noise_seed"])
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "freeze_00001_.png"
        cv2.imwrite(str(p), np.full((1280, 720, 3), 200, np.uint8))
        return runner.RunResult("j", "succeeded", 1.0, {"freeze_save": [p]})

    class Subject:
        def similarity(self, a, b):
            return 0.8

        def text_similarity(self, t, p):
            return 0.3 if "freeze" in str(p) else 0.2

    monkeypatch.setattr(runner, "run", run)
    monkeypatch.setattr(vision, "Subject", Subject)
    monkeypatch.setattr(pf, "depth", lambda img: depth_ramp())
    photo = product(tmp_path)
    first = pf.run(photo, 7, tmp_path / "a", text="NEW", cache=tmp_path / "cache")
    second = pf.run(photo, 7, tmp_path / "b", text="NOW -20%", cache=tmp_path / "cache")
    assert calls == [7] and not first["cached"] and second["cached"] and second["text"] == "NOW -20%"
    assert (tmp_path / "b" / "clip.mp4").stat().st_size > 10_000 and first["warnings"] == []


def test_a_still_that_loses_the_product_is_re_rolled_then_flagged(tmp_path, monkeypatch):

    calls = []

    def run(compiled, out_dir, **kw):
        calls.append(1)
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "freeze_00001_.png"
        cv2.imwrite(str(p), np.full((1280, 720, 3), 200, np.uint8))
        return runner.RunResult("j", "succeeded", 1.0, {"freeze_save": [p]})

    class Subject:
        def similarity(self, a, b):
            return 0.4

        def text_similarity(self, t, p):
            return 0.3

    monkeypatch.setattr(runner, "run", run)
    monkeypatch.setattr(vision, "Subject", Subject)
    monkeypatch.setattr(pf, "depth", lambda img: depth_ramp())
    rec = pf.run(product(tmp_path), 1, tmp_path / "o", cache=tmp_path / "c")
    assert len(calls) == 3 and any("Unverified" in w for w in rec["warnings"])


def test_the_preset_loads_and_its_graph_validates():
    from freefield.presets import load_presets

    presets, _ = load_presets()
    assert presets["product-freeze"].pipeline == "product_freeze"
    graph = json.loads(pf.GRAPH.read_text())
    assert node_check.validate(graph, {}, json.loads(node_check.SCHEMAS.read_text()), {"product.image"}) == []


def test_the_app_runs_a_product_freeze_with_the_offer_text(tmp_path, monkeypatch):
    from aiohttp import FormData
    from aiohttp.test_utils import TestClient, TestServer

    from freefield import server

    monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(runner, "make_client", lambda: object())
    seen = {}

    def fake(photo, seed, out, **kw):
        seen["text"] = kw["text"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "clip.mp4").write_bytes(b"mp4")
        (out / "still.png").write_bytes(b"png")
        return {"warnings": []}

    monkeypatch.setattr(pf, "run", fake)
    photo = product(tmp_path)

    async def go():
        async with TestClient(TestServer(server.make_app())) as client:
            fd = FormData()
            fd.add_field("preset", "product-freeze")
            fd.add_field("offer", "SPRING SALE")
            fd.add_field("photo", photo.read_bytes(), filename="mug.png")
            r = await client.post("/api/runs", data=fd)
            assert r.status == 202, await r.text()
            rid = (await r.json())["id"]
            for _ in range(200):
                data = await (await client.get(f"/api/runs/{rid}")).json()
                if data["status"] in ("succeeded", "failed"):
                    break
                await asyncio.sleep(0.02)
            assert data["status"] == "succeeded" and seen["text"] == "SPRING SALE", data

    asyncio.run(go())


def test_a_prompt_change_does_not_reuse_an_old_still(tmp_path, monkeypatch):
    photo = product(tmp_path)
    graph = tmp_path / "product_freeze.json"
    graph.write_text(pf.GRAPH.read_text())
    monkeypatch.setattr(pf, "GRAPH", graph)
    before = pf.cache_key(photo, 42)
    graph.write_text(pf.GRAPH.read_text().replace("calm empty space", "open space"))
    assert pf.cache_key(photo, 42) != before and pf.cache_key(photo, 42) == pf.cache_key(photo, 42)


def test_the_still_prompt_names_only_what_it_wants():
    text = json.loads(pf.GRAPH.read_text())["freeze_prompt"]["inputs"]["text"].lower()
    assert not any(w in f" {text} " for w in (" no ", "hand", "people", "watermark")), text
