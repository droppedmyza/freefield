import asyncio
import json
from pathlib import Path

from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer

from freefield import builder, node_check, presets, recipe, server, ui_export

ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / "eval" / "fixtures" / "heldout-1.png"


def setup(tmp_path, monkeypatch, errors=()):
    monkeypatch.setattr(builder, "DRAFTS", tmp_path / "drafts")
    monkeypatch.setattr(builder, "PRESETS", tmp_path / "presets")
    monkeypatch.setattr(presets, "PRESETS_DIR", tmp_path / "presets")
    monkeypatch.setattr(server, "PRESETS_DIR", tmp_path / "presets")
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path / "runs")
    real_load = server.load_presets
    monkeypatch.setattr(server, "load_presets", lambda: real_load(tmp_path / "presets"))
    monkeypatch.setattr(builder, "load_schemas", lambda *a, **k: (json.loads(node_check.SCHEMAS.read_text()), "snapshot"))
    monkeypatch.setattr(ui_export, "export_layout", lambda *a, **k: None)
    if errors:
        monkeypatch.setattr(builder, "validate", lambda *a, **k: list(errors))

    def fake_preview(folder, photo, *, client=None, seed=42):
        (folder / "preview.mp4").write_bytes(b"x" * 20000)
        (folder / "poster.jpg").write_bytes(b"jpg")
        return builder._record(folder, {"status": "ready for approval", "preview": {"render_ok": True, "clip": True,
                                                                                    "face_match": 0.6, "passed": True}})

    monkeypatch.setattr(builder, "preview", fake_preview)


async def wait(client, rid):
    for _ in range(200):
        data = await (await client.get(f"/api/runs/{rid}")).json()
        if data["status"] in ("succeeded", "failed"):
            return data
        await asyncio.sleep(0.02)
    raise AssertionError("run did not finish")


def photo_form():
    fd = FormData()
    fd.add_field("photo", PHOTO.read_bytes(), filename="me.png")
    return fd


def test_describe_preview_approve_and_the_new_card_appears(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)

    async def go():
        async with TestClient(TestServer(server.make_app())) as client:
            r = await client.post("/api/build", data={"description": 'make me a robot, then I dance, with a reveal, titled "BOT"'})
            assert r.status == 201
            info = await r.json()
            assert info["id"] == "bot" and info["status"] == "drafted" and info["recipe"]["stages"][2] == "reveal"
            assert (await client.get(info["files"]["graph.json"])).status == 200
            r = await client.post("/api/build/bot/preview", data=photo_form())
            assert r.status == 202
            assert (await wait(client, (await r.json())["id"]))["status"] == "succeeded"
            info = await (await client.get("/api/build/bot")).json()
            assert info["status"] == "ready for approval" and "preview.mp4" in info["files"]
            r = await client.post("/api/build/bot/publish", data={"approve": "no"})
            assert r.status == 409 and "approval" in (await r.json())["error"]
            r = await client.post("/api/build/bot/publish", data={"approve": "yes"})
            assert r.status == 200 and (await r.json())["preset"] == "bot"
            cards = {p["id"]: p for p in (await (await client.get("/api/presets")).json())["presets"]}
            assert cards["bot"]["family"] == "built" and cards["bot"]["poster"] == "/presets/bot/poster.jpg"
            assert (await client.get("/presets/bot/poster.jpg")).status == 200

    asyncio.run(go())


def test_a_rejected_draft_says_why_and_will_not_render(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch, errors=["klein_model.unet_name: model 'x.safetensors' is not available (download it first)"])

    async def go():
        async with TestClient(TestServer(server.make_app())) as client:
            r = await client.post("/api/build", data={"description": "make me a robot"})
            assert r.status == 422 and "not available" in (await r.json())["errors"][0]
            r = await client.post("/api/build/robot/preview", data=photo_form())
            assert r.status == 409
            r = await client.post("/api/build/robot/publish", data={"approve": "yes"})
            assert r.status == 409
            assert (await client.get("/drafts/robot/..%2Fx")).status == 404

    asyncio.run(go())


def test_a_caption_typed_in_the_app_reaches_the_graph(tmp_path, monkeypatch):
    from freefield import runner

    setup(tmp_path, monkeypatch)
    seen = {}

    def fake_run(compiled, out_dir, **kw):
        out_dir.mkdir(parents=True, exist_ok=True)
        seen["caption"] = compiled.graph["caption"]["inputs"]["text"]
        clip = out_dir / "clip.mp4"
        clip.write_bytes(b"x")
        return runner.RunResult("j", "succeeded", 1.0, {"clip_save": [clip]})

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(runner, "make_client", lambda: object())
    r = recipe.parse("my mug stays the same, then the camera orbits it, with a caption")
    d = builder.draft(r, comfy_url=None)
    (d.folder / "preview.mp4").write_bytes(b"x" * 20000)
    builder._record(d.folder, {"preview": {"passed": True}})
    builder.publish(d.folder, approved=True)

    async def go():
        async with TestClient(TestServer(server.make_app())) as client:
            fd = photo_form()
            fd.add_field("preset", r.id)
            fd.add_field("caption", "LISBON")
            resp = await client.post("/api/runs", data=fd)
            assert resp.status == 202, await resp.text()
            assert (await wait(client, (await resp.json())["id"]))["status"] == "succeeded"
            assert seen["caption"] == "LISBON"

    asyncio.run(go())
