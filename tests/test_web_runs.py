import asyncio
import shutil
from pathlib import Path
from types import SimpleNamespace

from aiohttp.test_utils import TestClient, TestServer

from freefield import breakout, media, presets, runner, server

ROOT = Path(__file__).resolve().parents[1]
RETRY_CLIP = ROOT / "tests" / "fixtures" / "video" / "r16-out-of-the-photo--heldout-1-s1-end.mp4"


def test_a_re_rendered_clip_plays_in_the_web_app(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    gated = SimpleNamespace(id="p", pipeline=None, params={}, inputs={}, gate={"hole": [0, 0, 1, 1]})
    monkeypatch.setattr(server, "load_presets", lambda: ({"p": gated}, []))
    monkeypatch.setattr(presets, "compile_prompt", lambda *a: None)
    monkeypatch.setattr(runner, "make_client", lambda: object())

    def fake_gated(compiled, spec, photo, out_dir, **kw):
        clip = out_dir / "video-2" / "clip_save_0_clip_00005_.mp4"
        clip.parent.mkdir(parents=True)
        shutil.copy(RETRY_CLIP, clip)
        return runner.RunResult("j", "succeeded", 1.0, {"clip_save": [clip]})

    monkeypatch.setattr(breakout, "run_gated", fake_gated)
    workers = []
    real_worker = server.Worker
    monkeypatch.setattr(server, "Worker", lambda: workers.append(real_worker()) or workers[-1])

    async def go():
        async with TestClient(TestServer(server.make_app())) as client:
            run = server.Run(id="0123456789abcdef", preset="p", seed=1)
            workers[0].submit(run, tmp_path / "photo.png")
            for _ in range(200):
                if run.status in ("succeeded", "failed"):
                    break
                await asyncio.sleep(0.02)
            assert run.status == "succeeded", run.error
            data = await (await client.get(f"/api/runs/{run.id}")).json()
            url = data["files"]["clip_save"][0]
            assert url == f"/runs/{run.id}/video-2/clip_save_0_clip_00005_.mp4"
            resp = await client.get(url)
            assert resp.status == 200 and await resp.read() == RETRY_CLIP.read_bytes()
            for bad in ("../x.mp4", "plates/clip.mp4", "video-2/../../secret", "video-2/keyframes-0/a.png"):
                assert (await client.get(f"/runs/{run.id}/{bad}")).status == 404

    asyncio.run(go())


def test_your_clip_runs_through_realbreak_in_the_web_app(tmp_path, monkeypatch):
    from freefield import clip_breakout

    monkeypatch.setattr(server, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(runner, "make_client", lambda: object())
    short = ROOT / "tests" / "fixtures" / "video" / "r16-out-of-the-photo--heldout-1-s1-end.mp4"

    def fake_real(clip, seed, out, **kw):
        kw["on_stage"]("Painting you")
        (out / "clip.mp4").write_bytes(RETRY_CLIP.read_bytes())
        (out / "proof.png").write_bytes(b"png")
        return {"warnings": ["Unverified: test"]}

    monkeypatch.setattr(clip_breakout, "run", fake_real)

    async def post(client, path, name):
        from aiohttp import FormData
        fd = FormData()
        fd.add_field("preset", "out-of-the-frame-real")
        fd.add_field("seed", "3")
        fd.add_field("photo", path.read_bytes(), filename=name)
        return await client.post("/api/runs", data=fd)

    async def go():
        async with TestClient(TestServer(server.make_app())) as client:
            r = await post(client, short, "me.mp4")
            assert r.status == 400 and "6-10 s" in (await r.json())["error"]
            r = await post(client, ROOT / "eval" / "fresh" / "fresh-1.png", "me.png")
            assert r.status == 400 and "video clip" in (await r.json())["error"]
            monkeypatch.setattr(media, "video_seconds", lambda p: 7.0)
            r = await post(client, short, "me.mov")
            assert r.status == 202
            rid = (await r.json())["id"]
            for _ in range(200):
                data = await (await client.get(f"/api/runs/{rid}")).json()
                if data["status"] in ("succeeded", "failed"):
                    break
                await asyncio.sleep(0.02)
            assert data["status"] == "succeeded" and data["warnings"] == ["Unverified: test"], data
            resp = await client.get(data["files"]["clip_save"][0])
            assert resp.status == 200 and await resp.read() == RETRY_CLIP.read_bytes()

    asyncio.run(go())
