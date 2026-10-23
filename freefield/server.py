from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import re
import secrets
import shutil
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aiohttp import web

from . import builder, media, pipelines, recipe, runner, ui_export
from .media import VIDEO_EXT
from .presets import PACKAGE_DIR, PRESETS_DIR, Preset, PresetError, compile_prompt, load_presets

log = logging.getLogger(__name__)

ROOT = PACKAGE_DIR.parent
RUNS_DIR = ROOT / "outputs" / "runs"
WEB_DIR = PACKAGE_DIR / "web"
MAX_UPLOAD = 15 * 1024 * 1024
MAX_VIDEO = 200 * 1024 * 1024
MAX_SEED = 2**31 - 1
RUN_ID = re.compile(r"^[a-f0-9]{16}$")
SAFE_FILE = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")
ATTEMPT_DIR = re.compile(r"^(?:keyframes|video)-[0-9]{1,2}$")
DRAFT_FILES = ("graph.json", "workflow.json", "workflow_api.json", "graph.png", "preset.json", "preview.mp4", "poster.jpg")

STAGES = [("keyframe", "Creating your keyframe"), ("photo", "Reading your photo"), ("artwork", "Reading the artwork"),
          ("klein", "Loading the image model"), ("wan", "Loading the video model"), ("motion", "Reading the motion prompt"),
          ("video", "Animating"), ("split", "Compositing"), ("watermark", "Finishing"), ("clip", "Saving")]


class BadRequest(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def stage_label(node_id: str | None) -> str | None:
    if not node_id:
        return None
    return next((label for prefix, label in STAGES if node_id.startswith(prefix)), None)


@dataclass
class Run:
    id: str
    preset: str
    seed: int
    status: str = "queued"
    stage: str | None = None
    progress: float = 0.0
    step: int | None = None
    steps: int | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    seconds: float | None = None
    outputs: dict[str, list[str]] = field(default_factory=dict)
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    texts: dict[str, str] = field(default_factory=dict)
    checks: dict = field(default_factory=dict)

    def track(self, ev: dict) -> None:
        if ev["type"] != "progress":
            return
        label = stage_label(ev.get("node"))
        if label:
            self.stage = label
        if ev.get("steps"):
            self.step, self.steps = ev.get("step"), ev.get("steps")
        if ev.get("nodes_total"):
            self.progress = round(ev["nodes_done"] / ev["nodes_total"], 3)

    def finish(self, result: runner.RunResult) -> None:
        self.seconds = round(result.seconds, 1)
        self.warnings = list(result.warnings)
        self.checks = dict(result.checks)
        self.outputs = {k: [p.relative_to(RUNS_DIR / self.id).as_posix() for p in v] for k, v in result.files.items()}
        self.status = "succeeded" if result.ok else "failed"
        self.error = None if result.ok else (result.error or result.status)
        self.progress = 1.0 if result.ok else self.progress


class Worker:
    def __init__(self) -> None:
        self.runs: dict[str, Run] = {}
        self.q: queue.Queue[tuple[Run, Path]] = queue.Queue()
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, name="freefield-worker", daemon=True).start()

    def submit(self, run: Run, photo: Path) -> None:
        with self.lock:
            self.runs[run.id] = run
        self.q.put((run, photo))

    def queue_position(self, run_id: str) -> int:
        with self.lock:
            waiting = sorted((r for r in self.runs.values() if r.status == "queued"), key=lambda r: r.created)
        return next((i + 1 for i, r in enumerate(waiting) if r.id == run_id), 0)

    def _build_preview(self, run: Run, photo: Path) -> None:
        run.stage = "Rendering the preview"
        try:
            rec = builder.preview(builder.DRAFTS / run.preset[len("build:"):], photo, seed=run.seed)
            run.status = "succeeded"
            run.warnings = [] if rec["preview"]["passed"] else [f"The preview did not pass its checks: {rec['preview']}"]
        except Exception as e:
            log.exception("build preview %s crashed", run.id)
            run.status, run.error = "failed", f"{type(e).__name__}: {e}"
        run.seconds = round(time.time() - (run.started or time.time()), 1)
        run.progress = 1.0

    def _loop(self) -> None:
        client = None
        while True:
            run, source = self.q.get()
            run.status, run.started = "running", time.time()
            if run.preset.startswith("build:"):
                self._build_preview(run, source)
                continue
            try:
                presets, _ = load_presets()
                client = client or runner.make_client()
                result = pipelines.run_preset(presets[run.preset], source, run.seed, RUNS_DIR / run.id, texts=run.texts,
                                              client=client, on_progress=run.track,
                                              on_stage=lambda s, run=run: setattr(run, "stage", s))
                run.finish(result)
            except Exception as e:
                log.exception("run %s crashed", run.id)
                run.status, run.error = "failed", f"{type(e).__name__}: {e}"
                client = None
            new_run_folder(run).joinpath("run.json").write_text(json.dumps(asdict(run), indent=2))


WORKER = web.AppKey("worker", Worker)


def json_response(data, status: int = 200) -> web.Response:
    return web.json_response(data, status=status, dumps=lambda d: json.dumps(d, default=str))


@web.middleware
async def bad_requests(request: web.Request, handler):
    try:
        return await handler(request)
    except BadRequest as e:
        return json_response({"error": str(e)}, e.status)


def new_run_folder(run: Run) -> Path:
    folder = RUNS_DIR / run.id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def read_upload(form, limit: int, what: str) -> tuple[bytes, str, str]:
    upload = form.get("photo")
    if not isinstance(upload, web.FileField):
        raise BadRequest("attach a photo (field 'photo')")
    data = upload.file.read(limit + 1)
    if len(data) > limit:
        raise BadRequest(f"{what} is larger than {limit // (1024 * 1024)} MB", 413)
    name = Path(upload.filename or "upload").name
    return data, Path(name).suffix.lower(), name


def photo_image(data: bytes, max_side: int = 2048):
    try:
        img = media.open_photo(data)
    except media.MediaError as e:
        raise BadRequest(str(e)) from None
    img.thumbnail((max_side, max_side))
    return img


def parse_seed(form) -> int:
    raw = form.get("seed")
    try:
        seed = int(raw) if raw not in (None, "") else secrets.randbelow(MAX_SEED + 1)
    except (TypeError, ValueError):
        seed = -1
    if not 0 <= seed <= MAX_SEED:
        raise BadRequest(f"seed must be an integer between 0 and {MAX_SEED}")
    return seed


def parse_texts(form, preset: Preset) -> dict[str, str]:
    texts = {}
    for name, spec in preset.inputs.items():
        value = str(form.get(name, ""))
        if spec.get("type") != "text" or not value.strip():
            continue
        limit = int(spec.get("max_length", 500))
        if len(value) > limit:
            raise BadRequest(f"{spec.get('label', name)} is {len(value)} characters; the limit is {limit}")
        texts[name] = value
    return texts


def wants_clip(preset: Preset, suffix: str) -> bool:
    return preset.pipeline == "clip_breakout" or (preset.pipeline == "earth_zoom" and suffix in VIDEO_EXT)


def checked_upload(data: bytes, suffix: str, name: str, preset: Preset, run: Run) -> tuple[Path, str]:
    upload = new_run_folder(run) / f"input{suffix}"
    upload.write_bytes(data)
    try:
        return upload, pipelines.check_source(preset, upload, name)
    except media.MediaError as e:
        raise BadRequest(str(e)) from None


def photo_from_upload(data: bytes, preset: Preset, run: Run) -> Path:
    img = photo_image(data, 6000 if preset.pipeline == "iris_reveal" else 2048)
    photo = new_run_folder(run) / "photo.png"
    img.save(photo)
    if not preset.pipeline:
        try:
            compile_prompt(preset, {"photo": str(photo), **run.texts}, {"seed": run.seed})
        except PresetError as e:
            raise BadRequest(str(e)) from None
    return photo


async def index(_: web.Request) -> web.FileResponse:
    return web.FileResponse(WEB_DIR / "index.html")


async def list_presets(_: web.Request) -> web.Response:
    presets, errors = load_presets()
    items = []
    for p in presets.values():
        item = p.public()
        preview = next((n for n in ("preview.mp4", "preview.jpg", "preview.webp") if (p.folder / n).exists()), None)
        if preview:
            item["preview"] = f"/presets/{p.id}/{preview}"
        if (p.folder / "poster.jpg").exists():
            item["poster"] = f"/presets/{p.id}/poster.jpg"
        items.append(item)
    return json_response({"presets": items, "broken": errors})


async def preset_file(req: web.Request) -> web.StreamResponse:
    pid, name = req.match_info["pid"], req.match_info["name"]
    if not SAFE_FILE.match(pid) or not SAFE_FILE.match(name) or not (name.startswith("preview") or name == "poster.jpg"):
        raise web.HTTPNotFound()
    path = PRESETS_DIR / pid / name
    if not path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(path)


async def create_run(req: web.Request) -> web.Response:
    form = await req.post()
    preset_id = str(form.get("preset", ""))
    presets, _ = load_presets()
    if preset_id not in presets:
        raise BadRequest(f"unknown preset {preset_id!r}")
    preset = presets[preset_id]
    run = Run(id=secrets.token_hex(8), preset=preset_id, seed=parse_seed(form), texts=parse_texts(form, preset))
    suffix = Path(getattr(form.get("photo"), "filename", "") or "").suffix.lower()
    data, suffix, name = read_upload(form, MAX_VIDEO, "clip") if wants_clip(preset, suffix) else read_upload(form, MAX_UPLOAD, "photo")
    try:
        source, kind = checked_upload(data, suffix, name, preset, run)
        if kind == "image":
            source.unlink()
            source = photo_from_upload(data, preset, run)
    except BadRequest:
        shutil.rmtree(RUNS_DIR / run.id, ignore_errors=True)
        raise
    req.app[WORKER].submit(run, source)
    log.info("queued run %s preset=%s seed=%s", run.id, preset_id, run.seed)
    return json_response({"id": run.id}, 202)


async def get_run(req: web.Request) -> web.Response:
    rid = req.match_info["rid"]
    worker = req.app[WORKER]
    run = worker.runs.get(rid) if RUN_ID.match(rid) else None
    if run is None:
        return json_response({"error": "no such run"}, 404)
    data = asdict(run)
    data["queue_position"] = worker.queue_position(rid)
    data["files"] = {k: [f"/runs/{rid}/{n}" for n in v] for k, v in run.outputs.items()}
    return json_response(data)


async def run_file(req: web.Request) -> web.StreamResponse:
    rid, *dirs, name = [req.match_info["rid"], *req.match_info["name"].split("/")]
    if (not RUN_ID.match(rid) or not SAFE_FILE.match(name) or len(dirs) > 1
            or not all(ATTEMPT_DIR.match(d) for d in dirs)):
        raise web.HTTPNotFound()
    path = RUNS_DIR.joinpath(rid, *dirs, name)
    if not path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(path)


def draft_info(bid: str) -> dict | None:
    folder = builder.DRAFTS / bid
    if not SAFE_FILE.match(bid) or not (folder / "build.json").exists():
        return None
    info = json.loads((folder / "build.json").read_text())
    files = {n: f"/drafts/{bid}/{n}" for n in DRAFT_FILES if (folder / n).exists()}
    return {"id": bid, **info, "files": files}


async def build_draft(req: web.Request) -> web.Response:
    form = await req.post()
    loop = asyncio.get_running_loop()
    try:
        r = recipe.parse(str(form.get("description", "")))
        r.profile = "studio" if str(form.get("profile", "")) == "studio" else "8gb"
        d = await loop.run_in_executor(None, lambda: builder.draft(r))
    except recipe.BuildError as e:
        raise BadRequest(str(e)) from None
    if not d.errors:
        await loop.run_in_executor(None, ui_export.export_layout, d.folder)
    return json_response(draft_info(r.id), 201 if not d.errors else 422)


async def build_info(req: web.Request) -> web.Response:
    info = draft_info(req.match_info["bid"])
    return json_response(info) if info else json_response({"error": "no such draft"}, 404)


async def build_preview(req: web.Request) -> web.Response:
    bid = req.match_info["bid"]
    info = draft_info(bid)
    if info is None:
        return json_response({"error": "no such draft"}, 404)
    if info.get("errors"):
        raise BadRequest("this draft was rejected: " + "; ".join(info["errors"][:3]), 409)
    form = await req.post()
    data, _, _ = read_upload(form, MAX_UPLOAD, "photo")
    run = Run(id=secrets.token_hex(8), preset=f"build:{bid}", seed=42)
    photo = new_run_folder(run) / "photo.png"
    photo_image(data).save(photo)
    req.app[WORKER].submit(run, photo)
    return json_response({"id": run.id}, 202)


async def build_publish(req: web.Request) -> web.Response:
    bid = req.match_info["bid"]
    if draft_info(bid) is None:
        return json_response({"error": "no such draft"}, 404)
    form = await req.post()
    try:
        target = builder.publish(builder.DRAFTS / bid, approved=str(form.get("approve", "")) == "yes")
    except recipe.BuildError as e:
        raise BadRequest(str(e), 409) from None
    return json_response({"preset": target.name})


async def draft_file(req: web.Request) -> web.StreamResponse:
    bid, name = req.match_info["bid"], req.match_info["name"]
    if not SAFE_FILE.match(bid) or not SAFE_FILE.match(name):
        raise web.HTTPNotFound()
    path = builder.DRAFTS / bid / name
    if not path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(path)


async def health(_: web.Request) -> web.Response:
    base = os.environ.get("COMFY_BASE_URL", runner.DEFAULT_BASE_URL)
    try:
        with urllib.request.urlopen(f"{base}/api/v2/health", timeout=3) as r:
            upstream = json.loads(r.read())
    except Exception as e:
        upstream = {"status": "down", "error": str(e)}
    return json_response({"comfy_api": base, "upstream": upstream})


async def stop_exports(_: web.Application) -> None:
    ui_export.stop_live_exports()


def make_app() -> web.Application:
    app = web.Application(client_max_size=MAX_VIDEO + 1024 * 1024, middlewares=[bad_requests])
    app[WORKER] = Worker()
    app.on_shutdown.append(stop_exports)
    app.add_routes([
        web.get("/", index),
        web.get("/api/presets", list_presets),
        web.get("/presets/{pid}/{name}", preset_file),
        web.post("/api/runs", create_run),
        web.get("/api/runs/{rid}", get_run),
        web.get("/runs/{rid}/{name:.+}", run_file),
        web.post("/api/build", build_draft),
        web.get("/api/build/{bid}", build_info),
        web.post("/api/build/{bid}/preview", build_preview),
        web.post("/api/build/{bid}/publish", build_publish),
        web.get("/drafts/{bid}/{name}", draft_file),
        web.get("/api/health", health),
    ])
    app.router.add_static("/static/", WEB_DIR)
    return app


def serve(host: str = "127.0.0.1", port: int = 8190) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"FreeField on http://{host}:{port}  (Comfy API: {runner.DEFAULT_BASE_URL} unless COMFY_BASE_URL is set)")
    web.run_app(make_app(), host=host, port=port, print=None)
