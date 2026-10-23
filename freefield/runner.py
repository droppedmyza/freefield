from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8189"

ProgressFn = Callable[[dict], None]


@dataclass(frozen=True)
class CompiledRun:
    graph: dict[str, Any]
    files: list[tuple[str, str, Path]]
    outputs: list[str]


@dataclass
class RunResult:
    job_id: str
    status: str
    seconds: float
    files: dict[str, list[Path]] = field(default_factory=dict)
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "succeeded" and any(self.files.values())


def make_client():
    from comfy_sdk import Comfy

    if not os.environ.get("COMFY_BASE_URL") and not os.environ.get("COMFY_API_KEY"):
        os.environ["COMFY_BASE_URL"] = DEFAULT_BASE_URL
    return Comfy(timeout=120.0, client_info="freefield/0.1")


def submit(client, compiled: CompiledRun):
    workflow = client.workflows.from_json(compiled.graph)
    for node_id, input_name, path in compiled.files:
        workflow.set_input(node_id, input_name, client.assets.from_file(path))
    return client.submit(workflow)


def stream_progress(job, on_progress: ProgressFn, started: float, timeout: float) -> None:
    from comfy_sdk import ComfyError, OutputReady, Progress, StatusChange

    on_progress({"type": "status", "status": "submitted", "job_id": job.id})
    try:
        for ev in job.events():
            if isinstance(ev, Progress):
                on_progress({"type": "progress", "value": ev.value, "node": ev.current_node, "step": ev.step,
                             "steps": ev.steps, "nodes_done": ev.nodes_done, "nodes_total": ev.nodes_total})
            elif isinstance(ev, StatusChange):
                on_progress({"type": "status", "status": ev.status, "queue_position": ev.queue_position})
            elif isinstance(ev, OutputReady):
                on_progress({"type": "output", "node": ev.output.node_id})
            if time.monotonic() - started > timeout:
                break
    except ComfyError as e:
        log.warning("event stream ended early: %s", e)


def download(job, outputs: list[str], out_dir: Path) -> dict[str, list[Path]]:
    files: dict[str, list[Path]] = {}
    for node_id in outputs:
        for i, out in enumerate(job.get_outputs(node_id)):
            dest = out_dir / f"{node_id}_{i}_{out.name}"
            out.to_file(dest)
            files.setdefault(node_id, []).append(dest)
    return files


def run(compiled: CompiledRun, out_dir: Path, *, client=None, on_progress: ProgressFn | None = None,
        timeout: float = 3600.0, label: str = "freefield") -> RunResult:
    from comfy_sdk import ComfyError

    own_client = client is None
    client = client or make_client()
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    job_id = ""
    try:
        job = submit(client, compiled)
        job_id = job.id
        log.info("submitted %s job %s", label, job_id)
        if on_progress:
            stream_progress(job, on_progress, started, timeout)
        try:
            job = job.wait(timeout=max(1.0, timeout - (time.monotonic() - started)))
        except (ComfyError, TimeoutError) as e:
            return RunResult(job_id, "timeout" if isinstance(e, TimeoutError) else "failed", time.monotonic() - started,
                             error=f"{type(e).__name__}: {e}")
        result = RunResult(job_id, job.status, time.monotonic() - started)
        if job.status != "succeeded":
            result.error = str(job.error)
            return result
        result.files = download(job, compiled.outputs, out_dir)
        if not any(result.files.values()):
            result.error = f"job finished but output nodes {compiled.outputs} produced no files"
        return result
    except ComfyError as e:
        return RunResult(job_id, "failed", time.monotonic() - started, error=f"{type(e).__name__}: {e}")
    finally:
        if own_client:
            client.close()
