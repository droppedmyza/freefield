from __future__ import annotations

import json
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from .presets import fill_for_export

ROOT = Path(__file__).resolve().parent.parent

EXPORT_UI_GRACE = 30
live_exports: set = set()


def stop_live_exports(signum: int = signal.SIGTERM) -> None:
    for proc in list(live_exports):
        if proc.poll() is None:
            proc.send_signal(signum)


def run_export_ui(args: list[str], timeout: float) -> subprocess.CompletedProcess:
    import subprocess

    cmd = ["node", str(ROOT / "scripts" / "export_ui_workflow.mjs"), *args]
    proc = None
    caught: list[tuple[int, float]] = []
    previous = {}

    def forward(signum, frame):
        caught.append((signum, time.monotonic()))
        stop_live_exports(signum)

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGHUP):
            handler = signal.getsignal(signum)
            if handler not in (signal.SIG_IGN, None):
                previous[signum] = handler
                signal.signal(signum, forward)
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        live_exports.add(proc)
        if caught:
            stop_live_exports(caught[0][0])
        deadline = time.monotonic() + timeout
        while True:
            try:
                out, err = proc.communicate(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                if caught and time.monotonic() > caught[0][1] + EXPORT_UI_GRACE:
                    proc.kill()
                elif not caught and time.monotonic() > deadline:
                    raise subprocess.TimeoutExpired(cmd, timeout) from None
    except BaseException:
        if proc is not None:
            proc.terminate()
            try:
                proc.communicate(timeout=EXPORT_UI_GRACE)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
        raise
    finally:
        live_exports.discard(proc)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if caught:
            signal.raise_signal(caught[0][0])
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def export_layout(folder: Path, comfy_url: str = "http://127.0.0.1:8188", timeout: float = 240) -> str | None:
    if shutil.which("node") is None:
        return "node is not installed"
    fixed = json.loads((folder / "preset.json").read_text())["fixed"]
    api = fill_for_export(json.loads((folder / "graph.json").read_text()), fixed, {"photo.image": "your_photo.png"})
    (folder / "workflow_api.json").write_text(json.dumps(api, indent=2) + "\n")
    res = run_export_ui([str(folder / "workflow_api.json"), str(folder / "stages.json"), str(folder / "workflow.json"),
                         str(folder / "graph.png"), comfy_url], timeout)
    return None if res.returncode == 0 else (res.stderr or res.stdout).strip()[-300:]
