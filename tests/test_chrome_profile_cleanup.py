import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHROME = next((c for c in ("chromium", "google-chrome-stable", "google-chrome") if shutil.which(c)), None)
pytestmark = pytest.mark.skipif(not (shutil.which("node") and CHROME), reason="needs node and Chrome/Chromium")

FAKE_COMFY = """<!doctype html><script>
const nodes = [];
window.LiteGraph = { NODE_TITLE_HEIGHT: 30, registered_node_types: {} };
window.app = {
  graph: { _nodes: nodes, _groups: [], serialize: () => ({ nodes: [], links: [], groups: [] }) },
  canvas: { ds: {}, canvas: { width: 800, height: 600 }, setDirty() {}, draw() {} },
  loadApiJson: async (api) => { for (const id of Object.keys(api)) nodes.push({ id: Number(id), size: [200, 100], widgets: [] }); },
  loadGraphData: async () => {},
  graphToPrompt: async () => ({ output: {} }),
};
</script>"""


def port_busy(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture
def comfy(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text(FAKE_COMFY)
    (site / "blank.html").write_text("<!doctype html><title>no app</title>")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(site)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture
def tmp():
    with tempfile.TemporaryDirectory(prefix="ffct-") as d:
        yield Path(d)


def env_with(tmp: Path) -> dict:
    return {**os.environ, "TMPDIR": str(tmp)}


def chrome_using(profile_prefix: Path) -> list[int]:
    pids = []
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                if f"--user-data-dir={profile_prefix}".encode() in (p / "cmdline").read_bytes():
                    pids.append(int(p.name))
            except OSError:
                pass
    return pids


def export_inputs(tmp_path: Path) -> list[str]:
    api = tmp_path / "workflow_api.json"
    api.write_text(json.dumps({"painting_1": {"class_type": "Note", "inputs": {}}}))
    stages = tmp_path / "stages.json"
    stages.write_text(json.dumps([{"title": "1. Painting", "match": ["painting_"]}]))
    return [str(api), str(stages), str(tmp_path / "workflow.json"), str(tmp_path / "graph.png")]


def test_export_ui_workflow_removes_its_profile_after_a_run(tmp_path, tmp, comfy):
    if port_busy(9333):
        pytest.skip("port 9333 is in use by a running export")
    res = subprocess.run(["node", str(ROOT / "scripts" / "export_ui_workflow.mjs"), *export_inputs(tmp_path), comfy + "/index.html"],
                         env=env_with(tmp), capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "workflow.json").exists()
    assert list(tmp.glob("ffchrome-*")) == []
    assert chrome_using(tmp) == []


def test_export_ui_workflow_removes_its_profile_on_sigterm(tmp_path, tmp, comfy):
    if port_busy(9333):
        pytest.skip("port 9333 is in use by a running export")
    proc = subprocess.Popen(["node", str(ROOT / "scripts" / "export_ui_workflow.mjs"), *export_inputs(tmp_path), comfy + "/blank.html"],
                            env=env_with(tmp), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.time() + 30
    while time.time() < deadline and not (list(tmp.glob("ffchrome-*")) and chrome_using(tmp) and port_busy(9333)):
        time.sleep(0.2)
    assert list(tmp.glob("ffchrome-*")), "the script never made its profile"
    time.sleep(1)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=20) == 128 + signal.SIGTERM
    assert list(tmp.glob("ffchrome-*")) == []
    assert chrome_using(tmp) == []


def test_check_workflows_removes_every_profile_after_a_run(tmp_path, tmp, comfy):
    if any(port_busy(p) for p in (9401, 9402)):
        pytest.skip("ports 9401-9402 are in use by a running check")
    wf = tmp_path / "workflows"
    (wf / "proof").mkdir(parents=True)
    for name in ("a", "b"):
        (wf / f"{name}.json").write_text(json.dumps({"nodes": [], "links": []}))
    res = subprocess.run(["node", str(ROOT / "scripts" / "check_workflows.mjs"), str(wf), comfy + "/index.html"],
                         env=env_with(tmp), capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    assert len(json.loads((wf / "load-proof.json").read_text())["results"]) == 2
    assert list(tmp.glob("wf-*")) == []
    assert chrome_using(tmp) == []


FAKE_CHROME = """#!/bin/bash
for a in "$@"; do case $a in --version) exit 0;; --user-data-dir=*) d=${a#--user-data-dir=};; esac; done
trap '' TERM INT HUP
(while :; do mkdir -p "$d"; touch "$d/alive"; sleep 0.05; done) &
wait
"""


@pytest.fixture
def stubborn_chrome(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "chromium"
    fake.write_text(FAKE_CHROME)
    fake.chmod(0o755)
    return str(bin_dir)


def run_until_signalled(argv: list[str], tmp: Path, prefix: str, bin_dir: str) -> tuple[int, float]:
    env = {**env_with(tmp), "PATH": bin_dir + os.pathsep + os.environ["PATH"]}
    proc = subprocess.Popen(argv, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 10
    while time.time() < deadline and not (any(tmp.glob(prefix + "*/alive")) and chrome_using(tmp)):
        time.sleep(0.05)
    assert chrome_using(tmp), "the fake browser never started"
    started = time.time()
    proc.send_signal(signal.SIGTERM)
    time.sleep(0.3)
    proc.send_signal(signal.SIGINT)
    code = proc.wait(timeout=30)
    return code, time.time() - started


def test_export_ui_workflow_kills_a_browser_that_ignores_sigterm_before_removing_its_profile(tmp_path, tmp, stubborn_chrome):
    argv = ["node", str(ROOT / "scripts" / "export_ui_workflow.mjs"), *export_inputs(tmp_path), "http://127.0.0.1:9/"]
    code, took = run_until_signalled(argv, tmp, "ffchrome-", stubborn_chrome)
    assert code == 128 + signal.SIGTERM
    assert took >= 4, "the profile was removed without waiting for the browser to exit"
    assert chrome_using(tmp) == []
    time.sleep(0.5)
    assert list(tmp.glob("ffchrome-*")) == []


def test_check_workflows_kills_a_browser_that_ignores_sigterm_before_removing_its_profile(tmp_path, tmp, stubborn_chrome):
    wf = tmp_path / "workflows"
    (wf / "proof").mkdir(parents=True)
    (wf / "a.json").write_text(json.dumps({"nodes": [], "links": []}))
    argv = ["node", str(ROOT / "scripts" / "check_workflows.mjs"), str(wf), "http://127.0.0.1:9/"]
    code, took = run_until_signalled(argv, tmp, "wf-", stubborn_chrome)
    assert code == 128 + signal.SIGTERM
    assert took >= 4, "the profile was removed without waiting for the browser to exit"
    assert chrome_using(tmp) == []
    time.sleep(0.5)
    assert list(tmp.glob("wf-*")) == []


DROP_GROUP_SIGKILL = """const kill = process.kill.bind(process);
process.kill = (pid, sig) => (pid < 0 && sig === "SIGKILL" ? true : kill(pid, sig));
"""

CRASH_WHEN_BROWSER_UP = """import { readdirSync, existsSync } from "node:fs";
const t = setInterval(() => {
  if (readdirSync(process.env.TMPDIR).some((d) => existsSync(`${process.env.TMPDIR}/${d}/alive`))) { clearInterval(t); throw new Error("injected crash"); }
}, 50);
"""


def stubborn_argv(name: str, tmp_path: Path) -> tuple[list[str], str]:
    if name == "export":
        return ["node", str(ROOT / "scripts" / "export_ui_workflow.mjs"), *export_inputs(tmp_path), "http://127.0.0.1:9/"], "ffchrome-"
    wf = tmp_path / "workflows"
    (wf / "proof").mkdir(parents=True)
    (wf / "a.json").write_text(json.dumps({"nodes": [], "links": []}))
    return ["node", str(ROOT / "scripts" / "check_workflows.mjs"), str(wf), "http://127.0.0.1:9/"], "wf-"


def start_with_preload(argv: list[str], tmp: Path, bin_dir: str, preload: Path) -> subprocess.Popen:
    env = {**env_with(tmp), "PATH": bin_dir + os.pathsep + os.environ["PATH"], "NODE_OPTIONS": f"--import={preload}"}
    return subprocess.Popen(argv, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def kill_groups_using(tmp: Path) -> None:
    for pgid in {os.getpgid(p) for p in chrome_using(tmp)}:
        os.killpg(pgid, signal.SIGKILL)


@pytest.mark.parametrize("name", ["export", "check"])
def test_a_browser_group_that_cannot_be_killed_keeps_its_profile(name, tmp_path, tmp, stubborn_chrome):
    preload = tmp_path / "drop-group-sigkill.mjs"
    preload.write_text(DROP_GROUP_SIGKILL)
    argv, prefix = stubborn_argv(name, tmp_path)
    proc = start_with_preload(argv, tmp, stubborn_chrome, preload)
    try:
        deadline = time.time() + 10
        while time.time() < deadline and not (any(tmp.glob(prefix + "*/alive")) and chrome_using(tmp)):
            time.sleep(0.05)
        assert chrome_using(tmp), "the fake browser never started"
        proc.send_signal(signal.SIGTERM)
        _, err = proc.communicate(timeout=30)
        assert proc.returncode == 128 + signal.SIGTERM
        assert "leaving" in err
        assert chrome_using(tmp)
        assert list(tmp.glob(prefix + "*"))
    finally:
        proc.kill()
        kill_groups_using(tmp)


@pytest.mark.parametrize("name", ["export", "check"])
def test_a_crash_still_kills_the_browser_and_removes_its_profile(name, tmp_path, tmp, stubborn_chrome):
    preload = tmp_path / "crash.mjs"
    preload.write_text(CRASH_WHEN_BROWSER_UP)
    argv, prefix = stubborn_argv(name, tmp_path)
    proc = start_with_preload(argv, tmp, stubborn_chrome, preload)
    try:
        _, err = proc.communicate(timeout=30)
        assert "injected crash" in err
        assert proc.returncode != 0
        assert chrome_using(tmp) == []
        time.sleep(0.5)
        assert list(tmp.glob(prefix + "*")) == []
    finally:
        proc.kill()
        kill_groups_using(tmp)


def test_a_timed_out_export_layout_leaves_no_browser_or_profile(tmp_path, tmp, stubborn_chrome, monkeypatch):
    from freefield import ui_export

    folder = tmp_path / "draft"
    folder.mkdir()
    (folder / "graph.json").write_text(json.dumps({"photo": {"class_type": "LoadImage", "inputs": {"image": "x.png"}}}))
    (folder / "preset.json").write_text(json.dumps({"fixed": {}}))
    (folder / "stages.json").write_text(json.dumps([{"title": "1. Photo", "match": ["photo"]}]))
    monkeypatch.setenv("TMPDIR", str(tmp))
    monkeypatch.setenv("PATH", stubborn_chrome + os.pathsep + os.environ["PATH"])
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            ui_export.export_layout(folder, "http://127.0.0.1:9/", timeout=2)
        assert chrome_using(tmp) == []
        time.sleep(0.5)
        assert list(tmp.glob("ffchrome-*")) == []
    finally:
        kill_groups_using(tmp)


EXPORTER = """
import json, sys
from pathlib import Path
from freefield import ui_export
folder = Path(sys.argv[1])
print(ui_export.export_layout(folder, "http://127.0.0.1:9/"), flush=True)
"""


def export_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "draft"
    folder.mkdir()
    (folder / "graph.json").write_text(json.dumps({"photo": {"class_type": "LoadImage", "inputs": {"image": "x.png"}}}))
    (folder / "preset.json").write_text(json.dumps({"fixed": {}}))
    (folder / "stages.json").write_text(json.dumps([{"title": "1. Photo", "match": ["photo"]}]))
    return folder


def processes_mentioning(text: str) -> list[int]:
    out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout
    return [int(line.split()[0]) for line in out.splitlines() if text in line]


def wait_for_browser(tmp: Path) -> None:
    deadline = time.time() + 10
    while time.time() < deadline and not (any(tmp.glob("ffchrome-*/alive")) and chrome_using(tmp)):
        time.sleep(0.05)
    assert chrome_using(tmp), "the fake browser never started"


def test_sigterm_to_the_python_exporter_alone_ends_node_the_browser_and_the_profile(tmp_path, tmp, stubborn_chrome):
    folder = export_folder(tmp_path)
    env = {**env_with(tmp), "PATH": stubborn_chrome + os.pathsep + os.environ["PATH"], "PYTHONPATH": str(ROOT)}
    parent = subprocess.Popen([sys.executable, "-c", EXPORTER, str(folder)], env=env, start_new_session=True,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_browser(tmp)
        assert processes_mentioning(str(folder / "workflow_api.json")), "node never started"
        parent.send_signal(signal.SIGTERM)
        assert parent.wait(timeout=60) == -signal.SIGTERM
        assert processes_mentioning(str(folder / "workflow_api.json")) == []
        assert chrome_using(tmp) == []
        assert list(tmp.glob("ffchrome-*")) == []
        time.sleep(0.5)
        assert list(tmp.glob("ffchrome-*")) == []
    finally:
        parent.kill()
        for pid in processes_mentioning(str(folder / "workflow_api.json")):
            os.kill(pid, signal.SIGKILL)
        kill_groups_using(tmp)


def test_server_shutdown_ends_a_running_export(tmp_path, tmp, stubborn_chrome, monkeypatch):
    import asyncio

    from aiohttp.test_utils import TestClient, TestServer

    from freefield import server, ui_export

    folder = export_folder(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp))
    monkeypatch.setenv("PATH", stubborn_chrome + os.pathsep + os.environ["PATH"])
    result = {}
    worker = threading.Thread(target=lambda: result.update(error=ui_export.export_layout(folder, "http://127.0.0.1:9/", timeout=120)))
    worker.start()
    try:
        wait_for_browser(tmp)

        async def start_and_shut_down():
            async with TestClient(TestServer(server.make_app())):
                pass

        asyncio.run(start_and_shut_down())
        worker.join(timeout=30)
        assert not worker.is_alive()
        assert result["error"] is not None
        assert processes_mentioning(str(folder / "workflow_api.json")) == []
        assert chrome_using(tmp) == []
        time.sleep(0.5)
        assert list(tmp.glob("ffchrome-*")) == []
    finally:
        ui_export.stop_live_exports(signal.SIGKILL)
        worker.join(timeout=10)
        kill_groups_using(tmp)
