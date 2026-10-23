import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows"
EXPORTED = [r["preset"] for r in json.loads((WORKFLOWS / "export.json").read_text())]


def exporter():
    spec = importlib.util.spec_from_file_location("export_workflows", ROOT / "scripts" / "export_workflows.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("pid", EXPORTED)
def test_every_exported_workflow_rebuilds_from_the_repo(pid):
    graph, data, assets = exporter().filled(pid)
    assert graph == json.loads((WORKFLOWS / f"{pid}.api.json").read_text())
    for src, name in assets:
        assert src.is_file() and (WORKFLOWS / "assets" / name).read_bytes() == src.read_bytes()


@pytest.mark.parametrize("pid", EXPORTED)
def test_every_ui_workflow_has_its_note_and_numbered_groups(pid):
    wf = json.loads((WORKFLOWS / f"{pid}.json").read_text())
    notes = [n for n in wf["nodes"] if n["type"] in ("MarkdownNote", "Note")]
    assert len(notes) == 1 and notes[0]["widgets_values"][0].startswith("# ")
    numbered = [g["title"].split(" ", 1)[0] for g in wf["groups"] if g["title"][:1].isdigit()]
    assert numbered and all(t.endswith(".") for t in numbered)


def test_the_load_proof_covers_every_workflow():
    proof = json.loads((WORKFLOWS / "load-proof.json").read_text())["results"]
    assert sorted(r["workflow"] for r in proof) == sorted(f"{pid}.json" for pid in EXPORTED)
    assert all(r["converts_back"] and not r["missing_node_types"] and r["loaded_nodes"] == r["file_nodes"] for r in proof)
