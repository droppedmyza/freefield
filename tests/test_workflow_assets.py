import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "workflows"
LOADERS = {"LoadImage": "image", "LoadImageMask": "image", "LoadVideo": "file"}
PLACEHOLDERS = {"your_photo.png", "your_clip.mp4"}


def shipped() -> dict[str, str]:
    index = json.loads((WF / "assets" / "assets.json").read_text())
    return {row["file"]: row["sha256"] for rows in index.values() for row in rows}


def test_every_file_an_exported_workflow_loads_is_shipped_or_a_placeholder():
    files = shipped()
    for api in sorted(WF.glob("*.api.json")):
        pid = api.name[: -len(".api.json")]
        for node in json.loads(api.read_text()).values():
            key = LOADERS.get(node["class_type"])
            if not key:
                continue
            name = node["inputs"][key]
            assert name in PLACEHOLDERS or (name in files and name.startswith(f"{pid}--")), (api.name, name)
        ui = json.loads((WF / f"{pid}.json").read_text())
        for node in ui["nodes"]:
            if node["type"] in LOADERS:
                name = node["widgets_values"][0]
                assert name in PLACEHOLDERS or name in files, (f"{pid}.json", name)


def test_the_shipped_files_match_their_hashes():
    for name, digest in shipped().items():
        assert hashlib.sha256((WF / "assets" / name).read_bytes()).hexdigest() == digest, name


def test_the_installer_copies_every_file_into_an_empty_input_folder(tmp_path):
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "install_workflow_assets.py"), "--input-dir", str(tmp_path)],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(shipped())


def test_the_installer_refuses_a_changed_file(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "x--plate.png").write_bytes(b"changed")
    (assets / "assets.json").write_text(json.dumps({"x": [{"file": "x--plate.png", "sha256": "0" * 64}]}))
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "install_workflow_assets.py"), "--input-dir", str(tmp_path),
                          "--assets", str(assets)], capture_output=True, text=True)
    assert res.returncode != 0 and "sha256" in res.stderr
