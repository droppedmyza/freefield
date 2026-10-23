from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from freefield.builder import models_of  # noqa: E402
from freefield.presets import GRAPHS_DIR, fill_for_export  # noqa: E402
from freefield.ui_export import run_export_ui  # noqa: E402

PRESETS = ROOT / "freefield" / "presets"
PLACEHOLDER = {"image": "your_photo.png", "video": "your_clip.mp4"}
AFTER_GRAPH = {
    "earth_zoom": ("This graph makes the three stills. FreeField's `freefield/earth_zoom.py` then flies down through them "
                   "with cloud cuts, adds your place name and a whoosh, and plays your own clip untouched after the dive."),
    "product_freeze": ("This graph makes the frozen-splash still. FreeField's `freefield/product_freeze.py` then measures "
                       "depth (Depth Anything V2), moves a parallax camera around it, adds your offer text and ends on a "
                       "card with your product photo."),
    "comic_bonk": ("This graph makes the cut-out and the clean background. FreeField's `freefield/comic_bonk.py` then "
                   "animates the mallet bonk, the squash and recoil, the BONK! burst and the sound."),
    "raven_intro": ("This graph is Raven Poof. FreeField's `freefield/raven_intro.py` checks the frame edges, re-renders a "
                    "take with a stray hand, matches the exposure to your photo and adds your title and a whoosh."),
}


def preset_dir(pid: str) -> Path:
    for base in (PRESETS, PRESETS / "_drafts"):
        if (base / pid / "preset.json").exists():
            return base / pid
    raise SystemExit(f"no preset {pid}")


def asset_name(pid: str, name: str) -> str:
    return f"{pid}--{name}"


def filled(pid: str) -> tuple[dict, dict, list[tuple[Path, str]]]:
    folder = preset_dir(pid)
    data = json.loads((folder / "preset.json").read_text())
    ref = data["graph"]
    graph = json.loads((folder / ref if (folder / ref).exists() else GRAPHS_DIR / ref).read_text())
    placeholders = {}
    for spec in data.get("inputs", {}).values():
        for bind in spec.get("bind", []):
            if bind.startswith("pipeline:"):
                continue
            kind = {"LoadVideo": "video", "LoadImage": "image"}.get(graph[bind.split(".", 1)[0]]["class_type"], spec["type"])
            placeholders[bind] = PLACEHOLDER.get(kind, "")
    for spec in data.get("params", {}).values():
        placeholders.update({bind: spec["default"] for bind in spec.get("bind", [])})
    assets = []

    def shelved(name: str) -> str:
        src = folder / name
        if not src.is_file():
            raise SystemExit(f"{pid}: asset {name} is missing from {folder}")
        assets.append((src, asset_name(pid, name)))
        return asset_name(pid, name)

    return fill_for_export(graph, data.get("fixed", {}), placeholders, shelved), data, assets


def stages(graph: dict, folder: Path) -> list[dict]:
    if (folder / "stages.json").exists():
        st = json.loads((folder / "stages.json").read_text())
        ids = list(graph)
        if all(any(i in s["match"] or any(i.startswith(m) for m in s["match"]) for s in st) for i in ids):
            return st
    colors = ["#335", "#353", "#533", "#553", "#355", "#535", "#444"]
    order: list[str] = []
    for nid in graph:
        head = nid.split("_")[0]
        if head not in order:
            order.append(head)
    return [{"title": h, "color": colors[i % len(colors)], "match": [h]} for i, h in enumerate(order)]


def markdown_safe(title: str) -> str:
    return re.sub(r"^(\w+)\. ", r"\1\\. ", title)


def source_flag(data: dict) -> str:
    kinds = {spec["type"] for spec in data.get("inputs", {}).values()}
    return "--clip walk.mp4" if "video" in kinds else "--photo me.jpg"


def hardware(data: dict) -> str:
    if "studio-profile" in data.get("tags", []) or any("14B" in m for m in data.get("models", [])):
        return "a 24 GB GPU (studio profile: Wan 2.2 I2V-A14B)"
    return "an 8 GB GPU"


def about_note(pid: str, data: dict, graph: dict, stage_list: list[dict], assets: list) -> str:
    lines = [f"# {data['name']}", "", data.get("description", ""), "",
             f"FreeField preset `{pid}`. Runs on {hardware(data)}. Open weights only.", "", "## Fill in"]
    for spec in data.get("inputs", {}).values():
        for bind in spec.get("bind", []):
            if bind.startswith("pipeline:"):
                lines.append(f"- {spec.get('label', 'Text')}: typed in the FreeField app (`--caption` on the command line)")
                continue
            node_id = bind.split(".", 1)[0]
            lines.append(f"- {spec.get('label', spec['type'])}: the node **{graph[node_id]['_meta']['title']}**")
    lines += ["", "## Stages, left to right"] + [f"- {markdown_safe(st['title'])}" for st in stage_list]
    lines += ["", "## Models"] + [f"- `{m}`" for m in models_of(graph)]
    lines += ["", "## What happens"]
    lines.append(AFTER_GRAPH.get(data.get("pipeline"), "This graph makes the finished video."))
    if data.get("gate"):
        lines.append("In the FreeField app the START, GRIP and OUT stills are checked before the video "
                     "(`freefield/gate.py`) and a bad one is re-rolled; in ComfyUI the graph runs once with the seed shown.")
    lines += ["", "## Run it"]
    if assets:
        lines.append(f"1. Install this workflow's files into ComfyUI's input folder: "
                     f"`python3 scripts/install_workflow_assets.py --url http://127.0.0.1:8188 {pid}`")
    lines += [f"{2 if assets else 1}. Pick your file in the input node above and press Run.", ""]
    if preset_dir(pid).parent.name == "_drafts":
        lines.append(f"This is a draft preset: it is not in the FreeField gallery yet, and its files are in "
                     f"`freefield/presets/_drafts/{pid}/`. Run it here in ComfyUI.")
    else:
        lines.append(f"Or run the whole effect in FreeField: `python -m freefield run {pid} {source_flag(data)}`")
    lines += ["",
              f"License: {data.get('license', 'see the README')}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    out = Path(sys.argv[1])
    url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8188"
    ids = sys.argv[3:]
    out.mkdir(parents=True, exist_ok=True)
    report = []
    shelf = out / "assets"
    manifest = json.loads((shelf / "assets.json").read_text()) if (shelf / "assets.json").exists() else {}
    for pid in ids:
        graph, data, assets = filled(pid)
        shelf.mkdir(exist_ok=True)
        manifest[pid] = []
        for src, name in assets:
            shutil.copy(src, shelf / name)
            manifest[pid].append({"file": name, "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                                  "from": src.relative_to(ROOT).as_posix()})
        api = out / f"{pid}.api.json"
        api.write_text(json.dumps(graph, indent=2) + "\n")
        stage_list = stages(graph, preset_dir(pid))
        st = out / f".{pid}.stages.json"
        st.write_text(json.dumps(stage_list, indent=2))
        note = out / f".{pid}.about.md"
        note.write_text(about_note(pid, data, graph, stage_list, assets))
        res = run_export_ui([str(api), str(st), str(out / f"{pid}.json"), str(out / f"{pid}.png"), url, str(note)], 300)
        st.unlink()
        note.unlink()
        report.append({"preset": pid, "name": data["name"], "pipeline": data.get("pipeline"), "nodes": len(graph),
                       "assets": [name for _, name in assets],
                       "ui_export": res.returncode == 0, "error": (res.stderr or res.stdout)[-300:] if res.returncode else None})
        print(json.dumps(report[-1]), flush=True)
    (shelf / "assets.json").write_text(json.dumps(manifest, indent=2) + "\n")
    old = json.loads((out / "export.json").read_text()) if (out / "export.json").exists() else []
    fresh = {r["preset"]: r for r in report}
    merged = [fresh.pop(r["preset"], r) for r in old] + list(fresh.values())
    (out / "export.json").write_text(json.dumps(merged, indent=2) + "\n")


if __name__ == "__main__":
    main()
