from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from freefield import runner  # noqa: E402
from freefield.runner import CompiledRun  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Run an exported ComfyUI API workflow through the Comfy SDK")
    ap.add_argument("workflow", help="an *.api.json file, e.g. workflows/raven-poof.api.json")
    ap.add_argument("--file", action="append", default=[], metavar="NODE.INPUT=PATH",
                    help="upload a local file into a node input, e.g. photo.image=me.png")
    ap.add_argument("--set", action="append", default=[], metavar="NODE.INPUT=VALUE", help="override an input value (JSON)")
    ap.add_argument("--out", default="outputs/api-run")
    args = ap.parse_args()
    graph = json.loads(Path(args.workflow).read_text())
    files = []
    for item in args.file:
        target, path = item.split("=", 1)
        nid, inp = target.split(".", 1)
        if nid not in graph:
            raise SystemExit(f"no node {nid!r} in {args.workflow}")
        files.append((nid, inp, Path(path)))
    for item in args.set:
        target, value = item.split("=", 1)
        nid, inp = target.split(".", 1)
        graph[nid]["inputs"][inp] = json.loads(value)
    outputs = [k for k, n in graph.items() if n["class_type"] in ("SaveImage", "SaveVideo")]
    res = runner.run(CompiledRun(graph, files, outputs), Path(args.out), label=Path(args.workflow).stem)
    print(json.dumps({"status": res.status, "error": res.error, "seconds": round(res.seconds, 1),
                      "files": {k: [str(p) for p in v] for k, v in res.files.items()}}, indent=2))
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
