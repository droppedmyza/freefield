#!/usr/bin/env python3
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from freefield import runner  # noqa: E402
from freefield.runner import CompiledRun  # noqa: E402

STYLE = ("casual smartphone selfie photo, head and shoulders, looking at the camera, natural window light, "
         "realistic skin texture, slight smile, plain indoor background, sharp focus, 35mm photo")
PEOPLE = [
    ("dev-1", "a woman in her late twenties with curly dark hair and brown skin, wearing a denim jacket", 101),
    ("dev-2", "a man in his thirties with short blond hair and a light beard, wearing a grey hoodie", 202),
    ("dev-3", "an East Asian woman in her forties with a black bob haircut and round glasses, wearing a white shirt", 303),
    ("heldout-1", "a South Asian man in his twenties with wavy black hair, wearing a green t-shirt", 404),
    ("heldout-2", "an older man in his sixties with grey hair and a mustache, wearing a navy sweater", 505),
]

def main() -> int:
    graph = json.loads((ROOT / "freefield/graphs/portrait.json").read_text())
    out = ROOT / "eval/fixtures"
    manifest = []
    client = runner.make_client()
    for name, who, seed in PEOPLE:
        dest = out / f"{name}.png"
        if dest.exists():
            print("ok   ", dest.name); manifest.append({"name": name, "who": who, "seed": seed}); continue
        g = json.loads(json.dumps(graph))
        g["keyframe_prompt"]["inputs"]["text"] = f"{STYLE}, {who}"
        g["keyframe_noise"]["inputs"]["noise_seed"] = seed
        res = runner.run(CompiledRun(g, [], ["keyframe_save"]), ROOT / "outputs" / "fixtures", client=client, label="fixtures")
        if not res.ok:
            print("FAIL ", name, res.status, res.error); return 1
        shutil.copy(res.files["keyframe_save"][0], dest)
        print(f"made  {dest.name} in {res.seconds:.1f}s")
        manifest.append({"name": name, "who": who, "seed": seed})
    (out / "fixtures.json").write_text(json.dumps({"style": STYLE, "people": manifest}, indent=2) + "\n")
    client.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
