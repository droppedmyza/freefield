from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "workflows" / "assets"


def manifest(assets: Path, presets: list[str]) -> list[Path]:
    index = json.loads((assets / "assets.json").read_text())
    unknown = sorted(set(presets) - set(index))
    if unknown:
        raise SystemExit(f"no such preset in {assets / 'assets.json'}: {', '.join(unknown)}")
    files = []
    for pid in presets or sorted(index):
        for row in index[pid]:
            path = assets / row["file"]
            if not path.is_file():
                raise SystemExit(f"missing {path}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise SystemExit(f"{path.name} does not match its sha256 in assets.json")
            files.append(path)
    return files


def upload(url: str, path: Path, auth: str | None) -> dict:
    boundary = uuid.uuid4().hex
    body = b"".join([
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{path.name}\"\r\n"
        "Content-Type: application/octet-stream\r\n\r\n".encode(), path.read_bytes(), b"\r\n",
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n".encode(),
        f"--{boundary}--\r\n".encode()])
    req = urllib.request.Request(url.rstrip("/") + "/upload/image", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                                          "User-Agent": "freefield-install-assets/1.0"})
    if auth:
        req.add_header("Authorization", "Basic " + base64.b64encode(auth.encode()).decode())
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"upload of {path.name} failed: HTTP {e.code} {e.read()[:200]!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Put the files FreeField's exported workflows load (plates, masks, artwork, wipes, "
                                             "headers) into a ComfyUI input folder, by copying or by uploading to a running ComfyUI")
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--input-dir", type=Path, help="ComfyUI's input folder, e.g. ComfyUI/input")
    where.add_argument("--url", help="a running ComfyUI, e.g. http://127.0.0.1:8188; basic auth from COMFYUI_AUTH=user:password")
    ap.add_argument("--assets", type=Path, default=ASSETS)
    ap.add_argument("presets", nargs="*", help="only these presets (default: all)")
    a = ap.parse_args()
    files = manifest(a.assets, a.presets)
    done = []
    for path in files:
        if a.input_dir:
            if not a.input_dir.is_dir():
                raise SystemExit(f"{a.input_dir} is not a folder")
            shutil.copy(path, a.input_dir / path.name)
            done.append(path.name)
        else:
            done.append(upload(a.url, path, os.environ.get("COMFYUI_AUTH"))["name"])
    print(json.dumps({"installed": done, "to": str(a.input_dir or a.url)}, indent=1))
    if len(done) != len(files):
        sys.exit(1)


if __name__ == "__main__":
    main()
