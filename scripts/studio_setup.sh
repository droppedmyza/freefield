#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
W="${W:-/workspace}"
mkdir -p "$W"
cp "$ROOT/scripts/pod_setup.sh" "$W/pod_setup.sh"
bash "$W/pod_setup.sh"
mkdir -p "$ROOT/.runtime"
ln -sfn "$W/ComfyUI" "$ROOT/.runtime/ComfyUI"
pip install -q -e "$ROOT[checks,dev]"
bash "$ROOT/scripts/download_models.sh" eval matte
cd "$ROOT" && python -m freefield list
