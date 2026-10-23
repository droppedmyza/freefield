#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RT="$ROOT/.runtime"
COMFY_REF="${COMFY_REF:-bdc63e7}"
mkdir -p "$RT"
if [ ! -d "$RT/ComfyUI/.git" ]; then
  git clone https://github.com/comfyanonymous/ComfyUI.git "$RT/ComfyUI"
  git -C "$RT/ComfyUI" checkout -q "$COMFY_REF" || echo "warn: could not pin $COMFY_REF, using HEAD"
fi
if [ ! -x "$RT/venv/bin/python" ]; then
  uv venv --python 3.13 "$RT/venv"
fi
# shellcheck disable=SC1091
source "$RT/venv/bin/activate"
python -c "import torch" 2>/dev/null || uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
uv pip install -q -r "$RT/ComfyUI/requirements.txt"
CN="$RT/ComfyUI/custom_nodes"
if [ ! -d "$CN/ComfyUI-GGUF" ]; then
  git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git "$CN/ComfyUI-GGUF"
fi
uv pip install -q -r "$CN/ComfyUI-GGUF/requirements.txt"
uv pip install -q "comfy-sdk==0.5.0" "comfy-api-proxy==0.1.6" "opencv-python-headless==5.0.0.93" "onnxruntime==1.31.0" "open_clip_torch==3.3.0" "transformers==5.20.0" "pytest==9.1.1"
uv pip install -q --no-deps -e "$ROOT"
"$ROOT/scripts/download_models.sh" "$@"
echo "setup complete"
