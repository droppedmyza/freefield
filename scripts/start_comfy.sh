#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${COMFY_PORT:-8188}"
PROXY_PORT="${PROXY_PORT:-8189}"
mkdir -p "$ROOT/logs"
# shellcheck disable=SC1091
source "$ROOT/.runtime/venv/bin/activate"

DEFAULT_ARGS=(--disable-async-offload)

wait_for() {
  for _ in $(seq 1 "$2"); do
    curl -sf "$1" >/dev/null && return 0
    sleep 1
  done
  echo "not up: $1; tail of $3:" >&2; tail -30 "$3" >&2; return 1
}

if stats=$(curl -sf "http://127.0.0.1:$PORT/system_stats"); then
  echo "ComfyUI already running on :$PORT"
  for flag in "${DEFAULT_ARGS[@]}"; do
    case "$stats" in *"$flag"*) ;; *) echo "WARNING: the running ComfyUI lacks $flag, so renders are not deterministic. Restart it: scripts/stop_comfy.sh && scripts/start_comfy.sh" >&2 ;; esac
  done
else
  [ -f "$ROOT/logs/comfy.log" ] && mv -f "$ROOT/logs/comfy.log" "$ROOT/logs/comfy.prev.log"
  WRAP=()
  if [ -n "${COMFY_MEM_MAX-6G}" ] && command -v systemd-run >/dev/null; then
    WRAP=(systemd-run --user --scope --quiet --slice=background.slice -p "MemoryMax=${COMFY_MEM_MAX-6G}" --)
  fi
  (cd "$ROOT/.runtime/ComfyUI" && exec "${WRAP[@]}" nohup python main.py --listen 127.0.0.1 --port "$PORT" --disable-auto-launch \
    "${DEFAULT_ARGS[@]}" ${COMFY_ARGS:-} > "$ROOT/logs/comfy.log" 2>&1) &
  wait_for "http://127.0.0.1:$PORT/system_stats" 180 "$ROOT/logs/comfy.log"
  echo "ComfyUI up on :$PORT (log: logs/comfy.log)"
fi

if curl -sf "http://127.0.0.1:$PROXY_PORT/api/v2/health" >/dev/null; then
  echo "comfy-api-proxy already running on :$PROXY_PORT"
else
  nohup comfy-api-proxy --comfyui "http://127.0.0.1:$PORT" --port "$PROXY_PORT" \
    --comfyui-base-dir "$ROOT/.runtime/ComfyUI" --state-dir "$ROOT/.runtime/proxy-state" \
    --max-upload-mb 50 > "$ROOT/logs/proxy.log" 2>&1 &
  wait_for "http://127.0.0.1:$PROXY_PORT/api/v2/health" 30 "$ROOT/logs/proxy.log"
  echo "comfy-api-proxy up on :$PROXY_PORT (Comfy API v2; log: logs/proxy.log)"
fi
