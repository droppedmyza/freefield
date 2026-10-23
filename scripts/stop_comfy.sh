#!/usr/bin/env bash
set -u
stop() {
  local pids; pids=$(pgrep -f "$1" || true)
  [ -z "$pids" ] && { echo "$2 not running"; return; }
  kill $pids
  for _ in $(seq 1 20); do sleep 1; pgrep -f "$1" >/dev/null || { echo "$2 stopped"; return; }; done
  kill -9 $pids 2>/dev/null; echo "$2 killed"
}
stop "comfy-api-proxy --comfyui http://127.0.0.1:${COMFY_PORT:-8188}" "comfy-api-proxy"
stop "python main.py --listen 127.0.0.1 --port ${COMFY_PORT:-8188}" "ComfyUI"
