#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/.runtime/ComfyUI/models"
MANIFEST="$ROOT/scripts/models.tsv"
want=("$@")

in_groups() {
  [ ${#want[@]} -eq 0 ] && return 0
  local g
  for g in "${want[@]}"; do [ "$g" = "$1" ] && return 0; done
  return 1
}

grep -v '^#' "$MANIFEST" | while IFS=$'\t' read -r group dest url license; do
  [ -z "${group:-}" ] && continue
  in_groups "$group" || continue
  out="$MODELS/$dest"
  mkdir -p "$(dirname "$out")"
  remote=$(curl -sIL "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{printf "%.0f\n", n+0}')
  local_size=$(stat -c %s "$out" 2>/dev/null || echo 0)
  if [ "$remote" -gt 0 ] && [ "$local_size" -eq "$remote" ]; then
    echo "ok    $dest ($license)"
    continue
  fi
  echo "fetch $dest ($((remote/1000000)) MB, $license)"
  curl -fsSL --retry 5 --retry-delay 5 -C - -o "$out.part" "$url" || { echo "FAILED $dest" >&2; exit 1; }
  mv "$out.part" "$out"
done
