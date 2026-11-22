#!/usr/bin/env bash
# Push the working tree to remote build/test machines over rsync (no git needed).
# Usage: scripts/sync.sh host [host ...]
# LIGHTSIM_REMOTE_DIR sets the destination (default: lightsim, relative to the
# remote home), so several working copies can build side by side on one host.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ $# -eq 0 ]; then echo "usage: scripts/sync.sh host [host ...]" >&2; exit 2; fi
HOSTS=("$@")
for h in "${HOSTS[@]}"; do
  rsync -az --delete \
    --exclude target/ --exclude .venv/ --exclude '__pycache__/' --exclude '*.so' \
    --exclude 'benchmarks/results/' --exclude '.git/' --exclude '.claude/' --exclude '.venv-agent/' \
    "$ROOT/" "$h:${LIGHTSIM_REMOTE_DIR:-lightsim}/"
done
