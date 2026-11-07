#!/usr/bin/env bash
# Run a Tinker-style training script against the Isomorph server on this Mac.
#   ./scripts/tune.sh recipes/sl_basic.py
# The script imports isotinker (in client/) and finds the server by itself.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: ./scripts/tune.sh your_script.py [args]"; exit 1; }
SCRIPT="$1"; shift
HERE="$(cd "$(dirname "$0")/.." && pwd)"
if [ ! -f "$SCRIPT" ] && [ -f "$HERE/$SCRIPT" ]; then SCRIPT="$HERE/$SCRIPT"; fi
[ -f "$SCRIPT" ] || { echo "No such script: $SCRIPT"; exit 1; }
SCRIPT="$(cd "$(dirname "$SCRIPT")" && pwd)/$(basename "$SCRIPT")"
cd "$HERE"
[ -x .venv/bin/python ] || ./setup.sh
export PYTHONPATH="$HERE/client${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
exec .venv/bin/python "$SCRIPT" "$@"
