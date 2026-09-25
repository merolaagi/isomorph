#!/usr/bin/env bash
# Create or update the local Python environment. Safe to re-run after every upgrade.
set -euo pipefail
cd "$(dirname "$0")"
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then echo "Isomorph needs Python 3.10 or newer. Install one with: brew install python@3.12"; exit 1; fi
[ -d .venv ] || { echo "Creating .venv with $PY"; "$PY" -m venv .venv; }
HASH="$(shasum requirements.txt | cut -d' ' -f1)"
if [ ! -f .venv/.req-hash ] || [ "$(cat .venv/.req-hash)" != "$HASH" ]; then
  echo "Installing dependencies (torch is large; the first install takes a few minutes)"
  .venv/bin/python -m pip install --upgrade pip -q
  .venv/bin/python -m pip install -r requirements.txt -q
  echo "$HASH" > .venv/.req-hash
fi
chmod +x run.sh scripts/*.sh 2>/dev/null || true
.venv/bin/python -c "import torch; print('Isomorph', open('VERSION').read().strip(), 'ready. torch', torch.__version__, '| Apple GPU (MPS):', torch.backends.mps.is_available())"
