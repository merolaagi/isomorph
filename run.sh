#!/usr/bin/env bash
# Start the Isomorph server and open it in the browser.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${ISOMORPH_PORT:-47821}"
[ -x .venv/bin/python ] || ./setup.sh
# If Isomorph runs as a background service, restart that instead of starting a second copy.
if [ -f "$HOME/Library/LaunchAgents/com.merolaagi.isomorph.plist" ]; then
  ./scripts/service.sh restart
  open "http://127.0.0.1:$PORT" >/dev/null 2>&1 || true
  exit 0
fi
# Stop an older Isomorph server on this port (for example after an upgrade).
OLD="$(pgrep -f "uvicorn server.app:app .*--port $PORT" || true)"
if [ -n "$OLD" ]; then echo "Stopping previous Isomorph server ($OLD)"; kill $OLD 2>/dev/null || true; sleep 1; fi
# If something else holds the port, move up to the next free one.
port_busy() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }
TRIES=0
while port_busy "$PORT"; do
  TRIES=$((TRIES + 1))
  [ "$TRIES" -gt 20 ] && { echo "No free port found near $PORT. Set ISOMORPH_PORT to a free port."; exit 1; }
  echo "Port $PORT is in use by another program, trying $((PORT + 1))"
  PORT=$((PORT + 1))
done
# Let PyTorch run any operation the Apple GPU lacks on the CPU instead of failing.
export PYTORCH_ENABLE_MPS_FALLBACK=1
echo "Isomorph on http://127.0.0.1:$PORT  (Ctrl+C to stop)"
( sleep 2; open "http://127.0.0.1:$PORT" >/dev/null 2>&1 || xdg-open "http://127.0.0.1:$PORT" >/dev/null 2>&1 || true ) &
exec .venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port "$PORT"
