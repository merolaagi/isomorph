#!/usr/bin/env bash
# Keep Isomorph running in the background on macOS (starts at login, restarts if it stops).
#   ./scripts/service.sh install | restart | stop | uninstall | status | logs
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
LABEL="com.merolaagi.isomorph"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
PORT="${ISOMORPH_PORT:-47821}"
DOMAIN="gui/$(id -u)"
health() { for _ in $(seq 1 60); do curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/api/health" 2>/dev/null | grep -qE "200|401" && return 0; sleep 1; done; return 1; }

case "${1:-status}" in
  install)
    [ -x .venv/bin/python ] || ./setup.sh
    mkdir -p data "$HOME/Library/LaunchAgents"
    OLD="$(pgrep -f "uvicorn server.app:app .*--port $PORT" || true)"
    [ -n "$OLD" ] && kill $OLD 2>/dev/null || true
    cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>ProgramArguments</key><array>
    <string>$ROOT/.venv/bin/python</string><string>-m</string><string>uvicorn</string><string>server.app:app</string>
    <string>--host</string><string>127.0.0.1</string><string>--port</string><string>$PORT</string>
  </array>
  <key>EnvironmentVariables</key><dict>
    <key>PYTORCH_ENABLE_MPS_FALLBACK</key><string>1</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$ROOT/data/service.log</string>
  <key>StandardErrorPath</key><string>$ROOT/data/service.log</string>
</dict></plist>
PL
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    launchctl bootstrap "$DOMAIN" "$PLIST"
    health && echo "Isomorph service running on http://127.0.0.1:$PORT" || { echo "The service did not come up. See data/service.log"; exit 1; } ;;
  restart)
    [ -f "$PLIST" ] || { echo "Service not installed. Run: ./scripts/service.sh install"; exit 1; }
    launchctl kickstart -k "$DOMAIN/$LABEL"
    health && echo "Isomorph service restarted on http://127.0.0.1:$PORT" || { echo "The service did not come up. See data/service.log"; exit 1; } ;;
  stop) launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null && echo "Stopped (starts again at next login or with: ./scripts/service.sh install)." || echo "Not running." ;;
  uninstall) launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true; rm -f "$PLIST"; echo "Service removed." ;;
  status)
    if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
      launchctl print "$DOMAIN/$LABEL" | grep -E "^\s*(state|pid) =" || true
      curl -s -o /dev/null -w "Health check: HTTP %{http_code}\n" "http://127.0.0.1:$PORT/api/health" || true
    else echo "Not installed or not running."; fi ;;
  logs) tail -n 100 -f data/service.log ;;
  *) echo "Usage: $0 install|restart|stop|uninstall|status|logs"; exit 1 ;;
esac
