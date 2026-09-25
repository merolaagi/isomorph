#!/usr/bin/env bash
# Publish Isomorph through your existing Cloudflare tunnel.
#   ./scripts/tunnel.sh [hostname]      default: isomorph.fueldeskpro.com
# Adds an ingress rule to the cloudflared config (with a backup), validates it,
# restarts cloudflared, and prints the CNAME target for the DNS record.
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="${1:-isomorph.fueldeskpro.com}"
PORT="${ISOMORPH_PORT:-47821}"
CFG="${CLOUDFLARED_CONFIG:-/etc/cloudflared/config.yml}"
[ -f "$CFG" ] || CFG="$HOME/.cloudflared/config.yml"
[ -f "$CFG" ] || { echo "No cloudflared config found at /etc/cloudflared/config.yml or ~/.cloudflared/config.yml. Set CLOUDFLARED_CONFIG=/path/to/config.yml"; exit 1; }
PY="$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )"

# Tunnel id: from 'tunnel:', else from the credentials file name, else from 'cloudflared tunnel list'.
UUID_RE='[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
TVAL="$(awk -F': *' '/^tunnel:/{print $2}' "$CFG" | tr -d '"'"'"' ')"
TID="$(echo "$TVAL" | grep -oE "$UUID_RE" || true)"
[ -n "$TID" ] || TID="$(awk -F': *' '/^credentials-file:/{print $2}' "$CFG" | grep -oE "$UUID_RE" || true)"
if [ -z "$TID" ] && command -v cloudflared >/dev/null 2>&1; then
  TID="$(cloudflared tunnel list 2>/dev/null | awk -v n="$TVAL" '$2==n{print $1}' | grep -oE "$UUID_RE" || true)"
fi

if grep -qE "hostname: *$HOST *$" "$CFG"; then
  echo "Ingress for $HOST already present in $CFG."
else
  TMP="$(mktemp)"
  HOST="$HOST" PORT="$PORT" "$PY" - "$CFG" "$TMP" <<'PY'
import os, re, sys
src, dst = sys.argv[1], sys.argv[2]
lines = open(src).read().splitlines(keepends=True)
host, port = os.environ["HOST"], os.environ["PORT"]
# Match the indentation of existing '- hostname:' entries.
ind = next((re.match(r"^(\s*)- hostname:", l).group(1) for l in lines if re.match(r"^\s*- hostname:", l)), "  ")
rule = [f"{ind}- hostname: {host}\n", f"{ind}  service: http://127.0.0.1:{port}\n"]
# Insert before the catch-all rule (a '- service:' entry with no hostname), else append to ingress.
idx = next((i for i, l in enumerate(lines) if re.match(r"^\s*- service:\s*http_status:", l)), None)
if idx is None:
    if not any(l.startswith("ingress:") for l in lines):
        lines.append("ingress:\n")
    lines += rule + [f"{ind}- service: http_status:404\n"]
else:
    lines[idx:idx] = rule
open(dst, "w").write("".join(lines))
PY
  BAK="$CFG.bak.$(date +%Y%m%d%H%M%S)"
  if [ -w "$CFG" ]; then cp "$CFG" "$BAK"; cp "$TMP" "$CFG"; else sudo cp "$CFG" "$BAK"; sudo cp "$TMP" "$CFG"; fi
  rm -f "$TMP"
  echo "Added $HOST -> http://127.0.0.1:$PORT to $CFG (backup: $BAK)"
fi

if command -v cloudflared >/dev/null 2>&1; then
  cloudflared tunnel --config "$CFG" ingress validate
  cloudflared tunnel --config "$CFG" ingress rule "https://$HOST" | tail -n 2
  if sudo launchctl list 2>/dev/null | grep -q com.cloudflare.cloudflared; then
    sudo launchctl kickstart -k system/com.cloudflare.cloudflared && echo "Restarted cloudflared."
  elif launchctl list 2>/dev/null | grep -q cloudflared; then
    launchctl kickstart -k "gui/$(id -u)/$(launchctl list | awk '/cloudflared/{print $3; exit}')" && echo "Restarted cloudflared."
  else
    echo "Restart your running cloudflared process so it picks up the new rule."
  fi
fi

echo
echo "DNS record to add in Cloudflare (fueldeskpro.com):"
echo "  Type: CNAME   Name: ${HOST%%.*}   Target: ${TID:-<tunnel-id>}.cfargotunnel.com   Proxy: on (orange cloud)"
[ -n "$TID" ] || echo "  Could not read the tunnel id. Find it with: cloudflared tunnel list"

PW_SET="$("$PY" -c "import json,os;p='data/settings.json';print('yes' if os.path.exists(p) and json.load(open(p)).get('password') else '')" 2>/dev/null || true)"
[ -n "$PW_SET" ] || echo "WARNING: no password is set, so anyone with the link can use the app. Run: ./scripts/password.sh"
