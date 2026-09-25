#!/usr/bin/env bash
# Set, change or remove the password that protects Isomorph.
#   ./scripts/password.sh          prompt for a new password
#   ./scripts/password.sh --off    remove it (only safe when the app is not public)
# Sign in with any username and this password. Takes effect immediately.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
if [ "${1:-}" = "--off" ]; then PW=""; else
  if [ -n "${1:-}" ]; then PW="$1"; else
    read -r -s -p "New Isomorph password: " PW; echo
    read -r -s -p "Repeat it: " PW2; echo
    [ "$PW" = "$PW2" ] || { echo "The two entries differ. Nothing changed."; exit 1; }
  fi
  [ ${#PW} -ge 10 ] || { echo "Use at least 10 characters. Nothing changed."; exit 1; }
fi
PY="$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )"
PW="$PW" "$PY" - <<'PY'
import json, os, pathlib
p = pathlib.Path("data/settings.json")
s = json.loads(p.read_text()) if p.exists() else {}
pw = os.environ["PW"]
if pw: s["password"] = pw
else: s.pop("password", None)
p.write_text(json.dumps(s)); os.chmod(p, 0o600)
print("Password set. Sign in with any username." if pw else "Password removed.")
PY
