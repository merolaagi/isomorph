#!/usr/bin/env bash
# Build a release archive with a unique name: isomorph-v<version>-<date>-<content hash>.tar.gz
set -euo pipefail
cd "$(dirname "$0")/.."
VER="$(cat VERSION)"
OUT="${1:-dist}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  FILES="$(git ls-files --cached --others --exclude-standard)"
else
  FILES="$(find . -type f -not -path './.venv/*' -not -path './data/*' -not -path './dist/*' -not -path './.git/*' \
    -not -name '*.pyc' -not -path '*/__pycache__/*' -not -name '.DS_Store' | sed 's|^\./||' | sort)"
fi
HASH="$(echo "$FILES" | while read -r f; do cat "$f"; done | shasum | cut -c1-7)"
NAME="isomorph-v$VER-$(date +%Y%m%d)-$HASH.tar.gz"
STAGE="$(mktemp -d)"
mkdir -p "$STAGE/isomorph"
echo "$FILES" | while read -r f; do mkdir -p "$STAGE/isomorph/$(dirname "$f")"; cp -p "$f" "$STAGE/isomorph/$f"; done
tar -czf "$OUT/$NAME" -C "$STAGE" isomorph
rm -rf "$STAGE"
echo "$OUT/$NAME"
