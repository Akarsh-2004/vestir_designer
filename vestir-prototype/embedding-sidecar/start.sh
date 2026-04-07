#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export EMBEDDING_BIND="${EMBEDDING_BIND:-0.0.0.0}"
export EMBEDDING_PORT="${EMBEDDING_PORT:-8010}"
PY_BIN="${PY_BIN:-}"
if [[ -z "$PY_BIN" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PY_BIN=".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PY_BIN="$(command -v python)"
  else
    echo "No Python interpreter found. Install python3 or create .venv first." >&2
    exit 1
  fi
fi
exec "$PY_BIN" -m uvicorn app:app --host "$EMBEDDING_BIND" --port "$EMBEDDING_PORT"
