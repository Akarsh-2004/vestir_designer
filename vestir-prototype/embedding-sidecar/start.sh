#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export EMBEDDING_BIND="${EMBEDDING_BIND:-0.0.0.0}"
export EMBEDDING_PORT="${EMBEDDING_PORT:-8010}"
exec python -m uvicorn app:app --host "$EMBEDDING_BIND" --port "$EMBEDDING_PORT"
