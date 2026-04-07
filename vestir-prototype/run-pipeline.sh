#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISION_DIR="$ROOT_DIR/vision-sidecar"
EMBED_DIR="$ROOT_DIR/embedding-sidecar"
TRYON_DIR="$ROOT_DIR/tryon-sidecar"

# Load env from repo root (GEMINI_API_KEY, HF_TOKEN, sidecar URLs, etc.)
if [ -f "$ROOT_DIR/../.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/../.env"
  set +a
fi
if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

export VISION_SIDECAR_URL="${VISION_SIDECAR_URL:-http://127.0.0.1:8008}"
export EMBEDDING_SIDECAR_URL="${EMBEDDING_SIDECAR_URL:-http://127.0.0.1:8010}"
export TRYON_SIDECAR_URL="${TRYON_SIDECAR_URL:-http://127.0.0.1:8009}"

pids=()

cleanup() {
  echo ""
  echo "Stopping pipeline..."
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

start_service() {
  local name="$1"
  local cmd="$2"
  echo "Starting $name..."
  bash -lc "$cmd" &
  local pid=$!
  pids+=("$pid")
  echo "  -> $name pid: $pid"
}

wait_for_health() {
  local name="$1"
  local url="$2"
  local tries="${3:-30}"
  local i=0
  until curl -sf "$url" >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -ge "$tries" ]; then
      echo "ERROR: $name did not become healthy: $url"
      exit 1
    fi
    sleep 1
  done
  echo "  -> $name healthy at $url"
}

# Ensure node deps are present.
if [ ! -d "$ROOT_DIR/node_modules" ]; then
  echo "Installing Node dependencies..."
  (cd "$ROOT_DIR" && npm install)
fi

# Embedding sidecar.
if [ ! -d "$EMBED_DIR/.venv" ]; then
  python3 -m venv "$EMBED_DIR/.venv"
fi
start_service "embedding-sidecar" \
  "cd \"$EMBED_DIR\" && source .venv/bin/activate && pip install -q -r requirements.txt && ./start.sh"
wait_for_health "embedding-sidecar" "http://127.0.0.1:8010/health" 90

# Vision sidecar.
if [ ! -d "$VISION_DIR/.venv" ]; then
  python3 -m venv "$VISION_DIR/.venv"
fi
start_service "vision-sidecar" "cd \"$VISION_DIR\" && ./start.sh"
wait_for_health "vision-sidecar" "http://127.0.0.1:8008/health" 90

# Try-on sidecar (best-effort; does not block pipeline if model isn't wired).
if [ ! -d "$TRYON_DIR/.venv" ]; then
  python3 -m venv "$TRYON_DIR/.venv"
fi
start_service "tryon-sidecar" \
  "cd \"$TRYON_DIR\" && source .venv/bin/activate && pip install -q -r requirements.txt && uvicorn app:app --host 127.0.0.1 --port 8009 --reload"
wait_for_health "tryon-sidecar" "http://127.0.0.1:8009/health" 30 || true

# Main app (web + API).
echo "Starting web + API..."
cd "$ROOT_DIR"
npm run dev &
main_pid=$!
pids+=("$main_pid")

wait_for_health "api" "http://127.0.0.1:8787/api/health" 60

echo ""
echo "Pipeline ready:"
echo "  Web:       http://localhost:5173"
echo "  API:       http://127.0.0.1:8787/api/health"
echo "  Vision:    http://127.0.0.1:8008/health"
echo "  Embedding: http://127.0.0.1:8010/health"
echo "  Try-on:    http://127.0.0.1:8009/health"
echo ""
echo "Press Ctrl+C to stop everything."

wait "$main_pid"
