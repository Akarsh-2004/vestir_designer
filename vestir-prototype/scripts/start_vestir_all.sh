#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
VISION_DIR="$ROOT_DIR/vision-sidecar"
TRYON_DIR="$ROOT_DIR/tryon-sidecar"

echo "[vestir] prototype: $ROOT_DIR"
echo "[vestir] repo root: $REPO_ROOT"

# Repo root .env (HF_TOKEN, GEMINI, etc.) then vestir-prototype/.env
for envfile in "$REPO_ROOT/.env" "$ROOT_DIR/.env"; do
  if [ -f "$envfile" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$envfile"
    set +a
    echo "[vestir] loaded $envfile"
  fi
done

if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

cleanup() {
  echo
  echo "[vestir] stopping processes..."
  if [ -n "${VISION_PID:-}" ]; then
    kill "$VISION_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "${DEV_PID:-}" ]; then
    kill "$DEV_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "${TRYON_PID:-}" ]; then
    kill "$TRYON_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "[vestir] starting vision-sidecar..."
"$VISION_DIR/start.sh" >/tmp/vestir-vision.log 2>&1 &
VISION_PID=$!

for i in {1..60}; do
  if curl -fsS "http://127.0.0.1:8008/health" >/dev/null 2>&1; then
    echo "[vestir] vision-sidecar is up (http://127.0.0.1:8008/health)"
    break
  fi
  sleep 1
  if [ "$i" -eq 60 ]; then
    echo "[vestir] vision-sidecar failed to start. Tail log:"
    tail -n 80 /tmp/vestir-vision.log || true
    exit 1
  fi
done

TRYON_PID=""
if [ "${SKIP_TRYON_SIDECAR:-0}" = "1" ]; then
  echo "[vestir] skipping tryon-sidecar (SKIP_TRYON_SIDECAR=1)"
else
  if command -v lsof >/dev/null 2>&1; then
    for pid in $(lsof -ti :8009 2>/dev/null || true); do
      echo "[vestir] freeing port 8009 (stopping pid $pid)"
      kill "$pid" >/dev/null 2>&1 || true
    done
    sleep 0.5
  fi
  echo "[vestir] starting tryon-sidecar..."
  export TRYON_UVICORN_RELOAD="${TRYON_UVICORN_RELOAD:-0}"
  "$TRYON_DIR/start.sh" >/tmp/vestir-tryon.log 2>&1 &
  TRYON_PID=$!

  tryon_ok=0
  for i in {1..60}; do
    if curl -fsS "http://127.0.0.1:8009/health" >/dev/null 2>&1; then
      echo "[vestir] tryon-sidecar is up (http://127.0.0.1:8009/health)"
      tryon_ok=1
      break
    fi
    sleep 1
  done
  if [ "$tryon_ok" != "1" ]; then
    echo "[vestir] warning: tryon-sidecar did not become healthy in time (frontend will still start)."
    echo "[vestir] tail /tmp/vestir-tryon.log — last 40 lines:"
    tail -n 40 /tmp/vestir-tryon.log || true
    kill "$TRYON_PID" >/dev/null 2>&1 || true
    TRYON_PID=""
  fi
fi

echo "[vestir] starting web+api (npm run dev)..."
(
  cd "$ROOT_DIR"
  npm run dev
) &
DEV_PID=$!

echo "[vestir] all services started."
echo "[vestir] web: http://127.0.0.1:5173"
echo "[vestir] api: http://127.0.0.1:8787"
echo "[vestir] vision health: http://127.0.0.1:8008/health"
echo "[vestir] tryon health: http://127.0.0.1:8009/health"
echo
echo "[vestir] press Ctrl+C to stop everything."

wait "$DEV_PID"

