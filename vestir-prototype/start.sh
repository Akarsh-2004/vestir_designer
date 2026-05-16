#!/usr/bin/env bash
# One command to run the Vestir prototype: vision-sidecar (SAM/YOLO), optional try-on sidecar,
# then Vite + Node API (npm run dev).
#
# Usage (from anywhere):
#   ./vestir-prototype/start.sh
#   bash vestir-prototype/start.sh
#
# Env (optional):
#   SKIP_TRYON_SIDECAR=1   — do not start try-on sidecar (faster / fewer GPU deps)
#   SKIP_VISION_SIDECAR=1  — only web+api (SAM/YOLO features will be unavailable)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
VISION_DIR="$ROOT_DIR/vision-sidecar"
TRYON_DIR="$ROOT_DIR/tryon-sidecar"

echo "[vestir] prototype root: $ROOT_DIR"
echo "[vestir] monorepo root:  $REPO_ROOT"

# --- Load env: repo root first (HF_TOKEN, GEMINI, etc.), then prototype ---
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

# Node API defaults to this host; exporting keeps child processes explicit.
export VISION_SIDECAR_URL="${VISION_SIDECAR_URL:-http://127.0.0.1:8008}"
export TRYON_SIDECAR_URL="${TRYON_SIDECAR_URL:-http://127.0.0.1:8009}"

# --- First-time setup: .env + node_modules ---
if [ ! -f "$ROOT_DIR/.env" ] && [ -f "$ROOT_DIR/.env.example" ]; then
  cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
  echo "[vestir] created $ROOT_DIR/.env from .env.example — add GEMINI_API_KEY etc. if needed."
fi

if [ ! -d "$ROOT_DIR/node_modules" ]; then
  echo "[vestir] installing npm dependencies..."
  (cd "$ROOT_DIR" && npm install)
fi

cleanup() {
  echo
  echo "[vestir] stopping..."
  if [ -n "${VISION_PID:-}" ]; then
    kill "$VISION_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "${TRYON_PID:-}" ]; then
    kill "$TRYON_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "${DEV_PID:-}" ]; then
    # npm run dev → concurrently → vite + node; reap children first.
    pkill -P "$DEV_PID" >/dev/null 2>&1 || true
    kill "$DEV_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

VISION_PID=""
if [ "${SKIP_VISION_SIDECAR:-0}" = "1" ]; then
  echo "[vestir] SKIP_VISION_SIDECAR=1 — not starting vision-sidecar (SAM/YOLO offline)."
else
  echo "[vestir] starting vision-sidecar (logs: /tmp/vestir-vision.log)..."
  "$VISION_DIR/start.sh" >/tmp/vestir-vision.log 2>&1 &
  VISION_PID=$!

  for i in $(seq 1 90); do
    if curl -fsS "http://127.0.0.1:8008/health" >/dev/null 2>&1; then
      echo "[vestir] vision-sidecar is up — $VISION_SIDECAR_URL/health"
      break
    fi
    sleep 1
    if [ "$i" -eq 90 ]; then
      echo "[vestir] ERROR: vision-sidecar did not become healthy in time."
      tail -n 80 /tmp/vestir-vision.log || true
      exit 1
    fi
  done
fi

TRYON_PID=""
if [ "${SKIP_TRYON_SIDECAR:-0}" = "1" ]; then
  echo "[vestir] SKIP_TRYON_SIDECAR=1 — not starting try-on sidecar."
else
  if command -v lsof >/dev/null 2>&1; then
    for pid in $(lsof -ti :8009 2>/dev/null || true); do
      echo "[vestir] freeing port 8009 (pid $pid)"
      kill "$pid" >/dev/null 2>&1 || true
    done
    sleep 0.5
  fi
  echo "[vestir] starting try-on sidecar (logs: /tmp/vestir-tryon.log)..."
  export TRYON_UVICORN_RELOAD="${TRYON_UVICORN_RELOAD:-0}"
  "$TRYON_DIR/start.sh" >/tmp/vestir-tryon.log 2>&1 &
  TRYON_PID=$!

  tryon_ok=0
  for i in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:8009/health" >/dev/null 2>&1; then
      echo "[vestir] try-on sidecar is up — $TRYON_SIDECAR_URL/health"
      tryon_ok=1
      break
    fi
    sleep 1
  done
  if [ "$tryon_ok" != "1" ]; then
    echo "[vestir] warning: try-on sidecar did not become healthy (virtual try-on may be offline)."
    tail -n 40 /tmp/vestir-tryon.log || true
    kill "$TRYON_PID" >/dev/null 2>&1 || true
    TRYON_PID=""
  fi
fi

echo "[vestir] starting web + API (npm run dev)..."
(
  cd "$ROOT_DIR"
  npm run dev
) &
DEV_PID=$!

echo
echo "[vestir] —— URLs ——"
echo "[vestir]  App (Vite):     http://127.0.0.1:5173"
echo "[vestir]  API:            http://127.0.0.1:8787"
echo "[vestir]  Vision (SAM):  $VISION_SIDECAR_URL"
echo "[vestir]  Try-on:        $TRYON_SIDECAR_URL"
echo "[vestir]  Mannequin v2:  http://127.0.0.1:8787/api/items/mannequin-v2"
echo "[vestir] ——————————————"
echo "[vestir] Logs: tail -f /tmp/vestir-vision.log  (and /tmp/vestir-tryon.log if started)"
echo "[vestir] Press Ctrl+C to stop all services."
echo

wait "$DEV_PID"
