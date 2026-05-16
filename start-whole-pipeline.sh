#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_DIR="$ROOT_DIR/vestir-prototype"
SANDBOX_DIR="$ROOT_DIR/vestir_aistudio"

echo "[vestir] Starting main pipeline API (port 8787)..."
(
  cd "$MAIN_DIR"
  npm run dev:api
) &
MAIN_PID=$!

echo "[vestir] Starting sandbox API/UI (port 3000)..."
(
  cd "$SANDBOX_DIR"
  npm run dev
) &
SANDBOX_PID=$!

cleanup() {
  echo
  echo "[vestir] Stopping services..."
  kill "$MAIN_PID" >/dev/null 2>&1 || true
  kill "$SANDBOX_PID" >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

echo "[vestir] Main pipeline: http://127.0.0.1:8787"
echo "[vestir] Sandbox:       http://127.0.0.1:3000"
echo "[vestir] Press Ctrl+C to stop both."

wait
