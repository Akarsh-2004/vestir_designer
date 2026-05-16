#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANDBOX_DIR="$ROOT_DIR/vestir_aistudio"

echo "[vestir] Starting sandbox API/UI only (port 3000)..."
cd "$SANDBOX_DIR"
npm run dev
