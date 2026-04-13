#!/usr/bin/env bash
# Start full Vestir prototype: vision-sidecar, tryon-sidecar (optional), Vite + API.
# Run from anywhere:  ./start.sh   or   bash start.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/scripts/start_vestir_all.sh"
