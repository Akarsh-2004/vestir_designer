#!/usr/bin/env bash
# Back-compat entrypoint — delegates to the prototype root start.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/start.sh"
