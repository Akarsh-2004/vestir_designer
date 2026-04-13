#!/usr/bin/env bash
# Download fal/virtual-tryoff-lora into models/virtual-tryoff-lora.
# Uses HF_TOKEN from repo root .env and maps it to HUGGING_FACE_HUB_TOKEN for huggingface_hub.
set -euo pipefail

cd "$(dirname "$0")/.."

for envfile in "../../.env" "../.env"; do
  if [ -f "$envfile" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$envfile"
    set +a
  fi
done

if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

if [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
  echo "Error: set HF_TOKEN in ../../.env (repo root) or HUGGING_FACE_HUB_TOKEN" >&2
  exit 1
fi

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q huggingface_hub

python - <<'PY'
import os
from huggingface_hub import snapshot_download

os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ.get("HF_TOKEN", "").strip())
path = snapshot_download(
    repo_id="fal/virtual-tryoff-lora",
    local_dir="models/virtual-tryoff-lora",
    allow_patterns=["*diffusers.safetensors*", "*.json", "*.md", "*.txt"],
)
print("Downloaded to:", path)
PY
