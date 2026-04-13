#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Repo root .env (HF_TOKEN, etc.) then vestir-prototype/.env
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

# FLUX.2 + MPS: PyTorch's default MPS allocator cap (~20 GiB) often aborts large models.
# Use the full unified-memory pool on Apple Silicon unless the user already set a ratio.
if [ "$(uname -s)" = "Darwin" ]; then
  : "${PYTORCH_MPS_HIGH_WATERMARK_RATIO:=0.0}"
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO
fi

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install -q -r requirements.txt

echo "Using TRYON_BASE_MODEL_ID=${TRYON_BASE_MODEL_ID:-unset}"
echo "Using TRYON_LORA_PATH=${TRYON_LORA_PATH:-unset}"

PORT="${TRYON_SIDECAR_PORT:-8009}"
# Default: no --reload (avoids parent/child both binding :8009 and "address already in use").
export TRYON_UVICORN_RELOAD="${TRYON_UVICORN_RELOAD:-0}"
if [ "${TRYON_UVICORN_RELOAD}" = "0" ]; then
  exec uvicorn app:app --host 127.0.0.1 --port "$PORT"
fi
exec uvicorn app:app --host 127.0.0.1 --port "$PORT" --reload

