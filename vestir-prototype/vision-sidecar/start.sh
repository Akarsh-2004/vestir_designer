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

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install -q -r requirements.txt
pip install -q -r requirements-yolov8.txt

export VESTIR_YOLOV8_CONF="${VESTIR_YOLOV8_CONF:-0.3}"
export VESTIR_YOLOV8_IOU="${VESTIR_YOLOV8_IOU:-0.6}"
export VESTIR_YOLOV8_MAX_DET="${VESTIR_YOLOV8_MAX_DET:-30}"
export VESTIR_YOLOV8_IMGSZ="${VESTIR_YOLOV8_IMGSZ:-960}"

project_model_path="$(cd ../.. && pwd)/yolo11_fashion_best.pt"
default_model_path="$(pwd)/models/deepfashion2_yolov8s-seg.pt"
if [ -z "${VESTIR_YOLOV8_PT:-}" ]; then
  if [ -f "$project_model_path" ]; then
    export VESTIR_YOLOV8_PT="$project_model_path"
  else
    export VESTIR_YOLOV8_PT="$default_model_path"
  fi
fi

if [ "$VESTIR_YOLOV8_PT" = "$default_model_path" ] && [ ! -f "$VESTIR_YOLOV8_PT" ]; then
  mkdir -p "$(dirname "$VESTIR_YOLOV8_PT")"
  pip install -q huggingface_hub
  python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="Bingsu/adetailer",
    filename="deepfashion2_yolov8s-seg.pt",
    local_dir="models",
)
print("Downloaded deepfashion2_yolov8s-seg.pt")
PY
fi

echo "Using VESTIR_YOLOV8_PT=$VESTIR_YOLOV8_PT"
exec uvicorn app:app --host 127.0.0.1 --port 8008 --reload
