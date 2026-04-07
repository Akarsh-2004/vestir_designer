#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_df2_v1_lowmem.sh prepare
#   ./scripts/run_df2_v1_lowmem.sh setup-mps
#   ./scripts/run_df2_v1_lowmem.sh train
#   ./scripts/run_df2_v1_lowmem.sh resume

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DF2_ROOT_DEFAULT="/Users/akarshsaklani/Desktop/vestir/DeepFashion2/deepfashion2_original_images"
DATASET_ROOT_DEFAULT="$ROOT_DIR/datasets/df2_v1"

CMD="${1:-train}"
DF2_ROOT="${DF2_ROOT:-$DF2_ROOT_DEFAULT}"
DATASET_ROOT="${DATASET_ROOT:-$DATASET_ROOT_DEFAULT}"

if [[ -n "${VENV_DIR:-}" ]]; then
  PY="$VENV_DIR/bin/python"
elif [[ -x "$ROOT_DIR/.venv311/bin/python" ]]; then
  PY="$ROOT_DIR/.venv311/bin/python"
else
  PY="$ROOT_DIR/.venv/bin/python"
fi

if [[ "$CMD" == "setup-mps" ]]; then
  cd "$ROOT_DIR"
  python3.11 -m venv .venv311
  . .venv311/bin/activate
  pip install -U pip wheel
  pip install -r requirements.txt -r requirements-yolov8.txt
  python - <<'PY'
import torch
print("torch:", torch.__version__)
print("mps_available:", torch.backends.mps.is_available())
print("mps_built:", torch.backends.mps.is_built())
PY
  exit 0
fi

if [[ ! -x "$PY" ]]; then
  echo "Missing venv python at $PY"
  echo "Create it first:"
  echo "  ./scripts/run_df2_v1_lowmem.sh setup-mps"
  echo "  # or classic:"
  echo "  cd \"$ROOT_DIR\" && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt -r requirements-yolov8.txt"
  exit 1
fi

if [[ "$CMD" == "prepare" ]]; then
  "$PY" "$ROOT_DIR/scripts/prepare_df2_yolo_v1.py" \
    --df2-root "$DF2_ROOT" \
    --out-root "$DATASET_ROOT" \
    --max-train "${MAX_TRAIN:-12000}" \
    --max-val "${MAX_VAL:-2000}"
  exit 0
fi

DATA_YAML="$DATASET_ROOT/data.yaml"
if [[ ! -f "$DATA_YAML" ]]; then
  echo "Missing $DATA_YAML. Run prepare first:"
  echo "  ./scripts/run_df2_v1_lowmem.sh prepare"
  exit 1
fi

RESUME_FLAG=""
if [[ "$CMD" == "resume" ]]; then
  RESUME_FLAG="--resume"
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.mpl-cache}"

"$PY" "$ROOT_DIR/scripts/train_df2_v1.py" \
  --data "$DATA_YAML" \
  --project "${PROJECT_DIR:-$ROOT_DIR/runs/detect}" \
  --name "${RUN_NAME:-df2-v1-lowmem}" \
  --model "${BASE_MODEL:-yolov8n.pt}" \
  --epochs "${EPOCHS:-60}" \
  --imgsz "${IMGSZ:-640}" \
  --batch "${BATCH:-8}" \
  --workers "${WORKERS:-2}" \
  --device "${DEVICE:-mps}" \
  --patience "${PATIENCE:-20}" \
  --save-period "${SAVE_PERIOD:-1}" \
  ${RESUME_FLAG}

