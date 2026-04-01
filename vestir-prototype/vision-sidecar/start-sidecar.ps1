# Vision sidecar (FastAPI). Run from THIS folder — NOT from vestir\ alone.
# Full path on your machine is usually:
#   ...\vestir\vestir-prototype\vision-sidecar
#
# Usage:
#   .\start-sidecar.ps1
# Optional (fine-tuned fashion weights):
#   $env:VESTIR_YOLOV8_PT = "C:\path\to\runs\detect\train\weights\best.pt"
#   .\start-sidecar.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "Creating .venv ..."
    python -m venv .venv
}

& .\.venv\Scripts\Activate.ps1
pip install -q -r requirements.txt

if ($env:VESTIR_YOLOV8_PT -and (Test-Path $env:VESTIR_YOLOV8_PT)) {
    Write-Host "VESTIR_YOLOV8_PT is set; installing YOLOv8 deps..."
    pip install -q -r requirements-yolov8.txt
} else {
    Write-Host "Tip: set VESTIR_YOLOV8_PT to your trained best.pt for fashion detection."
}

Write-Host "Starting http://127.0.0.1:8008 ..."
uvicorn app:app --host 127.0.0.1 --port 8008
