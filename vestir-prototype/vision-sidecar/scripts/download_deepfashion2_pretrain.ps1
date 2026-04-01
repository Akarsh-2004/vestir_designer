# Mirrors tomsebastiank/.../weights/download_weights.sh for Windows (PowerShell).
# Upstream: https://github.com/tomsebastiank/Clothing-detection-and-attribute-identification-using-YoloV3-and-DeepFashion/tree/master/weights
#
# What GitHub actually contains: ONLY download_weights.sh — no yolov3-df2_last.weights in the repo.
#
# Downloads:
#   - darknet53.conv.74  → use as --pretrained_weights for train.py (per their README)
#   - yolov3.weights     → COCO 80-class; NOT valid for bundled yolov3-df2-13class.cfg (13 classes)
#   - yolov3-tiny.weights→ tiny COCO; same mismatch for DF2 cfg
#
# For Vestir OpenCV inference you still need a DF2-trained .weights (e.g. yolov3-df2_last.weights) from training.

$ErrorActionPreference = "Stop"
$sidecarRoot = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $sidecarRoot "vendor\deepfashion-weights"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

function Get-FileIfMissing($url, $name) {
    $path = Join-Path $dest $name
    if (Test-Path $path) {
        Write-Host "Skip (exists): $path"
        return $path
    }
    Write-Host "Downloading $name ..."
    Invoke-WebRequest -Uri $url -OutFile $path -UseBasicParsing
    return $path
}

# Same URLs as weights/download_weights.sh in upstream repo
Get-FileIfMissing "https://pjreddie.com/media/files/darknet53.conv.74" "darknet53.conv.74"
Get-FileIfMissing "https://pjreddie.com/media/files/yolov3.weights" "yolov3.weights"
Get-FileIfMissing "https://pjreddie.com/media/files/yolov3-tiny.weights" "yolov3-tiny.weights"

Write-Host ""
Write-Host "Saved under: $dest"
Write-Host "Use darknet53.conv.74 when running their train.py with DeepFashion2 labels."
Write-Host "Do NOT set VESTIR_DARKNET_WEIGHTS=yolov3.weights with our 13-class cfg — class count mismatch."
Write-Host "After training, point VESTIR_DARKNET_WEIGHTS at your checkpoints / exported DF2 .weights file."
