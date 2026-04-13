# Vision Sidecar (YOLO / parsing contract)

The Node API calls `POST /analyze` and falls back safely when YOLO weights are missing.

**Path on disk:** this folder is `vestir-prototype\vision-sidecar` (under your Vestir checkout), **not** `vestir\vision-sidecar` at the repo root.

## Run (Windows)

From the repo root `vestir\`:

```powershell
cd vestir-prototype\vision-sidecar
.\start-sidecar.ps1
```

Or manually:

```powershell
cd vestir-prototype\vision-sidecar
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8008 --reload
```

Set on the **Node** process: `VISION_SIDECAR_URL=http://127.0.0.1:8008`

## YOLOv8 (Ultralytics) — recommended for quick fine-tunes (Kaggle / Mac)

Workflows like [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) + a small fashion dataset (e.g. **Colorful Fashion** on Kaggle with ~2k train images) train in **minutes to a few hours** on a small GPU, and run well on **Apple Silicon** via PyTorch MPS after `pip install ultralytics`.

1. Train or export **`best.pt`** / **`last.pt`** (`model.train(...)` as in the notebook you found: `YOLO("yolov8m.pt")` + `data.yaml` with `nc` + `names`).
2. In the vision sidecar venv:

   ```powershell
   pip install -r requirements-yolov8.txt
   $env:VESTIR_YOLOV8_PT = "C:\path\to\runs\detect\train\weights\best.pt"
   $env:VESTIR_YOLOV8_CONF = "0.4"
   uvicorn app:app --host 127.0.0.1 --port 8008
   ```

3. `GET /health` shows `"yolov8_pt": true` when the file path exists.

`/analyze` tries **YOLOv8 first**, then **Darknet**, then the heuristic box. Response `model` is `yolov8-ultralytics` when the `.pt` path ran.

**Note:** Loading `yolov8m.pt` / COCO weights without fine-tuning will still predict **COCO classes** (person, car, …), not fashion — point `VESTIR_YOLOV8_PT` at your **fine-tuned** checkpoint.

## DeepFashion2 YOLOv3 (default cfg + names in this repo)

This tracks
[tomsebastiank/Clothing-detection-and-attribute-identification-using-YoloV3-and-DeepFashion](https://github.com/tomsebastiank/Clothing-detection-and-attribute-identification-using-YoloV3-and-DeepFashion).

### What is actually under their `weights/` folder on GitHub?

Only **[weights/download_weights.sh](https://github.com/tomsebastiank/Clothing-detection-and-attribute-identification-using-YoloV3-and-DeepFashion/blob/master/weights/download_weights.sh)** — it `wget`s:

| File | Purpose |
|------|---------|
| `darknet53.conv.74` | Backbone init for **training** (matches their `train.py --pretrained_weights`) |
| `yolov3.weights` | **COCO 80 classes** — unknown objects, **wrong** for our 13-class DF2 cfg |
| `yolov3-tiny.weights` | COCO tiny — same mismatch for DF2 inference |

There is **no** `yolov3-df2_last.weights` (or similar) committed; that file is produced **after** you train on [DeepFashion2](https://github.com/switchablenorms/DeepFashion2), as in their README “Testing” section (`detect.py --weights_path weights/yolov3-df2_last.weights`).

**Windows:** run our mirror script (downloads the same PJReddie files into `vision-sidecar/vendor/deepfashion-weights/`):

```powershell
cd vision-sidecar\scripts
.\download_deepfashion2_pretrain.ps1
```

- **Bundled in Vestir:** `models/yolov3-df2-13class.cfg` and `data/df2.names` (13 classes).
- **For OpenCV inference (`VESTIR_DARKNET_WEIGHTS`):** you need a **13-class** `.weights` trained with that cfg (train in their repo, then point env at the checkpoint you export / save in darknet format).

**Minimum to enable YOLO in this sidecar:** only set weights (cfg and names pick up bundled files automatically):

| Variable | Required | Description |
|----------|----------|-------------|
| `VESTIR_DARKNET_WEIGHTS` | **yes** (for real YOLO) | Path to `.weights` matching `yolov3-df2-13class.cfg` |
| `VESTIR_DARKNET_CFG` | no | Override cfg path (default: bundled `models/yolov3-df2-13class.cfg`) |
| `VESTIR_DARKNET_NAMES` | no | Override names (default: bundled `data/df2.names`) |
| `VESTIR_DARKNET_CONF` | no | Confidence (default `0.45`; their `detect.py` uses `0.8` — raise if you get too many boxes) |
| `VESTIR_DARKNET_NMS` | no | NMS (default `0.45`) |
| `VESTIR_DARKNET_INPUT` | no | Input size (default `416`) |
| `VESTIR_DARKNET_CUDA` | no | `1` for OpenCV CUDA DNN |

Legacy env names `MODANET_YOLO_*` still work as aliases.

`GET /health` includes `"yolo_darknet_weights": true` when cfg + weights resolve to real files.

Without weights, `/analyze` uses a **heuristic** full-frame box; response `model` will **not** be `deepfashion2-yolov3-opencv`.

## Other ModaNet / custom YOLOv3

Any Darknet **cfg + weights + names** with compatible `classes=` can be used by setting `VESTIR_DARKNET_CFG`, `VESTIR_DARKNET_WEIGHTS`, and `VESTIR_DARKNET_NAMES`. Response `model` will show `darknet-yolov3-opencv` if `df2.names` is not used.

## Test YOLO only

```powershell
cd vision-sidecar
.\.venv\Scripts\Activate.ps1
$env:VESTIR_DARKNET_WEIGHTS = "C:\path\to\yolov3-df2_last.weights"
python test_modanet_yolo.py "C:\path\to\photo.jpg"
```

Success: JSON includes `"model": "deepfashion2-yolov3-opencv"` and `"garments"` with labels like `long sleeve outwear`, `trousers`, etc.

Optional HTTP check (uvicorn running):

```powershell
python test_modanet_yolo.py "C:\path\to\photo.jpg" --serve
```

## Hybrid infer: LAB palette + masking + fine-tuning policy

- `/infer` clusters garment colors in **OpenCV LAB** space and maps centroids to **`fashion_color_vocab.json`** (CIELAB anchors). Replace anchors by sampling real garment crops per name for production stability.
- **`VESTIR_COLOR_MASK`**: `none` | `alpha` | `grabcut` | `all` (default **`all`**). `alpha` uses PNG alpha; `grabcut` uses a center-seeded GrabCut mask; `all` combines both when alpha exists.
- **`FASHION_COLOR_VOCAB_PATH`**: optional override for the JSON path.
- Do not fine-tune detection/CLIP until you meet the crop-count gate — see **[FINETUNING.md](./FINETUNING.md)**.

On the **Node** API, optional second-pass arbitration when models disagree: set `VESTIR_INFER_ARBITRATE=gemini_on_disagreement` (requires `GEMINI_API_KEY`). The browser extension calls infer with `skipArbitration: true` so match stays fast.

## Response shape

```json
{
  "person_count": 1,
  "multi_person": false,
  "garments": [{ "label": "...", "confidence": 0.9, "bbox": { "x1": 0, "y1": 0, "x2": 1, "y2": 1 } }],
  "privacy_regions": [],
  "model": "deepfashion2-yolov3-opencv"
}
```
