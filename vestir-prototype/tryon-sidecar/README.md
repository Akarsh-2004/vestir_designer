# Try-on sidecar (DeepFashion / diffusion-style virtual try-on)

Your Kaggle notebook ([Virtual TryOn Diffusion Model](https://www.kaggle.com/code/datascientist97/virtual-tryon-diffusion-model)) solves **image synthesis**: given a **person** and a **garment** image, generate the person **wearing** that garment.

That is **complementary** to the main Vestir pipeline (detect → parse → attributes). It does **not** replace human parsing or color inference; use it for **preview** experiences (e.g. “see this top on me”).

## Run (stub)

```bash
cd tryon-sidecar
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8009 --reload
```

## LoRA-based virtual try-on (Diffusers)

### Hugging Face auth

`huggingface_hub` expects **`HUGGING_FACE_HUB_TOKEN`**. If you only set **`HF_TOKEN`** in the repo root `.env`, the sidecar maps it automatically (see `app.py` and `start.sh`).

### FLUX.2 virtual try-off LoRA ([fal/virtual-tryoff-lora](https://huggingface.co/fal/virtual-tryoff-lora))

**Apple Silicon:** `start.sh` sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` so MPS can use unified memory (no quantization). On macOS, FLUX defaults to **MPS** when `TRYON_USE_MPS_FOR_FLUX` is unset, and **`TRYON_FLUX_DEVICE_MAP=balanced`** when unset (splits sub-models across CPU/MPS so no single ~15 GiB Metal buffer — avoids PyTorch **Invalid buffer size**). Optional **`TRYON_FLUX_DEVICE_MAP=mps`** often fails on this model; the sidecar auto-retries with `balanced` if it does. Still **fp16** weights (not INT8). CPU-only: `TRYON_USE_MPS_FOR_FLUX=0` (often OOM while loading).

Example env (also in repo root `.env` if you prefer):

- `TRYON_BASE_MODEL_ID=black-forest-labs/FLUX.2-klein-base-9B`
- `TRYON_LORA_PATH=fal/virtual-tryoff-lora` **or** a local folder after download
- `TRYON_LORA_WEIGHT_NAME=virtual-tryoff-lora_diffusers.safetensors`
- `TRYON_LORA_ADAPTER_NAME=vtoff`

Download weights locally (uses `HF_TOKEN` → hub token):

```bash
cd vestir-prototype/tryon-sidecar
./scripts/download_tryoff_lora.sh
```

Then point `TRYON_LORA_PATH` at `models/virtual-tryoff-lora` (path relative to the tryon-sidecar cwd, or absolute).

You can run this sidecar with a Diffusers inpainting base model + VTON LoRA:

- `TRYON_BASE_MODEL_ID` (example: `runwayml/stable-diffusion-inpainting` or FLUX.2 base above)
- `TRYON_LORA_PATH` (local path or HF repo id of your VTON LoRA)
- optional `TRYON_LORA_SCALE` (default `0.85`)
- optional `TRYON_STEPS` (default `28`)
- optional `TRYON_GUIDANCE` (default `7.5`)

When those env vars are set, `POST /tryon` uses the loaded LoRA pipeline.

## Wire the Kaggle model (custom)

1. Export the notebook’s inference path into a plain Python module (single `forward(person, garment) -> image` is enough).
2. Save checkpoints to a directory on disk.
3. Set envs above (or adapt `_run_tryon()` to your checkpoint format).
4. Implement/replace `_run_tryon()` in `app.py` for your exact conditioning strategy if needed.

GPU and exact dependencies (PyTorch, diffusers, xformers, etc.) should match your training/inference stack for best results.

## API contract (stable for the Node server)

- `POST /tryoff` (virtual **try-off** — extract garment from a worn photo)  
  Body: `{ "image_base64": "...", "garment_target": "outfit" | "tshirt" | "dress" | "pants" | "jacket", "seed": optional }`  
  Requires FLUX.2 base + [fal/virtual-tryoff-lora](https://huggingface.co/fal/virtual-tryoff-lora).  
  Node exposes this as `POST /api/items/tryoff` with `{ imageUrl, garmentTarget?, seed? }`.

- `GET /health` and `GET /tryoff/status` now include:
  - `tryoff_pipeline_ready`: model is loaded in memory
  - `download_progress`: FLUX cache progress (`present/total`, `percent`, `cache_size_gb`, `missing_shards`)
  - Use these fields to monitor first-time download/init.

- `POST /tryon`  
  Body: `{ "person_image_base64": "...", "garment_image_base64": "...", "seed": optional }`  
  Response when not wired: `{ "ok": true, "implemented": false, "result_image_base64": null, "message": "..." }`  
  Response when wired: `{ "ok": true, "implemented": true, "result_image_base64": "<base64>" }`

## Node

Set `TRYON_SIDECAR_URL=http://127.0.0.1:8009` and call `POST /api/tryon/preview` on the main API (see server docs in repo).
