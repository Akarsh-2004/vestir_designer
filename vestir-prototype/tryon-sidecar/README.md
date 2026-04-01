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

## Wire the Kaggle model

1. Export the notebook’s inference path into a plain Python module (single `forward(person, garment) -> image` is enough).
2. Save checkpoints to a directory on disk.
3. Set `TRYON_WEIGHTS_DIR` to that directory.
4. Implement `_run_tryon()` in `app.py`: decode bytes → same preprocessing as the notebook → model → encode result bytes.

GPU and exact dependencies (PyTorch, diffusers, xformers, etc.) **must match** what the notebook used; copy versions from the Kaggle kernel or `pip freeze` after a successful run.

## API contract (stable for the Node server)

- `POST /tryon`  
  Body: `{ "person_image_base64": "...", "garment_image_base64": "...", "seed": optional }`  
  Response when not wired: `{ "ok": true, "implemented": false, "result_image_base64": null, "message": "..." }`  
  Response when wired: `{ "ok": true, "implemented": true, "result_image_base64": "<base64>" }`

## Node

Set `TRYON_SIDECAR_URL=http://127.0.0.1:8009` and call `POST /api/tryon/preview` on the main API (see server docs in repo).
