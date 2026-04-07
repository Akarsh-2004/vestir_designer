# Embedding sidecar (fashion SigLIP)

Serves `GET /health` and `POST /embed` for the Node API when `EMBEDDING_SIDECAR_URL` is set.

**Default model:** [`Marqo/marqo-fashionSigLIP`](https://huggingface.co/Marqo/marqo-fashionSigLIP) (downloads from Hugging Face on first load).

**Lighter fallback:** `EMBEDDING_BACKEND=siglip` uses `google/siglip-base-patch16-224` (set `EMBEDDING_MODEL_ID=google/siglip-base-patch16-224` to pin).

## Setup

```bash
cd embedding-sidecar
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Pre-download weights (optional; also happens on first request):

```bash
pip install -r requirements-download.txt
python download_models.py
# or from repo root: npm run embed:download
```

## Run

```bash
export EMBEDDING_PORT=8010
./start.sh
```

Env:

| Variable | Default | Meaning |
|----------|---------|---------|
| `EMBEDDING_MODEL_ID` | `Marqo/marqo-fashionSigLIP` | Hugging Face repo id |
| `EMBEDDING_BACKEND` | (auto) | Set `siglip` to force Google SigLIP path |
| `EMBEDDING_BIND` | `0.0.0.0` | Listen address |
| `EMBEDDING_PORT` | `8010` | Port |

## Smoke test

With the sidecar running:

```bash
cd ..   # vestir-prototype root
export EMBEDDING_SIDECAR_URL=http://127.0.0.1:8010
npm run smoke:embed
```

Node API: set `EMBEDDING_SIDECAR_URL=http://127.0.0.1:8010` alongside `GEMINI_API_KEY` (for infer) as needed.
