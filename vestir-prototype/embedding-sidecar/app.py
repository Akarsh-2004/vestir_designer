"""
Fashion / multimodal embedding sidecar for Vestir.
Default: Marqo/marqo-fashionSigLIP (Hugging Face, trust_remote_code).
Fallback: google/siglip-base-patch16-224 if EMBEDDING_MODEL_ID starts with google/
  or EMBEDDING_BACKEND=siglip.
"""
from __future__ import annotations

import base64
import io
import os
from typing import Any, Literal, Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

DEFAULT_MODEL = "Marqo/marqo-fashionSigLIP"
DEFAULT_GOOGLE = "google/siglip-base-patch16-224"


def _model_id() -> str:
    return os.environ.get("EMBEDDING_MODEL_ID", DEFAULT_MODEL).strip()


def _backend() -> Literal["marqo", "siglip"]:
    mid = _model_id()
    if os.environ.get("EMBEDDING_BACKEND", "").lower() == "siglip":
        return "siglip"
    if mid.startswith("google/"):
        return "siglip"
    return "marqo"


def _resolved_model_id(backend: str) -> str:
    mid = _model_id()
    if backend == "siglip":
        return mid if mid.startswith("google/") else DEFAULT_GOOGLE
    return mid


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EmbedIn(BaseModel):
    text: Optional[str] = None
    image_base64: Optional[str] = None


_state: dict[str, Any] = {
    "backend": None,
    "model_id": None,
    "device": None,
    "model": None,
    "processor": None,
    "error": None,
}


def _pil_from_b64(b64: str) -> Image.Image:
    raw = base64.b64decode(b64, validate=False)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _load() -> None:
    if _state["model"] is not None:
        return
    device = _device()
    _state["device"] = str(device)
    backend = _backend()
    mid = _resolved_model_id(backend)
    _state["backend"] = backend
    _state["model_id"] = mid
    try:
        if backend == "siglip":
            from transformers import SiglipModel, SiglipProcessor

            proc = SiglipProcessor.from_pretrained(mid)
            model = SiglipModel.from_pretrained(mid)
            model.eval()
            model.to(device)
            _state["processor"] = proc
            _state["model"] = model
        else:
            from transformers import AutoModel, AutoProcessor

            proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
            model = AutoModel.from_pretrained(mid, trust_remote_code=True)
            model.eval()
            model.to(device)
            _state["processor"] = proc
            _state["model"] = model
    except Exception as e:
        _state["error"] = f"{type(e).__name__}: {e}"
        raise


def _embed_marqo(text: Optional[str], image: Optional[Image.Image]) -> list[float]:
    model = _state["model"]
    processor = _state["processor"]
    device = _device()

    tok_kw = {"padding": "max_length", "truncation": True, "max_length": 128}
    if text and image:
        batch = processor(text=[text], images=image, return_tensors="pt", **tok_kw)
        input_ids = batch["input_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        t = model.get_text_features(input_ids=input_ids, normalize=True)
        i = model.get_image_features(pixel_values=pixel_values, normalize=True)
        fused = F.normalize((t + i) / 2.0, dim=-1)
        return fused[0].detach().float().cpu().tolist()
    if text:
        batch = processor(text=[text], return_tensors="pt", **tok_kw)
        input_ids = batch["input_ids"].to(device)
        t = model.get_text_features(input_ids=input_ids, normalize=True)
        return t[0].detach().float().cpu().tolist()
    if image:
        batch = processor(images=image, return_tensors="pt")
        pv = batch["pixel_values"].to(device)
        i = model.get_image_features(pixel_values=pv, normalize=True)
        return i[0].detach().float().cpu().tolist()
    raise ValueError("need text and/or image")


def _embed_siglip(text: Optional[str], image: Optional[Image.Image]) -> list[float]:
    model = _state["model"]
    processor = _state["processor"]
    device = _device()

    if text and image:
        batch = processor(text=[text], images=image, padding="max_length", return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        t = model.get_text_features(**{k: batch[k] for k in batch if k in ("input_ids", "attention_mask")})
        i = model.get_image_features(pixel_values=batch["pixel_values"])
        t = F.normalize(t, dim=-1)
        i = F.normalize(i, dim=-1)
        fused = F.normalize((t + i) / 2.0, dim=-1)
        return fused[0].detach().float().cpu().tolist()
    if text:
        batch = processor(text=[text], padding="max_length", return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        t = model.get_text_features(**{k: batch[k] for k in batch if k in ("input_ids", "attention_mask")})
        return F.normalize(t, dim=-1)[0].detach().float().cpu().tolist()
    if image:
        batch = processor(images=image, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        i = model.get_image_features(pixel_values=batch["pixel_values"])
        return F.normalize(i, dim=-1)[0].detach().float().cpu().tolist()
    raise ValueError("need text and/or image")


app = FastAPI(title="Vestir embedding sidecar", version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    try:
        _load()
    except Exception:
        pass


@app.get("/health")
def health() -> dict:
    out: dict = {
        "ok": _state["model"] is not None,
        "backend": _state["backend"],
        "model_id": _state["model_id"],
        "device": _state["device"],
    }
    if _state["error"]:
        out["error"] = _state["error"]
    return out


@app.post("/embed")
def embed(body: EmbedIn) -> dict:
    if not body.text and not body.image_base64:
        raise HTTPException(400, "Provide `text` and/or `image_base64`")
    try:
        if _state["model"] is None:
            _load()
    except Exception as e:
        raise HTTPException(503, str(e)) from e

    image = _pil_from_b64(body.image_base64) if body.image_base64 else None
    text = (body.text or "").strip() or None

    try:
        if _state["backend"] == "siglip":
            vec = _embed_siglip(text, image)
        else:
            vec = _embed_marqo(text, image)
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}") from e

    return {"vector": vec, "model": _state["model_id"], "dim": len(vec)}


def main() -> None:
    import uvicorn

    host = os.environ.get("EMBEDDING_BIND", "0.0.0.0")
    port = int(os.environ.get("EMBEDDING_PORT", "8010"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
