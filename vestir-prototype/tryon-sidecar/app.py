"""
Virtual try-on sidecar (diffusion / warping models).

This is intentionally **not** the same job as garment attribute extraction.
Port the forward pass from your Kaggle training notebook (e.g. diffusion try-on)
into `_run_tryon()` once weights and preprocessing are available locally.

Reference workflow (typical for DeepFashion / VITON-family try-on):
  person image + garment image -> aligned agnostic representation -> generator -> composite
"""

from __future__ import annotations

import base64
import os
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Vestir Try-On Sidecar", version="0.1.0")


class TryOnRequest(BaseModel):
    person_image_base64: str
    garment_image_base64: str
    seed: int | None = Field(default=None, description="Optional RNG seed for diffusion")


def _run_tryon(person_png: bytes, garment_png: bytes, seed: int | None) -> bytes | None:
    """
    Replace with real inference from your exported Kaggle notebook / checkpoint.

    Expected return: PNG or JPEG bytes of the try-on result, or None if not configured.
    """
    weights = os.environ.get("TRYON_WEIGHTS_DIR", "").strip()
    if not weights:
        return None
    # Example: load torch model from TRYON_WEIGHTS_DIR and run forward pass.
    return None


@app.get("/health")
def health() -> dict[str, Any]:
    configured = bool(os.environ.get("TRYON_WEIGHTS_DIR", "").strip())
    return {"ok": True, "service": "tryon-sidecar", "model_configured": configured}


@app.post("/tryon")
def tryon(payload: TryOnRequest) -> dict[str, Any]:
    try:
        person = base64.b64decode(payload.person_image_base64)
        garment = base64.b64decode(payload.garment_image_base64)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"invalid_base64: {exc}"}

    result = _run_tryon(person, garment, payload.seed)
    if result is None:
        return {
            "ok": True,
            "implemented": False,
            "message": "Set TRYON_WEIGHTS_DIR and implement _run_tryon() using your Kaggle notebook export.",
            "result_image_base64": None,
        }

    return {
        "ok": True,
        "implemented": True,
        "message": None,
        "result_image_base64": base64.b64encode(result).decode("ascii"),
    }
