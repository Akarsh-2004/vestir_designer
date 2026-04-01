from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from darknet_yolo import darknet_yolo_configured, run_darknet_yolo
from yolov8_engine import run_yolov8, yolov8_configured

app = FastAPI(title="Vestir Vision Sidecar", version="0.1.0")


class AnalyzeRequest(BaseModel):
    image_base64: str
    stages: list[str] = Field(default_factory=list)


def _decode_image(image_base64: str) -> np.ndarray | None:
    try:
        raw = base64.b64decode(image_base64)
        arr = np.frombuffer(raw, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "vision-sidecar",
        "yolo_darknet_weights": darknet_yolo_configured(),
        "yolov8_pt": yolov8_configured(),
    }


@app.post("/analyze")
def analyze(payload: AnalyzeRequest) -> dict[str, Any]:
    image = _decode_image(payload.image_base64)
    if image is None:
        return {"person_count": 0, "garments": [], "privacy_regions": [], "warnings": ["decode_failed"]}

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return {"person_count": 0, "garments": [], "privacy_regions": [], "warnings": ["invalid_image"]}

    # YOLOv8 first (Ultralytics .pt — typical for Kaggle / Mac MPS fine-tunes); then Darknet .weights.
    y8 = run_yolov8(image)
    if y8 is not None:
        return y8
    yolo_out = run_darknet_yolo(image)
    if yolo_out is not None:
        return yolo_out

    # Baseline heuristic when ModaNet YOLOv3 weights are not configured.
    garments = [
        {
            "label": "garment",
            "confidence": 0.55,
            "bbox": {"x1": 0.08, "y1": 0.08, "x2": 0.92, "y2": 0.92},
        }
    ]
    return {
        "person_count": 1,
        "multi_person": False,
        "garments": garments,
        "privacy_regions": [],
        "model": "heuristic-bootstrap",
    }

