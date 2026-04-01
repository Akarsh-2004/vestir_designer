"""
Ultralytics YOLOv8 inference — matches Kaggle-style workflows (e.g. colorful fashion, custom data.yaml).

Set VESTIR_YOLOV8_PT to a trained checkpoint (best.pt / last.pt).
Install: pip install -r requirements-yolov8.txt
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

_cached_pt: str | None = None
_cached_model: Any = None


def _yolo_weight_spec_configured(spec: str) -> bool:
    """
    True if spec is an existing .pt file, or an Ultralytics hub id (e.g. yolov8n.pt)
    that YOLO() can download — so VESTIR_YOLOV8_PT is not limited to disk paths.
    """
    p = spec.strip()
    if not p:
        return False
    if os.path.isfile(p):
        return True
    base = os.path.basename(p)
    if base != p:
        return False
    if not p.lower().endswith(".pt"):
        return False
    if os.path.sep in p or (os.path.altsep and os.path.altsep in p):
        return False
    return True


def yolov8_configured() -> bool:
    path = os.environ.get("VESTIR_YOLOV8_PT", "").strip()
    return _yolo_weight_spec_configured(path)


def _model():
    global _cached_pt, _cached_model
    path = os.environ["VESTIR_YOLOV8_PT"].strip()
    if _cached_model is not None and _cached_pt == path:
        return _cached_model
    from ultralytics import YOLO

    _cached_model = YOLO(path)
    _cached_pt = path
    return _cached_model


def run_yolov8(image_bgr: np.ndarray) -> dict[str, Any] | None:
    if not yolov8_configured():
        return None
    try:
        model = _model()
    except ImportError:
        return None

    conf = float(os.environ.get("VESTIR_YOLOV8_CONF", "0.4"))
    img_h, img_w = image_bgr.shape[:2]
    if img_h < 2 or img_w < 2:
        return None

    try:
        results = model.predict(source=image_bgr, conf=conf, verbose=False)
    except Exception:
        return None

    r = results[0]
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        return {
            "person_count": 0,
            "multi_person": False,
            "garments": [],
            "privacy_regions": [],
            "warnings": ["yolov8_no_detections"],
            "model": "yolov8-ultralytics",
        }

    raw_names = r.names
    if isinstance(raw_names, dict):
        names = {int(k): str(v) for k, v in raw_names.items()}
    else:
        names = {i: str(raw_names[i]) for i in range(len(raw_names))}
    garments: list[dict[str, Any]] = []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = xyxy[i]
        cid = int(clss[i])
        label = names.get(cid, f"class_{cid}")
        garments.append(
            {
                "label": label.replace("_", " "),
                "confidence": round(float(confs[i]), 3),
                "bbox": {
                    "x1": round(float(x1) / img_w, 4),
                    "y1": round(float(y1) / img_h, 4),
                    "x2": round(float(x2) / img_w, 4),
                    "y2": round(float(y2) / img_h, 4),
                },
            }
        )

    return {
        "person_count": 1 if garments else 0,
        "multi_person": False,
        "garments": garments,
        "privacy_regions": [],
        "warnings": [],
        "model": "yolov8-ultralytics",
    }
