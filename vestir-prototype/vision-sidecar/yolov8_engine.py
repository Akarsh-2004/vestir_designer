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

_GARMENT_KEYWORDS = {
    "clothing",
    "shirt",
    "t-shirt",
    "tee",
    "top",
    "blouse",
    "jacket",
    "coat",
    "hoodie",
    "sweater",
    "cardigan",
    "vest",
    "dress",
    "skirt",
    "pants",
    "trousers",
    "jeans",
    "shorts",
    "shoe",
    "sneaker",
    "boot",
    "bag",
    "handbag",
    "backpack",
    "belt",
    "scarf",
    "tie",
}


def _is_garment_like(label: str) -> bool:
    t = label.lower()
    return any(k in t for k in _GARMENT_KEYWORDS)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _torso_proxy_from_person_bbox(person_bbox: dict[str, float]) -> dict[str, float]:
    """
    Convert a person bbox to a torso-focused clothing proxy.
    This keeps wardrobe crops useful even when the detector is generic COCO.
    """
    x1 = _clamp01(person_bbox["x1"])
    y1 = _clamp01(person_bbox["y1"])
    x2 = _clamp01(person_bbox["x2"])
    y2 = _clamp01(person_bbox["y2"])
    w = max(0.01, x2 - x1)
    h = max(0.01, y2 - y1)

    # Torso window tuned for tops/dresses in full-body fashion photos.
    tx1 = _clamp01(x1 + 0.18 * w)
    tx2 = _clamp01(x2 - 0.18 * w)
    ty1 = _clamp01(y1 + 0.18 * h)
    ty2 = _clamp01(y1 + 0.78 * h)

    if tx2 <= tx1 or ty2 <= ty1:
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    return {"x1": tx1, "y1": ty1, "x2": tx2, "y2": ty2}


def _generic_proxy_from_bbox(bbox: dict[str, float]) -> dict[str, float]:
    """
    Conservative crop used only when YOLO gives non-garment classes and no person.
    Shrinks noisy full-frame detections toward the center.
    """
    x1 = _clamp01(bbox["x1"])
    y1 = _clamp01(bbox["y1"])
    x2 = _clamp01(bbox["x2"])
    y2 = _clamp01(bbox["y2"])
    w = max(0.01, x2 - x1)
    h = max(0.01, y2 - y1)

    gx1 = _clamp01(x1 + 0.14 * w)
    gx2 = _clamp01(x2 - 0.14 * w)
    gy1 = _clamp01(y1 + 0.10 * h)
    gy2 = _clamp01(y2 - 0.08 * h)
    if gx2 <= gx1 or gy2 <= gy1:
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    return {"x1": gx1, "y1": gy1, "x2": gx2, "y2": gy2}


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

    conf = float(os.environ.get("VESTIR_YOLOV8_CONF", "0.3"))
    iou = float(os.environ.get("VESTIR_YOLOV8_IOU", "0.6"))
    max_det = int(os.environ.get("VESTIR_YOLOV8_MAX_DET", "30"))
    imgsz_raw = os.environ.get("VESTIR_YOLOV8_IMGSZ", "").strip()
    imgsz = int(imgsz_raw) if imgsz_raw else None
    img_h, img_w = image_bgr.shape[:2]
    if img_h < 2 or img_w < 2:
        return None

    try:
        predict_kw: dict[str, Any] = {
            "source": image_bgr,
            "conf": conf,
            "iou": iou,
            "max_det": max_det,
            "verbose": False,
        }
        if imgsz is not None:
            predict_kw["imgsz"] = imgsz
        results = model.predict(**predict_kw)
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
    detections: list[dict[str, Any]] = []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = xyxy[i]
        cid = int(clss[i])
        label = names.get(cid, f"class_{cid}")
        detections.append(
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

    person_detections = [d for d in detections if d["label"].lower() == "person"]
    person_count = len(person_detections)

    garments = [d for d in detections if _is_garment_like(d["label"])]

    # If the model is generic (e.g. COCO) and returns only person/non-garment classes,
    # create torso clothing proxies so the wardrobe pipeline can still crop useful regions.
    warnings: list[str] = []
    if not garments and person_detections:
        garments = [
            {
                "label": "clothing",
                "confidence": round(max(0.35, min(0.9, d["confidence"] * 0.72)), 3),
                "bbox": _torso_proxy_from_person_bbox(d["bbox"]),
            }
            for d in person_detections
        ]
        warnings.append("yolov8_person_to_torso_proxy")
    elif not garments and detections:
        # Last-resort fallback for generic COCO checkpoints that return only scene objects.
        # Keep the highest-confidence large detection and center-shrink it.
        best = max(
            detections,
            key=lambda d: (
                d["confidence"],
                (d["bbox"]["x2"] - d["bbox"]["x1"]) * (d["bbox"]["y2"] - d["bbox"]["y1"]),
            ),
        )
        garments = [
            {
                "label": "clothing",
                "confidence": round(max(0.3, min(0.85, best["confidence"] * 0.65)), 3),
                "bbox": _generic_proxy_from_bbox(best["bbox"]),
            }
        ]
        warnings.append("yolov8_generic_bbox_proxy")

    return {
        "person_count": person_count,
        "multi_person": person_count > 1,
        "garments": garments,
        "privacy_regions": [],
        "warnings": warnings,
        "model": "yolov8-ultralytics",
    }
