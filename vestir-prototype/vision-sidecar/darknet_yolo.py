"""
YOLOv3 (Darknet .cfg + .weights) via OpenCV DNN.

Primary use case: DeepFashion2 13-class configs from
https://github.com/tomsebastiank/Clothing-detection-and-attribute-identification-using-YoloV3-and-DeepFashion

Bundled: models/yolov3-df2-13class.cfg + data/df2.names (classes). You only need a trained
*.weights file (see that repo: train.py / checkpoints, then export or use darknet-format weights
compatible with detect.py --weights_path).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent
_BUNDLED_CFG = _ROOT / "models" / "yolov3-df2-13class.cfg"
_BUNDLED_NAMES = _ROOT / "data" / "df2.names"


def _resolve_cfg() -> str:
    for key in ("VESTIR_DARKNET_CFG", "MODANET_YOLO_CFG"):
        v = os.environ.get(key, "").strip()
        if v and os.path.isfile(v):
            return v
    if _BUNDLED_CFG.is_file():
        return str(_BUNDLED_CFG)
    return ""


def _resolve_weights() -> str:
    for key in ("VESTIR_DARKNET_WEIGHTS", "MODANET_YOLO_WEIGHTS"):
        v = os.environ.get(key, "").strip()
        if v:
            return v
    return ""


def _resolve_names() -> str | None:
    for key in ("VESTIR_DARKNET_NAMES", "MODANET_YOLO_NAMES"):
        v = os.environ.get(key, "").strip()
        if v and os.path.isfile(v):
            return v
    if _BUNDLED_NAMES.is_file():
        return str(_BUNDLED_NAMES)
    return None


def _model_label() -> str:
    names = _resolve_names() or ""
    if "df2.names" in names.replace("\\", "/").split("/")[-1].lower():
        return "deepfashion2-yolov3-opencv"
    return "darknet-yolov3-opencv"


def _env_float(keys: tuple[str, ...], default: float) -> float:
    for k in keys:
        v = os.environ.get(k, "").strip()
        if v:
            return float(v)
    return default


def _env_int(keys: tuple[str, ...], default: int) -> int:
    for k in keys:
        v = os.environ.get(k, "").strip()
        if v:
            return int(v)
    return default


def _env_cuda() -> bool:
    for k in ("VESTIR_DARKNET_CUDA", "MODANET_YOLO_CUDA"):
        if os.environ.get(k, "").lower() in ("1", "true", "yes"):
            return True
    return False


def _load_class_names(names_path: str | None) -> list[str]:
    if names_path and os.path.isfile(names_path):
        with open(names_path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    return []


def darknet_yolo_configured() -> bool:
    cfg = _resolve_cfg()
    wts = _resolve_weights()
    return bool(cfg and wts and os.path.isfile(cfg) and os.path.isfile(wts))


def run_darknet_yolo(image_bgr: np.ndarray) -> dict[str, Any] | None:
    if not darknet_yolo_configured():
        return None

    cfg = _resolve_cfg()
    weights = _resolve_weights()
    names_path = _resolve_names()
    conf_threshold = _env_float(("VESTIR_DARKNET_CONF", "MODANET_YOLO_CONF"), 0.45)
    nms_threshold = _env_float(("VESTIR_DARKNET_NMS", "MODANET_YOLO_NMS"), 0.45)
    inp_w = _env_int(("VESTIR_DARKNET_INPUT", "MODANET_YOLO_INPUT"), 416)
    inp_h = inp_w

    model_tag = _model_label()

    try:
        net = cv2.dnn.readNetFromDarknet(cfg, weights)
    except cv2.error:
        return None

    if _env_cuda():
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except cv2.error:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    ln = net.getLayerNames()
    try:
        out_names = net.getUnconnectedOutLayersNames()
    except AttributeError:
        out_idx = net.getUnconnectedOutLayers()
        out_names = [ln[int(i) - 1] for i in np.asarray(out_idx).reshape(-1)]

    img_h, img_w = image_bgr.shape[:2]
    if img_h < 2 or img_w < 2:
        return None

    blob = cv2.dnn.blobFromImage(image_bgr, 1 / 255.0, (inp_w, inp_h), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_names)

    class_names = _load_class_names(names_path)
    n_classes = len(class_names)
    if n_classes == 0:
        return None

    boxes: list[list[int]] = []
    confidences: list[float] = []
    class_ids: list[int] = []

    for out in outs:
        if out is None:
            continue
        for det in out:
            if len(det) < 6:
                continue
            scores = np.array(det[5:], dtype=np.float32)
            if scores.size == 0:
                continue
            if n_classes and scores.size > n_classes:
                scores = scores[:n_classes]
            cid = int(np.argmax(scores))
            if cid < 0:
                continue
            conf = float(scores[cid]) * float(det[4])
            if conf < conf_threshold:
                continue
            cx, cy, bw, bh = det[0:4]
            box_w = max(1, int(bw * img_w))
            box_h = max(1, int(bh * img_h))
            center_x = int(cx * img_w)
            center_y = int(cy * img_h)
            x = max(0, center_x - box_w // 2)
            y = max(0, center_y - box_h // 2)
            if x >= img_w or y >= img_h:
                continue
            boxes.append([x, y, min(box_w, img_w - x), min(box_h, img_h - y)])
            confidences.append(conf)
            class_ids.append(cid)

    if not boxes:
        return {
            "person_count": 0,
            "multi_person": False,
            "garments": [],
            "privacy_regions": [],
            "warnings": ["yolo_no_detections"],
            "model": model_tag,
        }

    idx = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    garments: list[dict[str, Any]] = []
    if idx is not None and len(idx) > 0:
        for i in np.array(idx).flatten():
            bi = int(i)
            x, y, bw, bh = boxes[bi]
            x2 = min(img_w, x + bw)
            y2 = min(img_h, y + bh)
            cid = class_ids[bi]
            label = class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}"
            garments.append(
                {
                    "label": label,
                    "confidence": round(float(confidences[bi]), 3),
                    "bbox": {
                        "x1": round(x / img_w, 4),
                        "y1": round(y / img_h, 4),
                        "x2": round(x2 / img_w, 4),
                        "y2": round(y2 / img_h, 4),
                    },
                }
            )

    return {
        "person_count": 1 if garments else 0,
        "multi_person": False,
        "garments": garments,
        "privacy_regions": [],
        "warnings": [],
        "model": model_tag,
    }
