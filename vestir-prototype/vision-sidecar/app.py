from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI
from huggingface_hub import hf_hub_download
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoProcessor, pipeline

from darknet_yolo import darknet_yolo_configured, run_darknet_yolo
from fashion_tags import (
    build_fashion_tag_text_features,
    env_fashion_tag_config,
    fashion_tag_inference_payload,
    load_fashion_tag_entries,
)
from garment_color import _mask_mode, extract_garment_palette, load_fashion_color_entries
from yolov8_engine import run_yolov8, yolov8_configured

app = FastAPI(title="Vestir Vision Sidecar", version="0.1.0")


class AnalyzeRequest(BaseModel):
    image_base64: str
    stages: list[str] = Field(default_factory=list)


class InferRequest(BaseModel):
    image_base64: str


class FaceDetectRequest(BaseModel):
    image_base64: str


class SamSegmentRequest(BaseModel):
    image_base64: str
    boxes: list[dict[str, float]] = Field(default_factory=list)
    balanced: bool = True


LABELS = [
    "tshirt",
    "shirt",
    "top",
    "jacket",
    "jeans",
    "pants",
    "pyjama",
    "skirt",
    "dress",
    "frock",
    "shorts",
    "capri",
    "sweater",
    "sweatshirt",
    "hoodie",
    "vest",
    "coat",
    "blazer",
    "cap",
]

CATEGORY_MAP = {
    "tshirt": "Tops",
    "shirt": "Tops",
    "top": "Tops",
    "sweater": "Tops",
    "sweatshirt": "Tops",
    "hoodie": "Tops",
    "vest": "Tops",
    "jacket": "Outerwear",
    "coat": "Outerwear",
    "blazer": "Outerwear",
    "jeans": "Bottoms",
    "pants": "Bottoms",
    "pyjama": "Bottoms",
    "shorts": "Bottoms",
    "capri": "Bottoms",
    "skirt": "Bottoms",
    "dress": "Tops",
    "frock": "Tops",
}


def _local_files_only() -> bool:
    return str(os.environ.get("HF_LOCAL_FILES_ONLY", "")).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class HybridInferModels:
    siglip_model_id: str
    florence_model_id: str
    siglip_threshold: float
    siglip_openclip: Any | None
    siglip_model: Any
    siglip_preprocess: Any | None
    siglip_tokens: Any | None
    siglip_classifier: Any | None
    florence_processor: Any
    florence_model: Any
    fashion_tags_enabled: bool
    fashion_tag_threshold: float
    fashion_tag_max_per_layer: int
    fashion_tag_total_max: int
    fashion_tag_entries: list[Any]
    fashion_tag_text_features: Any | None


_MODELS: HybridInferModels | None = None
_MODELS_ERROR: str | None = None
_FACE_MODEL = None
_FACE_MODEL_ERROR: str | None = None
_FASHION_COLOR_ENTRIES: list | None = None
_SAM_PREDICTOR: Any | None = None
_SAM_ERROR: str | None = None


def _load_models() -> HybridInferModels:
    global _MODELS, _MODELS_ERROR
    if _MODELS is not None:
        return _MODELS
    if _MODELS_ERROR is not None:
        raise RuntimeError(_MODELS_ERROR)

    siglip_model_id = (
        os.environ.get("SIGLIP_FASHION_MODEL_ID", "").strip()
        or os.environ.get("SIGLIP_MODEL_ID", "").strip()
        or "Marqo/marqo-fashionSigLIP"
    )
    florence_model_id = (
        os.environ.get("FLORENCE_FASHION_MODEL_ID", "").strip()
        or os.environ.get("FLORENCE_MODEL_ID", "").strip()
        or "microsoft/Florence-2-base"
    )
    siglip_threshold = float(os.environ.get("SIGLIP_THRESHOLD", "0.14"))
    local_only = _local_files_only()
    ft_enabled, ft_threshold, ft_max_layer, ft_total_max, ft_vocab_path = env_fashion_tag_config()
    ft_entries: list[Any] = load_fashion_tag_entries(ft_vocab_path) if ft_enabled else []
    ft_text_features: Any | None = None

    try:
        if "marqo-fashionsiglip" in siglip_model_id.lower():
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(f"hf-hub:{siglip_model_id}")
            model.eval()
            tok = open_clip.get_tokenizer(f"hf-hub:{siglip_model_id}")(LABELS)
            siglip_openclip = open_clip
            siglip_model = model
            siglip_preprocess = preprocess
            siglip_tokens = tok
            siglip_classifier = None
            if ft_enabled and ft_entries:
                ft_batch = int(os.environ.get("FASHION_TAG_TEXT_BATCH", "48"))
                ft_text_features = build_fashion_tag_text_features(
                    open_clip,
                    model,
                    f"hf-hub:{siglip_model_id}",
                    ft_entries,
                    ft_batch,
                )
        else:
            siglip_openclip = None
            siglip_model = None
            siglip_preprocess = None
            siglip_tokens = None
            siglip_classifier = pipeline(
                "zero-shot-image-classification",
                model=siglip_model_id,
                device=-1,
                local_files_only=local_only,
            )
            ft_enabled = False
            ft_entries = []
            ft_text_features = None

        florence_processor = AutoProcessor.from_pretrained(
            florence_model_id,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        florence_model = AutoModelForCausalLM.from_pretrained(
            florence_model_id,
            trust_remote_code=True,
            local_files_only=local_only,
        ).eval()
    except Exception as exc:
        _MODELS_ERROR = f"hybrid model load failed: {exc}"
        raise RuntimeError(_MODELS_ERROR) from exc

    _MODELS = HybridInferModels(
        siglip_model_id=siglip_model_id,
        florence_model_id=florence_model_id,
        siglip_threshold=siglip_threshold,
        siglip_openclip=siglip_openclip,
        siglip_model=siglip_model,
        siglip_preprocess=siglip_preprocess,
        siglip_tokens=siglip_tokens,
        siglip_classifier=siglip_classifier,
        florence_processor=florence_processor,
        florence_model=florence_model,
        fashion_tags_enabled=bool(ft_enabled and ft_entries and ft_text_features is not None),
        fashion_tag_threshold=ft_threshold,
        fashion_tag_max_per_layer=ft_max_layer,
        fashion_tag_total_max=ft_total_max,
        fashion_tag_entries=ft_entries,
        fashion_tag_text_features=ft_text_features,
    )
    return _MODELS


def _decode_pil(image_base64: str) -> Image.Image:
    raw = base64.b64decode(image_base64)
    return Image.open(BytesIO(raw)).convert("RGB")


def _load_hf_face_model():
    global _FACE_MODEL, _FACE_MODEL_ERROR
    if _FACE_MODEL is not None:
        return _FACE_MODEL
    if _FACE_MODEL_ERROR is not None:
        raise RuntimeError(_FACE_MODEL_ERROR)
    model_id = os.environ.get("HF_FACE_MODEL_ID", "").strip() or "arnabdhar/YOLOv8-Face-Detection"
    filename_hint = os.environ.get("HF_FACE_MODEL_FILENAME", "").strip()
    candidate_names = [filename_hint] if filename_hint else []
    candidate_names.extend(["model.pt", "best.pt", "weights/best.pt", "yolov8n-face.pt"])
    try:
        from ultralytics import YOLO

        last_err: Exception | None = None
        model_path = None
        for name in candidate_names:
            if not name:
                continue
            try:
                model_path = hf_hub_download(repo_id=model_id, filename=name)
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
        if model_path is None:
            raise RuntimeError(f"could not locate weights in {model_id}; tried {candidate_names}: {last_err}")
        _FACE_MODEL = YOLO(model_path)
        return _FACE_MODEL
    except Exception as exc:  # noqa: BLE001
        _FACE_MODEL_ERROR = f"hf-face-model load failed: {exc}"
        raise RuntimeError(_FACE_MODEL_ERROR) from exc


def _detect_faces_hf(image_bgr: np.ndarray) -> list[dict[str, Any]]:
    model = _load_hf_face_model()
    h, w = image_bgr.shape[:2]
    result = model.predict(source=image_bgr, conf=0.22, verbose=False)[0]
    out: list[dict[str, Any]] = []
    if result.boxes is None:
        return out
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0].item())
        x1 = max(0.0, min(float(x1), float(w - 1)))
        y1 = max(0.0, min(float(y1), float(h - 1)))
        x2 = max(x1 + 1.0, min(float(x2), float(w)))
        y2 = max(y1 + 1.0, min(float(y2), float(h)))
        out.append(
            {
                "confidence": round(conf, 4),
                "bbox": {
                    "x1": round(x1 / w, 4),
                    "y1": round(y1 / h, 4),
                    "x2": round(x2 / w, 4),
                    "y2": round(y2 / h, 4),
                },
                "source": "hf_yolo_face",
            }
        )
    return out


def _detect_faces_haar(image_bgr: np.ndarray) -> list[dict[str, Any]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    h, w = gray.shape[:2]
    dets = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(20, 20))
    out = []
    for (x, y, bw, bh) in dets:
        out.append(
            {
                "confidence": 0.7,
                "bbox": {
                    "x1": round(float(x) / w, 4),
                    "y1": round(float(y) / h, 4),
                    "x2": round(float(x + bw) / w, 4),
                    "y2": round(float(y + bh) / h, 4),
                },
                "source": "opencv_haar",
            }
        )
    return out


def _siglip_scores(models: HybridInferModels, image: Image.Image) -> dict[str, float]:
    if models.siglip_openclip is not None:
        with torch.no_grad():
            image_tensor = models.siglip_preprocess(image).unsqueeze(0)
            image_features = models.siglip_model.encode_image(image_tensor, normalize=True)
            text_features = models.siglip_model.encode_text(models.siglip_tokens, normalize=True)
            probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0].tolist()
        out = {LABELS[i]: float(probs[i]) for i in range(len(LABELS)) if float(probs[i]) >= models.siglip_threshold}
        if not out:
            top_idx = int(np.argmax(probs))
            out = {LABELS[top_idx]: float(probs[top_idx])}
        return out

    assert models.siglip_classifier is not None
    result = models.siglip_classifier(image, candidate_labels=LABELS)
    scored = {str(r["label"]): float(r["score"]) for r in result}
    out = {k: v for k, v in scored.items() if v >= models.siglip_threshold}
    if not out and result:
        out = {str(result[0]["label"]): float(result[0]["score"])}
    return out


def _florence_caption_analysis(models: HybridInferModels, image: Image.Image) -> tuple[set[str], str, str]:
    task_prompt = "<MORE_DETAILED_CAPTION>"
    inputs = models.florence_processor(text=task_prompt, images=image, return_tensors="pt")
    with torch.no_grad():
        generated_ids = models.florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=64,
            num_beams=2,
        )
    text = models.florence_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].lower()
    found = {label for label in LABELS if label in text}
    if "frock" in found:
        found.add("dress")
    pattern = _infer_pattern_from_caption(text)
    return found, text, pattern


def _infer_pattern_from_caption(caption: str) -> str:
    """Map Florence caption keywords to schema pattern (Gemini PATTERN_ENUM compatible)."""
    if any(k in caption for k in ("plaid", "tartan", "checkered", "gingham")):
        if "gingham" in caption or "check" in caption:
            return "check"
        return "plaid"
    if any(k in caption for k in ("stripe", "striped", "pinstripe")):
        return "stripe"
    if any(k in caption for k in ("floral", "flower", "botanical", "paisley")):
        return "floral"
    if any(k in caption for k in ("graphic", "logo", "print", "slogan", "illustration")):
        return "graphic"
    if any(k in caption for k in ("texture", "ribbed", "quilted", "corduroy", "velvet", "sequin", "mesh", "lace")):
        return "texture"
    if "camouflage" in caption or "camo" in caption:
        return "mixed"
    return "solid"


_SUPER_GROUPS: dict[str, set[str]] = {
    "top": {"tshirt", "shirt", "top", "sweater", "sweatshirt", "hoodie", "vest"},
    "dress": {"dress", "frock"},
    "bottom": {"jeans", "pants", "pyjama", "shorts", "capri", "skirt"},
    "outerwear": {"jacket", "coat", "blazer"},
}


def _super_category_scores(
    siglip_scores: dict[str, float], florence_labels: set[str], aspect_ratio: float
) -> tuple[str, dict[str, float]]:
    scores = {g: 0.0 for g in _SUPER_GROUPS}
    for lbl, sc in siglip_scores.items():
        for g, mem in _SUPER_GROUPS.items():
            if lbl in mem:
                scores[g] += float(sc)
    for lbl in florence_labels:
        for g, mem in _SUPER_GROUPS.items():
            if lbl in mem:
                scores[g] += 0.22
    if aspect_ratio >= 0.88:
        scores["bottom"] += 0.2
    elif aspect_ratio <= 0.74:
        scores["top"] += 0.12
        scores["dress"] += 0.12
        scores["outerwear"] += 0.08
    winner = max(scores.items(), key=lambda x: x[1])[0]
    return winner, scores


def _apply_super_category_gate(merged: dict[str, float], winner_super: str) -> None:
    allowed: set[str] = set()
    for g, mem in _SUPER_GROUPS.items():
        if g == winner_super:
            allowed |= mem
    allowed.add("cap")
    for lbl in list(merged.keys()):
        if lbl not in allowed:
            merged[lbl] = max(0.0, merged[lbl] * 0.38)


def _attribute_disagreement(
    item_type: str,
    florence_labels: set[str],
    voted_category: str,
) -> bool:
    icat = CATEGORY_MAP.get(item_type, "Tops")
    top_like = {"shirt", "tshirt", "top", "sweatshirt", "hoodie", "sweater", "vest"}
    bottom_like = {"jeans", "pants", "shorts", "capri", "pyjama", "skirt"}
    if item_type in top_like and ("dress" in florence_labels or "frock" in florence_labels):
        return True
    if item_type in {"dress", "frock"}:
        if not (florence_labels & {"dress", "frock", "skirt"}):
            if florence_labels & top_like:
                return True
    if voted_category == "Bottoms" and icat == "Tops" and item_type in bottom_like:
        return False
    if voted_category == "Tops" and icat == "Bottoms" and item_type in top_like:
        return False
    if voted_category == "Bottoms" and icat == "Tops":
        return True
    if voted_category == "Tops" and icat == "Bottoms":
        return True
    return False


def _to_hsl(r: float, g: float, b: float) -> dict[str, float]:
    r_n = r / 255.0
    g_n = g / 255.0
    b_n = b / 255.0
    mx = max(r_n, g_n, b_n)
    mn = min(r_n, g_n, b_n)
    l = (mx + mn) / 2.0
    if mx == mn:
        return {"h": 0.0, "s": 0.0, "l": float(l)}
    d = mx - mn
    s = d / (2.0 - mx - mn) if l > 0.5 else d / (mx + mn)
    if mx == r_n:
        h = (g_n - b_n) / d + (6.0 if g_n < b_n else 0.0)
    elif mx == g_n:
        h = (b_n - r_n) / d + 2.0
    else:
        h = (r_n - g_n) / d + 4.0
    h = (h / 6.0) * 360.0
    return {"h": float(h), "s": float(s), "l": float(l)}


def _get_fashion_color_entries() -> list:
    global _FASHION_COLOR_ENTRIES
    if _FASHION_COLOR_ENTRIES is None:
        _FASHION_COLOR_ENTRIES = load_fashion_color_entries()
    return _FASHION_COLOR_ENTRIES


def _extract_palette(image: Image.Image, top_k: int = 3) -> list[dict[str, Any]]:
    return extract_garment_palette(
        image,
        top_k=top_k,
        to_hsl_fn=_to_hsl,
        color_entries=_get_fashion_color_entries(),
    )


def _is_monochrome_like(image: Image.Image) -> bool:
    arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    sat = arr[:, :, 1].astype(np.float32)
    # Low median saturation usually indicates black/white or heavily desaturated filter.
    return float(np.median(sat)) < 26.0


def _rebalance_monochrome_primary(palette: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not palette:
        return palette
    # In monochrome frames, large black background often dominates area.
    # Prefer the lightest non-trivial neutral cluster as primary garment color.
    boosted = sorted(
        palette,
        key=lambda c: (float(c.get("hsl", {}).get("l", 0.0)), float(c.get("coverage_pct", 0.0))),
        reverse=True,
    )
    candidate = next((c for c in boosted if float(c.get("coverage_pct", 0.0)) >= 0.12), boosted[0])
    if candidate is palette[0]:
        return palette
    rest = [c for c in palette if c is not candidate]
    total = float(candidate.get("coverage_pct", 0.0)) + sum(float(c.get("coverage_pct", 0.0)) for c in rest)
    if total <= 0:
        return [candidate, *rest]
    reweighted = [candidate, *rest]
    for c in reweighted:
        c["coverage_pct"] = float(c.get("coverage_pct", 0.0)) / total
    return reweighted


def _vote_category(siglip_scores: dict[str, float], florence_labels: set[str], aspect_ratio: float) -> str:
    votes: dict[str, float] = {"Tops": 0.0, "Bottoms": 0.0, "Outerwear": 0.0}
    for label, score in siglip_scores.items():
        cat = CATEGORY_MAP.get(label, "Tops")
        if cat in votes:
            votes[cat] += 0.9 * float(score)
    for label in florence_labels:
        cat = CATEGORY_MAP.get(label, "Tops")
        if cat in votes:
            votes[cat] += 0.55

    # Shape prior for cropped garments:
    # wide-ish crops are often bottoms; tall/narrow crops are often tops/outerwear.
    if aspect_ratio >= 0.88:
        votes["Bottoms"] += 0.35
    elif aspect_ratio <= 0.72:
        votes["Tops"] += 0.2
        votes["Outerwear"] += 0.12

    return max(votes.items(), key=lambda x: x[1])[0]


def _merge_primary(
    siglip_scores: dict[str, float],
    florence_labels: set[str],
    aspect_ratio: float,
    winner_super: str,
) -> tuple[str, float]:
    merged: dict[str, float] = {}
    for k, v in siglip_scores.items():
        merged[k] = max(merged.get(k, 0.0), 0.7 * float(v))
    for label in florence_labels:
        merged[label] = max(merged.get(label, 0.0), merged.get(label, 0.0) + 0.3)

    _apply_super_category_gate(merged, winner_super)

    # Dress/frock false-positives often get pulled toward "shirt" on partial crops.
    # If Florence explicitly sees dress-like cues, bias toward dress-like classes.
    dress_like = {"dress", "frock", "skirt"}
    top_like = {"shirt", "tshirt", "top", "sweatshirt", "hoodie", "sweater"}
    has_dress_like_signal = any(label in florence_labels for label in dress_like)
    if has_dress_like_signal:
        for label in dress_like:
            if label in florence_labels:
                merged[label] = max(merged.get(label, 0.0), merged.get(label, 0.0) + 0.45)
        for label in top_like:
            if label in merged:
                merged[label] = max(0.0, merged[label] - 0.18)
        if "frock" in merged:
            merged["dress"] = max(merged.get("dress", 0.0), merged["frock"] * 0.95)

    # Bottomwear confusion: "shorts/jeans/pants" can be mistaken as "shirt"
    # on tight crops with strong textures. Bias toward bottom labels when present.
    bottom_like = {"shorts", "jeans", "pants", "skirt", "capri", "pyjama"}
    has_bottom_signal = any(label in florence_labels for label in bottom_like)
    if has_bottom_signal:
        for label in bottom_like:
            if label in florence_labels:
                merged[label] = max(merged.get(label, 0.0), merged.get(label, 0.0) + 0.42)
        for label in {"shirt", "tshirt", "top"}:
            if label in merged:
                merged[label] = max(0.0, merged[label] - 0.2)

    # Category vote gate: if overall votes say Bottoms, down-rank pure top labels.
    voted_category = _vote_category(siglip_scores, florence_labels, aspect_ratio)
    if voted_category == "Bottoms":
        for label in {"shirt", "tshirt", "top", "sweatshirt", "hoodie"}:
            if label in merged:
                merged[label] = max(0.0, merged[label] - 0.28)
    elif voted_category in {"Tops", "Outerwear"}:
        for label in {"shorts", "pants", "jeans", "capri", "pyjama"}:
            if label in merged:
                merged[label] = max(0.0, merged[label] - 0.2)

    if not merged:
        return "top", 0.3
    best = max(merged.items(), key=lambda x: x[1])
    return best[0], float(min(1.0, best[1]))


def _decode_image(image_base64: str) -> np.ndarray | None:
    try:
        raw = base64.b64decode(image_base64)
        arr = np.frombuffer(raw, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None


def _load_sam_predictor():
    global _SAM_PREDICTOR, _SAM_ERROR
    if _SAM_PREDICTOR is not None:
        return _SAM_PREDICTOR
    if _SAM_ERROR is not None:
        raise RuntimeError(_SAM_ERROR)
    try:
        from segment_anything import SamPredictor, sam_model_registry  # type: ignore

        model_type = (os.environ.get("SAM_MODEL_TYPE", "").strip() or "vit_b").lower()
        checkpoint = os.environ.get("SAM_CHECKPOINT", "").strip()
        if not checkpoint:
            default_checkpoint = os.path.join(os.path.dirname(__file__), "weights", "sam_vit_b_01ec64.pth")
            if os.path.exists(default_checkpoint):
                checkpoint = default_checkpoint
        if not checkpoint:
            raise RuntimeError("SAM_CHECKPOINT not configured")
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
        _SAM_PREDICTOR = SamPredictor(sam)
        return _SAM_PREDICTOR
    except Exception as exc:  # noqa: BLE001
        _SAM_ERROR = f"sam_unavailable: {exc}"
        raise RuntimeError(_SAM_ERROR) from exc


def _largest_polygon_from_mask(mask: np.ndarray, width: int, height: int) -> list[dict[str, float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.003 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    out: list[dict[str, float]] = []
    for p in approx[:, 0]:
        x = float(np.clip(p[0] / max(1, width), 0.0, 1.0))
        y = float(np.clip(p[1] / max(1, height), 0.0, 1.0))
        out.append({"x": round(x, 4), "y": round(y, 4)})
    return out


def _grabcut_mask_from_bbox(image: np.ndarray, bbox: dict[str, float]) -> np.ndarray:
    h, w = image.shape[:2]
    x1 = int(max(0, min(1, float(bbox.get("x1", 0.0)))) * w)
    y1 = int(max(0, min(1, float(bbox.get("y1", 0.0)))) * h)
    x2 = int(max(0, min(1, float(bbox.get("x2", 1.0)))) * w)
    y2 = int(max(0, min(1, float(bbox.get("y2", 1.0)))) * h)
    x = max(0, min(w - 1, min(x1, x2)))
    y = max(0, min(h - 1, min(y1, y2)))
    bw = max(1, abs(x2 - x1))
    bh = max(1, abs(y2 - y1))
    rect = (x, y, bw, bh)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    out = np.where((mask == 1) | (mask == 3), 255, 0).astype("uint8")
    return out


def _norm_bbox_to_xyxy(b: dict[str, float], w: int, h: int) -> tuple[int, int, int, int]:
    x1 = int(max(0, min(1, float(b.get("x1", 0.0)))) * w)
    y1 = int(max(0, min(1, float(b.get("y1", 0.0)))) * h)
    x2 = int(max(0, min(1, float(b.get("x2", 1.0)))) * w)
    y2 = int(max(0, min(1, float(b.get("y2", 1.0)))) * h)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _iou_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    ua = float(max(1, (ax2 - ax1) * (ay2 - ay1)) + max(1, (bx2 - bx1) * (by2 - by1)) - inter)
    return inter / ua


def _center_inside(box: tuple[int, int, int, int], region: tuple[int, int, int, int]) -> bool:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def _yolo_garments_touching_prompt(
    image: np.ndarray, prompt_xyxy: tuple[int, int, int, int]
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Garment pixel boxes overlapping the SAM prompt (IoU or center-in-prompt)."""
    if not yolov8_configured():
        return []
    try:
        y8 = run_yolov8(image)
    except Exception:
        return []
    if not y8 or not isinstance(y8.get("garments"), list):
        return []
    h, w = image.shape[:2]
    out: list[tuple[tuple[int, int, int, int], float]] = []
    for g in y8["garments"]:
        bb = g.get("bbox")
        if not isinstance(bb, dict):
            continue
        box = _norm_bbox_to_xyxy(bb, w, h)
        if _iou_xyxy(box, prompt_xyxy) < 0.02 and not _center_inside(box, prompt_xyxy):
            continue
        conf = float(g.get("confidence", 0.5))
        out.append((box, conf))
    return out


def _mask_garment_overlap_score(
    m_bool: np.ndarray, garments: list[tuple[tuple[int, int, int, int], float]]
) -> float:
    score = 0.0
    for box, conf in garments:
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(m_bool.shape[1], x2), min(m_bool.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        region = m_bool[y1:y2, x1:x2]
        score += float(region.sum()) * conf
    return score


@app.get("/health")
def health() -> dict[str, Any]:
    models_ok = False
    model_error = None
    try:
        _load_models()
        models_ok = True
    except Exception as exc:  # pragma: no cover - health diagnostics only
        model_error = str(exc)
    return {
        "ok": True,
        "service": "vision-sidecar",
        "yolo_darknet_weights": darknet_yolo_configured(),
        "yolov8_pt": yolov8_configured(),
        "hybrid_infer_ready": models_ok,
        "hybrid_infer_error": model_error,
        "face_model_id": os.environ.get("HF_FACE_MODEL_ID", "").strip() or "arnabdhar/YOLOv8-Face-Detection",
        "sam_available": _SAM_PREDICTOR is not None and _SAM_ERROR is None,
        "sam_error": _SAM_ERROR,
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


@app.post("/infer")
def infer(payload: InferRequest) -> dict[str, Any]:
    try:
        models = _load_models()
        image = _decode_pil(payload.image_base64)
        siglip_scores = _siglip_scores(models, image)
        florence_labels, _florence_caption, pattern_hint = _florence_caption_analysis(models, image)
        w, h = image.size
        aspect_ratio = float(w) / float(max(1, h))
        winner_super, super_scores = _super_category_scores(siglip_scores, florence_labels, aspect_ratio)
        item_type, cls_conf = _merge_primary(siglip_scores, florence_labels, aspect_ratio, winner_super)
        category = CATEGORY_MAP.get(item_type, "Tops")
        voted_category = _vote_category(siglip_scores, florence_labels, aspect_ratio)
        attribute_disagreement = _attribute_disagreement(item_type, florence_labels, voted_category)
        siglip_sorted = sorted(siglip_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        palette = _extract_palette(image, top_k=3)
        if _is_monochrome_like(image):
            palette = _rebalance_monochrome_primary(palette)
        if not palette:
            palette = [
                {
                    "name": "Taupe",
                    "hex": "#8B7355",
                    "hsl": {"h": 34.0, "s": 0.24, "l": 0.44},
                    "coverage_pct": 1.0,
                }
            ]
        color_primary = str(palette[0]["name"])
        color_secondary = str(palette[1]["name"]) if len(palette) > 1 else None
        tag_payload: dict[str, Any] = {
            "fashion_tags": [],
            "fashion_tags_scored": [],
            "fashion_tags_by_layer": {},
            "fashion_descriptor": "",
        }
        if (
            models.fashion_tags_enabled
            and models.fashion_tag_text_features is not None
            and models.siglip_preprocess is not None
        ):
            tag_payload = fashion_tag_inference_payload(
                models.siglip_model,
                models.siglip_preprocess,
                image,
                models.fashion_tag_entries,
                models.fashion_tag_text_features,
                threshold=models.fashion_tag_threshold,
                max_per_layer=models.fashion_tag_max_per_layer,
                total_max=models.fashion_tag_total_max,
            )
        design_tags_ui = [t.replace("_", " ") for t in tag_payload.get("fashion_tags", [])[:28]]
        _uncertain: list[str] = []
        if cls_conf < 0.6:
            _uncertain.append("item_type")
        if attribute_disagreement:
            for fld in ("item_type", "category"):
                if fld not in _uncertain:
                    _uncertain.append(fld)
        result = {
            "schema_version": 3 if tag_payload.get("fashion_tags") else 2,
            "item_type": item_type,
            "subtype": item_type,
            "category": category,
            "color_primary": color_primary,
            "color_secondary": color_secondary,
            "color_primary_hsl": palette[0]["hsl"],
            "color_secondary_hsl": palette[1]["hsl"] if len(palette) > 1 else None,
            "color_palette": palette,
            "dominant_colors": [str(c["hex"]) for c in palette][:3],
            "pattern": pattern_hint,
            "fit": None,
            "material": "Cotton",
            "material_confidence": 0.62,
            "formality": 5,
            "season": ["spring", "summer", "autumn"],
            "season_weights": {"spring": 0.7, "summer": 0.7, "autumn": 0.6, "winter": 0.4},
            "occasions": ["casual_weekend"],
            "style_archetype": "smart_casual",
            "confidence_overall": round(float(max(0.4, cls_conf)), 3),
            "uncertainty": {
                "requires_user_confirmation": bool(cls_conf < 0.6 or attribute_disagreement),
                "uncertain_fields": _uncertain,
                "attribute_disagreement": attribute_disagreement,
                "blocks_embedding": attribute_disagreement,
            },
            "quality": {
                "blur_score": 0.85,
                "lighting_score": 0.85,
                "framing": "flat_lay",
                "occlusion_visible_pct": 0.9,
                "accepted": True,
                "warnings": [],
            },
            **tag_payload,
            **({"gemini_design_tags": design_tags_ui} if design_tags_ui else {}),
            "metadata": {
                "provider": "vision-sidecar",
                "model": f"hybrid(yolo+florence+siglip:{models.siglip_model_id})",
                "latency_ms": 0,
                "version": "1.0.0",
                "debug": {
                    "siglip_scores": dict(siglip_sorted),
                    "siglip_top3": [k for k, _ in siglip_sorted[:3]],
                    "florence_labels": sorted(list(florence_labels)),
                    "aspect_ratio": round(aspect_ratio, 3),
                    "category_vote": voted_category,
                    "super_category": winner_super,
                    "super_category_scores": {k: round(v, 4) for k, v in super_scores.items()},
                    "attribute_disagreement": attribute_disagreement,
                    "fashion_tag_count": len(models.fashion_tag_entries),
                    "fashion_tags_enabled": models.fashion_tags_enabled,
                    "color_mask_mode": _mask_mode(),
                },
            },
        }
        return {"ok": True, "result": result}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.post("/faces")
def detect_faces(payload: FaceDetectRequest) -> dict[str, Any]:
    image = _decode_image(payload.image_base64)
    if image is None:
        return {"ok": False, "error": "decode_failed", "faces": []}
    try:
        faces = _detect_faces_hf(image)
        return {"ok": True, "faces": faces, "backend": "hf"}
    except Exception:
        faces = _detect_faces_haar(image)
        return {"ok": True, "faces": faces, "backend": "opencv_haar_fallback"}


@app.post("/sam/segment")
def sam_segment(payload: SamSegmentRequest) -> dict[str, Any]:
    image = _decode_image(payload.image_base64)
    if image is None:
        return {"ok": False, "error": "decode_failed", "segments": []}
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return {"ok": False, "error": "invalid_image", "segments": []}
    segments: list[dict[str, Any]] = []
    backend = "grabcut_fallback"
    try:
        predictor = _load_sam_predictor()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)
        for idx, bbox in enumerate(payload.boxes):
            x1 = float(max(0, min(1, bbox.get("x1", 0.0))) * w)
            y1 = float(max(0, min(1, bbox.get("y1", 0.0))) * h)
            x2 = float(max(0, min(1, bbox.get("x2", 1.0))) * w)
            y2 = float(max(0, min(1, bbox.get("y2", 1.0))) * h)
            input_box = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
            multimask = bool(payload.balanced)
            masks, scores, _ = predictor.predict(box=input_box, multimask_output=multimask)
            scores_np = np.atleast_1d(np.asarray(scores, dtype=np.float64))
            masks_np = np.asarray(masks)
            if masks_np.ndim == 2:
                masks_np = masks_np[np.newaxis, ...]
            prompt_xyxy = (
                int(max(0, min(w - 1, input_box[0]))),
                int(max(0, min(h - 1, input_box[1]))),
                int(max(1, min(w, input_box[2]))),
                int(max(1, min(h, input_box[3]))),
            )
            garments_hint = _yolo_garments_touching_prompt(image, prompt_xyxy)
            best_idx = int(np.argmax(scores_np))
            if garments_hint and masks_np.shape[0] > 1:
                best_score = -1.0
                for mi in range(masks_np.shape[0]):
                    m_bool = masks_np[mi] > 0.5
                    ov = _mask_garment_overlap_score(m_bool, garments_hint)
                    if ov > best_score:
                        best_score = ov
                        best_idx = mi
                if best_score <= 0.0:
                    best_idx = int(np.argmax(scores_np))
            mask = (masks_np[best_idx].astype(np.uint8) * 255)
            polygon = _largest_polygon_from_mask(mask, w, h)
            seg_score = float(scores_np.flat[min(best_idx, max(0, scores_np.size - 1))])
            segments.append(
                {
                    "id": f"seg-{idx+1}",
                    "score": seg_score,
                    "bbox": bbox,
                    "polygon": polygon,
                }
            )
        backend = "sam2"
    except Exception:
        for idx, bbox in enumerate(payload.boxes):
            try:
                mask = _grabcut_mask_from_bbox(image, bbox)
                polygon = _largest_polygon_from_mask(mask, w, h)
            except Exception:
                polygon = []
            segments.append(
                {
                    "id": f"seg-{idx+1}",
                    "score": 0.45,
                    "bbox": bbox,
                    "polygon": polygon,
                }
            )
    return {"ok": True, "backend": backend, "segments": segments}

