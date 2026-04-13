import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, pipeline
from ultralytics import YOLO


LABELS: List[str] = [
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

UPPERWEAR = {"tshirt", "shirt", "top", "sweater", "sweatshirt", "hoodie", "vest"}
BOTTOMWEAR = {"jeans", "pants", "pyjama", "shorts", "capri", "skirt"}
DRESSWEAR = {"dress", "frock"}
OUTERWEAR = {"jacket", "coat", "blazer"}


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _clip_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Tuple[int, int, int, int]:
    ix1 = max(0, min(int(round(x1)), w - 1))
    iy1 = max(0, min(int(round(y1)), h - 1))
    ix2 = max(0, min(int(round(x2)), w))
    iy2 = max(0, min(int(round(y2)), h))
    if ix2 <= ix1:
        ix2 = min(w, ix1 + 1)
    if iy2 <= iy1:
        iy2 = min(h, iy1 + 1)
    return ix1, iy1, ix2, iy2


def _iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def _center(box: Sequence[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _point_in_box(pt: Tuple[float, float], box: Sequence[int]) -> bool:
    x, y = pt
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def _category_for_label(label: str) -> str:
    if label in UPPERWEAR:
        return "upperwear"
    if label in BOTTOMWEAR:
        return "bottomwear"
    if label in DRESSWEAR:
        return "dress"
    if label in OUTERWEAR:
        return "outerwear"
    return "other"


def _dominant_colors_bgr(crop_bgr: np.ndarray, top_k: int = 2) -> List[Tuple[str, float]]:
    if crop_bgr.size == 0:
        return []
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)
    if pixels.shape[0] < 50:
        return []

    # Drop low-information near-gray pixels from clustering dominance.
    a = pixels[:, 1]
    b = pixels[:, 2]
    sat_like = np.sqrt((a - 128.0) ** 2 + (b - 128.0) ** 2)
    keep = sat_like > 8.0
    use = pixels[keep] if np.count_nonzero(keep) > 100 else pixels

    k = min(3, max(1, use.shape[0] // 2000 + 1))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
    _, labels, centers = cv2.kmeans(use, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(-1)

    palette = {
        "black": np.array([20, 128, 128], dtype=np.float32),
        "white": np.array([245, 128, 128], dtype=np.float32),
        "gray": np.array([145, 128, 128], dtype=np.float32),
        "red": np.array([136, 208, 195], dtype=np.float32),
        "orange": np.array([170, 165, 195], dtype=np.float32),
        "yellow": np.array([230, 115, 210], dtype=np.float32),
        "green": np.array([165, 80, 170], dtype=np.float32),
        "olive": np.array([130, 118, 150], dtype=np.float32),
        "blue": np.array([82, 200, 20], dtype=np.float32),
        "navy": np.array([45, 165, 110], dtype=np.float32),
        "purple": np.array([90, 180, 130], dtype=np.float32),
        "pink": np.array([190, 165, 145], dtype=np.float32),
        "brown": np.array([85, 145, 165], dtype=np.float32),
        "beige": np.array([210, 130, 145], dtype=np.float32),
    }

    out: List[Tuple[str, float]] = []
    total = max(1, labels.shape[0])
    for ci in range(k):
        ratio = float(np.count_nonzero(labels == ci)) / float(total)
        c = centers[ci]
        best_name = "unknown"
        best_dist = 1e9
        for name, ref in palette.items():
            dist = float(np.linalg.norm(c - ref))
            if dist < best_dist:
                best_dist = dist
                best_name = name
        out.append((best_name, ratio))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_k]


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    label: Optional[str] = None


class SiglipClassifier:
    def __init__(self, model_id: str, threshold: float = 0.14):
        self.model_id = model_id
        self.threshold = threshold
        self.is_openclip = "marqo-fashionsiglip" in model_id.lower()
        self.candidate_labels = LABELS
        self.local_only = _env_flag("HF_LOCAL_FILES_ONLY", default=False)
        if self.is_openclip:
            import open_clip

            self.open_clip = open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(f"hf-hub:{model_id}")
            self.tokenizer = open_clip.get_tokenizer(f"hf-hub:{model_id}")
            self.text_tokens = self.tokenizer(self.candidate_labels)
            self.model.eval()
        else:
            self.clf = pipeline(
                "zero-shot-image-classification",
                model=model_id,
                device=-1,
                local_files_only=self.local_only,
            )

    def predict(self, image: Image.Image) -> Dict[str, float]:
        if self.is_openclip:
            with torch.no_grad():
                image_tensor = self.preprocess(image).unsqueeze(0)
                image_features = self.model.encode_image(image_tensor, normalize=True)
                text_features = self.model.encode_text(self.text_tokens, normalize=True)
                probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0].tolist()
            scores = {self.candidate_labels[i]: float(probs[i]) for i in range(len(self.candidate_labels))}
            out = {k: v for k, v in scores.items() if v >= self.threshold}
            if not out:
                top_i = int(np.argmax(probs))
                out = {self.candidate_labels[top_i]: float(probs[top_i])}
            return out

        result = self.clf(image, candidate_labels=self.candidate_labels)
        scored = {r["label"]: float(r["score"]) for r in result}
        out = {k: v for k, v in scored.items() if v >= self.threshold}
        if not out and result:
            out = {result[0]["label"]: float(result[0]["score"])}
        return out


class FlorenceClassifier:
    def __init__(self, model_id: str):
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, local_files_only=_env_flag("HF_LOCAL_FILES_ONLY", default=False)
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, local_files_only=_env_flag("HF_LOCAL_FILES_ONLY", default=False)
        ).eval()

    def predict_labels(self, image: Image.Image) -> Set[str]:
        task_prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=64,
                num_beams=2,
            )
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].lower()
        found = set()
        for label in LABELS:
            if label in text:
                found.add(label)
        if "frock" in found:
            found.add("dress")
        return found


def _select_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def _detect_faces(image_bgr: np.ndarray, detector: cv2.CascadeClassifier) -> List[Detection]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(24, 24))
    out: List[Detection] = []
    for (x, y, w, h) in faces:
        out.append(Detection(bbox=(int(x), int(y), int(x + w), int(y + h)), confidence=1.0, label="face"))
    return out


def _blur_faces(
    image_bgr: np.ndarray,
    faces: List[Detection],
    method: str = "gaussian",
    kernel: int = 41,
    roi_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    if kernel % 2 == 0:
        kernel += 1
    for det in faces:
        x1, y1, x2, y2 = det.bbox
        if roi_boxes:
            if max((_iou((x1, y1, x2, y2), rb) for rb in roi_boxes), default=0.0) <= 0.0:
                continue
        x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, w, h)
        face = out[y1:y2, x1:x2]
        if face.size == 0:
            continue
        if method == "pixelate":
            small = cv2.resize(face, (max(4, face.shape[1] // 12), max(4, face.shape[0] // 12)), interpolation=cv2.INTER_LINEAR)
            blur = cv2.resize(small, (face.shape[1], face.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            blur = cv2.GaussianBlur(face, (kernel, kernel), sigmaX=0)
        out[y1:y2, x1:x2] = blur
    return out


def _parse_roi_list(roi_args: List[str]) -> List[Tuple[int, int, int, int]]:
    rois: List[Tuple[int, int, int, int]] = []
    for raw in roi_args:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 4:
            continue
        try:
            x, y, w, h = [int(float(p)) for p in parts]
            rois.append((x, y, x + w, y + h))
        except Exception:
            continue
    return rois


def _interactive_rois(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    boxes = cv2.selectROIs("Select regions and press Enter", image_bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    out: List[Tuple[int, int, int, int]] = []
    for b in boxes:
        x, y, w, h = [int(v) for v in b]
        if w > 1 and h > 1:
            out.append((x, y, x + w, y + h))
    return out


def _associate_to_person(garment_box: Tuple[int, int, int, int], persons: List[Detection]) -> Optional[int]:
    if not persons:
        return None
    c = _center(garment_box)
    best_idx = None
    best_score = -1.0
    for i, p in enumerate(persons):
        iou = _iou(garment_box, p.bbox)
        score = iou + (0.15 if _point_in_box(c, p.bbox) else 0.0)
        if score > best_score:
            best_idx = i
            best_score = score
    return best_idx if best_score > 0.0 else None


def _pick_primary(siglip_scores: Dict[str, float], florence_labels: Set[str]) -> Tuple[str, float]:
    merged: Dict[str, float] = {}
    for lbl, sc in siglip_scores.items():
        merged[lbl] = max(merged.get(lbl, 0.0), 0.65 * sc)
    for lbl in florence_labels:
        merged[lbl] = max(merged.get(lbl, 0.0), merged.get(lbl, 0.0) + 0.35)
    if not merged:
        return "unknown", 0.0
    best = max(merged.items(), key=lambda x: x[1])
    return best[0], float(min(1.0, best[1]))


def _detect_yolo(model: YOLO, image_path: Path, conf: float) -> List[Detection]:
    result = model.predict(source=str(image_path), conf=conf, verbose=False)[0]
    names = result.names
    out: List[Detection] = []
    if result.boxes is None:
        return out
    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        score = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        out.append(Detection(bbox=(int(x1), int(y1), int(x2), int(y2)), confidence=score, label=str(names.get(cls_id, ""))))
    return out


def run_pipeline(
    image_path: Path,
    out_dir: Path,
    garment_model_path: Path,
    person_model_path: str,
    siglip_model_id: str,
    florence_model_id: str,
    roi_boxes: List[Tuple[int, int, int, int]],
    interactive_roi: bool,
    blur_faces: bool,
    blur_method: str,
) -> Dict[str, object]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    h, w = image_bgr.shape[:2]

    if interactive_roi:
        roi_boxes = _interactive_rois(image_bgr)

    person_model = YOLO(person_model_path)
    garment_model = YOLO(str(garment_model_path))
    siglip = SiglipClassifier(siglip_model_id, threshold=0.14)
    florence = FlorenceClassifier(florence_model_id)
    face_detector = _select_face_detector()

    person_raw = _detect_yolo(person_model, image_path, conf=0.25)
    persons = [d for d in person_raw if (d.label or "").lower() == "person"]
    garments = _detect_yolo(garment_model, image_path, conf=0.2)

    if roi_boxes:
        garments = [g for g in garments if max((_iou(g.bbox, rb) for rb in roi_boxes), default=0.0) > 0.0]
        persons = [p for p in persons if max((_iou(p.bbox, rb) for rb in roi_boxes), default=0.0) > 0.0]

    pil_img = Image.open(image_path).convert("RGB")
    person_items: Dict[int, List[dict]] = {i: [] for i in range(len(persons))}
    unassigned: List[dict] = []

    for g in garments:
        x1, y1, x2, y2 = _clip_box(*g.bbox, w, h)
        crop_pil = pil_img.crop((x1, y1, x2, y2))
        crop_bgr = image_bgr[y1:y2, x1:x2]
        siglip_scores = siglip.predict(crop_pil)
        florence_labels = florence.predict_labels(crop_pil)
        primary_label, conf = _pick_primary(siglip_scores, florence_labels)
        color_preds = _dominant_colors_bgr(crop_bgr, top_k=2)
        assoc = _associate_to_person((x1, y1, x2, y2), persons)

        item = {
            "bbox": [x1, y1, x2, y2],
            "detector_label": g.label or "",
            "detector_confidence": round(g.confidence, 4),
            "category": _category_for_label(primary_label),
            "type": primary_label,
            "classification_confidence": round(conf, 4),
            "siglip_scores": {k: round(v, 4) for k, v in sorted(siglip_scores.items(), key=lambda x: x[1], reverse=True)[:5]},
            "florence_labels": sorted(list(florence_labels)),
            "colors": [{"name": c, "ratio": round(r, 4)} for c, r in color_preds],
            "primary_color": color_preds[0][0] if color_preds else "unknown",
            "is_multicolor": bool(len(color_preds) > 1 and color_preds[1][1] >= 0.2),
        }
        if assoc is None:
            unassigned.append(item)
        else:
            person_items[assoc].append(item)

    faces = _detect_faces(image_bgr, face_detector)
    blurred_out = image_bgr
    if blur_faces:
        blurred_out = _blur_faces(image_bgr, faces, method=blur_method, roi_boxes=roi_boxes)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_img = out_dir / f"{image_path.stem}.anonymized{image_path.suffix}"
    out_json = out_dir / f"{image_path.stem}.pipeline.json"
    cv2.imwrite(str(out_img), blurred_out)

    payload = {
        "image_path": str(image_path),
        "image_size": {"width": w, "height": h},
        "models": {
            "person_detector": person_model_path,
            "garment_detector": str(garment_model_path),
            "siglip": siglip_model_id,
            "florence": florence_model_id,
            "face_detector": "opencv_haar_frontalface_default",
        },
        "roi_boxes_xyxy": [list(b) for b in roi_boxes],
        "people": [
            {
                "person_id": i,
                "bbox": list(p.bbox),
                "confidence": round(p.confidence, 4),
                "garments": person_items.get(i, []),
            }
            for i, p in enumerate(persons)
        ],
        "unassigned_garments": unassigned,
        "faces_detected": [{"bbox": list(f.bbox), "confidence": f.confidence} for f in faces],
        "anonymized_image_path": str(out_img),
    }
    out_json.write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Vestir end-to-end pipeline: detect, classify, color, associate, blur faces.")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--out-dir", default="/tmp/vestir_pipeline_out")
    parser.add_argument(
        "--garment-model-path",
        default=os.getenv("YOLO_MODEL_PATH", "/Users/akarshsaklani/Desktop/vestir/yolo11_fashion_best.pt"),
    )
    parser.add_argument("--person-model-path", default=os.getenv("PERSON_MODEL_PATH", "yolov8n.pt"))
    parser.add_argument(
        "--siglip-model-id",
        default=(
            os.getenv("SIGLIP_FASHION_MODEL_ID", "").strip()
            or os.getenv("SIGLIP_MODEL_ID", "").strip()
            or "Marqo/marqo-fashionSigLIP"
        ),
    )
    parser.add_argument(
        "--florence-model-id",
        default=(
            os.getenv("FLORENCE_FASHION_MODEL_ID", "").strip()
            or os.getenv("FLORENCE_MODEL_ID", "").strip()
            or "microsoft/Florence-2-base"
        ),
    )
    parser.add_argument(
        "--roi",
        action="append",
        default=[],
        help="Optional ROI as x,y,w,h. Repeat for multiple.",
    )
    parser.add_argument("--interactive-roi", action="store_true", help="Use GUI to select one or more regions.")
    parser.add_argument("--no-face-blur", action="store_true")
    parser.add_argument("--blur-method", choices=("gaussian", "pixelate"), default="gaussian")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    garment_model_path = Path(args.garment_model_path).expanduser().resolve()
    if not image_path.exists():
        raise RuntimeError(f"Input image does not exist: {image_path}")
    if not garment_model_path.exists():
        raise RuntimeError(f"Garment model not found: {garment_model_path}")

    if args.siglip_model_id.lower() == "google/siglip-base-patch16-224":
        raise RuntimeError(
            "Base SigLIP is disabled in this pipeline. Use your fashion-finetuned SigLIP model id "
            "(set SIGLIP_FASHION_MODEL_ID or pass --siglip-model-id)."
        )

    roi_boxes = _parse_roi_list(args.roi)
    payload = run_pipeline(
        image_path=image_path,
        out_dir=out_dir,
        garment_model_path=garment_model_path,
        person_model_path=args.person_model_path,
        siglip_model_id=args.siglip_model_id,
        florence_model_id=args.florence_model_id,
        roi_boxes=roi_boxes,
        interactive_roi=args.interactive_roi,
        blur_faces=not args.no_face_blur,
        blur_method=args.blur_method,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

