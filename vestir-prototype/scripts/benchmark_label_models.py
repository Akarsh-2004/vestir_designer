import argparse
import base64
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
import torch
from dotenv import load_dotenv
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, pipeline
from ultralytics import YOLO


DATASET_DIR = Path("/Users/akarshsaklani/Desktop/vestir/for label testing dataset ")
ROOT_DIR = Path("/Users/akarshsaklani/Desktop/vestir")
YOLO_MODEL_PATH = Path(
    os.getenv("YOLO_MODEL_PATH", str(ROOT_DIR / "yolo11_fashion_best.pt")) or str(ROOT_DIR / "yolo11_fashion_best.pt")
)

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

VISION_TAG_PROMPT = (
    "List clothing item types seen in this image as JSON array of lowercase words only. "
    "Allowed labels: " + ", ".join(LABELS)
)

ALIASES = {
    "tee": "tshirt",
    "t shirt": "tshirt",
    "t-shirt": "tshirt",
    "trouser": "pants",
    "trousers": "pants",
    "gown": "dress",
    "giown": "dress",
    "frok": "frock",
    "seater": "sweater",
    "sweat shirt": "sweatshirt",
    "pj": "pyjama",
}


def normalize_text(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_labels(text: str) -> Set[str]:
    text = normalize_text(text)
    for k, v in ALIASES.items():
        text = text.replace(k, v)
    found = set()
    for label in LABELS:
        if label in text:
            found.add(label)
    if "frock" in found:
        found.add("dress")
    return found


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class Metrics:
    hit_rate: float
    exact_match: float
    avg_jaccard: float
    avg_ai_semantic: Optional[float] = None


def evaluate(preds: Dict[str, Set[str]], gt: Dict[str, Set[str]]) -> Metrics:
    names = sorted(gt.keys())
    hit = 0
    exact = 0
    jac_sum = 0.0
    for n in names:
        p = preds.get(n, set())
        g = gt[n]
        if p & g:
            hit += 1
        if p == g:
            exact += 1
        jac_sum += jaccard(p, g)
    total = max(1, len(names))
    return Metrics(hit / total, exact / total, jac_sum / total)


def _parse_json_from_llm(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _normalize_score_dict(raw: dict, expected_keys: List[str]) -> Dict[str, float]:
    """Map LLM JSON to expected model keys (handles nested 'scores', case drift)."""
    if not isinstance(raw, dict):
        return {}
    if "scores" in raw and isinstance(raw["scores"], dict):
        raw = raw["scores"]
    lowered = {str(k).strip().lower(): v for k, v in raw.items()}
    out: Dict[str, float] = {}
    for k in expected_keys:
        lk = k.lower()
        v = lowered.get(lk)
        if v is None:
            for cand, val in lowered.items():
                if lk in cand or cand in lk:
                    v = val
                    break
        if isinstance(v, (int, float)):
            out[k] = max(0.0, min(1.0, float(v)))
        else:
            out[k] = 0.0
    return out


def _parse_retry_after_seconds(message: str) -> Optional[float]:
    m = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)\s*s", message, re.I)
    if m:
        return float(m.group(1)) + 0.5
    return None


def _gemini_post_generate(url: str, payload: dict, timeout: int = 120, max_retries: int = 12) -> dict:
    """POST to Gemini generateContent; honor 429 Retry-After / message and back off."""
    last_text = ""
    for attempt in range(max_retries):
        response = requests.post(url, json=payload, timeout=timeout)
        last_text = response.text[:1200]
        if response.status_code == 200:
            return response.json()
        if response.status_code == 429:
            try:
                err = response.json()
                msg = json.dumps(err)
            except Exception:
                msg = response.text
            wait = _parse_retry_after_seconds(msg) or min(90.0, 5.0 * (2**attempt))
            if os.getenv("BENCHMARK_VERBOSE_GEMINI", "").strip() == "1":
                print(f"[gemini] 429 backoff sleep {wait:.1f}s (attempt {attempt + 1})", flush=True)
            time.sleep(wait)
            continue
        if response.status_code in (503, 500):
            time.sleep(min(30.0, 2.0 * (attempt + 1)))
            continue
        response.raise_for_status()
    raise RuntimeError(f"Gemini HTTP failed after retries: {last_text}")


def _gemini_extract_text(data: dict) -> str:
    if not isinstance(data, dict):
        raise ValueError("Invalid Gemini response")
    fb = data.get("promptFeedback")
    if isinstance(fb, dict) and fb.get("blockReason"):
        raise RuntimeError(f"Gemini blocked prompt: {fb.get('blockReason')}")
    cands = data.get("candidates") or []
    if not cands:
        raise RuntimeError(f"No candidates: {data.get('error', data)}")
    c0 = cands[0]
    reason = c0.get("finishReason")
    if reason and reason in ("SAFETY", "RECITATION", "BLOCKLIST"):
        raise RuntimeError(f"Gemini finishReason={reason}")
    content = c0.get("content") or {}
    parts = content.get("parts") or []
    texts = []
    for p in parts:
        if isinstance(p, dict) and p.get("text"):
            texts.append(p["text"])
    if not texts:
        raise RuntimeError(f"No text in response: {json.dumps(c0)[:500]}")
    return "\n".join(texts)


def ai_compare_models_to_filename(
    api_key: str,
    filename_stem: str,
    model_predictions: Dict[str, Set[str]],
    model: str = "gemini-2.5-flash",
    timeout: int = 45,
) -> Dict[str, float]:
    """
    One Gemini call per image: score how well each model's tag set matches the
    filename ground truth, allowing typos and synonyms (frock≈dress, tee≈tshirt, etc.).
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload_sets = {k: sorted(v) for k, v in model_predictions.items()}
    prompt = f"""You are scoring clothing label agreement for a benchmark.

Ground truth is ONLY the filename text below (it may have spelling mistakes, abbreviations, or informal words). Do not assume anything not written there.

Filename text (ground truth): "{filename_stem}"

Each model produced a set of canonical clothing tags:
{json.dumps(payload_sets, indent=2)}

For EACH model key above, output one number from 0.0 to 1.0:
- 1.0 = predicted tags describe the same clothing items as the filename (synonyms and typos OK).
- 0.5 = partial overlap (some items right, some wrong or missing).
- 0.0 = clearly wrong or unrelated.

Rules:
- Treat frock, dress, gown as the same class; tee/t-shirt/tshirt/shirt/top as overlapping when context fits; pyjama/pajama/lounge pants as overlapping; jeans/pants/trousers when appropriate.
- If filename mentions multiple items, reward predictions that cover those items even if wording differs.
- If filename is ambiguous, be lenient when predictions are plausible.

Return ONLY a JSON object with the same keys as the models, float values. Example: {{"yolo":0.8,"siglip":0.7}}
"""
    expected = list(model_predictions.keys())
    gen_with_schema = {
        "temperature": 0.1,
        "maxOutputTokens": 512,
        "responseMimeType": "application/json",
        "responseSchema": {
            "type": "object",
            "properties": {name: {"type": "number"} for name in expected},
            "required": expected,
        },
    }
    gen_json_only = {
        "temperature": 0.1,
        "maxOutputTokens": 512,
        "responseMimeType": "application/json",
    }
    gen_plain = {"temperature": 0.1, "maxOutputTokens": 512}
    last_err: Optional[Exception] = None
    for gen_cfg in (gen_with_schema, gen_json_only, gen_plain):
        payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen_cfg}
        try:
            data = _gemini_post_generate(url, payload, timeout=timeout)
            text = _gemini_extract_text(data)
            parsed = _parse_json_from_llm(text)
            if isinstance(parsed, list):
                last_err = RuntimeError(f"Expected JSON object, got list: {text[:200]}")
                continue
            return _normalize_score_dict(parsed, expected)
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(f"Gemini compare failed after retries: {last_err}")


def _build_semantic_batch_input(
    images_chunk: List[Path],
    all_preds: Dict[str, Dict[str, Set[str]]],
) -> Dict[str, dict]:
    model_keys = list(all_preds.keys())
    batch_input: Dict[str, dict] = {}
    for img in images_chunk:
        batch_input[img.name] = {
            "ground_truth_filename_text": img.stem,
            "predictions": {m: sorted(all_preds[m][img.name]) for m in model_keys},
        }
    return batch_input


def _semantic_compare_prompt_text(batch_input: Dict[str, dict], model_keys: List[str]) -> str:
    return f"""You are scoring clothing label agreement for a benchmark batch.

You will receive one JSON object "images" keyed by image filename. Each entry has:
- ground_truth_filename_text: human label text (may have typos)
- predictions: maps model name -> list of canonical clothing tags

For EVERY image key, output scores 0.0-1.0 per model:
- 1.0 = that model's tags match the filename description (synonyms/typos OK: frock≈dress, tee≈tshirt, pyjama≈pajama, jeans≈pants when appropriate).
- 0.5 = partial match.
- 0.0 = wrong or missing items.

Return ONLY a JSON object with the SAME top-level keys as the input image filenames.
Each value must be a JSON object with exactly these keys: {json.dumps(model_keys)}
and float values.

Input:
{json.dumps({"images": batch_input}, indent=2)}
"""


def _aggregate_parsed_semantic_scores(
    parsed: dict,
    images_chunk: List[Path],
    model_keys: List[str],
    verbose_ai_errors: bool,
) -> Dict[str, float]:
    """Mean score per model over images_chunk (missing rows count as 0 for that image)."""
    sums: Dict[str, float] = {k: 0.0 for k in model_keys}
    n = len(images_chunk)
    for img in images_chunk:
        row = _find_row_for_image(parsed, img)
        if row is None:
            if verbose_ai_errors:
                print(f"[ai-compare batch] missing row for {img.name}", flush=True)
            continue
        normed = _normalize_score_dict(row, model_keys)
        for k in model_keys:
            sums[k] += normed.get(k, 0.0)
    return {k: sums[k] / max(1, n) for k in model_keys}


def _ollama_chat_semantic_json(
    base_url: str,
    model: str,
    prompt: str,
    timeout: int = 600,
    verbose: bool = False,
) -> dict:
    """Ask Ollama for JSON (text-only; no images). Uses native JSON mode when supported."""
    base = base_url.rstrip("/")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_predict": 32768},
    }
    r = requests.post(f"{base}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    text = (data.get("message") or {}).get("content") or ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if verbose:
            print(f"[ollama semantic] raw (parse fallback): {text[:400]}…", flush=True)
        return _parse_json_from_llm(text)


def collect_ai_semantic_scores_batched_ollama(
    base_url: str,
    ollama_model: str,
    images: List[Path],
    all_preds: Dict[str, Dict[str, Set[str]]],
    verbose_ai_errors: bool = False,
) -> Dict[str, float]:
    """
    Same semantic task as Gemini batch, via local Ollama (text-only). No API quota.
    Chunks large sets so smaller models do not truncate JSON.
    """
    model_keys = list(all_preds.keys())
    chunk_sz = max(1, int(os.getenv("OLLAMA_SEMANTIC_CHUNK", "10") or "10"))
    chunks: List[List[Path]] = [images[i : i + chunk_sz] for i in range(0, len(images), chunk_sz)]
    weighted: List[tuple[int, Dict[str, float]]] = []
    delay_chunk = float(os.getenv("OLLAMA_SEMANTIC_DELAY_SEC", "0") or "0")
    for ci, chunk in enumerate(chunks):
        if ci > 0 and delay_chunk > 0:
            time.sleep(delay_chunk)
        batch_input = _build_semantic_batch_input(chunk, all_preds)
        prompt = _semantic_compare_prompt_text(batch_input, model_keys)
        try:
            parsed = _ollama_chat_semantic_json(
                base_url, ollama_model, prompt, timeout=600, verbose=verbose_ai_errors
            )
            if "images" in parsed and isinstance(parsed["images"], dict):
                parsed = parsed["images"]
            if "scores" in parsed and isinstance(parsed["scores"], dict):
                parsed = parsed["scores"]
        except Exception as exc:
            if verbose_ai_errors:
                print(f"[ollama semantic] chunk {ci + 1}/{len(chunks)} failed: {exc}", flush=True)
            raise
        chunk_avg = _aggregate_parsed_semantic_scores(parsed, chunk, model_keys, verbose_ai_errors)
        weighted.append((len(chunk), chunk_avg))
    total_n = sum(w for w, _ in weighted)
    return {
        k: sum(w * avgs[k] for w, avgs in weighted) / max(1, total_n)
        for k in model_keys
    }


def _find_row_for_image(parsed: dict, img: Path) -> Optional[dict]:
    if not isinstance(parsed, dict):
        return None
    if img.name in parsed and isinstance(parsed[img.name], dict):
        return parsed[img.name]
    if img.stem in parsed and isinstance(parsed[img.stem], dict):
        return parsed[img.stem]
    for key, val in parsed.items():
        if not isinstance(val, dict):
            continue
        if key == img.name or key.endswith(img.name) or img.name in key:
            return val
    return None


def collect_ai_semantic_scores_batched(
    api_key: str,
    images: List[Path],
    all_preds: Dict[str, Dict[str, Set[str]]],
    model_id: str = "gemini-2.5-flash",
    verbose_ai_errors: bool = False,
) -> Dict[str, float]:
    """
    One Gemini call for the whole dataset (fits free-tier RPM much better than 1 call/image).
    Reads GEMINI_API_KEY from environment (load via dotenv from repo .env).
    """
    model_keys = list(all_preds.keys())
    batch_input = _build_semantic_batch_input(images, all_preds)
    prompt = _semantic_compare_prompt_text(batch_input, model_keys)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
    gen_cfg = {
        "temperature": 0.1,
        "maxOutputTokens": 8192,
        "responseMimeType": "application/json",
    }
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen_cfg}
    try:
        data = _gemini_post_generate(url, payload, timeout=180)
        text = _gemini_extract_text(data)
        parsed = _parse_json_from_llm(text)
        if "images" in parsed and isinstance(parsed["images"], dict):
            parsed = parsed["images"]
        if "scores" in parsed and isinstance(parsed["scores"], dict):
            parsed = parsed["scores"]
    except Exception as exc:
        if verbose_ai_errors:
            print(f"[ai-compare batch] {exc}", flush=True)
        raise

    return _aggregate_parsed_semantic_scores(parsed, images, model_keys, verbose_ai_errors)


def collect_ai_semantic_scores_per_image(
    api_key: str,
    images: List[Path],
    all_preds: Dict[str, Dict[str, Set[str]]],
    verbose_ai_errors: bool = False,
    delay_sec: float = 3.6,
    model_id: str = "gemini-2.5-flash",
) -> Dict[str, float]:
    """One API call per image — slow; use delay_sec to reduce 429s on free tier."""
    sums: Dict[str, float] = {k: 0.0 for k in all_preds}
    counts: Dict[str, int] = {k: 0 for k in all_preds}
    first_error: Optional[str] = None
    for idx, img in enumerate(images):
        if idx > 0 and delay_sec > 0:
            time.sleep(delay_sec)
        stem = img.stem
        per_image = {name: preds[img.name] for name, preds in all_preds.items()}
        try:
            scores = ai_compare_models_to_filename(api_key, stem, per_image, model=model_id)
        except Exception as exc:
            if first_error is None:
                first_error = f"{img.name}: {exc!s}"
            if verbose_ai_errors:
                print(f"[ai-compare] {img.name}: {exc}", flush=True)
            scores = {k: 0.0 for k in per_image}
        for k, s in scores.items():
            if k in sums:
                sums[k] += s
                counts[k] += 1
    if first_error and verbose_ai_errors:
        print(f"[ai-compare] First error was: {first_error}", flush=True)
    out_avg = {k: (sums[k] / counts[k] if counts[k] else 0.0) for k in sums}
    if all(v == 0.0 for v in out_avg.values()) and first_error:
        print(
            f"Warning: AI semantic scores are all 0 (every request failed). First error: {first_error}. "
            "Try batched mode (default), increase delay, or enable billing for higher quota.",
            flush=True,
        )
    return out_avg


def collect_ai_semantic_scores(
    images: List[Path],
    all_preds: Dict[str, Dict[str, Set[str]]],
    *,
    backend: str,
    api_key: str,
    ollama_base: str,
    ollama_semantic_model: str,
    verbose_ai_errors: bool = False,
    per_image: bool = False,
    model_id: Optional[str] = None,
) -> Tuple[Dict[str, float], str]:
    """
    Semantic judge: filename text vs each model's tag set.
    backend: gemini (cloud), ollama (local text LLM, chunked JSON), auto (Gemini batch if key else Ollama;
    on Gemini batch failure, try Ollama).
    """
    b = (backend or "auto").strip().lower()
    if b not in ("auto", "gemini", "ollama"):
        b = "auto"
    mid = (model_id or os.getenv("GEMINI_AI_MODEL", "").strip() or "gemini-2.5-flash").strip()

    def _ollama() -> Dict[str, float]:
        return collect_ai_semantic_scores_batched_ollama(
            ollama_base,
            ollama_semantic_model,
            images,
            all_preds,
            verbose_ai_errors=verbose_ai_errors,
        )

    if b == "ollama":
        if per_image:
            print(
                "Note: --ai-per-image applies to Gemini only; using Ollama batched semantic compare.",
                flush=True,
            )
        return _ollama(), "ollama"

    if b == "gemini":
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for semantic-backend=gemini")
        if per_image:
            delay = float(os.getenv("GEMINI_COMPARE_DELAY_SEC", "3.6") or "3.6")
            return (
                collect_ai_semantic_scores_per_image(
                    api_key, images, all_preds, verbose_ai_errors, delay_sec=delay, model_id=mid
                ),
                "gemini",
            )
        try:
            return (
                collect_ai_semantic_scores_batched(
                    api_key, images, all_preds, model_id=mid, verbose_ai_errors=verbose_ai_errors
                ),
                "gemini",
            )
        except Exception as exc:
            print(
                f"Warning: batched AI semantic compare failed ({exc!s}); falling back to per-image with delay.",
                flush=True,
            )
            delay = float(os.getenv("GEMINI_COMPARE_DELAY_SEC", "3.6") or "3.6")
            return (
                collect_ai_semantic_scores_per_image(
                    api_key, images, all_preds, verbose_ai_errors, delay_sec=delay, model_id=mid
                ),
                "gemini",
            )

    # auto
    if per_image and api_key:
        delay = float(os.getenv("GEMINI_COMPARE_DELAY_SEC", "3.6") or "3.6")
        return (
            collect_ai_semantic_scores_per_image(
                api_key, images, all_preds, verbose_ai_errors, delay_sec=delay, model_id=mid
            ),
            "gemini",
        )
    if api_key:
        try:
            return (
                collect_ai_semantic_scores_batched(
                    api_key, images, all_preds, model_id=mid, verbose_ai_errors=verbose_ai_errors
                ),
                "gemini",
            )
        except Exception as exc:
            print(
                f"Warning: Gemini batched semantic compare failed ({exc!s}); trying Ollama…",
                flush=True,
            )
    return _ollama(), "ollama"


def _bench_progress(stage: str, idx: int, total: int) -> None:
    if total <= 0:
        return
    if idx == 0 or (idx + 1) % 5 == 0 or idx + 1 == total:
        print(f"[benchmark] {stage} {idx + 1}/{total} …", flush=True)


def _print_prediction(tag: str, image_name: str, labels: Set[str], enabled: bool) -> None:
    if not enabled:
        return
    print(f"[pred] {tag:10} | {image_name} -> {sorted(labels)}", flush=True)


def predict_yolo(images: List[Path], print_predictions: bool = False) -> Dict[str, Set[str]]:
    print(f"[benchmark] YOLO ({len(images)} images, model={YOLO_MODEL_PATH}) …", flush=True)
    if not YOLO_MODEL_PATH.exists():
        raise RuntimeError(
            f"YOLO model not found at {YOLO_MODEL_PATH}. "
            "Set env YOLO_MODEL_PATH=/abs/path/to/best.pt (or desired checkpoint)."
        )
    model = YOLO(str(YOLO_MODEL_PATH))
    out: Dict[str, Set[str]] = {}
    for idx, img in enumerate(images):
        _bench_progress("YOLO", idx, len(images))
        result = model.predict(source=str(img), conf=0.2, verbose=False)[0]
        labels = set()
        names = result.names
        for cid in result.boxes.cls.tolist() if result.boxes is not None else []:
            labels |= extract_labels(str(names.get(int(cid), "")))
        out[img.name] = labels
        _print_prediction("yolo", img.name, labels, print_predictions)
    return out


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def predict_siglip(images: List[Path], print_predictions: bool = False) -> Dict[str, Set[str]]:
    print(f"[benchmark] SigLIP ({len(images)} images, can be slow on CPU) …", flush=True)
    siglip_model_id = (
        os.getenv("SIGLIP_MODEL_ID", "").strip()
        or os.getenv("SIGLIP_FASHION_MODEL_ID", "").strip()
        or "google/siglip-base-patch16-224"
    )
    if "google/siglip-base-patch16-224" in siglip_model_id:
        print(
            "Warning: using base SigLIP. Set SIGLIP_FASHION_MODEL_ID or SIGLIP_MODEL_ID to your fashion-finetuned model.",
            flush=True,
        )
    else:
        print(f"[benchmark] SigLIP model: {siglip_model_id}", flush=True)
    local_only = _env_flag("HF_LOCAL_FILES_ONLY", default=False)
    candidate_labels = LABELS
    use_embedding_path = "marqo-fashionsiglip" in siglip_model_id.lower()

    if use_embedding_path:
        # Marqo recommends OpenCLIP usage for this model family.
        import open_clip

        model, _, preprocess_val = open_clip.create_model_and_transforms(
            f"hf-hub:{siglip_model_id}"
        )
        tokenizer = open_clip.get_tokenizer(f"hf-hub:{siglip_model_id}")
        text_tokens = tokenizer(candidate_labels)
        model.eval()
    else:
        clf = pipeline(
            "zero-shot-image-classification",
            model=siglip_model_id,
            device=-1,
            local_files_only=local_only,
        )

    out: Dict[str, Set[str]] = {}
    for idx, img in enumerate(images):
        _bench_progress("SigLIP", idx, len(images))
        if use_embedding_path:
            image = Image.open(img).convert("RGB")
            with torch.no_grad():
                image_tensor = preprocess_val(image).unsqueeze(0)
                image_features = model.encode_image(image_tensor, normalize=True)
                text_features = model.encode_text(text_tokens, normalize=True)
                probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]
            labels = {
                candidate_labels[i]
                for i, p in enumerate(probs.tolist())
                if float(p) >= 0.14
            }
            if not labels:
                top_idx = int(torch.argmax(probs).item())
                labels = {candidate_labels[top_idx]}
        else:
            result = clf(str(img), candidate_labels=candidate_labels)
            labels = {r["label"] for r in result if float(r["score"]) >= 0.14}
            if not labels and result:
                labels = {result[0]["label"]}
        out[img.name] = labels
        _print_prediction("siglip", img.name, labels, print_predictions)
    return out


def predict_florence(images: List[Path], print_predictions: bool = False) -> Dict[str, Set[str]]:
    print(
        f"[benchmark] Florence: loading + running {len(images)} images "
        "(often several minutes; not frozen) …",
        flush=True,
    )
    model_id = (
        os.getenv("FLORENCE_MODEL_ID", "").strip()
        or os.getenv("FLORENCE_FASHION_MODEL_ID", "").strip()
        or "microsoft/Florence-2-base"
    )
    if "microsoft/Florence-2-base" in model_id:
        print(
            "Warning: using base Florence. Set FLORENCE_FASHION_MODEL_ID or FLORENCE_MODEL_ID to your fashion-finetuned model.",
            flush=True,
        )
    else:
        print(f"[benchmark] Florence model: {model_id}", flush=True)
    local_only = _env_flag("HF_LOCAL_FILES_ONLY", default=False)
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, local_files_only=local_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, local_files_only=local_only
    ).eval()
    out: Dict[str, Set[str]] = {}
    for idx, img in enumerate(images):
        _bench_progress("Florence", idx, len(images))
        image = Image.open(img).convert("RGB")
        task_prompt = "<MORE_DETAILED_CAPTION>"
        inputs = processor(text=task_prompt, images=image, return_tensors="pt")
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=64,
                num_beams=2,
            )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        labels = extract_labels(generated_text)
        out[img.name] = labels
        _print_prediction("florence", img.name, labels, print_predictions)
    return out


def predict_gemini(images: List[Path], api_key: str) -> Dict[str, Set[str]]:
    """Vision tagging; throttles requests to reduce Gemini free-tier 429s (~20 RPM)."""
    print(
        f"[benchmark] Gemini vision ({len(images)} images, ~{float(os.getenv('GEMINI_IMAGE_DELAY_SEC', '3.6') or '3.6'):.1f}s between calls) …",
        flush=True,
    )
    model = (os.getenv("GEMINI_AI_MODEL", "").strip() or "gemini-2.5-flash").strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    delay = float(os.getenv("GEMINI_IMAGE_DELAY_SEC", "3.6") or "3.6")
    out: Dict[str, Set[str]] = {}
    for idx, img in enumerate(images):
        _bench_progress("Gemini vision", idx, len(images))
        if idx > 0 and delay > 0:
            time.sleep(delay)
        mime = "image/jpeg"
        b64 = base64.b64encode(img.read_bytes()).decode("utf-8")
        prompt = VISION_TAG_PROMPT
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime, "data": b64}},
                ]
            }],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 256},
        }
        labels: Set[str] = set()
        try:
            data = _gemini_post_generate(url, payload, timeout=90)
            text = _gemini_extract_text(data)
            labels = extract_labels(text)
        except Exception:
            labels = set()
        out[img.name] = labels
    return out


def _ollama_local_model_names(base: str) -> Set[str]:
    try:
        r = requests.get(f"{base}/api/tags", timeout=8)
        r.raise_for_status()
        names: Set[str] = set()
        for m in r.json().get("models", []) or []:
            n = (m.get("name") or "").strip()
            if n:
                names.add(n)
                if ":" in n:
                    names.add(n.split(":", 1)[0])
        return names
    except Exception:
        return set()


def _ollama_model_available(requested: str, local: Set[str]) -> bool:
    if not requested:
        return False
    if requested in local:
        return True
    base = requested.split(":", 1)[0]
    for x in local:
        if x == base or x.startswith(base + ":") or base == x.split(":", 1)[0]:
            return True
    return False


def predict_ollama(
    images: List[Path],
    model: str,
    base_url: Optional[str] = None,
    print_predictions: bool = False,
) -> Dict[str, Set[str]]:
    """
    Local VLM via Ollama /api/chat (image + text). Same tagging prompt as Gemini for fair comparison.
    Env: OLLAMA_BASE_URL (default http://127.0.0.1:11434), OLLAMA_DELAY_SEC between images.
    """
    base = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434") or "http://127.0.0.1:11434").rstrip("/")
    delay = float(os.getenv("OLLAMA_DELAY_SEC", "0") or "0")
    timeout_sec = int(os.getenv("OLLAMA_TIMEOUT_SEC", "45") or "45")
    print(f"[benchmark] Ollama `{model}` ({len(images)} images @ {base}) …", flush=True)
    out: Dict[str, Set[str]] = {}
    for idx, img in enumerate(images):
        _bench_progress(f"Ollama {model}", idx, len(images))
        if idx > 0 and delay > 0:
            time.sleep(delay)
        b64 = base64.b64encode(img.read_bytes()).decode("utf-8")
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": VISION_TAG_PROMPT,
                    "images": [b64],
                }
            ],
        }
        labels: Set[str] = set()
        try:
            r = requests.post(f"{base}/api/chat", json=payload, timeout=timeout_sec)
            r.raise_for_status()
            data = r.json()
            text = (data.get("message") or {}).get("content") or ""
            labels = extract_labels(text)
        except Exception:
            labels = set()
        out[img.name] = labels
        _print_prediction("ollama", img.name, labels, print_predictions)
    return out


def hybrid_vote(*pred_maps: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    names = pred_maps[0].keys()
    out: Dict[str, Set[str]] = {}
    for name in names:
        votes: Dict[str, int] = {}
        for pred_map in pred_maps:
            for label in pred_map.get(name, set()):
                votes[label] = votes.get(label, 0) + 1
        keep = {label for label, count in votes.items() if count >= 2}
        if not keep and votes:
            keep = {max(votes.items(), key=lambda x: x[1])[0]}
        out[name] = keep
    return out


def print_metrics(tag: str, m: Metrics):
    ai = f" | ai_semantic={m.avg_ai_semantic:.3f}" if m.avg_ai_semantic is not None else ""
    print(
        f"{tag:14} | hit_rate={m.hit_rate:.3f} | exact_match={m.exact_match:.3f} | avg_jaccard={m.avg_jaccard:.3f}{ai}"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark clothing label models vs filename ground truth.")
    parser.add_argument(
        "--no-ai-compare",
        action="store_true",
        help="Disable AI semantic scoring (Gemini and/or Ollama; see --semantic-backend).",
    )
    parser.add_argument(
        "--semantic-backend",
        choices=("auto", "gemini", "ollama"),
        default=None,
        help="Judge filename vs predicted tags: gemini (API), ollama (local), auto (Gemini if key works else Ollama). "
        "Env: BENCHMARK_SEMANTIC_BACKEND. Ollama model: OLLAMA_SEMANTIC_MODEL (default llama3.2:3b).",
    )
    parser.add_argument(
        "--verbose-ai",
        action="store_true",
        help="Print per-image errors when AI semantic comparison fails (debug).",
    )
    parser.add_argument(
        "--ai-per-image",
        action="store_true",
        help="Use one Gemini call per image for semantic scoring (slow; hits free-tier limits easily). Default is one batched call.",
    )
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Skip local Ollama vision models (OLLAMA_VISION_MODEL_1 / _2).",
    )
    parser.add_argument(
        "--skip-gemini-vision",
        action="store_true",
        help="Skip Gemini image tagging model predictions in benchmark.",
    )
    parser.add_argument(
        "--print-predictions",
        action="store_true",
        help="Print per-image predictions for each model in terminal output.",
    )
    parser.add_argument(
        "--ollama-max-images",
        type=int,
        default=0,
        help="If >0, run Ollama vision only on first N images (rest left empty).",
    )
    args = parser.parse_args()
    load_dotenv(ROOT_DIR / ".env")
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    env_skip_gemini_vision = os.getenv("BENCHMARK_SKIP_GEMINI_VISION", "").strip().lower() in ("1", "true", "yes")
    skip_gemini_vision = bool(args.skip_gemini_vision or env_skip_gemini_vision)
    semantic_backend = (args.semantic_backend or os.getenv("BENCHMARK_SEMANTIC_BACKEND", "auto") or "auto").strip().lower()
    if semantic_backend not in ("auto", "gemini", "ollama"):
        semantic_backend = "auto"
    images = sorted([p for p in DATASET_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        raise RuntimeError(f"No images found in {DATASET_DIR}")
    gt = {img.name: extract_labels(img.stem) for img in images}

    approach_order = [
        "yolo",
        "siglip",
        "florence",
        "gemini",
        "ollama_1",
        "ollama_2",
        "hybrid",
    ]

    latency_seconds: Dict[str, float] = {}
    t0 = time.perf_counter()
    print(f"Images: {len(images)}")

    s = time.perf_counter()
    yolo_preds = predict_yolo(images, print_predictions=args.print_predictions)
    latency_seconds["yolo"] = time.perf_counter() - s

    s = time.perf_counter()
    siglip_preds = predict_siglip(images, print_predictions=args.print_predictions)
    latency_seconds["siglip"] = time.perf_counter() - s

    s = time.perf_counter()
    florence_preds = predict_florence(images, print_predictions=args.print_predictions)
    latency_seconds["florence"] = time.perf_counter() - s
    if skip_gemini_vision:
        s = time.perf_counter()
        gemini_preds = {img.name: set() for img in images}
        latency_seconds["gemini"] = time.perf_counter() - s
        print("Gemini vision: skipped (--skip-gemini-vision or BENCHMARK_SKIP_GEMINI_VISION=1)")
    else:
        s = time.perf_counter()
        gemini_preds = predict_gemini(images, api_key) if api_key else {img.name: set() for img in images}
        latency_seconds["gemini"] = time.perf_counter() - s

    ollama_m1 = (os.getenv("OLLAMA_VISION_MODEL_1", "llama3.2-vision:11b").strip() or "llama3.2-vision:11b")
    ollama_m2 = (os.getenv("OLLAMA_VISION_MODEL_2", "qwen2.5vl:7b").strip() or "qwen2.5vl:7b")
    ollama_base = (os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434") or "http://127.0.0.1:11434").rstrip("/")
    ollama_semantic_model = (
        os.getenv("OLLAMA_SEMANTIC_MODEL") or os.getenv("OLLAMA_MODEL") or "llama3.2:3b"
    ).strip() or "llama3.2:3b"

    if args.skip_ollama:
        s = time.perf_counter()
        ollama_1_preds = {img.name: set() for img in images}
        ollama_2_preds = {img.name: set() for img in images}
        latency_seconds["ollama_1"] = time.perf_counter() - s
        latency_seconds["ollama_2"] = 0.0
        print("Ollama: skipped (--skip-ollama)")
    else:
        local = _ollama_local_model_names(ollama_base)
        for label, mname in (
            ("OLLAMA_VISION_MODEL_1", ollama_m1),
            ("OLLAMA_VISION_MODEL_2", ollama_m2),
        ):
            if not _ollama_model_available(mname, local):
                print(
                    f"Warning: `{mname}` ({label}) not in `ollama list` at {ollama_base}. "
                    f"Run: ollama pull {mname.split(':', 1)[0]} — predictions will be empty.",
                )
        print(f"Ollama models: {ollama_m1} | {ollama_m2} ({ollama_base})")
        ollama_images = images
        if args.ollama_max_images and args.ollama_max_images > 0:
            ollama_images = images[: min(len(images), int(args.ollama_max_images))]
            print(f"Ollama vision image cap: {len(ollama_images)}/{len(images)}")

        s = time.perf_counter()
        ollama_1_part = predict_ollama(
            ollama_images, ollama_m1, ollama_base, print_predictions=args.print_predictions
        )
        ollama_1_preds = {img.name: set() for img in images}
        ollama_1_preds.update(ollama_1_part)
        latency_seconds["ollama_1"] = time.perf_counter() - s

        if ollama_m2 == ollama_m1:
            print("Ollama model_2 == model_1; reusing predictions to speed up.")
            ollama_2_preds = dict(ollama_1_preds)
            latency_seconds["ollama_2"] = 0.0
        else:
            s = time.perf_counter()
            ollama_2_part = predict_ollama(
                ollama_images, ollama_m2, ollama_base, print_predictions=args.print_predictions
            )
            ollama_2_preds = {img.name: set() for img in images}
            ollama_2_preds.update(ollama_2_part)
            latency_seconds["ollama_2"] = time.perf_counter() - s

    hybrid_preds = hybrid_vote(
        yolo_preds,
        siglip_preds,
        florence_preds,
        gemini_preds,
        ollama_1_preds,
        ollama_2_preds,
    )

    pred_maps = {
        "yolo": yolo_preds,
        "siglip": siglip_preds,
        "florence": florence_preds,
        "gemini": gemini_preds,
        "ollama_1": ollama_1_preds,
        "ollama_2": ollama_2_preds,
        "hybrid": hybrid_preds,
    }

    ai_semantic: Optional[Dict[str, float]] = None
    semantic_backend_used: Optional[str] = None
    env_off = os.getenv("BENCHMARK_AI_COMPARE", "").strip().lower() in ("0", "false", "no")
    can_semantic = not args.no_ai_compare and not env_off
    run_ai = False
    if can_semantic:
        if semantic_backend == "gemini" and not api_key:
            print(
                "\nTip: --semantic-backend gemini requires GEMINI_API_KEY. "
                "Use --semantic-backend ollama or auto with Ollama running (OLLAMA_SEMANTIC_MODEL).",
                flush=True,
            )
        else:
            run_ai = True
    if run_ai:
        mode_desc = (
            "Gemini 1 call/image"
            if args.ai_per_image and api_key
            else (
                "Gemini 1 batched call"
                if semantic_backend == "gemini" or (semantic_backend == "auto" and api_key)
                else "Ollama text judge (chunked)"
            )
        )
        print(
            f"\nRunning AI semantic comparison (requested={semantic_backend}; {mode_desc})…",
            flush=True,
        )
        try:
            s = time.perf_counter()
            ai_semantic, semantic_backend_used = collect_ai_semantic_scores(
                images,
                pred_maps,
                backend=semantic_backend,
                api_key=api_key,
                ollama_base=ollama_base,
                ollama_semantic_model=ollama_semantic_model,
                verbose_ai_errors=args.verbose_ai,
                per_image=args.ai_per_image,
            )
            latency_seconds["semantic_compare"] = time.perf_counter() - s
        except Exception as exc:
            print(f"Warning: AI semantic scoring failed: {exc}", flush=True)
            ai_semantic = None
            semantic_backend_used = None
            latency_seconds["semantic_compare"] = 0.0
        else:
            print(f"AI semantic backend used: {semantic_backend_used}", flush=True)
            print("AI semantic averages:", json.dumps({k: round(v, 3) for k, v in ai_semantic.items()}))
    else:
        latency_seconds["semantic_compare"] = 0.0

    scores: Dict[str, Metrics] = {}
    for name in approach_order:
        m = evaluate(pred_maps[name], gt)
        if ai_semantic is not None:
            m = Metrics(m.hit_rate, m.exact_match, m.avg_jaccard, ai_semantic.get(name))
        scores[name] = m

    print("\n=== Benchmark Results (6 approaches + hybrid over all 6) ===")
    for name in approach_order:
        print_metrics(name, scores[name])
    if not run_ai and api_key and (args.no_ai_compare or env_off):
        print("\nTip: AI semantic scoring is off (--no-ai-compare or BENCHMARK_AI_COMPARE=0).")
    if not api_key:
        print(
            "\nTip: Set GEMINI_API_KEY in .env at repo root "
            f"({ROOT_DIR / '.env'}) for Gemini image tagging; optional for semantic scoring if using Ollama/auto.",
        )
    if not args.skip_ollama:
        print(
            "\nTip: Ollama — vision: OLLAMA_VISION_MODEL_1 / _2, OLLAMA_BASE_URL, OLLAMA_DELAY_SEC. "
            "Semantic judge (no vision): OLLAMA_SEMANTIC_MODEL (default llama3.2:3b), OLLAMA_SEMANTIC_CHUNK, "
            "--semantic-backend ollama. Example: ollama pull llama3.2:3b",
        )
    latency_seconds["total"] = time.perf_counter() - t0
    n_images = max(1, len(images))
    print("\n=== Latency Report ===")
    for key in ("yolo", "siglip", "florence", "gemini", "ollama_1", "ollama_2", "semantic_compare", "total"):
        sec = latency_seconds.get(key, 0.0)
        per_img = sec / n_images if key != "total" else sec / n_images
        print(f"{key:16} | total={sec:.2f}s | per_image={per_img:.3f}s")

    report = {
        "dataset_dir": str(DATASET_DIR),
        "num_images": len(images),
        "semantic_compare": {
            "backend_requested": semantic_backend,
            "backend_used": semantic_backend_used,
            "ollama_semantic_model": ollama_semantic_model,
        },
        "ollama": {
            "base_url": ollama_base,
            "model_1": ollama_m1,
            "model_2": ollama_m2,
            "skipped": bool(args.skip_ollama),
        },
        "approach_order": approach_order,
        "latency_seconds": latency_seconds,
        "scores": {
            k: {
                "hit_rate": v.hit_rate,
                "exact_match": v.exact_match,
                "avg_jaccard": v.avg_jaccard,
                **({"avg_ai_semantic": v.avg_ai_semantic} if v.avg_ai_semantic is not None else {}),
            }
            for k, v in scores.items()
        },
    }
    if ai_semantic is not None:
        report["ai_semantic_averages"] = ai_semantic
    out_path = ROOT_DIR / "vestir-prototype" / "scripts" / "benchmark_label_models_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()

