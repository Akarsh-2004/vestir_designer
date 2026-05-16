"""
Fully generative mannequin pipeline.

Stages:
    1. Captioning (BLIP-2 / Florence / Gemini)
    2. Attribute normalization (LLM structured JSON + rule-based fallback)
    3. Prompt construction
    4. Image generation (FLUX try-off LoRA / SDXL)
    5. Post-processing (white background, center, shadow)
    6. Scoring (fashion SigLIP similarity)

Design goals:
    - Each stage implements a small, replaceable interface.
    - Pipeline never raises — it degrades (captioning fails → fallback tags,
      generator fails → retry with simplified prompt → final fallback = masked cutout).
    - Pipeline is stateless; callers pass an image buffer in, get an image buffer out
      plus structured metadata for logging / audit.

This module is intentionally import-light at module scope; heavy imports (torch,
diffusers, transformers) happen lazily inside Stage classes so unit tests can run
on CPU-only boxes without installing every model dep.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image, ImageFilter, ImageOps

_LOG = logging.getLogger("mannequin_pipeline")


def _default_vestir_gemini_flash_model() -> str:
    """Same env as Node `server/index.mjs` (VESTIR_INFER_FLASH_MODEL)."""
    raw = (os.environ.get("VESTIR_INFER_FLASH_MODEL") or "gemini-3-flash-preview").strip()
    return raw or "gemini-3-flash-preview"


# ---------------------------------------------------------------------------
# Shared data classes
# ---------------------------------------------------------------------------


@dataclass
class CaptionResult:
    text: str
    confidence: float
    provider: str
    raw: dict = field(default_factory=dict)


@dataclass
class GarmentAttributes:
    category: Optional[str] = None
    color: Optional[str] = None
    secondary_colors: list[str] = field(default_factory=list)
    fit: Optional[str] = None
    sleeve: Optional[str] = None
    neckline: Optional[str] = None
    length: Optional[str] = None
    pattern: Optional[str] = None
    pattern_details: Optional[str] = None  # free-text (e.g. "evenly spaced dark dots")
    text: Optional[str] = None
    material: Optional[str] = None
    details: list[str] = field(default_factory=list)
    caption_confidence: float = 0.0     # propagated from the captioner
    rejected_caption: bool = False      # true when caption was obviously too vague
    # Provenance + confidence for every visual field. Keys are field names,
    # values are 0..1 (from SigLIP softmax over the label set) or 0.0 if
    # derived from caption/rules only.
    field_confidences: dict[str, float] = field(default_factory=dict)
    field_sources: dict[str, str] = field(default_factory=dict)   # "siglip" | "caption" | "rules"
    field_topk: dict[str, list[list]] = field(default_factory=dict)  # for UI debugging

    def fill_ratio(self) -> float:
        core = [self.category, self.color, self.sleeve, self.neckline, self.fit]
        return sum(1 for x in core if x) / len(core)

    def visual_confidence(self) -> float:
        """Mean of SigLIP confidences for the core visual fields (0 if none)."""
        core = ("category", "color", "pattern", "sleeve", "fit")
        vals = [self.field_confidences.get(f, 0.0) for f in core]
        vals = [v for v in vals if v > 0.0]
        return sum(vals) / len(vals) if vals else 0.0

    def confidence(self) -> float:
        """
        Combined 0..1 confidence. Visual (SigLIP) classification now dominates;
        caption-based signals are a much weaker prior.
        """
        if self.rejected_caption:
            return 0.0
        visual = self.visual_confidence()
        caption = 0.4 * self.caption_confidence + 0.6 * self.fill_ratio()
        # If we have SigLIP on at least 3 fields, trust it heavily.
        trust_visual = sum(1 for v in self.field_confidences.values() if v > 0) >= 3
        if trust_visual:
            return round(0.7 * visual + 0.3 * caption, 3)
        return round(caption, 3)

    def is_sparse(self) -> bool:
        filled = [self.category, self.color, self.sleeve, self.neckline, self.fit]
        return sum(1 for x in filled if x) < 2


@dataclass
class StageLog:
    id: str
    status: str                       # "completed" | "skipped" | "partial" | "failed"
    duration_ms: int
    detail: Optional[str] = None


@dataclass
class CandidateScore:
    index: int
    garment_sim: float
    mannequin_sim: float
    person_sim: float
    margin: float              # mannequin_sim - person_sim
    skin_pixel_ratio: float
    rejected: bool
    reject_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Skin-tone pixel heuristic
# ---------------------------------------------------------------------------


def estimate_skin_pixel_ratio(image: Image.Image) -> float:
    """
    Crude but effective skin detector for rejecting SDXL outputs that still
    render forearms/hands. Works in YCbCr space (skin clusters tightly in
    Cb/Cr regardless of ethnicity) with an HSV cross-check to filter out
    warm-colored fabrics (beige / peach / brown clothes that trigger Cb/Cr).

    Returns fraction (0..1) of pixels classified as skin-like.

    Tuned empirically; thresholds err on the side of false positives so that
    true arms/hands are reliably caught by the pipeline's 3% ratio ceiling.
    """
    im = image.convert("RGB")
    # Downscale for speed — 256² is more than enough signal.
    im.thumbnail((256, 256), Image.Resampling.BILINEAR)
    ycbcr = im.convert("YCbCr")
    hsv = im.convert("HSV")

    y_data = list(ycbcr.getdata())
    hsv_data = list(hsv.getdata())
    rgb_data = list(im.getdata())
    total = len(y_data)
    if total == 0:
        return 0.0

    skin = 0
    for (y, cb, cr), (h, s, v), (r, g, b) in zip(y_data, hsv_data, rgb_data):
        # Classic Hsu/Abdel-Mottaleb YCbCr skin range.
        if not (80 <= y <= 235 and 77 <= cb <= 127 and 133 <= cr <= 173):
            continue
        # HSV cross-check: hue in warm band, modest saturation, not too dark.
        if not (
            (h <= 25 or h >= 230)   # warm hues (PIL HSV hue is 0..255)
            and 20 <= s <= 180
            and v >= 60
        ):
            continue
        # Reject near-white (bg) and near-black.
        if r > 240 and g > 240 and b > 240:
            continue
        if r < 25 and g < 25 and b < 25:
            continue
        # R > G > B is a reliable rule-of-thumb for most skin tones.
        if not (r > g > b or r > g):
            continue
        skin += 1

    return skin / total


@dataclass
class PipelineResult:
    image_bytes: Optional[bytes]
    mime_type: str
    attributes: GarmentAttributes
    attribute_confidence: float
    prompt: str
    negative_prompt: str
    score: Optional[float]
    candidates: list[CandidateScore]
    selected_index: Optional[int]
    fallback_recommended: bool
    fallback_reason: Optional[str]
    stages: list[StageLog]
    total_ms: int
    cache_hit: bool = False
    notes: list[str] = field(default_factory=list)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "mime_type": self.mime_type,
            "attributes": asdict(self.attributes),
            "attribute_confidence": self.attribute_confidence,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "score": self.score,
            "candidates": [asdict(c) for c in self.candidates],
            "selected_index": self.selected_index,
            "fallback_recommended": self.fallback_recommended,
            "fallback_reason": self.fallback_reason,
            "stages": [asdict(s) for s in self.stages],
            "total_ms": self.total_ms,
            "cache_hit": self.cache_hit,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Stage 1: Captioning
# ---------------------------------------------------------------------------


class Captioner:
    """Strategy interface. Implementations return CaptionResult or raise."""

    name: str = "base"

    def caption(self, image: Image.Image) -> CaptionResult:
        raise NotImplementedError


class GeminiCaptioner(Captioner):
    """Uses Google Generative AI (already configured in repo's .env)."""

    name = "gemini"

    _PROMPT = (
        "You are a fashion catalog annotator. Describe ONLY the garments visible "
        "in this photo for a product listing. One flowing sentence. Include: "
        "garment type, primary color, fit, sleeve type, neckline, length, "
        "patterns, any visible text/logo, fabric guess. Do NOT mention face, "
        "gender, pose, or background."
    )

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or _default_vestir_gemini_flash_model()

    def caption(self, image: Image.Image) -> CaptionResult:
        import google.generativeai as genai  # type: ignore

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.model)
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=92)
        resp = model.generate_content(
            [{"mime_type": "image/jpeg", "data": buf.getvalue()}, self._PROMPT]
        )
        text = (resp.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned empty caption")
        return CaptionResult(text=text, confidence=0.85, provider=self.name)


class Blip2Captioner(Captioner):
    """Fallback captioner using Salesforce/blip2-flan-t5-xl (lazy load)."""

    name = "blip2"

    def __init__(self, model_id: str = "Salesforce/blip2-opt-2.7b") -> None:
        self.model_id = model_id
        self._model = None
        self._processor = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        self._processor = Blip2Processor.from_pretrained(self.model_id)
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        if torch.cuda.is_available():
            self._model = self._model.to("cuda")

    def caption(self, image: Image.Image) -> CaptionResult:
        import torch
        self._ensure()
        assert self._processor is not None and self._model is not None
        prompt = "Question: describe the garment in detail. Answer:"
        inputs = self._processor(image, text=prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=96)
        text = self._processor.decode(out[0], skip_special_tokens=True).strip()
        return CaptionResult(text=text, confidence=0.55, provider=self.name)


class HeuristicCaptioner(Captioner):
    """Never-fail fallback. Produces a minimal caption from dominant color."""

    name = "heuristic"

    def caption(self, image: Image.Image) -> CaptionResult:
        small = image.convert("RGB").resize((32, 32))
        px = small.getdata()
        r = sum(p[0] for p in px) // len(px)
        g = sum(p[1] for p in px) // len(px)
        b = sum(p[2] for p in px) // len(px)
        text = f"a garment in rgb({r},{g},{b}) dominant color"
        return CaptionResult(text=text, confidence=0.15, provider=self.name)


def make_default_captioner() -> Captioner:
    """Gemini → BLIP-2 → heuristic chain wrapped in one Captioner."""

    class ChainCaptioner(Captioner):
        name = "chain"

        def caption(self, image: Image.Image) -> CaptionResult:
            errors: list[str] = []
            for impl_factory in (
                lambda: GeminiCaptioner(),
                lambda: Blip2Captioner(),
            ):
                try:
                    return impl_factory().caption(image)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{impl_factory.__name__}: {exc}")
                    _LOG.warning("captioner failed: %s", exc)
            _LOG.warning("falling back to heuristic captioner; errors=%s", errors)
            return HeuristicCaptioner().caption(image)

    return ChainCaptioner()


# ---------------------------------------------------------------------------
# Stage 2: Caption → GarmentAttributes
# ---------------------------------------------------------------------------


_SYNONYMS: dict[str, str] = {
    "tee": "t-shirt",
    "tees": "t-shirt",
    "t shirt": "t-shirt",
    "tshirt": "t-shirt",
    "jumper": "sweater",
    "pullover": "sweater",
    "coat": "jacket",
    "trousers": "pants",
    "slacks": "pants",
    "pant": "pants",
    "denim": "jeans",
}

_HUMAN_TOKENS = re.compile(
    r"\b(man|men|woman|women|boy|girl|lady|guy|model|person|people|face|smile|pose)\b",
    re.IGNORECASE,
)


def _clean_human_terms(text: str) -> str:
    return _HUMAN_TOKENS.sub("", text).strip()


def _apply_synonyms(text: str) -> str:
    t = text.lower()
    for src, dst in _SYNONYMS.items():
        t = re.sub(rf"\b{re.escape(src)}\b", dst, t)
    return t


class Normalizer:
    """Strategy interface. Returns GarmentAttributes."""

    name: str = "base"

    def normalize(
        self,
        caption: CaptionResult,
        image: Optional[Image.Image] = None,
    ) -> GarmentAttributes:
        raise NotImplementedError


class GeminiJsonNormalizer(Normalizer):
    """Uses Gemini to coerce the caption into strict JSON (single call, cheap)."""

    name = "gemini-json"

    _SCHEMA = (
        "Return STRICT JSON matching this schema:"
        '{"category": str, "color": str, "secondary_colors": [str], "fit": str, '
        '"sleeve": str, "neckline": str, "length": str, "pattern": str, '
        '"text": str, "material": str, "details": [str]}'
        "\nUnknown fields must be empty strings or empty arrays. "
        "Do NOT include person / body / pose / background fields. "
        "Category must use common nouns (t-shirt, kurta, hoodie, dress, jeans, etc.)."
    )

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or _default_vestir_gemini_flash_model()

    def normalize(
        self,
        caption: CaptionResult,
        image: Optional[Image.Image] = None,
    ) -> GarmentAttributes:
        import google.generativeai as genai  # type: ignore

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.model)
        prompt = (
            f"Caption: {_clean_human_terms(_apply_synonyms(caption.text))}\n\n"
            f"{self._SCHEMA}"
        )
        resp = model.generate_content(prompt)
        raw = (resp.text or "").strip()
        # Extract JSON even if the model wrapped it in code fences.
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise RuntimeError(f"Gemini did not return JSON: {raw[:160]}")
        data = json.loads(m.group(0))
        return GarmentAttributes(
            category=(data.get("category") or "").strip().lower() or None,
            color=(data.get("color") or "").strip().lower() or None,
            secondary_colors=[s for s in (data.get("secondary_colors") or []) if s],
            fit=(data.get("fit") or "").strip().lower() or None,
            sleeve=(data.get("sleeve") or "").strip().lower() or None,
            neckline=(data.get("neckline") or "").strip().lower() or None,
            length=(data.get("length") or "").strip().lower() or None,
            pattern=(data.get("pattern") or "").strip().lower() or None,
            text=(data.get("text") or "").strip() or None,
            material=(data.get("material") or "").strip().lower() or None,
            details=[d for d in (data.get("details") or []) if d],
        )


class RuleBasedNormalizer(Normalizer):
    """Backup: regex heuristics. Never raises."""

    name = "rules"

    _CATEGORY_TOKENS = (
        "t-shirt", "shirt", "blouse", "dress", "jeans", "pants",
        "hoodie", "sweater", "jacket", "kurta", "skirt", "shorts",
        "suit", "blazer",
    )

    _COLOR_TOKENS = (
        "black", "white", "grey", "gray", "charcoal", "navy", "blue",
        "red", "pink", "purple", "green", "olive", "beige", "cream",
        "yellow", "orange", "brown",
    )

    def normalize(
        self,
        caption: CaptionResult,
        image: Optional[Image.Image] = None,
    ) -> GarmentAttributes:
        text = _apply_synonyms(_clean_human_terms(caption.text)).lower()
        attrs = GarmentAttributes()
        for tok in self._CATEGORY_TOKENS:
            if re.search(rf"\b{re.escape(tok)}\b", text):
                attrs.category = tok
                break
        for tok in self._COLOR_TOKENS:
            if re.search(rf"\b{re.escape(tok)}\b", text):
                attrs.color = tok
                break
        if "long sleeve" in text or "full sleeve" in text:
            attrs.sleeve = "long sleeve"
        elif "short sleeve" in text or "half sleeve" in text:
            attrs.sleeve = "short sleeve"
        elif "sleeveless" in text or "no sleeve" in text:
            attrs.sleeve = "sleeveless"
        if "crew neck" in text:
            attrs.neckline = "crew neck"
        elif "v-neck" in text or "v neck" in text:
            attrs.neckline = "v-neck"
        elif "mandarin" in text or "collar" in text:
            attrs.neckline = "mandarin collar"
        if any(k in text for k in ("polka dot", "polka-dot", " dots ", " dotted ")):
            attrs.pattern = "polka dot"
        elif any(k in text for k in ("stripe", "striped")):
            attrs.pattern = "striped"
        elif "floral" in text:
            attrs.pattern = "floral"
        elif "check" in text or "plaid" in text:
            attrs.pattern = "checked"
        elif "graphic" in text or "print" in text:
            attrs.pattern = "graphic print"
        m = re.search(r"'([^']+)'|\"([^\"]+)\"", caption.text)
        if m:
            attrs.text = (m.group(1) or m.group(2) or "").strip() or None
        if "oversize" in text or "loose" in text:
            attrs.fit = "oversized"
        elif "slim" in text or "fitted" in text:
            attrs.fit = "slim"
        else:
            attrs.fit = "regular"
        # Mark every field provided here as caption-sourced so fusion knows
        # SigLIP should out-rank these.
        for f in ("category", "color", "pattern", "sleeve", "neckline", "fit", "text"):
            if getattr(attrs, f):
                attrs.field_sources[f] = "caption"
        return attrs


# ---------------------------------------------------------------------------
# SigLIP zero-shot attribute classifier
# ---------------------------------------------------------------------------

# Label sets. Every entry must be a SHORT natural-language phrase — SigLIP
# embeddings degrade sharply on technical jargon. Override via env if you
# want to extend vocab (comma-separated).
CATEGORY_LABELS: tuple[str, ...] = (
    "t-shirt", "blouse", "shirt", "polo shirt", "tank top", "crop top",
    "dress", "jumpsuit", "kurta", "hoodie", "sweatshirt", "sweater", "cardigan",
    "jacket", "coat", "blazer", "vest",
    "jeans", "pants", "trousers", "shorts", "skirt", "leggings",
)
COLOR_LABELS: tuple[str, ...] = (
    "white", "off-white", "cream", "beige", "tan",
    "light pink", "pink", "hot pink", "red", "maroon",
    "orange", "yellow", "mustard",
    "light blue", "blue", "navy blue",
    "light green", "green", "olive green",
    "purple", "lavender",
    "brown", "tan brown",
    "grey", "charcoal", "black",
    "multi-color",
)
PATTERN_LABELS: tuple[str, ...] = (
    "solid",
    "polka dot",
    "striped",
    "floral",
    "checked",
    "plaid",
    "graphic print",
    "text print",
    "animal print",
    "abstract print",
    "tie-dye",
    "color-blocked",
    "embroidered",
)
SLEEVE_LABELS: tuple[str, ...] = (
    "long sleeve",
    "three-quarter sleeve",
    "short sleeve",
    "cap sleeve",
    "sleeveless",
    "strapless",
)
FIT_LABELS: tuple[str, ...] = (
    "oversized fit",
    "loose fit",
    "regular fit",
    "slim fit",
    "fitted",
)
NECKLINE_LABELS: tuple[str, ...] = (
    "crew neck",
    "v-neck",
    "round neck",
    "collared",
    "mandarin collar",
    "boat neck",
    "square neck",
    "halter neck",
    "turtleneck",
    "scoop neck",
)
LENGTH_LABELS: tuple[str, ...] = (
    "cropped",
    "waist length",
    "hip length",
    "knee length",
    "mid length",
    "full length",
    "ankle length",
    "mini",
    "midi",
    "maxi",
)


@dataclass
class FieldPrediction:
    field: str
    top_label: str
    confidence: float                  # softmax prob of top-1 over this field
    topk: list[tuple[str, float]]      # [(label, prob), ...] sorted desc
    raw_top_cosine: float              # raw cosine of top-1 (unscaled)
    unknown: bool                      # True if below field threshold


class SiglipAttributeClassifier:
    """
    Zero-shot classifier over the Marqo fashion-SigLIP backbone. Reuses the
    singleton model held by FashionSiglipScorer to avoid double-loading.

    Each field has its own (labels, template, unknown_cosine_threshold). The
    template gives SigLIP the phrasing it was trained on ("a photo of ...")
    which improves separability meaningfully over bare labels.
    """

    # field -> (labels, template, unknown_cosine_threshold, softmax_temperature)
    FIELDS: dict[str, tuple[tuple[str, ...], str, float, float]] = {
        "category": (CATEGORY_LABELS, "a photo of a {label}", 0.15, 0.02),
        "color":    (COLOR_LABELS,    "a photo of a {label} colored garment", 0.15, 0.02),
        "pattern":  (PATTERN_LABELS,  "a photo of a garment with a {label} pattern", 0.14, 0.02),
        "sleeve":   (SLEEVE_LABELS,   "a photo of a garment with {label}", 0.14, 0.02),
        "fit":      (FIT_LABELS,      "a photo of a {label} garment", 0.12, 0.02),
        "neckline": (NECKLINE_LABELS, "a photo of a garment with a {label}", 0.12, 0.02),
        "length":   (LENGTH_LABELS,   "a photo of a {label} garment", 0.12, 0.02),
    }

    def __init__(self, scorer: "FashionSiglipScorer") -> None:
        self.scorer = scorer
        # (labels, template) -> text embeddings tensor; populated lazily
        self._text_cache: dict[tuple[tuple[str, ...], str], Any] = {}

    def _text_embeds(self, labels: tuple[str, ...], template: str):
        key = (labels, template)
        if key in self._text_cache:
            return self._text_cache[key]
        texts = tuple(template.format(label=l) for l in labels)
        embeds = self.scorer._embed_texts(texts)
        self._text_cache[key] = embeds
        return embeds

    def classify(self, image: Image.Image, field: str, k: int = 3) -> FieldPrediction:
        import torch

        labels, template, cos_threshold, temperature = self.FIELDS[field]
        img = self.scorer._embed_image(image)
        texts = self._text_embeds(labels, template)
        with torch.no_grad():
            # Both tensors are L2-normalized by the scorer, so matmul == cosine sim.
            cos = (img @ texts.T).squeeze(0)   # (N,)
            probs = torch.softmax(cos / temperature, dim=-1)
        k = min(k, len(labels))
        values, indices = probs.topk(k)
        topk = [
            (labels[i], float(v))
            for i, v in zip(indices.tolist(), values.tolist())
        ]
        top_cos = float(cos[indices[0]].item())
        top_label, top_prob = topk[0]
        return FieldPrediction(
            field=field,
            top_label=top_label,
            confidence=float(top_prob),
            topk=topk,
            raw_top_cosine=top_cos,
            unknown=top_cos < cos_threshold,
        )

    def classify_all(self, image: Image.Image) -> dict[str, FieldPrediction]:
        out: dict[str, FieldPrediction] = {}
        for f in self.FIELDS:
            try:
                out[f] = self.classify(image, f)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("classifier field=%s failed: %s", f, exc)
        return out


# ---------------------------------------------------------------------------
# Vision-first normalizer (SigLIP primary, caption only enriches)
# ---------------------------------------------------------------------------


_PATTERN_DETAIL_DESCRIPTIONS: dict[str, str] = {
    "polka dot": "evenly spaced round dots on the fabric",
    "striped": "parallel stripes running across the garment",
    "floral": "repeating floral motifs printed across the fabric",
    "checked": "a grid of intersecting colored lines forming small squares",
    "plaid": "classic plaid with intersecting stripes in multiple colors",
    "graphic print": "a bold graphic illustration printed on the front",
    "text print": "large stylized text printed across the fabric",
    "animal print": "an animal-skin pattern such as leopard or zebra",
    "abstract print": "an abstract multi-colored print",
    "tie-dye": "swirling tie-dye color bleeds",
    "color-blocked": "distinct solid color panels joined across the garment",
    "embroidered": "embroidered motifs stitched into the fabric",
    "solid": "a single uniform color with no print or pattern",
}


class VisionFirstNormalizer(Normalizer):
    """
    PRIMARY PATH: SigLIP zero-shot classification on the image.
    SUPPORT PATH: Gemini/rules on the caption — used ONLY to enrich
    non-visual fields (literal text on the garment, material guess,
    miscellaneous details) OR to override the classifier when the
    caption explicitly disagrees (e.g. caption says "polka dot" but
    classifier picked "solid").

    Fusion rules:
      visual fields (category, color, pattern, sleeve, fit, neckline, length):
        use classifier when confidence >= threshold
        else fall back to caption-derived value
        caption override: if caption explicitly names a pattern/sleeve
                          that is among the labels AND classifier is
                          not high-confidence, caption wins.
      text-only fields (text on garment, material, details):
        caption only.
    """

    name = "vision-first"

    # caption tokens we treat as strong signals even when SigLIP disagrees
    _PATTERN_CAPTION_TOKENS = {
        "polka dot": ("polka dot", "polka-dot", "polka dots", "polka-dots"),
        "striped": ("stripe", "striped", "stripes"),
        "floral": ("floral", "flowers"),
        "checked": ("checked", "check pattern", "gingham"),
        "plaid": ("plaid", "tartan"),
        "graphic print": ("graphic print", "graphic"),
        "text print": ("text print", "text on"),
        "animal print": ("leopard", "zebra", "animal print"),
    }

    _CLASSIFIER_CONFIDENCE_FLOOR = 0.40  # softmax prob floor per field

    def __init__(
        self,
        classifier: SiglipAttributeClassifier,
        gemini: GeminiJsonNormalizer | None = None,
        rules: RuleBasedNormalizer | None = None,
    ) -> None:
        self.classifier = classifier
        self.gemini = gemini or GeminiJsonNormalizer()
        self.rules = rules or RuleBasedNormalizer()

    def _caption_attrs(self, caption: CaptionResult) -> GarmentAttributes:
        try:
            attrs = self.gemini.normalize(caption)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("gemini enrichment failed: %s", exc)
            attrs = GarmentAttributes()
        try:
            rule = self.rules.normalize(caption)
            for fname in (
                "category", "color", "fit", "sleeve", "neckline",
                "length", "pattern", "text",
            ):
                if not getattr(attrs, fname):
                    setattr(attrs, fname, getattr(rule, fname))
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("rule-based enrichment failed: %s", exc)
        return attrs

    def _find_caption_pattern(self, caption_text: str) -> str | None:
        t = caption_text.lower()
        for canonical, tokens in self._PATTERN_CAPTION_TOKENS.items():
            if any(tok in t for tok in tokens):
                return canonical
        return None

    # ------------------------------------------------------------------

    def normalize(
        self,
        caption: CaptionResult,
        image: Optional[Image.Image] = None,
    ) -> GarmentAttributes:
        if image is None:
            _LOG.warning("VisionFirstNormalizer called without image; "
                         "falling back to caption-only chain.")
            return self._caption_attrs(caption)

        cap_attrs = self._caption_attrs(caption)
        preds = self.classifier.classify_all(image)

        # Strip the "fit" suffix from fit labels (classifier labels are
        # "slim fit" but downstream prompt template expects "slim-fit").
        def _clean_fit(label: str) -> str:
            return label.replace(" fit", "").strip()

        visual_values: dict[str, tuple[str, float]] = {}
        for fname, pred in preds.items():
            if pred.unknown or pred.confidence < self._CLASSIFIER_CONFIDENCE_FLOOR:
                continue
            label = pred.top_label
            if fname == "fit":
                label = _clean_fit(label)
            visual_values[fname] = (label, pred.confidence)

        # Caption override: pattern is the single most-missed field from image
        # alone because Gemini/BLIP captions often name the pattern explicitly.
        # If caption names a known pattern and classifier wasn't confident,
        # trust caption.
        caption_pattern = self._find_caption_pattern(caption.text)
        if caption_pattern:
            vp = visual_values.get("pattern")
            if vp is None or vp[1] < 0.55:
                visual_values["pattern"] = (caption_pattern, max(vp[1] if vp else 0.0, 0.55))

        # Start from classifier results for visual fields, then fill gaps
        # from caption-derived attributes.
        out = GarmentAttributes(
            caption_confidence=caption.confidence,
            secondary_colors=cap_attrs.secondary_colors,
            text=cap_attrs.text,
            material=cap_attrs.material,
            details=cap_attrs.details,
        )

        visual_fields = ("category", "color", "pattern", "sleeve", "fit", "neckline", "length")
        for fname in visual_fields:
            if fname in visual_values:
                label, conf = visual_values[fname]
                setattr(out, fname, label)
                out.field_confidences[fname] = round(conf, 3)
                out.field_sources[fname] = "siglip"
            elif getattr(cap_attrs, fname):
                setattr(out, fname, getattr(cap_attrs, fname))
                out.field_confidences[fname] = 0.0
                out.field_sources[fname] = cap_attrs.field_sources.get(fname, "caption")
            else:
                # Explicitly mark unknown — better than silently guessing.
                out.field_confidences[fname] = 0.0
                out.field_sources[fname] = "unknown"

        # Pattern details: used by the prompt builder for richer wording.
        if out.pattern and out.pattern in _PATTERN_DETAIL_DESCRIPTIONS:
            out.pattern_details = _PATTERN_DETAIL_DESCRIPTIONS[out.pattern]

        # Expose top-k for UI/debugging.
        out.field_topk = {
            fname: [[lbl, round(p, 3)] for lbl, p in preds[fname].topk]
            for fname in preds
        }

        # Non-visual fields (text, material): flag sources.
        if out.text:
            out.field_sources.setdefault("text", "caption")
        if out.material:
            out.field_sources.setdefault("material", "caption")

        return out


def make_default_normalizer(
    classifier: SiglipAttributeClassifier | None = None,
) -> Normalizer:
    """
    Default chain:
      - If a SigLIP classifier is available → VisionFirstNormalizer.
      - Else fall back to the old caption-first Gemini → rules chain.
    """
    if classifier is not None:
        return VisionFirstNormalizer(classifier=classifier)

    class ChainNormalizer(Normalizer):
        name = "chain-caption-only"

        def normalize(
            self,
            caption: CaptionResult,
            image: Optional[Image.Image] = None,
        ) -> GarmentAttributes:
            try:
                attrs = GeminiJsonNormalizer().normalize(caption)
                if attrs.is_sparse():
                    rule = RuleBasedNormalizer().normalize(caption)
                    for field_name in (
                        "category", "color", "fit", "sleeve", "neckline",
                        "length", "pattern", "text", "material",
                    ):
                        if not getattr(attrs, field_name):
                            setattr(attrs, field_name, getattr(rule, field_name))
                return attrs
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("normalizer Gemini path failed: %s", exc)
                return RuleBasedNormalizer().normalize(caption)

    return ChainNormalizer()


# ---------------------------------------------------------------------------
# Stage 3: Prompt builder
# ---------------------------------------------------------------------------


# Ghost-mannequin wording reads better on SDXL than "headless mannequin"
# because the phrase appears heavily in ecommerce product-photo training data.
# Constraints are deliberately repeated — SDXL rewards repetition.
_DEFAULT_PROMPT_TEMPLATE = (
    "A studio product image of a clothing item displayed on a ghost mannequin. "
    "The mannequin has NO arms, NO hands, NO head, NO legs, and NO visible human body parts. "
    "The garment is a {fit} {color} {category} with {sleeve} and {neckline}. "
    "It is {length}. "
    "{pattern_clause}"
    "{text_clause}"
    "{material_clause}"
    "{details_clause}"
    "Only the clothing is visible, floating naturally as in ghost mannequin photography. "
    "No human, no skin, no limbs, no person. "
    "Centered composition. Pure white background (#FFFFFF). "
    "Soft studio lighting. High detail fabric texture."
)

# Over-specified negative per SDXL best practices — SDXL responds well to
# enumerated body parts and to "realistic human" / "photo of person".
_NEGATIVE_PROMPT_DEFAULT = (
    "arms, hands, fingers, human, person, face, head, legs, body, skin, "
    "neck, shoulders, torso, model, pose, anatomy, realistic human, "
    "photo of person, ears, hair, nose, mouth, eyes, feet, toes, "
    "muscles, bone, belly button, armpit, wrist, forearm, elbow, knee, "
    "thigh, calf, back, chest, "
    "cluttered background, colored background, watermark, text overlay, "
    "blurry, deformed garment, extra sleeves, distorted collar, "
    "cartoon, illustration, low quality, lowres, nsfw"
)


class PromptBuilder:
    """Fills the generative-prompt template from GarmentAttributes."""

    def __init__(
        self,
        template: str | None = None,
        negative: str | None = None,
    ) -> None:
        self.template = template or os.environ.get("MANNEQUIN_PROMPT_TEMPLATE", _DEFAULT_PROMPT_TEMPLATE)
        self.negative = negative or os.environ.get("MANNEQUIN_NEGATIVE_PROMPT", _NEGATIVE_PROMPT_DEFAULT)

    def build(self, attrs: GarmentAttributes) -> tuple[str, str]:
        def clause(prefix: str, value: Optional[str], suffix: str = "") -> str:
            return f"{prefix}{value}{suffix}" if value else ""

        # Build the pattern clause with the richer detail description when
        # available. Emphasis + repetition helps SDXL actually render it.
        pattern_clause = ""
        if attrs.pattern and attrs.pattern != "solid":
            detail = attrs.pattern_details or f"clearly visible {attrs.pattern} motifs"
            pattern_clause = (
                f"The garment features a {attrs.pattern} pattern, with {detail}. "
                f"The {attrs.pattern} pattern is clearly visible and well-defined. "
            )
        elif attrs.pattern == "solid":
            pattern_clause = "The garment is a solid color with no visible print. "

        prompt = self.template.format(
            fit=attrs.fit or "regular-fit",
            color=attrs.color or "neutral-colored",
            category=attrs.category or "garment",
            sleeve=attrs.sleeve or "unspecified sleeve style",
            neckline=attrs.neckline or "unspecified neckline",
            length=attrs.length or "standard length",
            pattern_clause=pattern_clause,
            text_clause=clause('The garment has visible text: "', attrs.text, '". '),
            material_clause=clause("The fabric appears to be ", attrs.material, ". "),
            details_clause=(
                f"Notable details: {', '.join(attrs.details)}. "
                if attrs.details else ""
            ),
        )
        prompt = re.sub(r"\s+", " ", prompt).strip()
        return prompt, self.negative

    def simplify(self, attrs: GarmentAttributes) -> tuple[str, str]:
        """Retry prompt: minimal, high-signal only — still text-only."""
        minimal = (
            "Ghost mannequin product photography of a "
            f"{attrs.color or 'neutral'} {attrs.category or 'garment'}. "
            "Invisible mannequin — no arms, no hands, no head, no legs, no skin, no person. "
            "Only the clothing is visible, floating naturally. "
            "Pure white background. Centered front view. High detail fabric."
        ).strip()
        return re.sub(r"\s+", " ", minimal), self.negative


# ---------------------------------------------------------------------------
# Stage 4: Generator
# ---------------------------------------------------------------------------


class Generator:
    """
    Strategy interface. Text-to-image only by contract:
    `image` is intentionally NOT passed in — generators MUST be unconditional
    on the source image. The pipeline still holds the source image elsewhere
    for scoring.
    """

    name: str = "base"
    supports_batch: bool = False

    def generate(
        self,
        prompt: str,
        negative: str,
        seed: int | None,
        num_candidates: int = 1,
    ) -> list[bytes]:
        raise NotImplementedError


class SDXLGenerator(Generator):
    """
    Pure SDXL text-to-image. No source-image conditioning. Heavy first-load;
    kept warm via singleton. Honors `num_candidates` in a single pipeline call.
    """

    name = "sdxl"
    supports_batch = True
    _pipe = None

    def _ensure(self) -> None:
        if SDXLGenerator._pipe is not None:
            return
        import torch
        from diffusers import StableDiffusionXLPipeline  # type: ignore

        model_id = os.environ.get("SDXL_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=dtype)
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        elif getattr(__import__("torch").backends, "mps", None) and __import__("torch").backends.mps.is_available():
            pipe = pipe.to("mps")
        SDXLGenerator._pipe = pipe

    def generate(
        self,
        prompt: str,
        negative: str,
        seed: int | None,
        num_candidates: int = 1,
    ) -> list[bytes]:
        import torch
        self._ensure()
        assert SDXLGenerator._pipe is not None
        device = (
            "cuda" if torch.cuda.is_available()
            else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        )
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(seed))

        # Higher guidance + more steps materially improves prompt adherence,
        # which is exactly what the "no body parts" constraint needs.
        images = SDXLGenerator._pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=int(os.environ.get("SDXL_STEPS", "40")),
            guidance_scale=float(os.environ.get("SDXL_GUIDANCE", "7.5")),
            height=int(os.environ.get("SDXL_HEIGHT", "1024")),
            width=int(os.environ.get("SDXL_WIDTH", "768")),
            generator=generator,
            num_images_per_prompt=max(1, int(num_candidates)),
        ).images

        out_bytes: list[bytes] = []
        for im in images:
            b = BytesIO()
            im.save(b, format="PNG")
            out_bytes.append(b.getvalue())
        return out_bytes


class DalleGenerator(Generator):
    """
    OpenAI image generation fallback (gpt-image-1 / dall-e-3). Requires
    OPENAI_API_KEY. Pure text-to-image by nature.
    """

    name = "dalle"
    supports_batch = False

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("DALLE_MODEL", "gpt-image-1")

    def generate(
        self,
        prompt: str,
        negative: str,
        seed: int | None,
        num_candidates: int = 1,
    ) -> list[bytes]:
        # `openai` >=1.x; lazy imported so environments without it still boot.
        from openai import OpenAI  # type: ignore

        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=api_key)
        outs: list[bytes] = []
        # DALL·E-3 rejects n>1; loop instead to honor num_candidates.
        for _ in range(max(1, num_candidates)):
            resp = client.images.generate(
                model=self.model,
                prompt=f"{prompt}\n\nAvoid: {negative}",
                size=os.environ.get("DALLE_SIZE", "1024x1024"),
                n=1,
                response_format="b64_json",
            )
            b64 = resp.data[0].b64_json
            outs.append(base64.b64decode(b64))
        return outs


# ---------------------------------------------------------------------------
# Stage 5: Post-processing
# ---------------------------------------------------------------------------


class PostProcessor:
    """Enforces product-style output: white background, centered, consistent contrast."""

    def __init__(
        self,
        target_size: tuple[int, int] = (1024, 1024),
        background: tuple[int, int, int] = (255, 255, 255),
        shadow: bool = True,
    ) -> None:
        self.target_size = target_size
        self.background = background
        self.shadow = shadow

    def run(self, image_bytes: bytes) -> bytes:
        im = Image.open(BytesIO(image_bytes)).convert("RGB")
        im = ImageOps.autocontrast(im, cutoff=1)
        # Simple white-field normalization: pixels whose value is close to (>=250)
        # across all channels are forced to pure white.
        im = im.point(lambda p: 255 if p >= 250 else p)
        # Fit onto centered white canvas for consistent aspect across models.
        canvas = Image.new("RGB", self.target_size, self.background)
        im.thumbnail(self.target_size, Image.Resampling.LANCZOS)
        x = (self.target_size[0] - im.size[0]) // 2
        y = (self.target_size[1] - im.size[1]) // 2
        canvas.paste(im, (x, y))
        if self.shadow:
            # Soft ellipse shadow just under pasted content.
            shadow = Image.new("L", self.target_size, 0)
            from PIL import ImageDraw
            draw = ImageDraw.Draw(shadow)
            ellipse = (
                x + im.size[0] * 0.15,
                y + im.size[1] * 0.95,
                x + im.size[0] * 0.85,
                y + im.size[1] * 1.03,
            )
            draw.ellipse(ellipse, fill=80)
            shadow = shadow.filter(ImageFilter.GaussianBlur(16))
            gray = Image.new("RGB", self.target_size, (210, 210, 210))
            canvas = Image.composite(gray, canvas, shadow)
        out = BytesIO()
        canvas.save(out, format="JPEG", quality=92)
        return out.getvalue()


# ---------------------------------------------------------------------------
# Stage 6: Scoring
# ---------------------------------------------------------------------------


class FashionSiglipScorer:
    """
    Dual-purpose SigLIP scorer:
      - garment similarity:  cosine(embed(original), embed(generated))
      - human-presence probe: compare image→text similarity between a
        "mannequin wearing clothing" prompt and a "person with arms and body"
        prompt. If the "person" prompt wins by more than `person_margin`, the
        candidate is rejected as containing human parts.
    """

    name = "marqo-fashion-siglip"
    _model = None
    _preprocess = None

    _MANNEQUIN_PROMPTS = (
        "a headless armless mannequin displaying clothing on a white background",
        "a product photo of a garment on a featureless mannequin",
    )
    _HUMAN_PROMPTS = (
        "a person with visible arms and hands wearing clothing",
        "a human body with a face and limbs wearing clothing",
    )

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or os.environ.get(
            "SIGLIP_FASHION_MODEL_ID", "Marqo/marqo-fashionSigLIP"
        )

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        model.eval()
        if torch.cuda.is_available():
            model = model.to("cuda")
        FashionSiglipScorer._model = model
        FashionSiglipScorer._preprocess = processor

    # ------------------------------------------------------------------

    def _device(self):
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _embed_image(self, image: Image.Image):
        import torch
        self._ensure()
        inputs = FashionSiglipScorer._preprocess(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device()) for k, v in inputs.items()}
        with torch.no_grad():
            feats = FashionSiglipScorer._model.get_image_features(**inputs)
        return feats / feats.norm(dim=-1, keepdim=True)

    def _embed_texts(self, texts: tuple[str, ...]):
        import torch
        self._ensure()
        inputs = FashionSiglipScorer._preprocess(
            text=list(texts), return_tensors="pt", padding=True, truncation=True,
        )
        inputs = {k: v.to(self._device()) for k, v in inputs.items()}
        with torch.no_grad():
            feats = FashionSiglipScorer._model.get_text_features(**inputs)
        return feats / feats.norm(dim=-1, keepdim=True)

    # ------------------------------------------------------------------

    def garment_similarity(self, original: Image.Image, generated: Image.Image) -> float:
        a = self._embed_image(original)
        b = self._embed_image(generated)
        return float((a * b).sum().item())

    def human_score(self, image: Image.Image) -> dict[str, float]:
        """
        Returns a dict with:
          mannequin_sim  – mean similarity to mannequin prompts
          person_sim     – mean similarity to person prompts
          margin         – mannequin_sim - person_sim  (positive = looks like mannequin)
        """
        import torch
        img = self._embed_image(image)
        mann = self._embed_texts(self._MANNEQUIN_PROMPTS)
        pers = self._embed_texts(self._HUMAN_PROMPTS)
        mann_sim = float((img @ mann.T).mean().item())
        pers_sim = float((img @ pers.T).mean().item())
        return {
            "mannequin_sim": mann_sim,
            "person_sim": pers_sim,
            "margin": mann_sim - pers_sim,
        }


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class DiskCache:
    """Very small content-addressed cache: input-image hash → JPEG output bytes."""

    def __init__(self, root: str | None = None) -> None:
        base = root or os.environ.get("MANNEQUIN_CACHE_DIR", "/tmp/vestir-mannequin-cache")
        self.root = Path(base)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash(image_bytes: bytes, variant: str) -> str:
        h = hashlib.sha256()
        h.update(image_bytes)
        h.update(variant.encode("utf-8"))
        return h.hexdigest()

    def get(self, image_bytes: bytes, variant: str) -> bytes | None:
        path = self.root / f"{self._hash(image_bytes, variant)}.jpg"
        if path.exists():
            try:
                return path.read_bytes()
            except Exception:  # noqa: BLE001
                return None
        return None

    def put(self, image_bytes: bytes, variant: str, output: bytes) -> None:
        path = self.root / f"{self._hash(image_bytes, variant)}.jpg"
        try:
            path.write_bytes(output)
        except Exception:  # noqa: BLE001
            _LOG.warning("disk cache write failed: %s", path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    # SigLIP garment similarity floor (original ↔ candidate).
    score_threshold: float = 0.55
    # retries with simplified prompt when every candidate was rejected
    max_retries: int = 1
    # Over-sample to survive the 50–70% trash rate typical for SDXL txt2img.
    num_candidates: int = 6

    # Below this value we don't even attempt generation — hand off to legacy SAM.
    attribute_confidence_threshold: float = 0.45

    # ---- Human-presence rejection (strict, per v2.1) ----
    # A candidate is rejected if ANY of these holds:
    #   person_sim      > person_sim_max
    #   mannequin_sim   < mannequin_sim_min
    #   (mannequin_sim - person_sim) < human_margin_threshold
    #   skin_pixel_ratio > skin_pixel_ratio_max   (hard pixel check)
    person_sim_max: float = 0.20
    mannequin_sim_min: float = 0.25
    human_margin_threshold: float = 0.05
    skin_pixel_ratio_max: float = 0.03

    use_cache: bool = True
    variant: str = "v2.1-textonly"         # bump invalidates old caches
    skip_scoring_on_error: bool = True


class MannequinPipeline:
    """
    Tie all stages together. This is the class to import from app.py:

        from mannequin_pipeline import build_default_pipeline
        pipeline = build_default_pipeline(flux_runner=_run_tryoff)
        result = pipeline.run(image_bytes, seed=1234)
    """

    def __init__(
        self,
        captioner: Captioner,
        normalizer: Normalizer,
        prompt_builder: PromptBuilder,
        generator: Generator,
        post: PostProcessor,
        scorer: FashionSiglipScorer | None,
        cache: DiskCache | None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.captioner = captioner
        self.normalizer = normalizer
        self.prompt_builder = prompt_builder
        self.generator = generator
        self.post = post
        self.scorer = scorer
        self.cache = cache
        self.config = config or PipelineConfig()

    # ------------------------------------------------------------------

    def _timed(self, fn: Callable[[], Any], stage_id: str, stages: list[StageLog]) -> Any:
        started = time.time()
        try:
            out = fn()
            stages.append(StageLog(
                id=stage_id, status="completed",
                duration_ms=int((time.time() - started) * 1000),
            ))
            return out
        except Exception as exc:  # noqa: BLE001
            stages.append(StageLog(
                id=stage_id, status="failed",
                duration_ms=int((time.time() - started) * 1000),
                detail=str(exc),
            ))
            raise

    # ------------------------------------------------------------------

    _VAGUE_CAPTION = re.compile(
        r"^\s*(a|an|the)?\s*(person|man|woman|model|individual|someone)?\s*"
        r"(wearing|in)?\s*(clothes|clothing|outfit|garment|something)?\.?\s*$",
        re.IGNORECASE,
    )

    def _caption_is_vague(self, caption: CaptionResult) -> bool:
        text = caption.text.strip().lower()
        if len(text.split()) < 6:
            return True
        if self._VAGUE_CAPTION.match(text):
            return True
        return False

    def _empty_result(
        self,
        *,
        attrs: GarmentAttributes,
        prompt: str,
        negative: str,
        stages: list[StageLog],
        notes: list[str],
        total_started: float,
        fallback_reason: str,
        cache_hit: bool = False,
    ) -> PipelineResult:
        return PipelineResult(
            image_bytes=None, mime_type="",
            attributes=attrs, attribute_confidence=attrs.confidence(),
            prompt=prompt, negative_prompt=negative,
            score=None, candidates=[], selected_index=None,
            fallback_recommended=True, fallback_reason=fallback_reason,
            stages=stages,
            total_ms=int((time.time() - total_started) * 1000),
            cache_hit=cache_hit,
            notes=notes,
        )

    # ------------------------------------------------------------------

    def run(self, image_bytes: bytes, seed: int | None = None) -> PipelineResult:
        total_started = time.time()
        stages: list[StageLog] = []
        notes: list[str] = []

        # --- Cache short-circuit -------------------------------------------
        if self.config.use_cache and self.cache is not None:
            cached = self.cache.get(image_bytes, self.config.variant)
            if cached is not None:
                empty_attrs = GarmentAttributes()
                return PipelineResult(
                    image_bytes=cached,
                    mime_type="image/jpeg",
                    attributes=empty_attrs,
                    attribute_confidence=0.0,
                    prompt="",
                    negative_prompt=self.prompt_builder.negative,
                    score=None,
                    candidates=[], selected_index=None,
                    fallback_recommended=False, fallback_reason=None,
                    stages=[StageLog(id="cache", status="completed", duration_ms=0)],
                    total_ms=int((time.time() - total_started) * 1000),
                    cache_hit=True,
                    notes=["served from cache"],
                )

        # --- Ingest --------------------------------------------------------
        try:
            pil = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            _LOG.exception("pipeline: ingest failed")
            return self._empty_result(
                attrs=GarmentAttributes(), prompt="", negative=self.prompt_builder.negative,
                stages=[StageLog(id="ingest", status="failed", duration_ms=0, detail=str(exc))],
                notes=[f"could not open image: {exc}"],
                total_started=total_started,
                fallback_reason="ingest_failed",
            )

        # --- Stage 1: caption ---------------------------------------------
        try:
            caption = self._timed(lambda: self.captioner.caption(pil), "caption", stages)
            _LOG.info("pipeline caption provider=%s confidence=%.2f text=%r",
                      caption.provider, caption.confidence, caption.text[:160])
        except Exception as exc:  # noqa: BLE001
            notes.append(f"captioning failed hard: {exc}")
            return self._empty_result(
                attrs=GarmentAttributes(rejected_caption=True),
                prompt="", negative=self.prompt_builder.negative,
                stages=stages, notes=notes, total_started=total_started,
                fallback_reason="caption_failed",
            )

        if self._caption_is_vague(caption):
            notes.append(f"caption rejected as vague: {caption.text!r}")
            return self._empty_result(
                attrs=GarmentAttributes(caption_confidence=caption.confidence, rejected_caption=True),
                prompt="", negative=self.prompt_builder.negative,
                stages=stages, notes=notes, total_started=total_started,
                fallback_reason="caption_too_vague",
            )

        # --- Stage 2: normalize (vision-first) ----------------------------
        attrs = self._timed(
            lambda: self.normalizer.normalize(caption, image=pil),
            "normalize",
            stages,
        )
        attrs.caption_confidence = caption.confidence
        _LOG.info(
            "pipeline attrs=%s visual_conf=%.2f combined_conf=%.2f sources=%s",
            {k: getattr(attrs, k) for k in
             ("category", "color", "pattern", "sleeve", "fit", "neckline", "length")},
            attrs.visual_confidence(),
            attrs.confidence(),
            attrs.field_sources,
        )

        # --- Attribute-confidence gate ------------------------------------
        if attrs.confidence() < self.config.attribute_confidence_threshold:
            notes.append(
                f"attribute_confidence {attrs.confidence():.2f} < "
                f"{self.config.attribute_confidence_threshold:.2f}; "
                "recommending legacy SAM fallback"
            )
            return self._empty_result(
                attrs=attrs, prompt="", negative=self.prompt_builder.negative,
                stages=stages, notes=notes, total_started=total_started,
                fallback_reason="low_attribute_confidence",
            )

        # --- Stage 3: prompt ----------------------------------------------
        prompt, negative = self._timed(lambda: self.prompt_builder.build(attrs), "prompt", stages)
        _LOG.info("pipeline prompt=%r", prompt)

        # --- Stage 4: multi-candidate generation --------------------------
        def _gen(p: str, n: int) -> list[bytes]:
            return self.generator.generate(p, negative, seed, n) or []

        candidate_bytes: list[bytes] = self._timed(
            lambda: _gen(prompt, self.config.num_candidates),
            "generate",
            stages,
        )

        # Retry loop with simplified prompt if everything failed / was rejected.
        attempts = 1
        selected_index: Optional[int] = None
        candidates: list[CandidateScore] = []
        final_bytes: Optional[bytes] = None
        score: Optional[float] = None

        while True:
            # --- Stage 6: score + reject ----------------------------------
            candidates = []
            if candidate_bytes:
                for idx, raw in enumerate(candidate_bytes):
                    try:
                        cand_pil = Image.open(BytesIO(raw)).convert("RGB")

                        # Pixel-level skin-tone check runs regardless of scorer.
                        skin_ratio = estimate_skin_pixel_ratio(cand_pil)

                        if self.scorer is not None:
                            garm = self.scorer.garment_similarity(pil, cand_pil)
                            hs = self.scorer.human_score(cand_pil)
                        else:
                            garm = 0.0
                            hs = {"mannequin_sim": 0.0, "person_sim": 0.0, "margin": 0.0}

                        rejected = False
                        reason: Optional[str] = None

                        # --- Strict rejection cascade (v2.1) -----------------
                        if skin_ratio > self.config.skin_pixel_ratio_max:
                            rejected = True
                            reason = (
                                f"skin pixels {skin_ratio*100:.1f}% > "
                                f"{self.config.skin_pixel_ratio_max*100:.1f}% "
                                "(visible body parts)"
                            )
                        elif self.scorer is not None and hs["person_sim"] > self.config.person_sim_max:
                            rejected = True
                            reason = (
                                f"person_sim {hs['person_sim']:.3f} > "
                                f"{self.config.person_sim_max:.2f} (human detected)"
                            )
                        elif self.scorer is not None and hs["mannequin_sim"] < self.config.mannequin_sim_min:
                            rejected = True
                            reason = (
                                f"mannequin_sim {hs['mannequin_sim']:.3f} < "
                                f"{self.config.mannequin_sim_min:.2f} (not mannequin-like)"
                            )
                        elif self.scorer is not None and hs["margin"] < self.config.human_margin_threshold:
                            rejected = True
                            reason = (
                                f"margin {hs['margin']:.3f} < "
                                f"{self.config.human_margin_threshold:.2f} "
                                "(ambiguous mannequin/person)"
                            )
                        elif self.scorer is not None and garm < self.config.score_threshold:
                            rejected = True
                            reason = (
                                f"garment_sim {garm:.3f} < "
                                f"threshold {self.config.score_threshold:.2f}"
                            )

                        candidates.append(CandidateScore(
                            index=idx,
                            garment_sim=garm,
                            mannequin_sim=hs["mannequin_sim"],
                            person_sim=hs["person_sim"],
                            margin=hs["margin"],
                            skin_pixel_ratio=skin_ratio,
                            rejected=rejected,
                            reject_reason=reason,
                        ))
                    except Exception as exc:  # noqa: BLE001
                        notes.append(f"scoring candidate {idx} failed: {exc}")
                        candidates.append(CandidateScore(
                            index=idx, garment_sim=0.0,
                            mannequin_sim=0.0, person_sim=0.0, margin=0.0,
                            skin_pixel_ratio=0.0,
                            rejected=True, reject_reason=f"scoring_error: {exc}",
                        ))
                stages.append(StageLog(
                    id="score", status="completed", duration_ms=0,
                    detail=(
                        f"{len([c for c in candidates if not c.rejected])} accepted / "
                        f"{len(candidates)}"
                    ),
                ))
                # Emit per-candidate log lines so ops can see why things were rejected.
                for c in candidates:
                    _LOG.info(
                        "candidate %d: garment=%.3f mannequin=%.3f person=%.3f "
                        "margin=%.3f skin=%.3f  %s%s",
                        c.index, c.garment_sim, c.mannequin_sim, c.person_sim,
                        c.margin, c.skin_pixel_ratio,
                        "REJECTED" if c.rejected else "accepted",
                        f" ({c.reject_reason})" if c.reject_reason else "",
                    )

            accepted = [c for c in candidates if not c.rejected]
            if accepted:
                best = max(accepted, key=lambda c: c.garment_sim)
                selected_index = best.index
                score = best.garment_sim
                final_bytes = candidate_bytes[best.index]
                break

            # All rejected → retry once with simplified prompt.
            if attempts > self.config.max_retries:
                notes.append("all candidates rejected after retries")
                break
            attempts += 1
            prompt, negative = self.prompt_builder.simplify(attrs)
            notes.append(f"retrying with simplified prompt (attempt {attempts})")
            candidate_bytes = self._timed(
                lambda: _gen(prompt, self.config.num_candidates),
                f"generate_retry_{attempts}",
                stages,
            )

        if final_bytes is None:
            return PipelineResult(
                image_bytes=None, mime_type="",
                attributes=attrs, attribute_confidence=attrs.confidence(),
                prompt=prompt, negative_prompt=negative,
                score=score, candidates=candidates, selected_index=None,
                fallback_recommended=True, fallback_reason="all_candidates_rejected",
                stages=stages,
                total_ms=int((time.time() - total_started) * 1000),
                notes=notes,
            )

        # --- Stage 5: post-process ----------------------------------------
        post_bytes = self._timed(lambda: self.post.run(final_bytes), "post", stages)

        if self.config.use_cache and self.cache is not None:
            self.cache.put(image_bytes, self.config.variant, post_bytes)

        return PipelineResult(
            image_bytes=post_bytes,
            mime_type="image/jpeg",
            attributes=attrs, attribute_confidence=attrs.confidence(),
            prompt=prompt, negative_prompt=negative,
            score=score, candidates=candidates, selected_index=selected_index,
            fallback_recommended=False, fallback_reason=None,
            stages=stages,
            total_ms=int((time.time() - total_started) * 1000),
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_default_pipeline(
    config: PipelineConfig | None = None,
) -> MannequinPipeline:
    """
    Wire the default text-to-image-only chain.

    Generator selection is controlled by MANNEQUIN_GENERATOR:
      - "sdxl"  (default) — Stable Diffusion XL, pure text-to-image
      - "dalle"            — OpenAI gpt-image-1 / dall-e-3

    Image-conditioned generators (FLUX try-off, SD inpainting, img2img) are
    deliberately not wired here: this pipeline is spec'd as text-only.

    The SigLIP backbone is always instantiated because it powers BOTH the
    attribute classifier (primary source of truth) and candidate scoring
    (human-presence / garment-similarity rejection). MANNEQUIN_SCORE=0 only
    disables the per-candidate scoring, not the backbone.
    """
    gen_kind = os.environ.get("MANNEQUIN_GENERATOR", "sdxl").strip().lower()
    if gen_kind == "dalle":
        generator: Generator = DalleGenerator()
    elif gen_kind == "sdxl":
        generator = SDXLGenerator()
    else:
        raise RuntimeError(
            f"MANNEQUIN_GENERATOR={gen_kind!r} is not supported in the "
            "text-to-image-only pipeline. Use 'sdxl' or 'dalle'."
        )

    siglip_backbone = FashionSiglipScorer()
    classifier = SiglipAttributeClassifier(siglip_backbone)

    scorer: FashionSiglipScorer | None = siglip_backbone
    if os.environ.get("MANNEQUIN_SCORE", "1").strip() != "1":
        scorer = None  # candidate ranking disabled, but classifier still works

    return MannequinPipeline(
        captioner=make_default_captioner(),
        normalizer=make_default_normalizer(classifier=classifier),
        prompt_builder=PromptBuilder(),
        generator=generator,
        post=PostProcessor(),
        scorer=scorer,
        cache=DiskCache(),
        config=config,
    )
