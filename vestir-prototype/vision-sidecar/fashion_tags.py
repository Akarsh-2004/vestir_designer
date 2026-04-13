"""Layered fashion tag vocabulary + SigLIP (open_clip) embedding retrieval."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_VOCAB_PATH = Path(__file__).resolve().parent / "fashion_tag_vocab.json"

# Order for human-readable descriptor stacking
_LAYER_DESCRIPTOR_ORDER: tuple[str, ...] = (
    "garment",
    "color_base",
    "color_tone",
    "pattern_color",
    "fabric",
    "fit",
    "aesthetic",
    "details",
    "occasion",
    "season",
    "styling",
    "gender_expression",
    "context",
)


@dataclass(frozen=True)
class FashionTagEntry:
    layer: str
    tag_id: str
    embed_text: str


def load_fashion_tag_entries(path: Path | None = None) -> list[FashionTagEntry]:
    p = path or _VOCAB_PATH
    if not p.is_file():
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    tags = raw.get("tags")
    if not isinstance(tags, list):
        return []
    out: list[FashionTagEntry] = []
    for row in tags:
        if not isinstance(row, dict):
            continue
        layer = str(row.get("layer", "")).strip()
        tag_id = str(row.get("id", "")).strip()
        embed = str(row.get("embed", "")).strip() or tag_id.replace("_", " ")
        if not layer or not tag_id:
            continue
        out.append(FashionTagEntry(layer=layer, tag_id=tag_id, embed_text=embed))
    return out


def build_fashion_tag_text_features(
    open_clip_mod: Any,
    model: Any,
    tokenizer_source: str,
    entries: list[FashionTagEntry],
    batch_size: int,
) -> torch.Tensor | None:
    if not entries:
        return None
    tok = open_clip_mod.get_tokenizer(tokenizer_source)
    texts = [e.embed_text for e in entries]
    device = next(model.parameters()).device
    chunks: list[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tokens = tok(batch)
        if hasattr(tokens, "to"):
            tokens = tokens.to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens, normalize=True)
        chunks.append(feats.cpu())
    return torch.cat(chunks, dim=0)


def score_fashion_tags(
    model: Any,
    image_tensor_device: torch.Tensor,
    tag_features_cpu: torch.Tensor,
) -> list[float]:
    """Return sigmoid scores for each tag (multi-label SigLIP-style)."""
    device = next(model.parameters()).device
    img = image_tensor_device.to(device)
    with torch.no_grad():
        img_f = model.encode_image(img, normalize=True)
        tf = tag_features_cpu.to(device)
        # open_clip models expose logit_scale (learned temperature)
        ls = model.logit_scale.exp()
        logits = ls * (img_f @ tf.T)
        probs = torch.sigmoid(logits)[0].detach().cpu().tolist()
    return [float(x) for x in probs]


def select_tags(
    entries: list[FashionTagEntry],
    scores: list[float],
    *,
    threshold: float,
    max_per_layer: int,
    total_max: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], list[str]]:
    scored = [
        {"layer": e.layer, "tag": e.tag_id, "score": s}
        for e, s in zip(entries, scores, strict=True)
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)

    by_layer: dict[str, list[dict[str, Any]]] = {}
    picked: list[dict[str, Any]] = []
    for row in scored:
        if row["score"] < threshold:
            continue
        layer = row["layer"]
        if layer not in by_layer:
            by_layer[layer] = []
        if len(by_layer[layer]) >= max_per_layer:
            continue
        by_layer[layer].append({"tag": row["tag"], "score": round(row["score"], 4)})
        picked.append(row)
        if len(picked) >= total_max:
            break

    # If threshold was too strict, keep top-N globally
    if not picked and scored:
        for row in scored[:total_max]:
            layer = row["layer"]
            by_layer.setdefault(layer, []).append({"tag": row["tag"], "score": round(row["score"], 4)})
            picked.append(row)

    flat_tags = []
    seen: set[str] = set()
    for row in picked:
        if row["tag"] not in seen:
            seen.add(row["tag"])
            flat_tags.append(row["tag"])

    return picked, by_layer, flat_tags


def build_descriptor(by_layer: dict[str, list[dict[str, Any]]]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for layer in _LAYER_DESCRIPTOR_ORDER:
        for item in by_layer.get(layer, [])[:4]:
            t = str(item["tag"]).replace("_", " ")
            if t not in seen:
                seen.add(t)
                parts.append(t)
    return ", ".join(parts[:36])


def fashion_tag_inference_payload(
    model: Any,
    preprocess: Any,
    image_pil: Any,
    entries: list[FashionTagEntry],
    tag_features_cpu: torch.Tensor | None,
    *,
    threshold: float,
    max_per_layer: int,
    total_max: int,
) -> dict[str, Any]:
    if not entries or tag_features_cpu is None:
        return {
            "fashion_tags": [],
            "fashion_tags_scored": [],
            "fashion_tags_by_layer": {},
            "fashion_descriptor": "",
        }
    image_tensor = preprocess(image_pil).unsqueeze(0)
    scores = score_fashion_tags(model, image_tensor, tag_features_cpu)
    picked, by_layer, flat_tags = select_tags(
        entries,
        scores,
        threshold=threshold,
        max_per_layer=max_per_layer,
        total_max=total_max,
    )
    scored_out = [
        {"layer": r["layer"], "tag": r["tag"], "score": round(float(r["score"]), 4)} for r in picked
    ]
    return {
        "fashion_tags": flat_tags,
        "fashion_tags_scored": scored_out,
        "fashion_tags_by_layer": by_layer,
        "fashion_descriptor": build_descriptor(by_layer),
    }


def env_fashion_tag_config() -> tuple[bool, float, int, int, Path | None]:
    enabled = str(os.environ.get("FASHION_TAGS_ENABLE", "1")).strip().lower() not in {"0", "false", "no", "off"}
    threshold = float(os.environ.get("FASHION_TAG_THRESHOLD", "0.22"))
    max_per_layer = int(os.environ.get("FASHION_TAG_MAX_PER_LAYER", "6"))
    total_max = int(os.environ.get("FASHION_TAG_TOTAL_MAX", "32"))
    override = os.environ.get("FASHION_TAG_VOCAB_PATH", "").strip()
    path = Path(override) if override else None
    return enabled, threshold, max_per_layer, total_max, path
