"""LAB k-means garment palette, fashion vocabulary mapping, optional alpha + GrabCut masking."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

_VOCAB_PATH = Path(__file__).resolve().parent / "fashion_color_vocab.json"


def _mask_mode() -> str:
    raw = os.environ.get("VESTIR_COLOR_MASK", "all").strip().lower()
    if raw in {"none", "off", "0"}:
        return "none"
    if raw in {"alpha"}:
        return "alpha"
    if raw in {"grabcut"}:
        return "grabcut"
    if raw in {"all", "both", "1", "true", "yes"}:
        return "all"
    return "all"


def load_fashion_color_entries(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or Path(os.environ.get("FASHION_COLOR_VOCAB_PATH", "").strip() or _VOCAB_PATH)
    if not p.is_file():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    colors = data.get("colors")
    if not isinstance(colors, list):
        return []
    out: list[dict[str, Any]] = []
    for row in colors:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        try:
            L = float(row["L"])
            a = float(row["a"])
            b = float(row["b"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append({"name": name, "L": L, "a": a, "b": b})
    return out


def nearest_fashion_name(L: float, a: float, b: float, entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "Taupe"
    best = entries[0]["name"]
    best_d = 1e9
    for e in entries:
        d = (L - e["L"]) ** 2 + (a - e["a"]) ** 2 + (b - e["b"]) ** 2
        if d < best_d:
            best_d = d
            best = e["name"]
    return str(best)


def opencv_lab_u8_to_cielab(lab: np.ndarray) -> tuple[float, float, float]:
    """Approximate CIELAB from OpenCV 8-bit LAB (L*255/100, a,b centered at 128)."""
    L = float(lab[0]) * 100.0 / 255.0
    a = float(lab[1]) - 128.0
    b = float(lab[2]) - 128.0
    return L, a, b


def lab_u8_center_to_bgr(center: np.ndarray) -> tuple[int, int, int]:
    c = np.clip(np.round(center), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    bgr = cv2.cvtColor(c, cv2.COLOR_LAB2BGR)[0, 0]
    return int(bgr[2]), int(bgr[1]), int(bgr[0])


def _inner_crop_rect(w: int, h: int, ratio: float) -> tuple[int, int, int, int]:
    dx = int(w * ratio * 0.5)
    dy = int(h * ratio * 0.5)
    x0 = max(0, dx)
    y0 = max(0, dy)
    cw = max(8, w - 2 * dx)
    ch = max(8, h - 2 * dy)
    return x0, y0, cw, ch


def _grabcut_mask_bgr(bgr: np.ndarray, ratio: float = 0.12) -> np.ndarray:
    h, w = bgr.shape[:2]
    x0, y0, cw, ch = _inner_crop_rect(w, h, ratio)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    rect = (x0, y0, cw, ch)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.ones((h, w), dtype=bool)
    fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    return fg


def extract_garment_palette(
    image: Image.Image,
    *,
    top_k: int = 3,
    to_hsl_fn: Any,
    color_entries: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    k-means in OpenCV LAB space on (optionally masked) pixels; map centroids to fashion names via CIELAB distance.
    """
    entries = color_entries if color_entries is not None else load_fashion_color_entries()
    mode = _mask_mode()

    if image.mode == "RGBA":
        rgba = np.array(image)
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]
    else:
        rgb = np.array(image.convert("RGB"))
        alpha = None

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    include = np.ones((h, w), dtype=bool)
    if mode == "none":
        pass
    elif mode == "alpha":
        if alpha is not None:
            include = alpha > 25
    elif mode == "grabcut":
        gc = _grabcut_mask_bgr(bgr)
        include = gc if gc.sum() >= 32 else np.ones((h, w), dtype=bool)
    else:
        if alpha is not None:
            include &= alpha > 25
        gc = _grabcut_mask_bgr(bgr)
        if gc.sum() >= 32:
            include &= gc

    flat_lab = lab.reshape(-1, 3).astype(np.float32)
    flat_inc = include.reshape(-1)
    samples = flat_lab[flat_inc]
    if samples.shape[0] < 64:
        samples = flat_lab
    if samples.shape[0] < 16:
        return []

    k = min(max(1, top_k), 4, samples.shape[0] // 8)
    if k < 1:
        return []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
    _, labels, centers = cv2.kmeans(samples, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(-1)
    total = max(1, labels.shape[0])

    out: list[dict[str, Any]] = []
    for i in range(k):
        ratio = float(np.count_nonzero(labels == i)) / float(total)
        ctr = centers[i]
        L, a, b = opencv_lab_u8_to_cielab(ctr)
        r, g, b_rgb = lab_u8_center_to_bgr(ctr)
        name = nearest_fashion_name(L, a, b, entries)
        hsl = to_hsl_fn(float(r), float(g), float(b_rgb))
        out.append(
            {
                "name": name,
                "hex": f"#{r:02X}{g:02X}{b_rgb:02X}",
                "hsl": hsl,
                "coverage_pct": ratio,
            }
        )
    out.sort(key=lambda x: float(x["coverage_pct"]), reverse=True)
    return out[:top_k]
