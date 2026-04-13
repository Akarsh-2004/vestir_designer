"""
Virtual try-on sidecar (diffusion / warping models).

This is intentionally **not** the same job as garment attribute extraction.
Port the forward pass from your Kaggle training notebook (e.g. diffusion try-on)
into `_run_tryon()` once weights and preprocessing are available locally.

Reference workflow (typical for DeepFashion / VITON-family try-on):
  person image + garment image -> aligned agnostic representation -> generator -> composite
"""

from __future__ import annotations

import base64
import logging
import os
import sys
import threading
from io import BytesIO
from pathlib import Path
from typing import Any

# huggingface_hub reads HUGGING_FACE_HUB_TOKEN; repo .env often uses HF_TOKEN only.
_hf = os.environ.get("HF_TOKEN", "").strip()
if _hf and not os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip():
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf
# More reliable downloads on some networks: avoid Xet path + extend timeout.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

import numpy as np
from fastapi import FastAPI
from PIL import Image, ImageFilter
from pydantic import BaseModel, Field

app = FastAPI(title="Vestir Try-On Sidecar", version="0.1.0")


class TryOnRequest(BaseModel):
    person_image_base64: str
    garment_image_base64: str
    seed: int | None = Field(default=None, description="Optional RNG seed for diffusion")


class TryOffRequest(BaseModel):
    """Virtual try-off: one worn photo → garment(s) on white (fal/virtual-tryoff-lora + FLUX.2)."""

    image_base64: str
    garment_target: str = Field(
        default="outfit",
        description="outfit | ensemble | tshirt | dress | pants | jacket (or free text for custom phrase)",
    )
    seed: int | None = Field(default=None, description="Optional RNG seed")


_TRYOFF_TARGET_PHRASE: dict[str, str] = {
    "outfit": "full outfit in the reference image",
    "full_outfit": "full outfit in the reference image",
    "shirt": "t-shirt",
    "tshirt": "t-shirt",
    "tee": "t-shirt",
    "dress": "dress",
    "pants": "pants",
    "jeans": "pants",
    "jacket": "jacket",
    "coat": "jacket",
}

# Full prompts aligned with fal/virtual-tryoff-lora model card (FLUX.2 + TRYOFF prefix).
_TRYOFF_FLUX_CARD_SUFFIX = (
    "product photography style. NO HUMAN VISIBLE "
    "(the garments maintain their 3D form like an invisible mannequin)."
)

_TRYOFF_FLUX_PROMPTS: dict[str, str] = {
    "outfit": (
        "TRYOFF extract the outfit over a white background, "
        + _TRYOFF_FLUX_CARD_SUFFIX
    ),
    "tshirt": (
        "TRYOFF extract the t-shirt over a white background, "
        + _TRYOFF_FLUX_CARD_SUFFIX
    ),
    "dress": (
        "TRYOFF extract the dress over a white background, "
        + _TRYOFF_FLUX_CARD_SUFFIX
    ),
    "pants": (
        "TRYOFF extract the pants over a white background, "
        + _TRYOFF_FLUX_CARD_SUFFIX
    ),
    "jacket": (
        "TRYOFF extract the jacket over a white background, "
        + _TRYOFF_FLUX_CARD_SUFFIX
    ),
    "ensemble": (
        "TRYOFF extract full outfit in the reference image over a white background, "
        "high-end professional product photography. Present the outfit as a complete, vertically stacked "
        "ensemble arranged as if worn. The items are stacked as if worn. The top-layer garment is dominant, "
        "followed directly by the bottom-layer garment. The footwear is placed below the bottom-layer hem, "
        "aligning with where the feet would naturally be. Lighting: Clean, even, diffused studio lighting "
        "(softbox or beauty dish style). The illumination must highlight all varying textures "
        "(e.g., pebble leather, suede, knit, or canvas) without creating harsh shadows."
    ),
}


def _tryoff_flux_prompt(garment_target: str) -> str | None:
    key = garment_target.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "shirt": "tshirt",
        "tee": "tshirt",
        "full_outfit": "ensemble",
        "stacked_outfit": "ensemble",
        "outfit_stacked": "ensemble",
    }
    key = aliases.get(key, key)
    return _TRYOFF_FLUX_PROMPTS.get(key)


_TRYON_PIPE = None
_TRYON_PIPE_ERR: str | None = None
_LAST_TRYOFF_ERROR: str | None = None
_TRYOFF_WARMUP_ATTEMPTED = False
_TRYOFF_WARMUP_ERROR: str | None = None
_TRYON_INIT_LOCK = threading.Lock()

logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger("vestir.tryon-sidecar")


def _tryon_base_is_flux2() -> bool:
    bm = os.environ.get("TRYON_BASE_MODEL_ID", "").strip()
    return "flux.2" in bm.lower() or "flux2" in bm.lower()


def _flux_transformer_cpu_weights_to_mps(pipe: Any) -> None:
    """
    With accelerate `device_map=balanced`, some transformer submodules (e.g. time_guidance_embed) can remain on CPU
    while denoise runs on MPS, causing: Tensor for argument weight is on cpu but expected on mps.
    """
    import torch

    if _tryon_torch_device() != "mps":
        return
    t = getattr(pipe, "transformer", None)
    if t is None:
        return
    tgt = torch.device("mps")
    moved = 0
    for p in t.parameters():
        if getattr(p, "is_meta", False):
            continue
        if p.device.type == "cpu":
            p.data = p.data.to(tgt)
            moved += 1
    if moved:
        _LOG.info("Moved %d transformer parameters CPU→mps (balanced device_map)", moved)


def _flux_align_peft_lora_devices(pipe: Any) -> None:
    """
    PEFT can leave lora_A/lora_B on CPU while the base module runs on MPS/CUDA (device_map + safetensors load),
    which raises 'Tensor for argument weight is on cpu but expected on mps'. Move each component's LoRA tensors
    to that component's accelerator device (inferred from its first non-CPU parameter).
    """
    import torch

    for comp in ("transformer", "text_encoder"):
        mod = getattr(pipe, comp, None)
        if mod is None:
            continue
        ref_dev: torch.device | None = None
        for p in mod.parameters():
            if getattr(p, "is_meta", False):
                continue
            if p.device.type != "cpu":
                ref_dev = p.device
                break
        if ref_dev is None or ref_dev.type == "cpu":
            continue
        moved = 0
        for pname, p in mod.named_parameters():
            if "lora_" not in pname or getattr(p, "is_meta", False):
                continue
            if p.device != ref_dev:
                p.data = p.data.to(ref_dev)
                moved += 1
        if moved:
            _LOG.info("Aligned %d LoRA tensors to %s (%s)", moved, ref_dev, comp)


def _tryon_use_mps_for_flux_enabled() -> bool:
    """
    On macOS, default to Metal for FLUX so weights load into unified memory instead of
    spiking CPU RAM (which often ends in SIGKILL during from_pretrained).

    Override explicitly: TRYON_USE_MPS_FOR_FLUX=0 (CPU) or =1 (Metal). Unset = auto on Darwin.
    """
    raw = os.environ.get("TRYON_USE_MPS_FOR_FLUX")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if sys.platform != "darwin":
        return False
    try:
        import torch

        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001
        return False


def _tryon_flux_device_map(device: str) -> str | None:
    """
    Diffusers + Accelerate can place FLUX sub-models across devices during load.

    TRYON_FLUX_DEVICE_MAP: unset / balanced / mps / cpu / off (same precision; not quantization).
    Default on MPS: balanced — avoids putting a single ~15+ GiB shard entirely on Metal, which
    often raises PyTorch MPS "Invalid buffer size". Use mps only if your torch/macOS handles it.
    Default on CUDA: unset (normal load + .to(cuda)).
    """
    if not _tryon_base_is_flux2():
        return None
    raw = os.environ.get("TRYON_FLUX_DEVICE_MAP")
    key = (raw or "").strip().lower()
    if key in {"0", "false", "no", "off", "none"}:
        return None
    if key in {"balanced", "mps", "cpu"}:
        if key == "mps" and device != "mps":
            _LOG.warning("TRYON_FLUX_DEVICE_MAP=mps only applies when the pipeline device is MPS (got %s); ignoring", device)
            return None
        return key
    if key != "":
        _LOG.warning("Unknown TRYON_FLUX_DEVICE_MAP=%r; falling back to defaults", raw)
        return None
    if device == "mps":
        return "balanced"
    return None


def _tryon_torch_device() -> str:
    """
    Inference device for Diffusers: cuda > mps (Apple GPU) > cpu.
    Override with TRYON_TORCH_DEVICE or VESTIR_TORCH_DEVICE = cuda | mps | cpu.
    Disable Metal with TRYON_USE_MPS=0.

    On macOS, FLUX.2 defaults to MPS (TRYON_USE_MPS_FOR_FLUX unset) so loading uses unified
    memory; pair with PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 (see start.sh). Set
    TRYON_USE_MPS_FOR_FLUX=0 to force CPU (often OOM for FLUX.2-klein on modest RAM).
    """
    import torch

    override = os.environ.get("TRYON_TORCH_DEVICE", os.environ.get("VESTIR_TORCH_DEVICE", "")).strip().lower()
    if override == "cpu":
        return "cpu"
    if override == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        _LOG.warning("TRYON_TORCH_DEVICE=cuda but CUDA is not available; using CPU")
        return "cpu"
    if override == "mps":
        mps_ok = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        if mps_ok:
            return "mps"
        _LOG.warning("TRYON_TORCH_DEVICE=mps but MPS is not available; using CPU")
        return "cpu"

    use_mps = os.environ.get("TRYON_USE_MPS", "1").strip().lower() not in {"0", "false", "no", "off"}
    flux2 = _tryon_base_is_flux2()
    use_mps_for_flux = _tryon_use_mps_for_flux_enabled()
    if use_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        if flux2 and not use_mps_for_flux:
            _LOG.info(
                "FLUX.2 try-off: TRYON_USE_MPS_FOR_FLUX=0 — using CPU (may OOM while loading). "
                "Unset TRYON_USE_MPS_FOR_FLUX on macOS for Metal (default device_map=balanced), "
                "or set TRYON_TORCH_DEVICE=mps."
            )
        else:
            if flux2 and use_mps_for_flux and sys.platform == "darwin":
                _LOG.info(
                    "FLUX.2 on Apple Silicon: MPS + default TRYON_FLUX_DEVICE_MAP=balanced when unset "
                    "(fp16 weights; balanced avoids a single huge Metal buffer). start.sh sets PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0."
                )
            return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _tryon_pipeline_dtype(device: str, *, use_flux2: bool):
    """Pick torch_dtype for from_pretrained; MPS has weaker bfloat16 support than CUDA."""
    import torch

    if device == "cuda":
        return torch.bfloat16 if use_flux2 else torch.float16
    if device == "mps":
        if os.environ.get("TRYON_MPS_FLOAT32", "").strip().lower() in {"1", "true", "yes", "on"}:
            return torch.float32
        return torch.float16
    return torch.float32


def _tryon_device_status_label() -> str | None:
    """Device string for /health and /tryoff/status (pipeline device if loaded, else chosen target)."""
    try:
        if _TRYON_PIPE is not None:
            d = getattr(_TRYON_PIPE, "device", None)
            return str(d) if d is not None else None
        return _tryon_torch_device()
    except Exception:  # noqa: BLE001
        return None


def _pil_from_bytes(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data)).convert("RGB")


def _center_square(im: Image.Image, size: int = 768) -> Image.Image:
    w, h = im.size
    side = min(w, h)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    return im.crop((x0, y0, x0 + side, y0 + side)).resize((size, size), Image.Resampling.BICUBIC)


def _tryoff_sd_mask(size: tuple[int, int]) -> Image.Image:
    """
    For SD inpainting: white = repaint (background), black = keep (garment / subject).
    Simple center-weighted rectangle; edges become white studio background.
    """
    w, h = size
    arr = np.full((h, w), 255, dtype=np.uint8)
    # Keep most of the subject area untouched to preserve texture/shape details.
    # Only repaint a thin border so SD cleans the background to white.
    x1, x2 = int(0.06 * w), int(0.94 * w)
    y1, y2 = int(0.05 * h), int(0.95 * h)
    arr[y1:y2, x1:x2] = 0
    return Image.fromarray(arr, mode="L")


def _simple_garment_mask(person: Image.Image) -> Image.Image:
    """
    Lightweight torso mask for virtual try-on fallback.
    You should replace this with human parsing / segmentation masks for production.
    """
    arr = np.array(person.convert("RGB"))
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, x2 = int(0.18 * w), int(0.82 * w)
    y1, y2 = int(0.20 * h), int(0.80 * h)
    mask[y1:y2, x1:x2] = 255
    return Image.fromarray(mask, mode="L")


def _flux_download_progress() -> dict[str, Any]:
    """Best-effort progress snapshot for FLUX local cache."""
    base_model = os.environ.get("TRYON_BASE_MODEL_ID", "").strip()
    use_flux2 = "flux.2" in base_model.lower() or "flux2" in base_model.lower()
    if not use_flux2:
        return {"enabled": False, "reason": "not_flux2_base"}

    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--black-forest-labs--FLUX.2-klein-base-9B"
    snap_root = cache_root / "snapshots"
    if not snap_root.exists():
        return {"enabled": True, "cache_exists": False, "present": 0, "total": 8}

    snaps = [p for p in snap_root.iterdir() if p.is_dir()]
    if not snaps:
        return {"enabled": True, "cache_exists": True, "present": 0, "total": 8}
    snap = max(snaps, key=lambda p: p.stat().st_mtime)

    required = [
        "model_index.json",
        "text_encoder/model-00001-of-00004.safetensors",
        "text_encoder/model-00002-of-00004.safetensors",
        "text_encoder/model-00003-of-00004.safetensors",
        "text_encoder/model-00004-of-00004.safetensors",
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    ]
    present = [r for r in required if (snap / r).exists()]
    missing = [r for r in required if not (snap / r).exists()]

    size_bytes = 0
    try:
        for p in cache_root.rglob("*"):
            if p.is_file():
                size_bytes += p.stat().st_size
    except Exception:  # noqa: BLE001
        size_bytes = 0

    return {
        "enabled": True,
        "cache_exists": True,
        "snapshot": snap.name,
        "present": len(present),
        "total": len(required),
        "percent": round((len(present) / len(required)) * 100, 1),
        "missing_shards": missing,
        "cache_size_gb": round(size_bytes / (1024**3), 2),
    }


def _load_tryon_pipe():
    global _TRYON_PIPE, _TRYON_PIPE_ERR
    if _TRYON_PIPE is not None:
        return _TRYON_PIPE
    if _TRYON_PIPE_ERR is not None:
        raise RuntimeError(_TRYON_PIPE_ERR)

    # Single-flight init: prevent duplicate FLUX downloads/loads from concurrent requests.
    with _TRYON_INIT_LOCK:
        if _TRYON_PIPE is not None:
            return _TRYON_PIPE
        if _TRYON_PIPE_ERR is not None:
            raise RuntimeError(_TRYON_PIPE_ERR)

        base_model = os.environ.get("TRYON_BASE_MODEL_ID", "").strip()
        lora_path = os.environ.get("TRYON_LORA_PATH", "").strip()
        use_flux2_pre = "flux.2" in base_model.lower() or "flux2" in base_model.lower()
        local_tryoff = os.path.join(os.path.dirname(__file__), "models", "virtual-tryoff-lora")
        if lora_path in {"fal/virtual-tryoff-lora", "fal/virtual-tryoff-lora/"} and os.path.isdir(local_tryoff):
            lora_path = local_tryoff
        if not base_model:
            _TRYON_PIPE_ERR = "TRYON_BASE_MODEL_ID is missing"
            raise RuntimeError(_TRYON_PIPE_ERR)
        if use_flux2_pre and not lora_path:
            _TRYON_PIPE_ERR = "TRYON_LORA_PATH is missing (required for FLUX tryoff LoRA)"
            raise RuntimeError(_TRYON_PIPE_ERR)

        try:
            import torch
            import diffusers
            Flux2KleinPipeline = getattr(diffusers, "Flux2KleinPipeline", None)
            AutoPipelineForInpainting = getattr(diffusers, "AutoPipelineForInpainting", None)
        except Exception as exc:  # noqa: BLE001
            _TRYON_PIPE_ERR = f"Missing diffusers/torch deps: {exc}"
            raise RuntimeError(_TRYON_PIPE_ERR) from exc

        try:
            hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip() or None
            # Only args Flux2KleinPipeline.from_pretrained accepts (max_workers is for hub download, not here).
            load_kw: dict[str, Any] = {}
            if hf_token:
                load_kw["token"] = hf_token
            use_flux2 = "flux.2" in base_model.lower() or "flux2" in base_model.lower()
            device = _tryon_torch_device()
            dtype = _tryon_pipeline_dtype(device, use_flux2=use_flux2)
            low_mem_opt_out = os.environ.get("TRYON_LOW_CPU_MEM_USAGE", "").strip().lower() in {"0", "false", "no", "off"}
            flux_dm = _tryon_flux_device_map(device) if use_flux2 else None
            # device_map for FLUX is applied in the load loop below (with mps→balanced retry).
            if flux_dm is not None:
                _LOG.info(
                    "FLUX load: device_map=%s (dtype=%s; full-precision weights, not quantized)",
                    flux_dm,
                    dtype,
                )
            elif device == "cpu" and not low_mem_opt_out:
                load_kw["low_cpu_mem_usage"] = True
                _LOG.info(
                    "Using low_cpu_mem_usage=True for CPU load (reduces peak RAM; set TRYON_LOW_CPU_MEM_USAGE=0 to disable)"
                )
            elif use_flux2 and device == "mps" and not low_mem_opt_out:
                load_kw["low_cpu_mem_usage"] = True
                _LOG.info(
                    "FLUX on MPS without device_map: low_cpu_mem_usage=True; set TRYON_FLUX_DEVICE_MAP=balanced (default) or omit it"
                )
            elif use_flux2:
                load_kw["low_cpu_mem_usage"] = False
            if use_flux2:
                if Flux2KleinPipeline is None:
                    raise RuntimeError("Installed diffusers version does not expose Flux2KleinPipeline")
                map_try: str | None = flux_dm
                pipe = None
                while True:
                    flux_load = dict(load_kw)
                    if map_try is not None:
                        flux_load["device_map"] = map_try
                        flux_load["low_cpu_mem_usage"] = True
                    try:
                        pipe = Flux2KleinPipeline.from_pretrained(
                            base_model,
                            torch_dtype=dtype,
                            **flux_load,
                        )
                        break
                    except Exception as flux_exc:
                        err_low = str(flux_exc).lower()
                        if (
                            map_try == "mps"
                            and device == "mps"
                            and "invalid buffer size" in err_low
                        ):
                            _LOG.warning(
                                "device_map=mps exceeded MPS single-buffer limit (~15 GiB). Retrying with device_map=balanced "
                                "(same fp16 weights; split across CPU/MPS). Remove TRYON_FLUX_DEVICE_MAP=mps from .env to use the default."
                            )
                            map_try = "balanced"
                            continue
                        raise
                assert pipe is not None
            else:
                if AutoPipelineForInpainting is None:
                    raise RuntimeError("Installed diffusers version does not expose AutoPipelineForInpainting")
                pipe = AutoPipelineForInpainting.from_pretrained(
                    base_model,
                    torch_dtype=dtype,
                    safety_checker=None,
                    **load_kw,
                )
            if getattr(pipe, "_is_pipeline_device_mapped", lambda: False)():
                _LOG.info("Try-on/off pipeline: multi-component device_map active; skipping .to(%s)", device)
            else:
                pipe = pipe.to(device)
            _LOG.info("Try-on/off pipeline device=%s dtype=%s", device, dtype)
            # fal/virtual-tryoff-lora is trained for FLUX.2 — do not load it into SD 1.5 inpainting.
            if use_flux2:
                weight_name = os.environ.get("TRYON_LORA_WEIGHT_NAME", "").strip() or None
                adapter_name = os.environ.get("TRYON_LORA_ADAPTER_NAME", "").strip() or "vtoff"
                # Diffusers defaults LoRA load to low_cpu_mem_usage=True (peft), which often leaves adapter weights on CPU.
                # Any accelerator (MPS/CUDA) then hits "weight is on cpu but expected on mps/cuda" during inference.
                lora_low_cpu_mem = device == "cpu"
                if device != "cpu":
                    _LOG.info("FLUX LoRA load: low_cpu_mem_usage=False (device=%s)", device)
                if weight_name:
                    pipe.load_lora_weights(
                        lora_path,
                        weight_name=weight_name,
                        adapter_name=adapter_name,
                        low_cpu_mem_usage=lora_low_cpu_mem,
                    )
                else:
                    pipe.load_lora_weights(
                        lora_path,
                        adapter_name=adapter_name,
                        low_cpu_mem_usage=lora_low_cpu_mem,
                    )
                lora_scale = float(os.environ.get("TRYON_LORA_SCALE", "0.85"))
                pipe.set_adapters([adapter_name], adapter_weights=[lora_scale])
                _flux_align_peft_lora_devices(pipe)
                _flux_transformer_cpu_weights_to_mps(pipe)
                fuse_ok = os.environ.get("TRYON_FUSE_LORA", "1").strip().lower() in {"1", "true", "yes", "on"}
                # fuse_lora materializes merged weights; with accelerate device_map (e.g. balanced on MPS) that
                # often lands on CPU while the transformer runs on MPS → "weight is on cpu but expected on mps".
                if fuse_ok and getattr(pipe, "_is_pipeline_device_mapped", lambda: False)():
                    _LOG.warning(
                        "Skipping fuse_lora: multi-component device_map (PEFT adapters stay on each module's device)."
                    )
                    fuse_ok = False
                if fuse_ok:
                    pipe.fuse_lora(adapter_names=[adapter_name], lora_scale=lora_scale)
            _TRYON_PIPE = pipe
            return _TRYON_PIPE
        except Exception as exc:  # noqa: BLE001
            _TRYON_PIPE_ERR = f"Try-on pipeline init failed: {exc}"
            raise RuntimeError(_TRYON_PIPE_ERR) from exc


def reset_tryon_pipeline_cache() -> None:
    """Clear cached pipeline so the next load retries (e.g. after fixing HF token or disk)."""
    global _TRYON_PIPE, _TRYON_PIPE_ERR
    _TRYON_PIPE = None
    _TRYON_PIPE_ERR = None


def initialize_tryoff_pipeline(*, force_reload: bool = False) -> dict[str, Any]:
    """
    Load Diffusers weights into memory. Call at startup and/or POST /tryoff/warmup.
    """
    global _TRYOFF_WARMUP_ATTEMPTED, _TRYOFF_WARMUP_ERROR
    _TRYOFF_WARMUP_ATTEMPTED = True
    if force_reload:
        reset_tryon_pipeline_cache()
    try:
        _load_tryon_pipe()
        _TRYOFF_WARMUP_ERROR = None
        _LOG.info("Tryoff pipeline ready (%s)", os.environ.get("TRYON_BASE_MODEL_ID", ""))
        return {"ok": True, "pipeline_ready": True, "error": None}
    except Exception as e:  # noqa: BLE001
        err = str(e)
        _TRYOFF_WARMUP_ERROR = err
        _LOG.warning("Tryoff pipeline init failed: %s", err)
        return {"ok": False, "pipeline_ready": False, "error": err}


def _tryoff_startup_warmup_sync() -> None:
    flag = os.environ.get("TRYOFF_WARMUP_ON_START", "1").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        _LOG.info("TRYOFF_WARMUP_ON_START disabled; pipeline loads on first /tryoff")
        return
    initialize_tryoff_pipeline(force_reload=False)


def _run_tryon(person_png: bytes, garment_png: bytes, seed: int | None) -> bytes | None:
    """
    LoRA-based virtual try-on fallback using Diffusers inpainting.
    Env required:
    - TRYON_BASE_MODEL_ID (e.g. runwayml/stable-diffusion-inpainting)
    - TRYON_LORA_PATH (local path or HF repo for VTON LoRA)
    Optional:
    - TRYON_LORA_SCALE (default 0.85)
    - TRYON_STEPS (default 28)
    - TRYON_GUIDANCE (default 7.5)
    """
    base_model = os.environ.get("TRYON_BASE_MODEL_ID", "").strip()
    lora_path = os.environ.get("TRYON_LORA_PATH", "").strip()
    if not base_model:
        return None
    use_flux2_tryon = "flux.2" in base_model.lower() or "flux2" in base_model.lower()
    if use_flux2_tryon and not lora_path:
        return None
    try:
        import torch

        person = _center_square(_pil_from_bytes(person_png), 768)
        garment = _center_square(_pil_from_bytes(garment_png), 768)
        mask = _simple_garment_mask(person)
        pipe = _load_tryon_pipe()
        prompt = (
            "high quality fashion photo, person wearing the provided garment, "
            "natural fit, realistic fabric details, studio lighting"
        )
        negative_prompt = (
            "deformed body, extra limbs, blurry, low quality, artifacts, distorted clothes"
        )
        g = None
        if seed is not None:
            g = torch.Generator(device=pipe.device).manual_seed(int(seed))
        steps = int(os.environ.get("TRYON_STEPS", "28"))
        guidance = float(os.environ.get("TRYON_GUIDANCE", "7.5"))
        # Blend garment signal in prompt-time via preview strip to preserve API shape until
        # dedicated VTON garment conditioning is added.
        garment_hint = garment.resize((256, 256))
        person_np = np.array(person)
        hint_np = np.array(garment_hint.resize((person.width // 3, person.height // 3)))
        person_np[0:hint_np.shape[0], 0:hint_np.shape[1], :] = hint_np
        init_img = Image.fromarray(person_np)
        base_model = os.environ.get("TRYON_BASE_MODEL_ID", "").strip()
        use_flux2 = "flux.2" in base_model.lower() or "flux2" in base_model.lower()
        if use_flux2:
            tryoff_prompt = os.environ.get(
                "TRYON_PROMPT",
                (
                    "TRYOFF extract full outfit in the reference image over a white background, "
                    "high-end professional product photography. NO HUMAN VISIBLE (the garments "
                    "maintain their 3D form like an invisible mannequin)."
                ),
            )
            result = pipe(
                image=init_img,
                prompt=tryoff_prompt,
                height=1024,
                width=768,
                num_inference_steps=steps,
                guidance_scale=float(os.environ.get("TRYON_GUIDANCE", "5.0")),
                generator=g,
            ).images[0]
        else:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_img,
                mask_image=mask,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=g,
                strength=0.92,
            ).images[0]
        out = BytesIO()
        result.save(out, format="JPEG", quality=92)
        return out.getvalue()
    except Exception:  # noqa: BLE001
        return None


def _tryoff_phrase(garment_target: str) -> str:
    key = garment_target.strip().lower().replace(" ", "_").replace("-", "_")
    return _TRYOFF_TARGET_PHRASE.get(key, garment_target.strip())


def _run_tryoff(image_bytes: bytes, garment_target: str, seed: int | None) -> bytes | None:
    """
    Single-image virtual try-off (garment extraction on white).
    Requires FLUX.2 base + fal/virtual-tryoff-lora (same env as TRYON_*).
    """
    base_model = os.environ.get("TRYON_BASE_MODEL_ID", "").strip()
    lora_path = os.environ.get("TRYON_LORA_PATH", "").strip()
    if not base_model:
        global _LAST_TRYOFF_ERROR
        _LAST_TRYOFF_ERROR = "missing TRYON_BASE_MODEL_ID"
        return None
    use_flux2 = "flux.2" in base_model.lower() or "flux2" in base_model.lower()
    require_flux = os.environ.get("TRYOFF_REQUIRE_FLUX", "1").strip().lower() in {"1", "true", "yes", "on"}
    if require_flux and not use_flux2:
        _LAST_TRYOFF_ERROR = (
            "TRYOFF_REQUIRE_FLUX=1, but TRYON_BASE_MODEL_ID is not FLUX.2. "
            "Set TRYON_BASE_MODEL_ID=black-forest-labs/FLUX.2-klein-base-9B."
        )
        return None
    # Non-FLUX: real try-off via SD inpainting (no FLUX LoRA — incompatible arch).
    if not use_flux2:
        try:
            import torch

            pil = _pil_from_bytes(image_bytes)
            max_edge = int(os.environ.get("TRYOFF_SD_MAX_EDGE", "1024"))
            w, h = pil.size
            if max(w, h) > max_edge:
                scale = max_edge / float(max(w, h))
                pil = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

            phrase = _tryoff_phrase(garment_target)
            prompt = os.environ.get(
                "TRYOFF_SD_PROMPT",
                (
                    f"high-end e-commerce studio product photo of the {phrase}, isolated on pure white background, "
                    "ghost mannequin / invisible mannequin presentation, three-dimensional drape and natural garment volume, "
                    "accurate texture weave and stitching, realistic shadows, high micro-detail, no human visible"
                ),
            )
            negative = os.environ.get(
                "TRYOFF_SD_NEGATIVE",
                (
                    "face, hands, skin, person, body, mannequin visible, cluttered scene, text, logo watermark, "
                    "cartoon, illustration, flat graphic, low quality, blurry, deformed fabric"
                ),
            )
            mask = _tryoff_sd_mask(pil.size)
            pipe = _load_tryon_pipe()
            dev = getattr(pipe, "device", "cpu")
            g = None
            if seed is not None:
                g = torch.Generator(device=dev).manual_seed(int(seed))
            steps = int(os.environ.get("TRYOFF_SD_STEPS", os.environ.get("TRYON_STEPS", "36")))
            guidance = float(os.environ.get("TRYOFF_SD_GUIDANCE", os.environ.get("TRYON_GUIDANCE", "6.0")))
            strength = float(os.environ.get("TRYOFF_SD_STRENGTH", "0.92"))

            result = pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=pil,
                mask_image=mask,
                num_inference_steps=steps,
                guidance_scale=guidance,
                strength=strength,
                generator=g,
            ).images[0]

            if os.environ.get("TRYOFF_SD_UNSHARP", "1").strip().lower() in {"1", "true", "yes", "on"}:
                # Mild local contrast boost to retain textile detail in JPEG output.
                result = result.filter(ImageFilter.UnsharpMask(radius=1.2, percent=115, threshold=3))
            out = BytesIO()
            result.save(out, format="JPEG", quality=95)
            _LAST_TRYOFF_ERROR = None
            return out.getvalue()
        except Exception as exc:  # noqa: BLE001
            _LAST_TRYOFF_ERROR = f"SD tryoff failed: {exc}"
            # Passthrough so pipeline still runs if diffusion fails (e.g. OOM on CPU).
            if os.environ.get("TRYOFF_SD_PASSTHROUGH_ON_ERROR", "1").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                try:
                    pil = _pil_from_bytes(image_bytes)
                    max_edge = int(os.environ.get("TRYOFF_FALLBACK_MAX_EDGE", "1536"))
                    w, h = pil.size
                    if max(w, h) > max_edge:
                        scale = max_edge / float(max(w, h))
                        pil = pil.resize(
                            (max(1, int(w * scale)), max(1, int(h * scale))),
                            Image.Resampling.LANCZOS,
                        )
                    out = BytesIO()
                    pil.save(out, format="JPEG", quality=92)
                    return out.getvalue()
                except Exception:  # noqa: BLE001
                    return None
            return None
    if not lora_path:
        _LAST_TRYOFF_ERROR = "missing TRYON_LORA_PATH for FLUX tryoff"
        return None
    try:
        import torch

        pil = _pil_from_bytes(image_bytes)
        w, h = pil.size
        max_edge = int(os.environ.get("TRYOFF_MAX_INPUT_EDGE", "1536"))
        if max(w, h) > max_edge:
            scale = max_edge / float(max(w, h))
            pil = pil.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        card_prompt = _tryoff_flux_prompt(garment_target)
        if card_prompt:
            prompt = card_prompt
        else:
            phrase = _tryoff_phrase(garment_target)
            tpl = os.environ.get(
                "TRYOFF_PROMPT_TEMPLATE",
                (
                    "TRYOFF extract the {phrase} over a white background, product photography style. "
                    "NO HUMAN VISIBLE (the garments maintain their 3D form like an invisible mannequin)."
                ),
            )
            prompt = tpl.format(phrase=phrase) if "{phrase}" in tpl else tpl

        pipe = _load_tryon_pipe()
        # With accelerate `device_map`, `pipe.device` is often `meta` and `_execution_device` may still resolve to
        # meta when no offload hooks exist — latents/seed must use a real accelerator (mps/cuda) or we get CPU/MPS
        # mismatches during denoise.
        exec_d = getattr(pipe, "_execution_device", None)
        if exec_d is not None and getattr(exec_d, "type", "") not in ("", "meta"):
            gen_device = exec_d.type if exec_d.index is None else f"{exec_d.type}:{exec_d.index}"
        else:
            gen_device = _tryon_torch_device()
        g = None
        if seed is not None:
            g = torch.Generator(device=gen_device).manual_seed(int(seed))

        out_h = int(os.environ.get("TRYOFF_HEIGHT", "1024"))
        out_w = int(os.environ.get("TRYOFF_WIDTH", "768"))
        steps = int(os.environ.get("TRYON_STEPS", "28"))
        guidance = float(os.environ.get("TRYON_GUIDANCE", "5.0"))

        result = pipe(
            image=pil,
            prompt=prompt,
            height=out_h,
            width=out_w,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=g,
        ).images[0]
        out = BytesIO()
        result.save(out, format="JPEG", quality=92)
        _LAST_TRYOFF_ERROR = None
        return out.getvalue()
    except Exception as exc:  # noqa: BLE001
        import traceback

        _LAST_TRYOFF_ERROR = f"{exc}\n{traceback.format_exc()}"
        return None


@app.on_event("startup")
def _startup_tryoff_background() -> None:
    """
    Do not block ASGI startup on model download/load — otherwise /health never answers
    and vestir start.sh times out waiting for tryon-sidecar.
    """

    def _run() -> None:
        try:
            _tryoff_startup_warmup_sync()
        except Exception:  # noqa: BLE001
            _LOG.exception("tryoff background warmup thread crashed")

    threading.Thread(target=_run, name="vestir-tryoff-warmup", daemon=True).start()


@app.get("/health")
def health() -> dict[str, Any]:
    configured = bool(os.environ.get("TRYON_BASE_MODEL_ID", "").strip())
    bm = os.environ.get("TRYON_BASE_MODEL_ID", "").strip()
    flux2 = "flux.2" in bm.lower() or "flux2" in bm.lower()
    pipe_ready = _TRYON_PIPE is not None
    return {
        "ok": True,
        "service": "tryon-sidecar",
        "model_configured": configured,
        "tryoff_capable": bool(configured),
        "tryoff_pipeline_ready": pipe_ready,
        "tryoff_warmup_attempted": _TRYOFF_WARMUP_ATTEMPTED,
        "tryoff_warmup_error": _TRYOFF_WARMUP_ERROR,
        "torch_device": _tryon_device_status_label(),
        "base_model": bm or None,
        "lora_path": os.environ.get("TRYON_LORA_PATH", "").strip() or None,
        "mode": "flux2-lora" if flux2 else "sd-inpaint-tryoff",
        "download_progress": _flux_download_progress(),
    }


@app.get("/tryoff/status")
def tryoff_status() -> dict[str, Any]:
    bm = os.environ.get("TRYON_BASE_MODEL_ID", "").strip()
    flux2 = "flux.2" in bm.lower() or "flux2" in bm.lower()
    return {
        "pipeline_ready": _TRYON_PIPE is not None,
        "warmup_attempted": _TRYOFF_WARMUP_ATTEMPTED,
        "warmup_error": _TRYOFF_WARMUP_ERROR,
        "cached_load_error": _TRYON_PIPE_ERR,
        "torch_device": _tryon_device_status_label(),
        "base_model": bm or None,
        "mode": "flux2-lora" if flux2 else "sd-inpaint-tryoff",
        "warmup_on_start": os.environ.get("TRYOFF_WARMUP_ON_START", "1"),
        "download_progress": _flux_download_progress(),
    }


@app.post("/tryoff/warmup")
def tryoff_warmup(force: bool = True) -> dict[str, Any]:
    """Force reload Diffusers weights (e.g. after fixing HF access). Query: ?force=false to no-op if already loaded."""
    if not force and _TRYON_PIPE is not None:
        return {"ok": True, "pipeline_ready": True, "error": None, "note": "already loaded"}
    return {**initialize_tryoff_pipeline(force_reload=force), "note": None}


@app.post("/tryon")
def tryon(payload: TryOnRequest) -> dict[str, Any]:
    try:
        person = base64.b64decode(payload.person_image_base64)
        garment = base64.b64decode(payload.garment_image_base64)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"invalid_base64: {exc}"}

    result = _run_tryon(person, garment, payload.seed)
    if result is None:
        return {
            "ok": True,
            "implemented": False,
            "message": "Set TRYON_BASE_MODEL_ID and TRYON_LORA_PATH (Diffusers + VTON LoRA) to enable try-on.",
            "result_image_base64": None,
        }

    return {
        "ok": True,
        "implemented": True,
        "message": None,
        "result_image_base64": base64.b64encode(result).decode("ascii"),
    }


@app.post("/tryoff")
def tryoff(payload: TryOffRequest) -> dict[str, Any]:
    try:
        raw = base64.b64decode(payload.image_base64)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"invalid_base64: {exc}"}

    result = _run_tryoff(raw, payload.garment_target, payload.seed)
    if result is None:
        return {
            "ok": True,
            "implemented": False,
            "message": (
                "Try-off failed. Use FLUX.2 + fal/virtual-tryoff-lora, or SD inpainting "
                "(runwayml/stable-diffusion-inpainting) without the FLUX LoRA. See debug_error."
            ),
            "debug_error": _LAST_TRYOFF_ERROR,
            "result_image_base64": None,
        }

    return {
        "ok": True,
        "implemented": True,
        "message": None,
        "result_image_base64": base64.b64encode(result).decode("ascii"),
    }
