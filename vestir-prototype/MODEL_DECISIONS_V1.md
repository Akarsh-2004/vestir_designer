# Vestir V1 Model Decisions

This document captures what we are finalizing for V1, which models we are using, and which option gives the best operations tradeoff for your current device constraints.

## Finalized V1 Stack

## Detection + Privacy + Attributes
- **Garment detection (primary):** YOLO nano (`yolov8n`) fine-tuned on DeepFashion2.
- **Detection serving:** local Python sidecar (`vision-sidecar`) via `/analyze`.
- **Privacy redaction:** local face-aware masking before cloud inference (blur for worn/selfie scenes).
- **Color extraction:** local classical CV (HSL/LAB palette logic on cropped garment).
- **Attribute enrichment:** Gemini on already-cropped garments only (strict fallback when unavailable).
- **Reasoning text:** local/sidecar reasoning stage (Ollama path already present in backend).

## Training Configuration (V1)
- **Dataset mapping:** DeepFashion2 -> 4 classes:
  - `0=tops`, `1=outerwear`, `2=bottoms`, `3=dress`
- **Prepared subset for first stable run:** `8000 train / 1200 val`
- **Low-memory defaults:**
  - `imgsz=640`
  - `batch=4..8` (start 6 or 8, reduce if memory pressure)
  - `workers=1..2`
  - `save_period=1` (checkpoint every epoch)
- **Resumability:** always resume from `runs/detect/df2-v1-lowmem/weights/last.pt`

## Why this is finalized for V1
- Meets privacy requirement with local redaction before cloud use.
- Fits a constrained workstation better than larger models or segmentation-heavy stacks.
- Gives deterministic color extraction and avoids overloading Gemini.
- Maintains progress safely via checkpoints (power off / reboot safe).

## Model/Approach Comparison for Ops

## Option A: Existing Hybrid (YOLO sidecar + backend heuristics + Gemini)
- **Ops score:** High
- **Pros:** already integrated, resilient fallbacks, fast to ship.
- **Cons:** quality can drift if detector outputs weak boxes/classes.

## Option B: YOLO-only local-first (primary V1 direction)
- **Ops score:** Highest for current hardware
- **Pros:** privacy strongest, predictable cost/latency, no cloud dependency for detection.
- **Cons:** needs fine-tune quality and good checkpoint monitoring.

## Option C: Google Vision-centered detection
- **Ops score:** Medium
- **Pros:** easy startup for generic detection.
- **Cons:** cloud dependency/cost/privacy concerns; weaker garment-specific precision unless custom Vision training.

## Option D: Open-vocabulary + SAM segmentation
- **Ops score:** Lowest for V1
- **Pros:** high quality ceiling for overlaps/multi-person scenes.
- **Cons:** heavy compute + integration complexity, not ideal for first production pass on this device.

## Final Recommendation
- **V1 final choice:** Option B on top of existing hybrid wiring:
  - Local fine-tuned YOLO for garment detection
  - Local privacy masking
  - Local color extraction
  - Gemini only for cropped-garment attributes (with strict fallback contract)

This gives the best balance of privacy, quality, and operational stability on your current machine.

## What gives better ops in practice
- Smaller model + stable pipeline beats larger model + fragile runtime.
- Deterministic local color extraction reduces cloud retries and latency.
- Checkpoint-every-epoch + resume keeps long training jobs manageable.
- Strong fallback schema avoids silent failures in UI/DB.

## Out of Scope for V1
- Full segmentation-first production path (GroundingDINO+SAM).
- Fine-grained 13+ class production taxonomy in first release.
- Cloud-only perception pipeline.

## V2 Upgrade Path
- Move from bbox-first to mask-aware extraction for overlap-heavy images.
- Add per-person garment association in multi-person scenes.
- Expand class taxonomy once V1 metrics are stable.
