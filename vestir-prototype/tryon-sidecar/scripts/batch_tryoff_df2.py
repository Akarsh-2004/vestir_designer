#!/usr/bin/env python3
"""
Batch virtual try-off for DeepFashion2-style layout (df2_v1) or any image folder.

Requires tryon-sidecar running (default http://127.0.0.1:8009).

Examples:
  # All val images (slow) → output dir
  python3 scripts/batch_tryoff_df2.py \\
    --df2-root ../vision-sidecar/datasets/df2_v1 \\
    --split val \\
    --out-dir ./outputs/df2_tryoff_val

  # First 20 train images, faster SD settings via env
  TRYOFF_SD_STEPS=10 TRYOFF_SD_MAX_EDGE=512 \\
    python3 scripts/batch_tryoff_df2.py --df2-root ... --split train --limit 20 --out-dir ./out

  # Flat folder of images (no YOLO labels)
  python3 scripts/batch_tryoff_df2.py --images-dir /path/to/jpgs --out-dir ./out --garment-default outfit
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Must match vision-sidecar/datasets/df2_v1/data.yaml (class_id -> tryoff garment_target)
DF2_CLASS_TO_TRYOFF: dict[int, str] = {
    0: "outfit",  # tops — full upper context
    1: "jacket",  # outerwear
    2: "pants",  # bottoms
    3: "dress",
}


def _post_tryoff(
    base: str,
    image_bytes: bytes,
    garment_target: str,
    timeout_sec: float,
) -> tuple[bool, bytes | None, str | None]:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    body = json.dumps({"image_base64": b64, "garment_target": garment_target}).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/tryoff",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return False, None, f"HTTP {e.code}: {e.read().decode()[:400]}"
    except Exception as e:  # noqa: BLE001
        return False, None, str(e)

    if not payload.get("ok"):
        return False, None, payload.get("error") or "ok=false"

    b64out = payload.get("result_image_base64")
    if not b64out:
        return False, None, payload.get("debug_error") or payload.get("message") or "no image"

    implemented = bool(payload.get("implemented"))
    try:
        raw = base64.b64decode(b64out)
    except Exception as e:  # noqa: BLE001
        return False, None, f"decode: {e}"
    return implemented, raw, payload.get("debug_error") or payload.get("message")


def _first_label_class(label_path: Path) -> int | None:
    if not label_path.is_file():
        return None
    try:
        line = label_path.read_text().strip().splitlines()[0].split()
        if not line:
            return None
        return int(float(line[0]))
    except (IndexError, ValueError):
        return None


def _collect_df2_images(df2_root: Path, split: str) -> list[tuple[Path, Path | None]]:
    """Return list of (image_path, optional_label_path)."""
    images_root = df2_root / "images"
    labels_root = df2_root / "labels"
    out: list[tuple[Path, Path | None]] = []
    splits = ["train", "val"] if split == "both" else [split]
    for sp in splits:
        img_dir = images_root / sp
        lbl_dir = labels_root / sp
        if not img_dir.is_dir():
            print(f"Warning: missing {img_dir}", file=sys.stderr)
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            lbl = lbl_dir / f"{p.stem}.txt"
            out.append((p, lbl if lbl.is_file() else None))
    return out


def _collect_flat_dir(images_dir: Path) -> list[tuple[Path, Path | None]]:
    out: list[tuple[Path, Path | None]] = []
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            out.append((p, None))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch try-off for DF2 or a folder of images.")
    ap.add_argument(
        "--df2-root",
        type=Path,
        help="Path to df2_v1 (contains images/{train,val} and labels/{train,val})",
    )
    ap.add_argument(
        "--images-dir",
        type=Path,
        help="Flat directory of images (alternative to --df2-root)",
    )
    ap.add_argument(
        "--split",
        choices=("train", "val", "both"),
        default="val",
        help="DF2 split when using --df2-root",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write try-off JPEGs + manifest.jsonl",
    )
    ap.add_argument("--limit", type=int, default=0, help="Max images (0 = no limit)")
    ap.add_argument(
        "--garment-default",
        default="outfit",
        help="Tryoff target when no YOLO label (flat dir or missing .txt)",
    )
    ap.add_argument(
        "--tryoff-url",
        default=os.environ.get("TRYON_SIDECAR_URL", "http://127.0.0.1:8009").rstrip("/"),
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("TRYOFF_BATCH_TIMEOUT_SEC", "600")),
        help="Per-image HTTP timeout (seconds)",
    )
    args = ap.parse_args()

    if bool(args.df2_root) == bool(args.images_dir):
        ap.error("Specify exactly one of --df2-root or --images-dir")

    if args.df2_root:
        df2_root = args.df2_root.expanduser().resolve()
        pairs = _collect_df2_images(df2_root, args.split)
    else:
        images_dir = args.images_dir.expanduser().resolve()
        if not images_dir.is_dir():
            print(f"Not a directory: {images_dir}", file=sys.stderr)
            return 2
        pairs = _collect_flat_dir(images_dir)

    if not pairs:
        print("No images found.", file=sys.stderr)
        return 1

    if args.limit > 0:
        pairs = pairs[: args.limit]

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"

    base = args.tryoff_url.rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=15) as r:
            health = json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        print(f"Health check failed: {e}", file=sys.stderr)
        print("Start: cd tryon-sidecar && TRYON_UVICORN_RELOAD=0 ./start.sh", file=sys.stderr)
        return 1

    if not health.get("tryoff_pipeline_ready"):
        print("Warning: tryoff_pipeline_ready is false — first images may be slow or fail.", file=sys.stderr)

    ok_n = 0
    fail_n = 0
    with manifest.open("w", encoding="utf-8") as mf:
        for i, (img_path, lbl_path) in enumerate(pairs):
            stem = img_path.stem
            garment = args.garment_default
            if lbl_path is not None:
                cid = _first_label_class(lbl_path)
                if cid is not None and cid in DF2_CLASS_TO_TRYOFF:
                    garment = DF2_CLASS_TO_TRYOFF[cid]

            t0 = time.perf_counter()
            data = img_path.read_bytes()
            implemented, out_bytes, err = _post_tryoff(base, data, garment, args.timeout)
            dt = time.perf_counter() - t0

            out_name = f"{stem}_tryoff.jpg"
            out_path = out_dir / out_name
            rec = {
                "index": i,
                "source": str(img_path),
                "garment_target": garment,
                "implemented": implemented,
                "seconds": round(dt, 3),
                "output": str(out_path) if out_bytes else None,
                "error": err,
            }
            mf.write(json.dumps(rec) + "\n")

            if out_bytes:
                out_path.write_bytes(out_bytes)
                ok_n += 1
                impl_tag = "impl" if implemented else "fallback"
                print(f"[{i + 1}/{len(pairs)}] {impl_tag} {out_name} ({dt:.1f}s) target={garment}")
            else:
                fail_n += 1
                print(f"[{i + 1}/{len(pairs)}] FAIL {img_path.name}: {err}", file=sys.stderr)

    print(f"\nDone. wrote {ok_n} images, {fail_n} failures → {out_dir}")
    print(f"Manifest: {manifest}")
    return 0 if fail_n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
