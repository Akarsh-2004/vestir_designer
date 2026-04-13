import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Set

from dotenv import load_dotenv

import benchmark_label_models as bm


DF2_ROOT = Path("/Users/akarshsaklani/Desktop/vestir/vestir-prototype/vision-sidecar/datasets/df2_v1")

# Map DF2_v1 4-class training labels to benchmark canonical labels.
DF2_CLASS_TO_CANON = {
    0: {"top"},
    1: {"jacket"},
    2: {"pants"},
    3: {"dress"},
}


def _read_gt_from_yolo_txt(label_path: Path) -> Set[str]:
    if not label_path.exists():
        return set()
    out: Set[str] = set()
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cls = int(parts[0])
        except Exception:
            continue
        out |= DF2_CLASS_TO_CANON.get(cls, set())
    return out


def _collect_split(split: str, max_images: int) -> tuple[List[Path], Dict[str, Set[str]]]:
    img_dir = DF2_ROOT / "images" / split
    lbl_dir = DF2_ROOT / "labels" / split
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if max_images > 0:
        images = images[: max_images]
    gt: Dict[str, Set[str]] = {}
    for img in images:
        gt[img.name] = _read_gt_from_yolo_txt(lbl_dir / f"{img.stem}.txt")
    return images, gt


def _print_metrics(tag: str, m: bm.Metrics) -> None:
    print(
        f"{tag:14} | hit_rate={m.hit_rate:.3f} | exact_match={m.exact_match:.3f} | avg_jaccard={m.avg_jaccard:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DF2 small ops benchmark with annotation GT.")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--max-images", type=int, default=40, help="Total images for YOLO/SigLIP/Florence.")
    parser.add_argument("--ollama-max-images", type=int, default=25, help="Images to run for Ollama single model.")
    parser.add_argument("--print-predictions", action="store_true")
    parser.add_argument("--ollama-model", default="llama3.2-vision:11b")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-only", action="store_true", help="Run only Ollama model benchmark.")
    args = parser.parse_args()

    load_dotenv("/Users/akarshsaklani/Desktop/vestir/.env")

    images, gt = _collect_split(args.split, args.max_images)
    if not images:
        raise RuntimeError(f"No images found in split {args.split}")
    print(f"Split={args.split} | Images={len(images)} | Ollama cap={min(len(images), args.ollama_max_images)}")

    latency: Dict[str, float] = {}
    t0 = time.perf_counter()

    yolo_preds = {img.name: set() for img in images}
    siglip_preds = {img.name: set() for img in images}
    florence_preds = {img.name: set() for img in images}
    if not args.ollama_only:
        s = time.perf_counter()
        yolo_preds = bm.predict_yolo(images, print_predictions=args.print_predictions)
        latency["yolo"] = time.perf_counter() - s

        s = time.perf_counter()
        siglip_preds = bm.predict_siglip(images, print_predictions=args.print_predictions)
        latency["siglip"] = time.perf_counter() - s

        s = time.perf_counter()
        florence_preds = bm.predict_florence(images, print_predictions=args.print_predictions)
        latency["florence"] = time.perf_counter() - s

    # Single Ollama model only, capped subset for ops speed.
    ollama_images = images[: min(len(images), max(0, args.ollama_max_images))]
    s = time.perf_counter()
    ollama_part = bm.predict_ollama(
        ollama_images,
        args.ollama_model,
        args.ollama_base_url,
        print_predictions=args.print_predictions,
    )
    latency["ollama_single"] = time.perf_counter() - s
    ollama_preds = {img.name: set() for img in images}
    ollama_preds.update(ollama_part)

    hybrid_preds = bm.hybrid_vote(yolo_preds, siglip_preds, florence_preds, ollama_preds)

    scores = {"ollama_single": bm.evaluate(ollama_preds, gt)}
    if not args.ollama_only:
        scores.update(
            {
                "yolo": bm.evaluate(yolo_preds, gt),
                "siglip": bm.evaluate(siglip_preds, gt),
                "florence": bm.evaluate(florence_preds, gt),
                "hybrid": bm.evaluate(hybrid_preds, gt),
            }
        )

    print("\n=== DF2 Small Ops Benchmark ===")
    order = ("ollama_single",) if args.ollama_only else ("yolo", "siglip", "florence", "ollama_single", "hybrid")
    for name in order:
        _print_metrics(name, scores[name])

    latency["total"] = time.perf_counter() - t0
    n = max(1, len(images))
    print("\n=== Latency ===")
    lat_order = ("ollama_single", "total") if args.ollama_only else ("yolo", "siglip", "florence", "ollama_single", "total")
    for key in lat_order:
        sec = latency.get(key, 0.0)
        print(f"{key:14} | total={sec:.2f}s | per_image={sec / n:.3f}s")

    report = {
        "dataset_root": str(DF2_ROOT),
        "split": args.split,
        "num_images": len(images),
        "ollama_num_images": len(ollama_images),
        "ollama_model": args.ollama_model,
        "scores": {
            k: {
                "hit_rate": v.hit_rate,
                "exact_match": v.exact_match,
                "avg_jaccard": v.avg_jaccard,
            }
            for k, v in scores.items()
        },
        "latency_seconds": latency,
    }
    out = Path("/Users/akarshsaklani/Desktop/vestir/vestir-prototype/scripts/benchmark_df2_small_ops_report.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved report: {out}")


if __name__ == "__main__":
    main()

