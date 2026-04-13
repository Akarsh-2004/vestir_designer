import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

from dotenv import load_dotenv

import benchmark_label_models as bm


ROOT = Path("/Users/akarshsaklani/Desktop/vestir")
FILENAME_DATASET_DIR = ROOT / "for label testing dataset "
DF2_ROOT = ROOT / "vestir-prototype" / "vision-sidecar" / "datasets" / "df2_v1"
OUT_DIR = ROOT / "vestir-prototype" / "scripts"


DF2_CLASS_TO_CANON = {
    0: {"top"},
    1: {"jacket"},
    2: {"pants"},
    3: {"dress"},
}


def _sorted_join(values: Set[str]) -> str:
    return ",".join(sorted(values))


def _read_df2_gt(label_path: Path) -> Tuple[Set[str], List[int]]:
    if not label_path.exists():
        return set(), []
    class_ids: List[int] = []
    labels: Set[str] = set()
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cid = int(parts[0])
        except Exception:
            continue
        class_ids.append(cid)
        labels |= DF2_CLASS_TO_CANON.get(cid, set())
    return labels, class_ids


def _run_models(
    images: List[Path],
    ollama_model: str,
    ollama_base: str,
    ollama_max_images: int,
) -> Dict[str, Dict[str, Set[str]]]:
    yolo = bm.predict_yolo(images, print_predictions=False)
    siglip = bm.predict_siglip(images, print_predictions=False)
    florence = bm.predict_florence(images, print_predictions=False)

    ollama_preds = {img.name: set() for img in images}
    if ollama_max_images > 0:
        subset = images[: min(len(images), ollama_max_images)]
        part = bm.predict_ollama(subset, ollama_model, ollama_base, print_predictions=False)
        ollama_preds.update(part)

    hybrid = bm.hybrid_vote(yolo, siglip, florence, ollama_preds)
    return {
        "yolo": yolo,
        "siglip": siglip,
        "florence": florence,
        "ollama": ollama_preds,
        "hybrid": hybrid,
    }


def export_filename_dataset(models: Dict[str, Dict[str, Set[str]]]) -> Path:
    images = sorted([p for p in FILENAME_DATASET_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    out_csv = OUT_DIR / "filename_dataset_full_predictions.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "ground_truth_from_filename", "yolo", "siglip", "florence", "ollama", "hybrid"])
        for img in images:
            gt = bm.extract_labels(img.stem)
            w.writerow(
                [
                    img.name,
                    _sorted_join(gt),
                    _sorted_join(models["yolo"].get(img.name, set())),
                    _sorted_join(models["siglip"].get(img.name, set())),
                    _sorted_join(models["florence"].get(img.name, set())),
                    _sorted_join(models["ollama"].get(img.name, set())),
                    _sorted_join(models["hybrid"].get(img.name, set())),
                ]
            )
    return out_csv


def export_df2_dataset(models: Dict[str, Dict[str, Set[str]]], split: str, max_images: int) -> Path:
    img_dir = DF2_ROOT / "images" / split
    lbl_dir = DF2_ROOT / "labels" / split
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if max_images > 0:
        images = images[: max_images]

    out_csv = OUT_DIR / f"df2_{split}_full_predictions_with_annotations.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "file_name",
                "df2_annotation_class_ids",
                "ground_truth_mapped_labels",
                "yolo",
                "siglip",
                "florence",
                "ollama",
                "hybrid",
            ]
        )
        for img in images:
            gt_labels, class_ids = _read_df2_gt(lbl_dir / f"{img.stem}.txt")
            w.writerow(
                [
                    img.name,
                    ",".join(str(x) for x in class_ids),
                    _sorted_join(gt_labels),
                    _sorted_join(models["yolo"].get(img.name, set())),
                    _sorted_join(models["siglip"].get(img.name, set())),
                    _sorted_join(models["florence"].get(img.name, set())),
                    _sorted_join(models["ollama"].get(img.name, set())),
                    _sorted_join(models["hybrid"].get(img.name, set())),
                ]
            )
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Export full per-file prediction reports for filename and DF2 datasets.")
    parser.add_argument("--df2-split", choices=("train", "val"), default="val")
    parser.add_argument("--df2-max-images", type=int, default=40)
    parser.add_argument("--ollama-max-images", type=int, default=25)
    parser.add_argument("--ollama-model", default="llama3.2-vision:11b")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    filename_images = sorted([p for p in FILENAME_DATASET_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    df2_images = sorted(
        [p for p in (DF2_ROOT / "images" / args.df2_split).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if args.df2_max_images > 0:
        df2_images = df2_images[: args.df2_max_images]

    print(f"Running models on filename dataset: {len(filename_images)} images")
    filename_models = _run_models(filename_images, args.ollama_model, args.ollama_base_url, args.ollama_max_images)
    print(f"Running models on DF2 {args.df2_split}: {len(df2_images)} images")
    df2_models = _run_models(df2_images, args.ollama_model, args.ollama_base_url, args.ollama_max_images)

    out1 = export_filename_dataset(filename_models)
    out2 = export_df2_dataset(df2_models, args.df2_split, args.df2_max_images)

    summary = {
        "filename_dataset_report_csv": str(out1),
        "df2_report_csv": str(out2),
        "df2_split": args.df2_split,
        "df2_num_images": len(df2_images),
        "ollama_max_images": args.ollama_max_images,
    }
    out_summary = OUT_DIR / "full_prediction_reports_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    print(f"Saved: {out1}")
    print(f"Saved: {out2}")
    print(f"Saved: {out_summary}")


if __name__ == "__main__":
    main()

