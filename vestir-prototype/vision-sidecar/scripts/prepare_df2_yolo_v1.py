from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from PIL import Image


# DeepFashion2 category_id (1..13) -> compact wardrobe classes.
CLASS_MAP: dict[int, int] = {
    1: 0,  # short sleeve top
    2: 0,  # long sleeve top
    3: 1,  # short sleeve outwear
    4: 1,  # long sleeve outwear
    5: 1,  # vest
    6: 0,  # sling
    7: 2,  # shorts
    8: 2,  # trousers
    9: 2,  # skirt
    10: 3,  # short sleeve dress
    11: 3,  # long sleeve dress
    12: 3,  # vest dress
    13: 3,  # sling dress
}

CLASS_NAMES = ["tops", "outerwear", "bottoms", "dress"]


def iter_annos(anno_dir: Path) -> Iterable[Path]:
    for p in sorted(anno_dir.glob("*.json")):
        yield p


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_yolo_bbox(bbox: list[float], width: float, height: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    bw = max(1.0, float(x2) - float(x1))
    bh = max(1.0, float(y2) - float(y1))
    cx = float(x1) + bw / 2.0
    cy = float(y1) + bh / 2.0
    return cx / width, cy / height, bw / width, bh / height


def convert_split(
    split_name: str,
    image_dir: Path,
    anno_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    max_images: int,
) -> int:
    ensure_dir(out_images_dir)
    ensure_dir(out_labels_dir)
    converted = 0

    for anno_path in iter_annos(anno_dir):
        if max_images > 0 and converted >= max_images:
            break

        stem = anno_path.stem
        image_path = image_dir / f"{stem}.jpg"
        if not image_path.exists():
            continue

        try:
            with Image.open(image_path) as im:
                width, height = im.size
        except Exception:
            continue
        if width <= 1 or height <= 1:
            continue

        try:
            payload = json.loads(anno_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        label_lines: list[str] = []
        for key, value in payload.items():
            if not key.startswith("item") or not isinstance(value, dict):
                continue
            cid = int(value.get("category_id", 0))
            mapped = CLASS_MAP.get(cid)
            bbox = value.get("bounding_box")
            if mapped is None or not isinstance(bbox, list) or len(bbox) != 4:
                continue
            xc, yc, bw, bh = to_yolo_bbox(bbox, width, height)
            if bw <= 0 or bh <= 0:
                continue
            xc = min(1.0, max(0.0, xc))
            yc = min(1.0, max(0.0, yc))
            bw = min(1.0, max(0.0, bw))
            bh = min(1.0, max(0.0, bh))
            label_lines.append(f"{mapped} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if not label_lines:
            continue

        # Symlink images to avoid duplicating huge datasets.
        target_img = out_images_dir / image_path.name
        if target_img.exists() or target_img.is_symlink():
            target_img.unlink()
        os.symlink(image_path, target_img)

        (out_labels_dir / f"{stem}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        converted += 1

    print(f"{split_name}: converted {converted} images")
    return converted


def write_data_yaml(out_root: Path) -> Path:
    yaml_path = out_root / "data.yaml"
    content = "\n".join(
        [
            f"path: {out_root}",
            "train: images/train",
            "val: images/val",
            "names:",
            *[f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES)],
            "",
        ]
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare DeepFashion2 YOLO dataset for V1 low-memory finetune")
    parser.add_argument("--df2-root", required=True, help="Path to DeepFashion2/deepfashion2_original_images")
    parser.add_argument(
        "--out-root",
        default="datasets/df2_v1",
        help="Output dataset root (contains images/, labels/, data.yaml)",
    )
    parser.add_argument("--max-train", type=int, default=12000, help="Limit train image count (0 = all)")
    parser.add_argument("--max-val", type=int, default=2000, help="Limit val image count (0 = all)")
    args = parser.parse_args()

    df2_root = Path(args.df2_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    train_img_dir = df2_root / "train" / "image"
    train_anno_dir = df2_root / "train" / "annos"
    val_img_dir = df2_root / "validation" / "image"
    val_anno_dir = df2_root / "validation" / "annos"
    for p in (train_img_dir, train_anno_dir, val_img_dir, val_anno_dir):
        if not p.exists():
            raise FileNotFoundError(f"Missing required path: {p}")

    converted_train = convert_split(
        "train",
        train_img_dir,
        train_anno_dir,
        out_root / "images" / "train",
        out_root / "labels" / "train",
        args.max_train,
    )
    converted_val = convert_split(
        "val",
        val_img_dir,
        val_anno_dir,
        out_root / "images" / "val",
        out_root / "labels" / "val",
        args.max_val,
    )

    if converted_train == 0 or converted_val == 0:
        raise RuntimeError("No images converted. Check DF2 paths and annotations.")

    yaml_path = write_data_yaml(out_root)
    print(f"Wrote: {yaml_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

