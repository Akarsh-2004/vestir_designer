from __future__ import annotations

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="Low-memory DeepFashion2 V1 finetune with resume support")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--project", default="runs/detect", help="Ultralytics project directory")
    parser.add_argument("--name", default="df2-v1-lowmem", help="Run name")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model for first run")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="mps", help="mps/cpu/cuda:0")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--save-period", type=int, default=1, help="Checkpoint every N epochs")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt if available")
    args = parser.parse_args()

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    run_dir = Path(args.project).expanduser().resolve() / args.name
    last_ckpt = run_dir / "weights" / "last.pt"

    # Keep memory pressure predictable for a workstation.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")

    if args.resume and last_ckpt.exists():
        print(f"Resuming from checkpoint: {last_ckpt}")
        model = YOLO(str(last_ckpt))
        model.train(
            data=str(data_path),
            project=str(Path(args.project).expanduser().resolve()),
            name=args.name,
            device=args.device,
            resume=True,
        )
    else:
        print(f"Starting fresh from model: {args.model}")
        model = YOLO(args.model)
        model.train(
            data=str(data_path),
            project=str(Path(args.project).expanduser().resolve()),
            name=args.name,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            cache=False,
            amp=True,
            patience=args.patience,
            save=True,
            save_period=args.save_period,
            exist_ok=True,
            pretrained=True,
            verbose=True,
            plots=False,
        )

    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    print(f"Run dir: {run_dir}")
    print(f"best.pt: {best}")
    print(f"last.pt: {last}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

