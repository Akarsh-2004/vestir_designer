"""
Test YOLOv8 garment / outfit detection (Ultralytics) — matches what /analyze uses first.

Prereqs:
  pip install -r requirements.txt
  pip install -r requirements-yolov8.txt
  $env:VESTIR_YOLOV8_PT = "C:\\path\\to\\best.pt"

Usage:
  python test_yolov8_outfit.py path\\to\\outfit.jpg
  python test_yolov8_outfit.py photo.jpg --serve
  python test_yolov8_outfit.py photo.jpg --conf 0.25
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description="Test YOLOv8 outfit/garment detection (VESTIR_YOLOV8_PT)")
    parser.add_argument("image", help="Path to a test image (jpg/png) — full outfit or flat-lay")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="POST same image to http://127.0.0.1:8008/analyze (start uvicorn first)",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8008/analyze",
        help="Analyze URL when using --serve",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Override VESTIR_YOLOV8_CONF for this run only (default: env or 0.4)",
    )
    args = parser.parse_args()

    if args.conf is not None:
        os.environ["VESTIR_YOLOV8_CONF"] = str(args.conf)

    import cv2

    from yolov8_engine import run_yolov8, yolov8_configured

    bgr = cv2.imread(args.image)
    if bgr is None:
        print("Could not read image:", args.image, file=sys.stderr)
        return 1

    print("yolov8_configured:", yolov8_configured())
    if not yolov8_configured():
        print(
            "\nVESTIR_YOLOV8_PT must be set in THIS shell to a real file or hub model.\n"
            "Examples:\n"
            '  $env:VESTIR_YOLOV8_PT = "C:\\Users\\YOU\\runs\\detect\\train\\weights\\best.pt"\n'
            "Smoke test (COCO classes: person, car, … not clothing):\n"
            '  $env:VESTIR_YOLOV8_PT = "yolov8n.pt"\n'
            "Optional:\n"
            '  $env:VESTIR_YOLOV8_CONF = "0.35"\n'
            "Tip: placeholder paths like C:\\\\path\\\\to\\\\best.pt do not exist — use Explorer copy-as-path.\n",
            file=sys.stderr,
        )
        return 2

    out = run_yolov8(bgr)
    if out is None:
        print("run_yolov8 returned None (check ultralytics install and weights).", file=sys.stderr)
        return 3

    print(json.dumps(out, indent=2))
    n = len(out.get("garments") or [])
    print(f"\n# detections: {n}  model: {out.get('model')}", file=sys.stderr)

    if args.serve:
        ok, raw = _post_analyze(args.url, bgr)
        print("\n--- HTTP", args.url, "---")
        print("ok:", ok)
        print(raw)

    return 0


def _post_analyze(url: str, bgr) -> tuple[bool, str]:
    import cv2

    _, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    payload = json.dumps({"image_base64": base64.b64encode(buf.tobytes()).decode("ascii")}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return True, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return False, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    raise SystemExit(main())
