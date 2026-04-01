"""
Test ModaNet YOLOv3 locally (no Node server required).

Prereqs:
  1. pip install -r requirements.txt
  2. Download .cfg + .weights (+ optional .names) from:
     https://github.com/kritanjalijain/Clothing_Detection_YOLO (see their README / Google Drive)
  3. Set MODANET_YOLO_CFG and MODANET_YOLO_WEIGHTS (PowerShell examples below)

Usage:
  python test_modanet_yolo.py path\\to\\photo.jpg
  python test_modanet_yolo.py photo.jpg --serve   # also POST same image to running uvicorn
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description="Test ModaNet YOLO paths / OpenCV DNN")
    parser.add_argument("image", help="Path to a test image (jpg/png)")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="After local run, POST base64 to http://127.0.0.1:8008/analyze (start uvicorn first)",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8008/analyze",
        help="Analyze URL when using --serve",
    )
    args = parser.parse_args()

    import cv2

    from darknet_yolo import darknet_yolo_configured, run_darknet_yolo

    bgr = cv2.imread(args.image)
    if bgr is None:
        print("Could not read image:", args.image, file=sys.stderr)
        return 1

    print("darknet_yolo_configured:", darknet_yolo_configured())
    if not darknet_yolo_configured():
        print(
            "\nBundled DeepFashion2 cfg is in vision-sidecar/models/ if you ran setup.\n"
            "Set WEIGHTS only (PowerShell), then re-run:\n"
            '  $env:VESTIR_DARKNET_WEIGHTS = "C:\\path\\to\\yolov3-df2_last.weights"\n'
            "Optional overrides:\n"
            '  $env:VESTIR_DARKNET_CFG = "C:\\custom\\yolov3-custom.cfg"\n'
            '  $env:VESTIR_DARKNET_NAMES = "C:\\custom\\df2.names"\n'
            "Weights: train with tomsebastiank/Clothing-detection... or use compatible darknet .weights\n",
            file=sys.stderr,
        )
        return 2

    out = run_darknet_yolo(bgr)
    print(json.dumps(out, indent=2))

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
