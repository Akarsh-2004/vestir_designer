#!/usr/bin/env python3
"""
Smoke-test virtual tryoff only (tryon-sidecar on :8009).

Usage:
  cd tryon-sidecar && ./start.sh   # in another terminal
  python3 scripts/test_tryoff.py /path/to/photo.jpg

Env:
  TRYON_SIDECAR_URL  default http://127.0.0.1:8009
  GARMENT_TARGET     default outfit  (pants, dress, jacket, …)
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def main() -> int:
    base = os.environ.get("TRYON_SIDECAR_URL", "http://127.0.0.1:8009").rstrip("/")
    garment = os.environ.get("GARMENT_TARGET", "outfit").strip() or "outfit"

    if len(sys.argv) < 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2

    img_path = Path(sys.argv[1]).expanduser().resolve()
    if not img_path.is_file():
        print(f"Not a file: {img_path}", file=sys.stderr)
        return 2

    # --- health ---
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=10) as r:
            health = json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        print(f"Health check failed ({base}/health): {e}", file=sys.stderr)
        print("Start sidecar: cd tryon-sidecar && ./start.sh", file=sys.stderr)
        return 1

    print("health:", json.dumps(health, indent=2))
    if not health.get("tryoff_pipeline_ready"):
        print(
            "\nNote: tryoff_pipeline_ready is false — model may still be warming up.\n"
            "Wait and retry, or POST /tryoff/warmup?force=true\n",
            file=sys.stderr,
        )

    raw = img_path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    body = json.dumps({"image_base64": b64, "garment_target": garment}).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/tryoff",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print(f"\nPOST {base}/tryoff (garment_target={garment!r}, image={img_path.name}) …")
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()[:500]}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"Request failed: {e}", file=sys.stderr)
        return 1

    ok = payload.get("ok")
    implemented = payload.get("implemented")
    msg = payload.get("message")
    err = payload.get("debug_error")
    b64out = payload.get("result_image_base64")

    print("\n--- response ---")
    print(f"ok={ok!r} implemented={implemented!r}")
    if msg:
        print(f"message: {msg}")
    if err:
        print(f"debug_error: {err}")

    if b64out:
        out_path = Path(os.environ.get("TRYOFF_TEST_OUT", "/tmp/vestir-tryoff-test-out.jpg"))
        out_path.write_bytes(base64.b64decode(b64out))
        print(f"\nSaved output: {out_path}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
