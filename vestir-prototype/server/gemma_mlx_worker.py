from __future__ import annotations

import base64
import json
import sys
from io import BytesIO
from typing import Any

from PIL import Image


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _load_runtime():
    try:
        from mlx_vlm import generate, load  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"mlx_vlm import failed: {exc}") from exc

    model_id = (sys.argv[1] if len(sys.argv) > 1 else "").strip() or "mlx-community/gemma-3-4b-it-4bit"
    model, processor = load(model_id)
    return model_id, model, processor, generate


def main() -> int:
    try:
        model_id, model, processor, generate_fn = _load_runtime()
        _emit({"type": "ready", "ok": True, "model": model_id})
    except Exception as exc:  # noqa: BLE001
        _emit({"type": "ready", "ok": False, "error": str(exc)})
        return 1

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        req_id = "unknown"
        try:
            payload = json.loads(line)
            req_id = str(payload.get("id", "unknown"))
            prompt = str(payload.get("prompt", "")).strip()
            image_b64 = str(payload.get("image_base64", "")).strip()
            max_tokens = int(payload.get("max_tokens", 220))
            if not image_b64:
                raise ValueError("image_base64 missing")
            if not prompt:
                raise ValueError("prompt missing")

            image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
            output = generate_fn(
                model=model,
                processor=processor,
                image=image,
                prompt=prompt,
                max_tokens=max(80, max_tokens),
                verbose=False,
            )
            _emit({"type": "result", "id": req_id, "ok": True, "text": str(output)})
        except Exception as exc:  # noqa: BLE001
            _emit({"type": "result", "id": req_id, "ok": False, "error": str(exc)})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

