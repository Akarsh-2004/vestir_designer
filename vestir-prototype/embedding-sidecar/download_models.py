#!/usr/bin/env python3
"""Pre-download Hugging Face weights into the local cache (for offline demos)."""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get("EMBEDDING_MODEL_ID", "Marqo/marqo-fashionSigLIP"),
        help="HF model id (default: Marqo fashion SigLIP)",
    )
    args = parser.parse_args()
    model_id = args.model
    trust = model_id.startswith("Marqo/")

    from huggingface_hub import snapshot_download

    print("Downloading snapshot:", model_id, "trust_remote_code=", trust, flush=True)
    snapshot_download(repo_id=model_id, trust_remote_code=trust)
    print("OK:", model_id, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
