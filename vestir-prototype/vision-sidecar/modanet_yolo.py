"""Backward-compatible imports; prefer darknet_yolo in new code."""

from darknet_yolo import darknet_yolo_configured as modanet_yolo_configured
from darknet_yolo import run_darknet_yolo as run_modanet_yolo

__all__ = ["modanet_yolo_configured", "run_modanet_yolo"]
