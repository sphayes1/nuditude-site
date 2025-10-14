"""
Segmentation utilities for clothing-mask generation.

This module is intentionally lightweight to provide a clear integration point
for the YOLOv8 + SAM workflow described in `ai_outfit_swap_design.md`.
The initial implementation returns `None` so that the caller can fall back to
the heuristic mask until the heavy models are wired in.
"""

from __future__ import annotations

from typing import Optional, Tuple

from PIL import Image

# Flag to indicate whether advanced segmentation is available.
SEGMENTATION_READY = False


def load_models() -> None:
    """
    Placeholder for model initialization.

    TODO:
        - Load YOLOv8 garment detector (e.g., via Ultralytics or ONNXRuntime).
        - Load Segment Anything model (ViT-B) and create predictor.
        - Populate global state so repeated jobs reuse the weights.
    """
    # Future implementation will set SEGMENTATION_READY = True after loading.
    pass


def generate_clothing_mask(image: Image.Image) -> Tuple[Optional[Image.Image], dict]:
    """
    Produce an RGBA/L mask highlighting clothing regions.

    Returns:
        (mask, diagnostics)

        mask: PIL.Image in mode "L" or "RGBA" representing the clothing area.
        diagnostics: dict containing useful metadata (model confidences,
                     polygons, etc.). The default implementation provides
                     a reason for fallback.
    """
    diagnostics = {
        "segmentation_used": False,
        "reason": "Segmentation models not yet integrated",
    }
    return None, diagnostics

