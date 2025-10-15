"""
Segmentation utilities for clothing-mask generation.

This module provides two segmentation approaches:
1. YOLOv8 + SAM: Advanced garment detection with precise polygon segmentation
2. DeepLabV3-ResNet50: Lightweight person segmentation fallback

Both approaches exclude the face/upper-head region (when face bounding box data
is provided) so that subsequent inpainting focuses on clothing.

All callers should treat this module as the single source of truth for mask generation.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torchvision import transforms
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)

# Conditional imports for YOLOv8 + SAM (may not be available)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

# Global state ---------------------------------------------------------------

# DeepLabV3 fallback
SEGMENTATION_READY = False
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL = None
_TRANSFORM = None

# YOLOv8 + SAM advanced segmentation
YOLO_SAM_READY = False
_YOLO_MODEL = None
_SAM_PREDICTOR = None

# Pascal VOC class index for "person"
PERSON_CLASS_INDEX = 15

# Configuration from environment variables
PERSON_THRESHOLD = float(os.getenv("SEGMENTATION_PERSON_THRESHOLD", "0.4"))
FACE_MASK_PADDING = float(os.getenv("SEGMENTATION_FACE_PADDING", "0.1"))
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.5"))
SAM_MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_b")
SAM_CHECKPOINT_PATH = os.getenv("SAM_CHECKPOINT_PATH", "/workspace/models/sam/sam_vit_b_01ec64.pth")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")  # Default to nano model

# YOLOv8 garment class IDs (COCO dataset approximate mapping)
# For general clothing detection, we'll use person detection then SAM refinement
GARMENT_CLASSES = [0]  # 0 = person in COCO (we'll use SAM to refine the body region)


def load_models() -> None:
    """
    Load the DeepLabV3 segmentation model into memory (idempotent fallback).
    """
    global SEGMENTATION_READY, _MODEL, _TRANSFORM

    if SEGMENTATION_READY and _MODEL is not None:
        return

    try:
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        _MODEL = deeplabv3_resnet50(weights=weights).to(_DEVICE)
        _MODEL.eval()
        _TRANSFORM = weights.transforms()
        SEGMENTATION_READY = True
        print("✓ DeepLabV3-ResNet50 segmentation loaded")
    except Exception as exc:
        print(f"✗ Failed to load DeepLabV3 model: {exc}")


def load_yolo_sam_models() -> None:
    """
    Load YOLOv8 + SAM models for advanced garment segmentation (idempotent).
    """
    global YOLO_SAM_READY, _YOLO_MODEL, _SAM_PREDICTOR

    if YOLO_SAM_READY and _YOLO_MODEL is not None and _SAM_PREDICTOR is not None:
        return

    if not YOLO_AVAILABLE or not SAM_AVAILABLE:
        print("✗ YOLOv8 or SAM not available (missing dependencies)")
        return

    try:
        # Load YOLOv8 for person/garment detection
        print(f"Loading YOLOv8 model from {YOLO_MODEL_PATH}...")
        _YOLO_MODEL = YOLO(YOLO_MODEL_PATH)
        _YOLO_MODEL.to(_DEVICE)

        # Load SAM for precise segmentation
        print(f"Loading SAM model ({SAM_MODEL_TYPE}) from {SAM_CHECKPOINT_PATH}...")
        if not os.path.exists(SAM_CHECKPOINT_PATH):
            raise FileNotFoundError(f"SAM checkpoint not found: {SAM_CHECKPOINT_PATH}")

        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(_DEVICE)
        _SAM_PREDICTOR = SamPredictor(sam)

        YOLO_SAM_READY = True
        print("✓ YOLOv8 + SAM segmentation loaded successfully")
    except Exception as exc:
        print(f"✗ Failed to load YOLOv8/SAM models: {exc}")
        print("  Will fall back to DeepLabV3 segmentation")


def _tensor_from_image(image: Image.Image) -> torch.Tensor:
    if _TRANSFORM is None:
        raise RuntimeError("Segmentation transform not initialized.")
    pil_rgb = image.convert("RGB")
    return _TRANSFORM(pil_rgb).unsqueeze(0).to(_DEVICE)


def _apply_face_cutout(mask: Image.Image, face_bbox: Optional[Tuple[float, float, float, float]]) -> None:
    if face_bbox is None:
        return
    draw = ImageDraw.Draw(mask)
    _, y1, _, y2 = face_bbox
    padding = FACE_MASK_PADDING * mask.height
    cutoff = int(min(mask.height, y2 + padding))
    draw.rectangle([(0, 0), (mask.width, cutoff)], fill=0)


def _post_process(mask_array: np.ndarray, image_size: Tuple[int, int]) -> Image.Image:
    # Resize to original resolution and apply blur/threshold to smooth edges
    mask_img = Image.fromarray((mask_array * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.resize(image_size, Image.Resampling.BILINEAR)
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=6))
    mask_img = mask_img.point(lambda p: 255 if p > 64 else 0)
    return mask_img


def _merge_masks(masks: List[np.ndarray]) -> np.ndarray:
    """Merge multiple binary masks into a single mask using logical OR."""
    if not masks:
        return np.zeros((512, 512), dtype=bool)

    combined = masks[0].astype(bool)
    for mask in masks[1:]:
        combined = np.logical_or(combined, mask.astype(bool))

    return combined.astype(np.uint8)


def generate_clothing_mask_yolo_sam(
    image: Image.Image,
    face_bbox: Optional[Tuple[float, float, float, float]] = None
) -> Tuple[Optional[Image.Image], Dict[str, object]]:
    """
    Advanced garment segmentation using YOLOv8 detection + SAM refinement.

    Args:
        image: PIL Image (RGB or convertible)
        face_bbox: Optional (x1, y1, x2, y2) bounding box from FaceID detection

    Returns:
        (mask_image, diagnostics)
    """
    diagnostics: Dict[str, object] = {
        "segmentation_used": False,
        "reason": "YOLOv8+SAM not initialized",
        "method": "yolo_sam",
    }

    if not YOLO_SAM_READY or _YOLO_MODEL is None or _SAM_PREDICTOR is None:
        return None, diagnostics

    try:
        # Convert PIL to numpy array (RGB format for YOLOv8)
        image_np = np.array(image.convert("RGB"))

        # Run YOLOv8 detection to find person(s)
        results = _YOLO_MODEL(image_np, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)

        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            diagnostics.update({
                "segmentation_used": False,
                "reason": "No persons detected by YOLOv8",
                "detections": 0,
            })
            return None, diagnostics

        # Extract bounding boxes for detected persons
        boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confidences = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        # Filter for person class (0 in COCO)
        person_boxes = [
            (box, conf) for box, conf, cls in zip(boxes, confidences, classes)
            if int(cls) in GARMENT_CLASSES
        ]

        if not person_boxes:
            diagnostics.update({
                "segmentation_used": False,
                "reason": "No person class detections",
                "detections": len(boxes),
            })
            return None, diagnostics

        # Use SAM to refine each detected person into a precise mask
        _SAM_PREDICTOR.set_image(image_np)

        sam_masks = []
        detection_data = []

        for box, conf in person_boxes:
            x1, y1, x2, y2 = box
            # SAM expects input_box as [x1, y1, x2, y2]
            input_box = np.array([x1, y1, x2, y2])

            masks, scores, _ = _SAM_PREDICTOR.predict(
                box=input_box,
                multimask_output=False  # Single best mask
            )

            if masks is not None and len(masks) > 0:
                sam_masks.append(masks[0])  # Take the first (and only) mask
                detection_data.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "sam_score": float(scores[0]) if len(scores) > 0 else 0.0,
                })

        if not sam_masks:
            diagnostics.update({
                "segmentation_used": False,
                "reason": "SAM failed to generate masks",
                "person_detections": len(person_boxes),
            })
            return None, diagnostics

        # Merge all person masks
        merged_mask = _merge_masks(sam_masks)

        # Convert to PIL Image
        mask_img = Image.fromarray((merged_mask * 255).astype(np.uint8), mode="L")

        # Apply face cutout to preserve facial features
        _apply_face_cutout(mask_img, face_bbox)

        # Post-process: smooth edges
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=4))
        mask_img = mask_img.point(lambda p: 255 if p > 128 else 0)

        coverage = np.array(mask_img).astype(bool).mean()

        diagnostics.update({
            "segmentation_used": True,
            "reason": "yolo_sam",
            "method": "YOLOv8 + SAM",
            "mask_coverage": float(coverage),
            "person_detections": len(person_boxes),
            "sam_masks_generated": len(sam_masks),
            "detections": detection_data,
            "device": str(_DEVICE),
        })

        return mask_img, diagnostics

    except Exception as exc:
        diagnostics.update({
            "segmentation_used": False,
            "reason": f"YOLOv8+SAM exception: {exc}",
            "method": "yolo_sam",
        })
        return None, diagnostics


def generate_clothing_mask(
    image: Image.Image,
    face_bbox: Optional[Tuple[float, float, float, float]] = None
) -> Tuple[Optional[Image.Image], Dict[str, object]]:
    """
    Produce a grayscale mask highlighting clothing regions.

    Args:
        image: PIL Image (RGB or convertible)
        face_bbox: Optional (x1, y1, x2, y2) bounding box from FaceID detection

    Returns:
        (mask_image, diagnostics)
    """
    diagnostics: Dict[str, object] = {
        "segmentation_used": False,
        "reason": "Segmentation model unavailable",
    }

    if not SEGMENTATION_READY or _MODEL is None or _TRANSFORM is None:
        return None, diagnostics

    try:
        with torch.no_grad():
            input_tensor = _tensor_from_image(image)
            output = _MODEL(input_tensor)["out"][0]
            probabilities = torch.softmax(output, dim=0)[PERSON_CLASS_INDEX]
            mask_array = (probabilities >= PERSON_THRESHOLD).cpu().numpy()

        if mask_array.mean() < 0.01:
            diagnostics.update(
                {
                    "segmentation_used": False,
                    "reason": "Person mask too small; falling back",
                    "mask_coverage": float(mask_array.mean()),
                }
            )
            return None, diagnostics

        mask_img = _post_process(mask_array, image.size)
        _apply_face_cutout(mask_img, face_bbox)

        coverage = np.array(mask_img).astype(bool).mean()
        diagnostics.update(
            {
                "segmentation_used": True,
                "reason": "deeplabv3-resnet50",
                "mask_coverage": float(coverage),
                "threshold": PERSON_THRESHOLD,
                "device": str(_DEVICE),
            }
        )
        return mask_img, diagnostics
    except Exception as exc:
        diagnostics.update(
            {
                "segmentation_used": False,
                "reason": f"Segmentation exception: {exc}",
            }
        )
        return None, diagnostics
