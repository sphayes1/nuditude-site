# RunPod Segmentation Setup Guide

## ⚠️ Critical Detection Policy

**MANDATORY REQUIREMENT:**
- **Jobs are automatically rejected if no face or person is detected in the input image.**
- This applies to all generations across the entire project.
- The worker will return an error immediately without consuming compute resources.

**Detection Logic:**
- ✅ **Pass**: Face detected by InsightFace **OR** person detected by YOLOv8/DeepLabV3
- ❌ **Reject**: Neither face nor person detected

**Error Response Example:**
```json
{
  "error": "No face or person detected in the input image. This application requires a clearly visible person in the photo. Please upload a different image with a person clearly visible.",
  "rejection_reason": "no_detection",
  "face_detected": false,
  "person_detected": false,
  "segmentation": {...}
}
```

**Rationale:**
1. **Quality Control** — Prevents wasted compute on invalid inputs
2. **Safety/Compliance** — Ensures proper subject consent verification
3. **User Experience** — Fast feedback on unsuitable images

---

## Quick Start: Enable YOLOv8 + SAM

### 1. Update RunPod Template

Navigate to your RunPod template configuration and update:

**Container Image:**
```
spencer45/runpod-sdxl-ipadapter:faceid-latest
```

**Image Digest (for verification):**
```
sha256:e673a5f638822c68bf2229962b46c1a479bc4b558bcc9ad3538e997adad5550f
```

### 2. Add Environment Variable

In the RunPod template **Environment Variables** section, add:

```bash
USE_SEGMENTATION=1
```

This enables the YOLOv8 + SAM segmentation pipeline with automatic fallback to DeepLabV3 and heuristic masks.

---

## Advanced Configuration (Optional)

### Fine-Tuning Segmentation Parameters

Add any of these environment variables to customize segmentation behavior:

```bash
# YOLOv8 person detection confidence threshold (default: 0.5)
YOLO_CONFIDENCE_THRESHOLD=0.6

# YOLOv8 model size (default: yolov8n.pt - nano model)
# Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)
YOLO_MODEL_PATH=yolov8s.pt

# SAM model type (default: vit_b)
# Options: vit_b (base, 375MB), vit_l (large, 1.2GB), vit_h (huge, 2.4GB)
SAM_MODEL_TYPE=vit_b

# Face cutout padding multiplier (default: 0.1)
SEGMENTATION_FACE_PADDING=0.15

# DeepLabV3 person threshold for fallback (default: 0.4)
SEGMENTATION_PERSON_THRESHOLD=0.5
```

### Memory & Performance Considerations

| Configuration | VRAM Usage | Speed | Quality | Best For |
|--------------|------------|-------|---------|----------|
| **YOLOv8n + SAM ViT-B** | +1.5GB | ~2-3s | ⭐⭐⭐⭐⭐ | Balanced (recommended) |
| **YOLOv8s + SAM ViT-L** | +2.5GB | ~4-5s | ⭐⭐⭐⭐⭐ | Maximum quality |
| **DeepLabV3 fallback** | +500MB | ~0.5s | ⭐⭐⭐⭐ | Speed priority |
| **Heuristic only** | 0MB | <0.1s | ⭐⭐ | Minimal overhead |

---

## Verification

### Check Logs on Worker Startup

When the worker initializes, you should see:

```
============================================================
Initializing SDXL + FaceID Worker
============================================================
Using device: cuda
Advanced segmentation enabled: True
Attempting to load YOLOv8 + SAM segmentation models...
Loading YOLOv8 model from yolov8n.pt...
Loading SAM model (vit_b) from /workspace/models/sam/sam_vit_b_01ec64.pth...
✓ YOLOv8 + SAM segmentation loaded successfully
✓ YOLOv8 + SAM ready for advanced segmentation
Worker ready! Waiting for jobs...
============================================================
```

### Verify in Generation Logs

After a generation completes, check the job response for:

```json
{
  "segmentation": {
    "segmentation_used": true,
    "method": "YOLOv8 + SAM",
    "person_detections": 1,
    "sam_masks_generated": 1,
    "mask_coverage": 0.487,
    "detections": [
      {
        "bbox": [124.5, 89.2, 645.8, 1021.3],
        "confidence": 0.94,
        "sam_score": 0.98
      }
    ],
    "device": "cuda"
  }
}
```

**Key indicators:**
- `"segmentation_used": true` — Segmentation succeeded
- `"method": "YOLOv8 + SAM"` — Using advanced pipeline (not fallback)
- `"confidence"` and `"sam_score"` — Quality metrics (higher is better)

---

## Troubleshooting

### Issue: Segmentation not loading

**Symptoms:**
```
✗ YOLOv8 or SAM not available (missing dependencies)
```

**Solution:**
1. Verify you're using the latest image digest: `sha256:0baef19c5a9c78b39549a5a6a8e8afe88dbfb3efbe86d406e584ca45bda28eb9`
2. Check that `USE_SEGMENTATION=1` is set in environment variables
3. Restart the RunPod worker

### Issue: Job rejected with "No face or person detected"

**Symptoms:**
```json
{
  "error": "No face or person detected in the input image...",
  "rejection_reason": "no_detection"
}
```

**This is EXPECTED BEHAVIOR** — not a bug. The system is working as designed.

**Solution:**
1. **Verify image quality:**
   - Ensure the person is clearly visible and well-lit
   - Person should occupy at least 20-30% of the image
   - Avoid extreme angles, heavy occlusion, or silhouettes

2. **Check image format:**
   - Use standard formats (JPEG, PNG)
   - Ensure image is not corrupted
   - Min resolution: 512x512 pixels recommended

3. **If using valid images but still rejected:**
   - Lower detection thresholds (advanced users only):
     ```bash
     YOLO_CONFIDENCE_THRESHOLD=0.3
     SEGMENTATION_PERSON_THRESHOLD=0.3
     ```
   - Note: Lowering too much may cause false positives on non-person images

**Important:** This rejection is a safety feature, not an error. Invalid images are rejected immediately to save compute resources.

---

### Issue: Falling back to DeepLabV3

**Symptoms:**
```json
{
  "method": "deeplabv3-resnet50",
  "reason": "YOLOv8+SAM failed"
}
```

**Solution:**
- This is expected behavior if YOLOv8/SAM encounters detection issues but DeepLabV3 succeeds
- Job still completes successfully (DeepLabV3 is a valid fallback)
- Check that input image contains a clearly visible person
- Try lowering `YOLO_CONFIDENCE_THRESHOLD` to 0.3 to make YOLOv8 more sensitive

### Issue: High VRAM usage / OOM errors

**Symptoms:**
- Worker crashes during segmentation
- CUDA out of memory errors

**Solution:**
1. Use smaller models:
   ```bash
   YOLO_MODEL_PATH=yolov8n.pt  # Use nano instead of small/medium
   SAM_MODEL_TYPE=vit_b         # Use base instead of large/huge
   ```
2. Reduce batch size or image resolution if needed
3. Consider using DeepLabV3-only mode (lighter VRAM footprint)

---

## Performance Benchmarks

Tested on **RTX 4090 (24GB VRAM)**:

| Scenario | Total Time | Segmentation Overhead | Mask Quality |
|----------|-----------|----------------------|--------------|
| **YOLOv8n + SAM ViT-B** | ~15s | +2.1s | Excellent |
| **YOLOv8s + SAM ViT-L** | ~17s | +3.8s | Outstanding |
| **DeepLabV3 fallback** | ~13s | +0.6s | Very Good |
| **Heuristic mask** | ~12s | <0.1s | Good |

*Note: Total time includes inpainting (~10-12s baseline). Overhead is purely segmentation.*

---

## Next Steps

1. **Update RunPod template** with new image and `USE_SEGMENTATION=1`
2. **Test with sample images** via your frontend
3. **Monitor admin dashboard** for segmentation diagnostics
4. **Compare results** against previous heuristic masks
5. **Fine-tune parameters** based on your specific use case

---

**Last Updated:** October 13, 2025
**Docker Image:** `spencer45/runpod-sdxl-ipadapter:faceid-latest`
**Digest:** `sha256:e673a5f638822c68bf2229962b46c1a479bc4b558bcc9ad3538e997adad5550f`
**Includes:** Mandatory face/person detection enforcement
