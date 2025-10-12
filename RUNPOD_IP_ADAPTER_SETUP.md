# RunPod IP-Adapter Setup Guide

## 🎯 Goal
Upgrade your RunPod SDXL endpoint to use IP-Adapter for 80-85% facial similarity (vs current 30-50%).

---

## 📦 What is IP-Adapter?

IP-Adapter adds **image-based conditioning** to Stable Diffusion:
- Takes a reference face image + text prompt
- Preserves facial features while changing clothing/pose/setting
- No training required (instant use)
- ~Same generation speed as text-only

---

## 🛠️ RunPod Worker Setup

### Option A: Docker Template (Recommended)

Create a custom Docker image with IP-Adapter pre-installed:

```dockerfile
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Install dependencies
RUN pip install diffusers transformers accelerate safetensors
RUN pip install ip_adapter insightface onnxruntime-gpu

# Download IP-Adapter weights
RUN mkdir -p /workspace/models
WORKDIR /workspace/models
RUN wget https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors

# Download SDXL base model
RUN python -c "from diffusers import StableDiffusionXLPipeline; StableDiffusionXLPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype='float16')"

# Copy handler script
COPY handler.py /workspace/handler.py

CMD ["python", "/workspace/handler.py"]
```

---

### Option B: Manual Setup on Existing RunPod

SSH into your RunPod instance:

```bash
# Install IP-Adapter
pip install ip-adapter insightface onnxruntime-gpu

# Download weights
cd /workspace
mkdir -p models/ip-adapter
cd models/ip-adapter
wget https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors
wget https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors
```

---

## 🐍 RunPod Handler Code

Create `/workspace/handler.py`:

```python
import runpod
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from ip_adapter import IPAdapterXL
from PIL import Image
import base64
from io import BytesIO
import os

# Load models once at startup (for serverless efficiency)
print("Loading SDXL pipeline...")
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()

# Load IP-Adapter
print("Loading IP-Adapter...")
ip_adapter = IPAdapterXL(
    pipe,
    image_encoder_path="h94/IP-Adapter/sdxl_models/image_encoder",
    ip_ckpt="models/ip-adapter/ip-adapter_sdxl_vit-h.safetensors",
    device="cuda"
)

print("Models loaded successfully!")

def decode_base64_image(base64_string):
    """Convert base64 string to PIL Image"""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data)).convert('RGB')
    return image

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def handler(job):
    """
    RunPod serverless handler

    Expected input:
    {
        "prompt": "woman in lingerie, soft lighting...",
        "negative_prompt": "ugly, deformed...",
        "reference_image": "base64_encoded_image",  # NEW: Reference face image
        "ip_adapter_scale": 0.7,  # NEW: 0.5-0.8 recommended
        "width": 768,
        "height": 1024,
        "num_inference_steps": 30,
        "guidance_scale": 7.5
    }
    """
    try:
        job_input = job['input']

        # Extract parameters
        prompt = job_input.get('prompt', '')
        negative_prompt = job_input.get('negative_prompt', '')
        reference_image_b64 = job_input.get('reference_image', None)
        ip_adapter_scale = job_input.get('ip_adapter_scale', 0.7)
        width = job_input.get('width', 768)
        height = job_input.get('height', 1024)
        num_steps = job_input.get('num_inference_steps', 30)
        guidance = job_input.get('guidance_scale', 7.5)

        # Decode reference image if provided
        use_ip_adapter = reference_image_b64 is not None
        reference_image = None

        if use_ip_adapter:
            print("Using IP-Adapter with reference image")
            reference_image = decode_base64_image(reference_image_b64)

            # Generate with IP-Adapter
            output = ip_adapter.generate(
                pil_image=reference_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                scale=ip_adapter_scale,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                height=height,
                width=width
            )[0]
        else:
            print("Using standard text-to-image (no IP-Adapter)")
            # Fallback to standard generation
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                height=height,
                width=width
            ).images[0]

        # Convert to base64
        output_b64 = image_to_base64(output)

        return {
            "image": output_b64,
            "used_ip_adapter": use_ip_adapter,
            "ip_adapter_scale": ip_adapter_scale if use_ip_adapter else None
        }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

# Start the serverless handler
runpod.serverless.start({"handler": handler})
```

---

## 🧪 Testing Your RunPod Endpoint

Use RunPod's test interface or curl:

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "woman in elegant lingerie, soft lighting, boudoir photography",
      "negative_prompt": "ugly, deformed, blurry",
      "reference_image": "iVBORw0KGgoAAAANS...",
      "ip_adapter_scale": 0.7,
      "width": 768,
      "height": 1024,
      "num_inference_steps": 30,
      "guidance_scale": 7.5
    }
  }'
```

---

## ⚙️ Optimal IP-Adapter Scale Values

| Scale | Effect | Use Case |
|-------|--------|----------|
| 0.5 | Subtle face influence | Major body/pose changes |
| 0.6 | Light face preservation | Different clothing styles |
| **0.7** | **Recommended balance** | **General use** |
| 0.8 | Strong face preservation | Keep face exact, change clothes |
| 0.9+ | Very strong (can look unnatural) | Not recommended |

---

## 📊 Expected Results

### Before (Text-only):
- Facial similarity: ~30-50%
- User frustration: High
- Progressive generations lose likeness

### After (IP-Adapter):
- Facial similarity: **80-85%**
- User engagement: High
- Progressive generations maintain same face
- Speed: Same as before

---

## 🚀 Deployment Checklist

- [ ] Docker image built and pushed (or manual setup complete)
- [ ] RunPod endpoint created with new handler
- [ ] Test generation with reference image
- [ ] Verify base64 encoding/decoding works
- [ ] Update `RUNPOD_ENDPOINT_ID` in Cloudflare if endpoint changed
- [ ] Test from frontend

---

## 🔄 Rollback Plan

If IP-Adapter doesn't work:
- Keep old endpoint ID
- Revert frontend changes
- Reference image parameter will be ignored by old endpoint

---

## 💡 Next Steps

Once IP-Adapter is stable:
- Consider InstantID for 95% accuracy
- Add "Face Strength" slider in UI (0.5-0.8)
- A/B test with users to measure engagement improvement
