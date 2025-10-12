"""
RunPod Serverless Handler with IP-Adapter FaceID Plus (SDXL)
Locks identity using face embeddings from InsightFace.
"""

import runpod
import torch
import traceback
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

# InsightFace for face detection/embeddings
from insightface.app import FaceAnalysis

# FaceID Plus for SDXL (import path can vary by version)
try:
    from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlusXL
except Exception:  # fallback if packaged differently
    from ip_adapter import IPAdapterFaceIDPlusXL

print("=" * 60)
print("Initializing SDXL + FaceID Worker")
print("=" * 60)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load VAE (better quality)
print("Loading VAE...")
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16
)

# Load SDXL pipeline
print("Loading SDXL pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to(device)

# Memory optimizations
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()

print("SDXL pipeline loaded successfully!")

print("Loading FaceID (IP-Adapter) ...")
try:
    # Start InsightFace (antelopev2)
    face_app = FaceAnalysis(
        name="antelopev2",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    # Load FaceID Plus SDXL adapter
    ip_adapter = IPAdapterFaceIDPlusXL(
        pipe,
        image_encoder_path="/workspace/models/image_encoder",
        ip_ckpt="/workspace/models/ip-adapter/ip-adapter-faceid-plus_sdxl.bin",
        device=device
    )
    print("✓ FaceID loaded successfully!")
    FACEID_AVAILABLE = True
except Exception as e:
    print(f"⚠ FaceID failed to load: {e}")
    print("Will fallback to text-only generation")
    FACEID_AVAILABLE = False

print("=" * 60)
print("Worker ready! Waiting for jobs...")
print("=" * 60)


def decode_base64_image(base64_string: str):
    """Convert base64 string to PIL Image."""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        # Resize if too large (for memory)
        max_size = 1024
        if image.width > max_size or image.height > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            print(f"Resized reference image to {image.size}")
        return image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG", optimize=True)
    return base64.b64encode(buffered.getvalue()).decode()


def pil_to_bgr(pil_image: Image.Image):
    arr = np.array(pil_image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr[:, :, ::-1].copy()


def get_face_embedding(pil_image: Image.Image):
    try:
        bgr = pil_to_bgr(pil_image)
        faces = face_app.get(bgr)
        if not faces:
            print("No face detected in reference image")
            return None
        # choose largest face
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        emb = faces[0].normed_embedding
        if emb is None:
            return None
        return torch.tensor(emb, device=device).unsqueeze(0).float()
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


def handler(job):
    """
    RunPod serverless handler

    Expected input:
    {
        "prompt": "portrait, studio lighting",
        "negative_prompt": "ugly, deformed, blurry",
        "reference_image": "base64_string",   # optional, for FaceID
        "ip_adapter_scale": 0.8,               # 0.7–0.9 identity strength
        "width": 768,
        "height": 1024,
        "num_inference_steps": 28,
        "guidance_scale": 4.5,
        "seed": null
    }
    """
    job_input = job.get("input", {})

    try:
        prompt = job_input.get("prompt", "")
        negative_prompt = job_input.get("negative_prompt", "ugly, deformed, blurry, low quality, distorted")
        reference_image_b64 = job_input.get("reference_image")
        ip_adapter_scale = float(job_input.get("ip_adapter_scale", 0.8))
        width = int(job_input.get("width", 768))
        height = int(job_input.get("height", 1024))
        num_steps = int(job_input.get("num_inference_steps", 28))
        guidance = float(job_input.get("guidance_scale", 4.5))
        seed = job_input.get("seed")

        print("\n" + "=" * 60)
        print("Job received:")
        print(f"  Prompt: {prompt[:100]}...")
        print(f"  Reference image: {bool(reference_image_b64)}")
        print(f"  FaceID scale: {ip_adapter_scale}")
        print(f"  Size: {width}x{height}")
        print(f"  Steps: {num_steps}, Guidance: {guidance}")
        print("=" * 60 + "\n")

        if not prompt:
            return {"error": "Prompt is required"}

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(seed))
            print(f"Using seed: {seed}")

        use_faceid = FACEID_AVAILABLE and bool(reference_image_b64)
        faceid_embeds = None

        if use_faceid:
            print("Using FaceID for identity preservation ...")
            reference_image = decode_base64_image(reference_image_b64)
            if reference_image is None:
                print("Failed to decode reference image, fallback to text-only")
                use_faceid = False
            else:
                faceid_embeds = get_face_embedding(reference_image)
                if faceid_embeds is None:
                    print("No embedding extracted, fallback to text-only")
                    use_faceid = False

        if use_faceid:
            print(f"Generating with FaceID (id_scale={ip_adapter_scale}) ...")
            output_image = ip_adapter.generate(
                faceid_embeds=faceid_embeds,
                prompt=prompt,
                negative_prompt=negative_prompt,
                id_scale=ip_adapter_scale,
                image_prompt_scale=0.0,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                height=height,
                width=width,
                generator=generator
            )[0]
        else:
            print("Generating with text-only (no FaceID) ...")
            output_image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                height=height,
                width=width,
                generator=generator
            ).images[0]

        print("Converting output to base64 ...")
        output_b64 = image_to_base64(output_image)
        print("✓ Generation complete!")

        return {
            "image": output_b64,
            "used_faceid": use_faceid,
            "id_scale": ip_adapter_scale if use_faceid else None,
            "width": width,
            "height": height,
            "num_inference_steps": num_steps
        }

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print("\nError occurred:")
        print(error_trace)
        return {"error": error_msg, "traceback": error_trace}


print("\nStarting RunPod serverless handler ...")
runpod.serverless.start({"handler": handler})