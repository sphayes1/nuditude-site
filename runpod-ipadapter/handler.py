"""
RunPod Serverless Handler with IP-Adapter FaceID (SDXL)
Locks identity using face embeddings from InsightFace.
"""

import base64
import os
import traceback
from io import BytesIO

import numpy as np
import runpod
import torch
from PIL import Image
from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis

# Dynamically select the FaceID class/weight available in ip_adapter package
try:
    from ip_adapter import ip_adapter_faceid as faceid_module
except ImportError:
    import ip_adapter.ip_adapter_faceid as faceid_module  # type: ignore

FACEID_CLASS = None
FACEID_FILENAME = None
FACEID_CANDIDATES = [
    ("IPAdapterFaceIDPlusV2XL", "ip-adapter-faceid-plusv2_sdxl.bin"),
    ("IPAdapterFaceIDPlusXL", "ip-adapter-faceid-plusv2_sdxl.bin"),
    ("IPAdapterFaceIDXL", "ip-adapter-faceid_sdxl.bin")
]
for class_name, filename in FACEID_CANDIDATES:
    if hasattr(faceid_module, class_name):
        FACEID_CLASS = getattr(faceid_module, class_name)
        FACEID_FILENAME = filename
        print(f"Selected FaceID class: {class_name} with weight {filename}")
        break
if FACEID_CLASS is None or FACEID_FILENAME is None:
    raise ImportError("No suitable FaceID class available in ip_adapter.faceid")

print("=" * 60)
print("Initializing SDXL + FaceID Worker")
print("=" * 60)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading VAE...")
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16
)

print("Loading SDXL pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to(device)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()
print("SDXL pipeline loaded successfully!")

print("Loading FaceID (IP-Adapter) ...")
try:
    FACEID_PATH = f"/workspace/models/ip-adapter/{FACEID_FILENAME}"
    os.makedirs(os.path.dirname(FACEID_PATH), exist_ok=True)
    size = os.path.getsize(FACEID_PATH) if os.path.exists(FACEID_PATH) else 0
    if size < 1024 * 1024:
        print(f"FaceID weight missing or too small (size={size}). Downloading via huggingface_hub...")
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            print("Warning: HUGGINGFACE_TOKEN not set; attempting anonymous download (may fail).")
        try:
            hf_hub_download(
                repo_id="h94/IP-Adapter-FaceID",
                filename=FACEID_FILENAME,
                token=token,
                local_dir=os.path.dirname(FACEID_PATH),
                local_dir_use_symlinks=False
            )
        except Exception as de:
            print(f"hf_hub_download error: {de}")
            raise
except Exception as e:
    print(f"FaceID preflight warning: {e}")

try:
    size = os.path.getsize(FACEID_PATH) if os.path.exists(FACEID_PATH) else 0
    print(f"FaceID weight at {FACEID_PATH}, size={size} bytes")
    if size < 1024 * 1024:
        raise Exception(f"FaceID weight file is too small ({size} bytes) or missing")

    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    ip_adapter = FACEID_CLASS(
        pipe,
        image_encoder_path="/workspace/models/image_encoder",
        ip_ckpt=FACEID_PATH,
        device=device
    )
    print("Ã¢Å“â€œ FaceID loaded successfully!")
    FACEID_AVAILABLE = True
except Exception as e:
    print("FaceID failed to load:")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("Full traceback:")
    traceback.print_exc()
    print("Will fallback to text-only generation")
    FACEID_AVAILABLE = False

print("=" * 60)
print("Worker ready! Waiting for jobs...")
print("=" * 60)


def decode_base64_image(base64_string: str):
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        max_size = 1024
        if image.width > max_size or image.height > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            print(f"Resized reference image to {image.size}")
        return image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


def image_to_base64(image: Image.Image) -> str:
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
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        emb = faces[0].normed_embedding
        if emb is None:
            return None
        return torch.tensor(emb, device=device).unsqueeze(0).float()
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


def handler(job):
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
        print("Ã¢Å“â€œ Generation complete!")

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