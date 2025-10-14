"""
RunPod Serverless Handler with IP-Adapter FaceID (SDXL)
Locks identity using face embeddings from InsightFace.
"""

import base64
import inspect
import os
import traceback
from io import BytesIO

import numpy as np
import runpod
import torch
from PIL import Image, ImageDraw, ImageFilter
from diffusers import AutoencoderKL, StableDiffusionXLInpaintPipeline
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis

try:
    from ip_adapter import ip_adapter_faceid as faceid_module
except ImportError:
    import ip_adapter.ip_adapter_faceid as faceid_module  # type: ignore

MASTER_PROMPT_PATH = os.getenv("MASTER_PROMPT_PATH", "/workspace/config/master_prompt.txt")
DEFAULT_MASTER_PROMPT = os.getenv("MASTER_PROMPT_TEXT", "").strip()
MASTER_PROMPT_CACHE = {"text": DEFAULT_MASTER_PROMPT, "mtime": None}

NEGATIVE_PROMPT_PATH = os.getenv("MASTER_NEGATIVE_PROMPT_PATH", "/workspace/config/negative_prompt.txt")
DEFAULT_NEGATIVE_PROMPT = os.getenv(
    "MASTER_NEGATIVE_PROMPT_TEXT",
    "ugly, deformed, blurry, low quality, distorted"
).strip()
NEGATIVE_PROMPT_CACHE = {"text": DEFAULT_NEGATIVE_PROMPT, "mtime": None}
ALLOW_USER_PROMPT_ENV = os.getenv("ALLOW_USER_PROMPT")

FACEID_CANDIDATES = [
    ("IPAdapterFaceIDPlusV2XL", "ip-adapter-faceid-plusv2_sdxl.bin"),
    ("IPAdapterFaceIDPlusXL", "ip-adapter-faceid-plusv2_sdxl.bin"),
    ("IPAdapterFaceIDXL", "ip-adapter-faceid_sdxl.bin"),
]
AVAILABLE_FACEID_CANDIDATES = [
    (getattr(faceid_module, name), weight, name)
    for name, weight in FACEID_CANDIDATES
    if hasattr(faceid_module, name)
]

if not AVAILABLE_FACEID_CANDIDATES:
    print("Warning: no FaceID classes found in ip_adapter; running text-only.")

print("=" * 60)
print("Initializing SDXL + FaceID Worker")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if MASTER_PROMPT_PATH:
    try:
        master_dir = os.path.dirname(MASTER_PROMPT_PATH)
        if master_dir:
            os.makedirs(master_dir, exist_ok=True)
    except Exception as e:
        print(f"Warning: unable to ensure master prompt directory {MASTER_PROMPT_PATH}: {e}")
if NEGATIVE_PROMPT_PATH:
    try:
        negative_dir = os.path.dirname(NEGATIVE_PROMPT_PATH)
        if negative_dir:
            os.makedirs(negative_dir, exist_ok=True)
    except Exception as e:
        print(f"Warning: unable to ensure negative prompt directory {NEGATIVE_PROMPT_PATH}: {e}")

print(f"Master prompt path: {MASTER_PROMPT_PATH or 'disabled'}")
print(f"Negative prompt path: {NEGATIVE_PROMPT_PATH or 'disabled'}")

print("Loading VAE...")
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16
)

print("Loading SDXL inpaint pipeline...")
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to(device)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()
print("SDXL pipeline loaded successfully!")

FACEID_AVAILABLE = False
face_app = None
ip_adapter = None
faceid_error = None

print("Loading FaceID (IP-Adapter) ...")
for faceid_class, weight_filename, class_name in AVAILABLE_FACEID_CANDIDATES:
    print(f"Trying FaceID class {class_name} with weight {weight_filename}")
    faceid_path = f"/workspace/models/ip-adapter/{weight_filename}"
    os.makedirs(os.path.dirname(faceid_path), exist_ok=True)

    size = os.path.getsize(faceid_path) if os.path.exists(faceid_path) else 0
    if size < 1024 * 1024:
        print(f"FaceID weight needs download (current size={size}).")
        download_kwargs = dict(
            repo_id="h94/IP-Adapter-FaceID",
            filename=weight_filename,
            local_dir=os.path.dirname(faceid_path),
            local_dir_use_symlinks=False,
        )
        token = os.getenv("HUGGINGFACE_TOKEN")
        if token:
            download_kwargs["token"] = token
        else:
            print("Warning: HUGGINGFACE_TOKEN not set; attempting anonymous download.")
        try:
            hf_hub_download(**download_kwargs)
        except Exception as download_error:
            print(f"hf_hub_download error for {weight_filename}: {download_error}")
            faceid_error = download_error
            continue
        size = os.path.getsize(faceid_path) if os.path.exists(faceid_path) else 0
        print(f"Downloaded FaceID weight size={size} bytes")
        if size < 1024 * 1024:
            faceid_error = Exception("FaceID weight still too small after download")
            print(faceid_error)
            continue

    try:
        face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        face_app.prepare(ctx_id=0, det_size=(640, 640))

        try:
            init_signature = inspect.signature(faceid_class.__init__)
            parameters = set(init_signature.parameters.keys())
        except (TypeError, ValueError):
            init_signature = None
            parameters = set()

        args = []
        kwargs = {"ip_ckpt": faceid_path, "device": device}

        if "sd_pipe" in parameters:
            kwargs["sd_pipe"] = pipe
        elif "pipe" in parameters:
            kwargs["pipe"] = pipe
        elif "pipeline" in parameters:
            kwargs["pipeline"] = pipe
        else:
            args.append(pipe)

        if "image_encoder_path" in parameters:
            kwargs["image_encoder_path"] = "/workspace/models/image_encoder"

        if "torch_dtype" in parameters:
            kwargs["torch_dtype"] = torch.float16

        ip_adapter = faceid_class(*args, **kwargs)

        print(f"FaceID loaded successfully using {class_name}!")
        FACEID_AVAILABLE = True
        faceid_error = None
        break
    except Exception as load_error:
        print(f"Failed to load {class_name}: {load_error}")
        traceback.print_exc()
        faceid_error = load_error
        face_app = None
        ip_adapter = None
        continue

if not FACEID_AVAILABLE:
    print(f"FaceID unavailable: {faceid_error}")
    print("Continuing in text-only mode.")

print("=" * 60)
print("Worker ready! Waiting for jobs...")
print("=" * 60)

def parse_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default

def load_prompt_from_source(label: str, path: str | None, cache: dict, default_value: str) -> str:
    cached_value = cache.get("text", "") or default_value
    if not path:
        return cached_value

    try:
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            if cache.get("mtime") != mtime:
                with open(path, "r", encoding="utf-8") as prompt_file:
                    new_value = prompt_file.read().strip()
                cache["text"] = new_value or default_value
                cache["mtime"] = mtime
        else:
            cache["text"] = cache.get("text", "") or default_value
    except Exception as exc:
        print(f"Warning: Failed to load {label} prompt from {path}: {exc}")

    return cache.get("text", "") or default_value

def get_master_prompt() -> str:
    return load_prompt_from_source("master", MASTER_PROMPT_PATH, MASTER_PROMPT_CACHE, DEFAULT_MASTER_PROMPT)

def get_negative_prompt_default() -> str:
    return load_prompt_from_source("negative", NEGATIVE_PROMPT_PATH, NEGATIVE_PROMPT_CACHE, DEFAULT_NEGATIVE_PROMPT)

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
        if face_app is None:
            return None, None
        bgr = pil_to_bgr(pil_image)
        faces = face_app.get(bgr)
        if not faces:
            print("No face detected in reference image")
            return None, None
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        emb = faces[0].normed_embedding
        if emb is None:
            return None, tuple(faces[0].bbox)
        return torch.tensor(emb, device=device).unsqueeze(0).float(), tuple(faces[0].bbox)
    except Exception as e:
        print(f"Embedding error: {e}")
        return None, None

def generate_inpaint_mask(image: Image.Image, face_bbox=None) -> Image.Image:
    width, height = image.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    if face_bbox:
        _, y1, _, y2 = face_bbox
        y_start = max(0, min(height, int(y2 + 0.05 * height)))
    else:
        y_start = int(height * 0.35)

    # Expand slightly to ensure shoulders are covered
    draw.rectangle([(0, y_start), (width, height)], fill=255)

    # Feather the mask edges for smoother blending
    return mask.filter(ImageFilter.GaussianBlur(radius=12))

def handler(job):
    job_input = job.get("input", {})

    try:
        prompt_raw = str(job_input.get("prompt", "") or "").strip()
        negative_prompt_raw = job_input.get("negative_prompt")
        reference_image_b64 = job_input.get("reference_image")
        ip_adapter_scale = float(job_input.get("ip_adapter_scale", 0.8))
        width = int(job_input.get("width", 768))
        height = int(job_input.get("height", 1024))
        num_steps = int(job_input.get("num_inference_steps", 28))
        guidance = float(job_input.get("guidance_scale", 4.5))
        seed_raw = job_input.get("seed")

        use_master_prompt = parse_bool(job_input.get("use_master_prompt"), True)
        allow_user_prompt_default = parse_bool(ALLOW_USER_PROMPT_ENV, False)
        allow_user_prompt = parse_bool(job_input.get("allow_user_prompt"), allow_user_prompt_default)
        master_prompt_override = job_input.get("master_prompt")
        if use_master_prompt:
            if master_prompt_override is not None:
                master_prompt = str(master_prompt_override).strip()
            else:
                master_prompt = get_master_prompt().strip()
        else:
            master_prompt = ""

        user_prompt = prompt_raw if allow_user_prompt else ""
        if prompt_raw and not allow_user_prompt:
            print("User prompt provided but ignored because allow_user_prompt is disabled.")

        prompt_parts = [part for part in (master_prompt, user_prompt) if part]
        prompt = ", ".join(prompt_parts).strip()

        if negative_prompt_raw is None or str(negative_prompt_raw).strip() == "":
            negative_prompt = get_negative_prompt_default()
        else:
            negative_prompt = str(negative_prompt_raw)

        seed_value = None
        if seed_raw is not None:
            try:
                seed_value = int(seed_raw)
            except (TypeError, ValueError):
                print(f"Invalid seed value provided ({seed_raw}); ignoring.")
                seed_value = None

        print("\n" + "=" * 60)
        print("Job received:")
        print(f"  Prompt: {prompt[:100]}...")
        print(f"  Reference image: {bool(reference_image_b64)}")
        print(f"  FaceID scale: {ip_adapter_scale}")
        print(f"  Requested size: {width}x{height}")
        print(f"  Steps: {num_steps}, Guidance: {guidance}")
        print(f"  Master prompt active: {bool(master_prompt)}")
        print(f"  User prompt allowed: {allow_user_prompt}")
        print("=" * 60 + "\n")

        if not prompt:
            return {"error": "Prompt is required"}

        if seed_value is not None:
            print(f"Using seed: {seed_value}")

        if not reference_image_b64:
            return {"error": "reference_image is required for inpainting"}

        reference_image = decode_base64_image(reference_image_b64)
        if reference_image is None:
            return {"error": "Failed to decode reference image"}

        if reference_image.width != width or reference_image.height != height:
            print(f"Resizing reference image from {reference_image.size} to {(width, height)}")
            reference_image = reference_image.resize((width, height), Image.Resampling.LANCZOS)
        width = reference_image.width
        height = reference_image.height

        use_faceid = FACEID_AVAILABLE
        faceid_embeds = None
        face_bbox = None

        if use_faceid:
            print("Using FaceID for identity preservation ...")
            faceid_embeds, face_bbox = get_face_embedding(reference_image)
            if faceid_embeds is None:
                print("No embedding extracted; falling back to inpaint without FaceID")
                use_faceid = False

        mask_image = generate_inpaint_mask(reference_image, face_bbox)
        mask_b64 = image_to_base64(mask_image.convert("L"))

        generator = None
        if seed_value is not None:
            generator = torch.Generator(device=device).manual_seed(seed_value)

        if use_faceid and ip_adapter is not None and faceid_embeds is not None:
            print(f"Generating with FaceID (id_scale={ip_adapter_scale}) ...")
            faceid_kwargs = dict(
                faceid_embeds=faceid_embeds,
                prompt=prompt,
                negative_prompt=negative_prompt,
                scale=ip_adapter_scale,
                num_samples=1,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                image=reference_image,
                mask_image=mask_image,
                generator=generator,
            )
            if seed_value is not None:
                faceid_kwargs["seed"] = seed_value
            output_images = ip_adapter.generate(**faceid_kwargs)
            output_image = output_images[0]
        else:
            print("Generating with inpainting (no FaceID) ...")
            output_image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                image=reference_image,
                mask_image=mask_image,
                generator=generator
            ).images[0]

        print("Converting output to base64 ...")
        output_b64 = image_to_base64(output_image)
        print("Generation complete.")

        return {
            "image": output_b64,
            "used_faceid": use_faceid and ip_adapter is not None and faceid_embeds is not None,
            "id_scale": ip_adapter_scale if use_faceid else None,
            "width": width,
            "height": height,
            "num_inference_steps": num_steps,
            "prompt": prompt,
            "master_prompt": master_prompt,
            "user_prompt_allowed": allow_user_prompt,
            "mask_image": mask_b64
        }

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print("\nError occurred:")
        print(error_trace)
        return {"error": error_msg, "traceback": error_trace}

print("\nStarting RunPod serverless handler ...")
runpod.serverless.start({"handler": handler})
