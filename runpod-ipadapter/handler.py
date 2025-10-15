"""
RunPod Serverless Handler with IP-Adapter FaceID (SDXL)
Locks identity using face embeddings from InsightFace.
"""

print("=" * 60)
print("🚀 Handler script starting...")
print("=" * 60, flush=True)

import base64
import inspect
import os
import sys
import traceback
from io import BytesIO

import numpy as np
import runpod
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from diffusers import AutoencoderKL, StableDiffusionXLInpaintPipeline, ControlNetModel
from diffusers.pipelines import StableDiffusionXLControlNetInpaintPipeline
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis
from controlnet_aux import MidasDetector
from segmentation import (
    SEGMENTATION_READY,
    YOLO_SAM_READY,
    generate_clothing_mask,
    generate_clothing_mask_yolo_sam,
    load_models,
    load_yolo_sam_models,
)
import segmentation as segmentation_module

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
MIN_MASK_FRACTION = float(os.getenv("MASK_MIN_FRACTION", "0.45"))
FACE_PADDING_FRACTION = float(os.getenv("MASK_FACE_PADDING", "0.05"))
USE_ADVANCED_SEGMENTATION = os.getenv("USE_SEGMENTATION", "1").lower() in {"1", "true", "yes"}  # Default to ON
USE_CONTROLNET = os.getenv("USE_CONTROLNET", "1").lower() in {"1", "true", "yes"}  # Default to ON
CONTROLNET_TYPE = os.getenv("CONTROLNET_TYPE", "depth")  # Or "openpose" for poses like your flexing arm

FACEID_CANDIDATES = [
    # Try PlusV2 first (best quality) - note: might be named differently in library
    ("IPAdapterFaceIDPlusV2", "ip-adapter-faceid-plusv2_sdxl.bin"),
    ("IPAdapterFaceIDPlusV2SDXL", "ip-adapter-faceid-plusv2_sdxl.bin"),
    ("IPAdapterFaceIDPlusV2XL", "ip-adapter-faceid-plusv2_sdxl.bin"),
    # Then try Plus
    ("IPAdapterFaceIDPlus", "ip-adapter-faceid-plusv2_sdxl.bin"),
    ("IPAdapterFaceIDPlusXL", "ip-adapter-faceid-plusv2_sdxl.bin"),
    # Finally fall back to basic
    ("IPAdapterFaceIDXL", "ip-adapter-faceid_sdxl.bin"),
    ("IPAdapterFaceID", "ip-adapter-faceid_sdxl.bin"),
]
AVAILABLE_FACEID_CANDIDATES = [
    (getattr(faceid_module, name), weight, name)
    for name, weight in FACEID_CANDIDATES
    if hasattr(faceid_module, name)
]

# Debug: Print what FaceID classes are actually available
print(f"Available FaceID classes: {[name for _, _, name in AVAILABLE_FACEID_CANDIDATES]}")

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
print(f"Advanced segmentation enabled: {USE_ADVANCED_SEGMENTATION}")

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

# Initialize ControlNet if enabled
CONTROLNET_AVAILABLE = False
controlnet_pipe = None
depth_processor = None
if USE_CONTROLNET:
    try:
        print(f"Loading ControlNet ({CONTROLNET_TYPE}) for pose/depth preservation...")
        if CONTROLNET_TYPE == "depth":
            controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-depth-sdxl-1.0",
                torch_dtype=torch.float16,
                variant="fp16"
            )
            depth_processor = MidasDetector.from_pretrained("lllyasviel/Annotators")
            print("✓ Depth estimator loaded")
        elif CONTROLNET_TYPE == "openpose":
            controlnet = ControlNetModel.from_pretrained(
                "thibaud/controlnet-openpose-sdxl-1.0",
                torch_dtype=torch.float16
            )
            from controlnet_aux import OpenposeDetector
            depth_processor = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            print("✓ OpenPose detector loaded")
        elif CONTROLNET_TYPE == "canny":
            controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-canny-sdxl-1.0",
                torch_dtype=torch.float16
            )
            depth_processor = None  # Canny uses cv2 directly
            print("✓ Canny edge detector will use OpenCV")
        else:
            raise ValueError(f"Unsupported CONTROLNET_TYPE: {CONTROLNET_TYPE}")

        # Create ControlNet inpaint pipeline
        controlnet_pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to(device)
        controlnet_pipe.enable_xformers_memory_efficient_attention()
        controlnet_pipe.enable_vae_slicing()
        CONTROLNET_AVAILABLE = True
        print(f"✓ ControlNet ({CONTROLNET_TYPE}) ready for depth/pose preservation")
    except Exception as controlnet_error:
        print(f"❌ ControlNet initialization failed: {controlnet_error}")
        traceback.print_exc()
        CONTROLNET_AVAILABLE = False
        controlnet_pipe = None
        depth_processor = None

FACEID_AVAILABLE = False
face_app = None
ip_adapter = None
faceid_error = None
SEGMENTATION_AVAILABLE = False
YOLO_SAM_AVAILABLE = False

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
# Attempt to initialize advanced segmentation models.
if USE_ADVANCED_SEGMENTATION:
    # Try loading YOLOv8 + SAM first (preferred)
    try:
        print("Attempting to load YOLOv8 + SAM segmentation models...")
        load_yolo_sam_models()
        # Access the module attribute directly to get the current value
        YOLO_SAM_AVAILABLE = segmentation_module.YOLO_SAM_READY
        if YOLO_SAM_AVAILABLE:
            print("✓ YOLOv8 + SAM ready for advanced segmentation")
        else:
            print("YOLOv8 + SAM not available, trying DeepLabV3 fallback...")
    except Exception as yolo_sam_error:
        YOLO_SAM_AVAILABLE = False
        print(f"YOLOv8 + SAM initialization failed: {yolo_sam_error}")
        traceback.print_exc()

    # Fall back to DeepLabV3 if YOLOv8+SAM unavailable
    if not YOLO_SAM_AVAILABLE:
        try:
            print("Loading DeepLabV3 segmentation fallback...")
            load_models()
            # Access the module attribute directly to get the current value
            SEGMENTATION_AVAILABLE = segmentation_module.SEGMENTATION_READY
            if SEGMENTATION_AVAILABLE:
                print("✓ DeepLabV3 loaded as fallback segmentation")
        except Exception as segmentation_error:
            SEGMENTATION_AVAILABLE = False
            USE_ADVANCED_SEGMENTATION = False
            print(f"Failed to load DeepLabV3 fallback: {segmentation_error}")
            traceback.print_exc()

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

def generate_inpaint_mask(image: Image.Image, face_bbox=None, face_padding_fraction=None):
    """
    Generate an inpainting mask for clothing replacement.

    Args:
        image: The input PIL Image
        face_bbox: Optional face bounding box (x1, y1, x2, y2)
        face_padding_fraction: Optional padding above face as fraction of image height.
                               If None, uses FACE_PADDING_FRACTION constant.

    Returns:
        Tuple of (mask_image, y_start_pixel)
    """
    if face_padding_fraction is None:
        face_padding_fraction = FACE_PADDING_FRACTION

    width, height = image.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    threshold = int(height * MIN_MASK_FRACTION)

    if face_bbox:
        _, y1, _, y2 = face_bbox
        computed = int(y2 + face_padding_fraction * height)
        y_start = max(threshold, min(height, computed))
    else:
        y_start = max(threshold, int(height * 0.35))

    # Expand slightly to ensure shoulders are covered
    draw.rectangle([(0, y_start), (width, height)], fill=255)

    # Feather the mask edges for smoother blending
    blurred = mask.filter(ImageFilter.GaussianBlur(radius=20))
    return blurred, y_start

def generate_control_image(image: Image.Image, control_type: str):
    """Generate control conditioning image (depth, pose, or canny edge)."""
    try:
        if control_type == "depth" and depth_processor is not None:
            depth_image = depth_processor(image)
            return depth_image
        elif control_type == "openpose" and depth_processor is not None:
            pose_image = depth_processor(image)
            return pose_image
        elif control_type == "canny":
            import cv2
            image_array = np.array(image)
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(edges_rgb)
        else:
            return None
    except Exception as e:
        print(f"Error generating {control_type} control image: {e}")
        traceback.print_exc()
        return None

def handler(job):
    job_input = job.get("input", {})

    try:
        # Simplified: Only master prompt and negative prompt
        negative_prompt_raw = job_input.get("negative_prompt")
        reference_image_b64 = job_input.get("reference_image")
        # Lower FaceID scale to prevent body warping (0.5-0.6 recommended for anatomy preservation)
        ip_adapter_scale = float(job_input.get("ip_adapter_scale", 0.55))
        width = int(job_input.get("width", 768))
        height = int(job_input.get("height", 1024))
        num_steps = int(job_input.get("num_inference_steps", 100))
        guidance = float(job_input.get("guidance_scale", 9.0))
        seed_raw = job_input.get("seed")

        # Face padding for mask coverage (higher = more area above face)
        face_padding_override = job_input.get("face_padding")
        if face_padding_override is not None:
            face_padding_fraction = float(face_padding_override)
            print(f"Using custom face padding: {face_padding_fraction}")
        else:
            face_padding_fraction = FACE_PADDING_FRACTION

        # Simplified: Use master prompt only (no user prompt complexity)
        master_prompt_override = job_input.get("master_prompt")
        if master_prompt_override is not None:
            master_prompt = str(master_prompt_override).strip()
        else:
            master_prompt = get_master_prompt().strip()

        # For now, ignore user prompts and just use master prompt
        prompt = master_prompt

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
        print(f"  Steps: {num_steps}, Guidance: {guidance}, Strength: {strength}")
        print(f"  Master prompt: {bool(master_prompt)}")
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
            else:
                print(f"Face bbox detected: {face_bbox}")

        segmentation_diagnostics = {}
        mask_image = None
        mask_start_px = None
        person_detected = False  # Track if any person detection method succeeds

        if USE_ADVANCED_SEGMENTATION:
            # Priority 1: Try YOLOv8 + SAM (most accurate)
            if YOLO_SAM_AVAILABLE:
                try:
                    print("Attempting YOLOv8 + SAM segmentation...")
                    mask_candidate, segmentation_diagnostics = generate_clothing_mask_yolo_sam(
                        reference_image, face_bbox
                    )
                    if mask_candidate is not None:
                        print("✓ YOLOv8 + SAM mask generated successfully")
                        mask_image = mask_candidate.convert("L")
                        person_detected = True  # YOLOv8 successfully detected a person
                except Exception as yolo_sam_error:
                    segmentation_diagnostics = {
                        "segmentation_used": False,
                        "reason": f"YOLOv8+SAM error: {yolo_sam_error}",
                        "method": "yolo_sam_failed"
                    }
                    print(f"YOLOv8+SAM segmentation failed: {yolo_sam_error}")
                    traceback.print_exc()

            # Priority 2: Fall back to DeepLabV3 (if YOLOv8+SAM failed or unavailable)
            if mask_image is None and SEGMENTATION_AVAILABLE:
                try:
                    print("Attempting DeepLabV3 segmentation fallback...")
                    mask_candidate, segmentation_diagnostics = generate_clothing_mask(
                        reference_image, face_bbox
                    )
                    if mask_candidate is not None:
                        print("✓ DeepLabV3 fallback mask generated")
                        mask_image = mask_candidate.convert("L")
                        person_detected = True  # DeepLabV3 successfully detected a person
                except Exception as segmentation_error:
                    segmentation_diagnostics = {
                        "segmentation_used": False,
                        "reason": f"DeepLabV3 error: {segmentation_error}",
                        "method": "deeplabv3_failed"
                    }
                    print(f"DeepLabV3 segmentation failed: {segmentation_error}")
                    traceback.print_exc()

        # Priority 3: Heuristic mask (final fallback)
        if mask_image is None:
            print("Using heuristic rectangle mask (all segmentation methods unavailable or failed)")
            mask_image, mask_start_px = generate_inpaint_mask(reference_image, face_bbox, face_padding_fraction)
            segmentation_diagnostics.setdefault("segmentation_used", False)
            segmentation_diagnostics.setdefault("reason", "Heuristic mask applied")
            segmentation_diagnostics.setdefault("method", "heuristic")
        else:
            mask_start_px = None
            segmentation_diagnostics.setdefault("segmentation_used", True)

        # Invert the mask to target the clothing area for inpainting
        mask_image = ImageOps.invert(mask_image.convert("L"))


        # ⚠️ CRITICAL: Enforce face/person detection requirement
        # If neither face NOR person was detected, reject the job
        face_detected = (faceid_embeds is not None and face_bbox is not None)

        if not face_detected and not person_detected:
            error_message = (
                "No face or person detected in the input image. "
                "This application requires a clearly visible person in the photo. "
                "Please upload a different image with a person clearly visible."
            )
            print(f"❌ JOB REJECTED: {error_message}")
            return {
                "error": error_message,
                "rejection_reason": "no_detection",
                "face_detected": False,
                "person_detected": False,
                "segmentation": segmentation_diagnostics
            }

        mask_start_fraction = round(mask_start_px / height, 4) if height and mask_start_px is not None else None
        if mask_start_fraction is not None:
            print(f"Mask starts at pixel row {mask_start_px} (~{mask_start_fraction} of height)")
        else:
            print("Mask start fraction unavailable (advanced segmentation provided explicit mask).")
        mask_b64 = image_to_base64(mask_image.convert("L"))

        generator = None
        if seed_value is not None:
            generator = torch.Generator(device=device).manual_seed(seed_value)

        strength = float(job_input.get("strength", 0.5))  # Lower = preserve more of original

        if use_faceid and ip_adapter is not None and faceid_embeds is not None:
            # CRITICAL: Generate ControlNet conditioning BEFORE FaceID generation
            control_image = None
            if CONTROLNET_AVAILABLE and controlnet_pipe is not None:
                print(f"Generating ControlNet {CONTROLNET_TYPE} conditioning for pose preservation...")
                control_image = generate_control_image(reference_image, CONTROLNET_TYPE)
                if control_image is not None:
                    print(f"✓ ControlNet {CONTROLNET_TYPE} image generated successfully")
                else:
                    print(f"⚠️ ControlNet {CONTROLNET_TYPE} image generation failed, continuing without it")

            print(f"Generating with FaceID (id_scale={ip_adapter_scale}, strength={strength}, controlnet={'YES' if control_image is not None else 'NO'}) ...")
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
            )

            # Add ControlNet conditioning if available
            if control_image is not None:
                faceid_kwargs["control_image"] = control_image
                faceid_kwargs["controlnet_conditioning_scale"] = 0.6  # Moderate control to preserve pose

            if seed_value is not None:
                faceid_kwargs["seed"] = seed_value
            # Try to add strength if the FaceID method supports it
            try:
                faceid_kwargs["strength"] = strength
                output_images = ip_adapter.generate(**faceid_kwargs)
            except TypeError as e:
                # If strength or control_image not supported, try without them
                error_msg = str(e).lower()
                if "strength" in error_msg:
                    print("Note: FaceID adapter doesn't support strength parameter")
                    faceid_kwargs.pop("strength", None)
                if "control" in error_msg:
                    print("Note: FaceID adapter doesn't support ControlNet parameters")
                    faceid_kwargs.pop("control_image", None)
                    faceid_kwargs.pop("controlnet_conditioning_scale", None)
                output_images = ip_adapter.generate(**faceid_kwargs)
            output_image = output_images[0]
        else:
            # Use ControlNet if available for better structure preservation
            if CONTROLNET_AVAILABLE and controlnet_pipe is not None:
                print(f"Generating with ControlNet {CONTROLNET_TYPE} (no FaceID, strength={strength}) ...")
                control_image = generate_control_image(reference_image, CONTROLNET_TYPE)
                if control_image is not None:
                    output_image = controlnet_pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance,
                        image=reference_image,
                        mask_image=mask_image,
                        control_image=control_image,
                        strength=strength,
                        controlnet_conditioning_scale=0.5,  # Control strength (0-1)
                        generator=generator
                    ).images[0]
                else:
                    print("⚠️ Control image generation failed, falling back to regular inpainting")
                    output_image = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance,
                        image=reference_image,
                        mask_image=mask_image,
                        strength=strength,
                        generator=generator
                    ).images[0]
            else:
                print(f"Generating with inpainting (no FaceID, strength={strength}) ...")
                output_image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance,
                    image=reference_image,
                    mask_image=mask_image,
                    strength=strength,
                    generator=generator
                ).images[0]

        print("Converting output to base64 ...")
        output_b64 = image_to_base64(output_image)
        print("Generation complete.")

        # Convert numpy types to Python native types for JSON serialization
        def convert_to_json_serializable(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Check if ControlNet was actually used (either with FaceID or standalone)
        used_controlnet_flag = False
        if use_faceid and CONTROLNET_AVAILABLE:
            # ControlNet was attempted with FaceID - check if control_image was generated
            used_controlnet_flag = True  # We attempted it, even if it might have failed
        elif not use_faceid and CONTROLNET_AVAILABLE and controlnet_pipe is not None:
            used_controlnet_flag = True

        response = {
            "image": output_b64,
            "used_faceid": use_faceid and ip_adapter is not None and faceid_embeds is not None,
            "used_controlnet": used_controlnet_flag,
            "controlnet_type": CONTROLNET_TYPE if CONTROLNET_AVAILABLE else None,
            "id_scale": ip_adapter_scale if use_faceid else None,
            "strength": strength,
            "guidance_scale": guidance,
            "width": width,
            "height": height,
            "num_inference_steps": num_steps,
            "prompt": prompt,
            "mask_image": mask_b64,
            "mask_start_fraction": mask_start_fraction,
            "face_bbox": [float(x) for x in face_bbox] if face_bbox else None,
            "segmentation": convert_to_json_serializable(segmentation_diagnostics)
        }
        return response

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print("\nError occurred:")
        print(error_trace)
        return {"error": error_msg, "traceback": error_trace}

print("\nStarting RunPod serverless handler ...", flush=True)
try:
    runpod.serverless.start({"handler": handler})
except Exception as startup_error:
    print(f"\n❌ FATAL ERROR: Handler failed to start: {startup_error}")
    traceback.print_exc()
    sys.exit(1)
