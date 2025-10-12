"""
RunPod Serverless Handler with IP-Adapter Support
Provides SDXL generation with facial preservation using IP-Adapter
"""

import runpod
import torch
import traceback
import base64
from io import BytesIO
from PIL import Image
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from ip_adapter import IPAdapterXL

print("=" * 60)
print("Initializing SDXL + IP-Adapter Worker")
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

# Enable memory optimizations
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()

print("SDXL pipeline loaded successfully!")

# Load IP-Adapter
print("Loading IP-Adapter...")
try:
    ip_adapter = IPAdapterXL(
        pipe,
        image_encoder_path="/workspace/models/image_encoder",
        ip_ckpt="/workspace/models/ip-adapter/ip-adapter_sdxl_vit-h.safetensors",
        device=device
    )
    print("✓ IP-Adapter loaded successfully!")
    IP_ADAPTER_AVAILABLE = True
except Exception as e:
    print(f"⚠ IP-Adapter failed to load: {e}")
    print("Will fallback to text-only generation")
    IP_ADAPTER_AVAILABLE = False

print("=" * 60)
print("Worker ready! Waiting for jobs...")
print("=" * 60)


def decode_base64_image(base64_string):
    """Convert base64 string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        # Decode base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data)).convert('RGB')

        # Resize if too large (for memory efficiency)
        max_size = 1024
        if image.width > max_size or image.height > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            print(f"Resized reference image to {image.size}")

        return image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG", optimize=True)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def handler(job):
    """
    RunPod serverless handler

    Expected input:
    {
        "prompt": "woman in lingerie, soft lighting, boudoir photography",
        "negative_prompt": "ugly, deformed, blurry",
        "reference_image": "base64_string",  # Optional - for IP-Adapter
        "ip_adapter_scale": 0.7,  # Optional - 0.5 to 0.8
        "width": 768,
        "height": 1024,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "seed": null  # Optional - for reproducibility
    }
    """
    job_input = job.get('input', {})

    try:
        # Extract parameters
        prompt = job_input.get('prompt', '')
        negative_prompt = job_input.get('negative_prompt', 'ugly, deformed, blurry, low quality, distorted')
        reference_image_b64 = job_input.get('reference_image', None)
        ip_adapter_scale = float(job_input.get('ip_adapter_scale', 0.7))
        width = int(job_input.get('width', 768))
        height = int(job_input.get('height', 1024))
        num_steps = int(job_input.get('num_inference_steps', 30))
        guidance = float(job_input.get('guidance_scale', 7.5))
        seed = job_input.get('seed', None)

        print(f"\n{'='*60}")
        print(f"Job received:")
        print(f"  Prompt: {prompt[:100]}...")
        print(f"  Reference image: {bool(reference_image_b64)}")
        print(f"  IP-Adapter scale: {ip_adapter_scale}")
        print(f"  Size: {width}x{height}")
        print(f"  Steps: {num_steps}, Guidance: {guidance}")
        print(f"{'='*60}\n")

        # Validate prompt
        if not prompt:
            return {"error": "Prompt is required"}

        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(seed))
            print(f"Using seed: {seed}")

        # Check if we should use IP-Adapter
        use_ip_adapter = IP_ADAPTER_AVAILABLE and reference_image_b64 is not None

        if use_ip_adapter:
            print("🎨 Using IP-Adapter for facial preservation...")

            # Decode reference image
            reference_image = decode_base64_image(reference_image_b64)

            if reference_image is None:
                print("⚠ Failed to decode reference image, falling back to text-only")
                use_ip_adapter = False

        # Generate image
        if use_ip_adapter:
            # Generate with IP-Adapter
            print(f"Generating with IP-Adapter (scale={ip_adapter_scale})...")
            output_image = ip_adapter.generate(
                pil_image=reference_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                scale=ip_adapter_scale,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                height=height,
                width=width,
                generator=generator
            )[0]
        else:
            # Standard text-to-image
            print("Generating with text-only (no IP-Adapter)...")
            output_image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                height=height,
                width=width,
                generator=generator
            ).images[0]

        # Convert to base64
        print("Converting output to base64...")
        output_b64 = image_to_base64(output_image)

        print("✓ Generation complete!")

        return {
            "image": output_b64,
            "used_ip_adapter": use_ip_adapter,
            "ip_adapter_scale": ip_adapter_scale if use_ip_adapter else None,
            "width": width,
            "height": height,
            "num_inference_steps": num_steps
        }

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()

        print(f"\n❌ Error occurred:")
        print(error_trace)

        return {
            "error": error_msg,
            "traceback": error_trace
        }


# Start the serverless handler
print("\n🚀 Starting RunPod serverless handler...")
runpod.serverless.start({"handler": handler})
