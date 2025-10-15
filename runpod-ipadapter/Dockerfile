# IP-Adapter enabled SDXL for RunPod Serverless
# Using public RunPod PyTorch base image

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Install IP-Adapter from GitHub (PyPI version lacks FaceID support)
RUN pip install --no-cache-dir \
    git+https://github.com/tencent-ailab/IP-Adapter.git \
    insightface==0.7.3 \
    onnxruntime-gpu==1.16.3 \
    opencv-python-headless==4.8.1.78

# Create workspace directory
WORKDIR /workspace

# Download IP-Adapter weights at build time for faster cold starts
RUN mkdir -p /workspace/models/ip-adapter && \
    curl -L --fail -o /workspace/models/ip-adapter/ip-adapter_sdxl_vit-h.safetensors \
    https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors && \
    test -s /workspace/models/ip-adapter/ip-adapter_sdxl_vit-h.safetensors

# FaceID weight is downloaded at runtime by handler.py (see lines 58-90)
# This allows for robust retry logic and multiple fallback URLs

# Download image encoder
RUN mkdir -p /workspace/models/image_encoder && \
    curl -L --fail -o /workspace/models/image_encoder/config.json \
    https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/config.json && \
    curl -L --fail -o /workspace/models/image_encoder/model.safetensors \
    https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors && \
    test -s /workspace/models/image_encoder/model.safetensors

# Download SAM checkpoint (ViT-B for balance of speed/quality)
RUN mkdir -p /workspace/models/sam && \
    curl -L --fail -o /workspace/models/sam/sam_vit_b_01ec64.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth && \
    test -s /workspace/models/sam/sam_vit_b_01ec64.pth

# Copy handler and segmentation module
COPY handler.py /workspace/handler.py
COPY segmentation.py /workspace/segmentation.py
COPY requirements.txt /workspace/requirements.txt

# Install any additional requirements
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Set Python to unbuffered mode for better logging
ENV PYTHONUNBUFFERED=1

# Expose ports (RunPod serverless uses these)
EXPOSE 8000

# Start handler
CMD ["python", "-u", "/workspace/handler.py"]

