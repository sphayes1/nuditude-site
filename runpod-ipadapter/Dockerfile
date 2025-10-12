# IP-Adapter enabled SDXL for RunPod Serverless
# Based on RunPod's A1111 worker image

FROM registry.runpod.net/runpod-workers-worker-a1111-main-dockerfile:022c30933

# Install IP-Adapter dependencies
RUN pip install --no-cache-dir \
    ip-adapter==0.1.0 \
    insightface==0.7.3 \
    onnxruntime-gpu==1.16.3 \
    opencv-python-headless==4.8.1.78

# Create workspace directory
WORKDIR /workspace

# Download IP-Adapter weights at build time for faster cold starts
RUN mkdir -p /workspace/models/ip-adapter && \
    wget -q --show-progress -O /workspace/models/ip-adapter/ip-adapter_sdxl_vit-h.safetensors \
    https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors

# Download image encoder
RUN mkdir -p /workspace/models/image_encoder && \
    wget -q --show-progress -O /workspace/models/image_encoder/config.json \
    https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/config.json && \
    wget -q --show-progress -O /workspace/models/image_encoder/model.safetensors \
    https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors

# Copy handler
COPY handler.py /workspace/handler.py
COPY requirements.txt /workspace/requirements.txt

# Install any additional requirements
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Set Python to unbuffered mode for better logging
ENV PYTHONUNBUFFERED=1

# Expose ports (RunPod serverless uses these)
EXPOSE 8000

# Start handler
CMD ["python", "-u", "/workspace/handler.py"]
