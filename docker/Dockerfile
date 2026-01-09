# RailSafeNet Final - Docker Image
# Base: NVIDIA TensorRT for optimized inference
# Compatible with: x86_64 servers (TITAN RTX, RTX 30xx, etc.)
# For Jetson: use nvcr.io/nvidia/l4t-tensorrt:r8.5.2-runtime
FROM nvcr.io/nvidia/tensorrt:23.10-py3

# Metadata
LABEL maintainer="RailSafeNet Team"
LABEL description="Production rail detection system for Jetson AGX Orin"
LABEL version="2.0"

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:0
ENV PYTHONUNBUFFERED=1
ENV ASSETS_DIR=/app/assets
ENV CONFIG_DIR=/app/config
ENV RESULTS_DIR=/app/results

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-opencv \
    libopencv-dev \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    v4l-utils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip
RUN pip3 install --upgrade pip

# Install PyTorch with CUDA support
# Using CUDA 12.1 wheels (compatible with CUDA 12.x)
RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other Python dependencies
RUN pip3 install --no-cache-dir \
    "albumentations>=1.3.0" \
    "pycuda>=2022.1" \
    "numpy>=1.21.0,<2.0.0" \
    "scipy>=1.7.0" \
    "pyyaml>=6.0" \
    "tqdm>=4.64.0" \
    "opencv-python>=4.5.0" \
    "scikit-learn>=1.0.0" \
    "scikit-image>=0.19.0"

# Create application directory
WORKDIR /app

# Copy application code
COPY videoAssessor_final.py /app/
COPY videoAssessor.py /app/
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY config/ /app/config/

# Create directories for assets (to be mounted)
RUN mkdir -p /app/assets/models_pretrained/segformer/optimized \
    && mkdir -p /app/assets/models_pretrained/yolo \
    && mkdir -p /app/assets/crop \
    && mkdir -p /app/results \
    && mkdir -p /app/output

# Set permissions
RUN chmod +x /app/videoAssessor_final.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import cv2; import numpy; print('OK')" || exit 1

# Default entry point
ENTRYPOINT ["python3", "/app/videoAssessor_final.py"]

# Default command (video mode)
CMD ["--mode", "video"]
