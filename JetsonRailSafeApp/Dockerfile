# Base image for Jetson Orin (JetPack 5.1 / L4T 35.2.1)
# Includes PyTorch, TorchVision, Torchaudio, TensorRT, etc.
FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:0
ENV ASSETS_DIR=/app/assets

# Install system dependencies
# python3-pyqt5 is installed via apt to ensure compatibility with system libraries
RUN apt-get update && apt-get install -y \
    python3-pyqt5 \
    python3-opencv \
    libopencv-dev \
    qt5-default \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Note: PyTorch and TensorRT are already in the base image
RUN pip3 install --no-cache-dir \
    albumentations \
    pycuda \
    numpy \
    tqdm

# Create app directory
WORKDIR /app

# Copy application code
COPY . /app

# Create assets directory (to be mounted)
RUN mkdir -p /app/assets

# Set entrypoint
CMD ["python3", "main.py"]
