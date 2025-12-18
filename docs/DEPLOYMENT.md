# RailSafeNet LiDAR - Deployment Guide

## Deployment Target

**Primary Target Platform**: **Nvidia AGX Orin**

This system is designed to be deployed on the Nvidia AGX Orin edge AI platform, packaged as a Docker container for consistent deployment and easy management.

---

## Deployment Strategy

### Overview

The RailSafeNet LiDAR system will be:
1. Developed and tested on **Nvidia Titan RTX** (development environment)
2. Packaged into a **Docker container** with all dependencies
3. Deployed to **Nvidia AGX Orin** (production environment)

### Architecture

```
Development (Nvidia Titan RTX)
    ↓
Docker Container Build
    ↓
Container Registry
    ↓
Production Deployment (Nvidia AGX Orin)
```

---

## Target Hardware Specifications

### Nvidia AGX Orin

| Specification | Details |
|--------------|---------|
| **GPU** | Nvidia Ampere architecture (2048 CUDA cores, 64 Tensor cores) |
| **CPU** | 12-core ARM Cortex-A78AE |
| **Memory** | 32GB or 64GB LPDDR5 |
| **AI Performance** | Up to 275 TOPS |
| **Power** | 15W - 60W configurable |
| **CUDA Support** | CUDA 11.4+ |
| **TensorRT** | TensorRT 8.5+ |
| **JetPack** | JetPack 5.x or 6.x |

### Performance Expectations

- **Target FPS**: 25-30 FPS (real-time processing)
- **Frame Processing Time**: < 40ms per frame
- **Segmentation Inference**: < 40ms
- **Detection Inference**: < 25ms
- **Power Consumption**: 30-40W typical (configurable)

---

## Docker Containerization

### Container Requirements

#### Base Image
- **Base**: `nvcr.io/nvidia/l4t-tensorrt:r8.5.2-runtime`
  - Includes CUDA, cuDNN, TensorRT optimized for Jetson/AGX platforms
  - ARM64 architecture support
  - JetPack SDK integration

#### System Dependencies
```dockerfile
# CUDA Runtime
CUDA 11.4+
cuDNN 8.6+
TensorRT 8.5+

# Python Runtime
Python 3.8+
pip 21.0+

# Video Processing
OpenCV 4.5+ (with CUDA support)
GStreamer (for video streaming)

# System Libraries
libcudnn8
libnvinfer8
libnvonnxparsers8
```

#### Python Dependencies
```
numpy>=1.21.0
opencv-python>=4.5.0 (CUDA-enabled build)
tensorrt>=8.0.0
pycuda>=2021.1
torch>=1.10.0 (optional, if needed)
pytest>=7.0.0 (development only)
pytest-cov>=3.0.0 (development only)
```

### Container Structure

```
/app/
├── src/                      # Application source code
│   ├── rail_detection/
│   ├── processing/
│   └── utils/
├── assets/
│   └── models_pretrained/    # TensorRT engine files
│       ├── segformer/
│       │   └── *.engine
│       └── yolo/
│           └── *.engine
├── config/                   # Configuration files
│   └── production.yaml
├── logs/                     # Application logs
└── scripts/                  # Deployment scripts
    └── entrypoint.sh
```

---

## Build Process

### Step 1: Prepare TensorRT Engines

**On Development Machine (Titan RTX)**:
1. Train/fine-tune models
2. Export to ONNX format
3. Convert to TensorRT engines for **AGX Orin architecture**

**Important**: TensorRT engines must be built specifically for the target GPU architecture.

```bash
# Convert ONNX to TensorRT for AGX Orin (Ampere architecture)
trtexec --onnx=model.onnx \
        --saveEngine=model.engine \
        --fp16 \
        --workspace=4096 \
        --minShapes=input:1x3x512x896 \
        --optShapes=input:1x3x512x896 \
        --maxShapes=input:1x3x512x896
```

### Step 2: Build Docker Image

**Dockerfile** (to be created in Phase 7):
```dockerfile
# Multi-stage build for optimized image size
FROM nvcr.io/nvidia/l4t-tensorrt:r8.5.2-runtime AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ /app/src/
COPY assets/ /app/assets/
COPY config/ /app/config/
COPY scripts/ /app/scripts/

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Expose ports (if needed for monitoring)
EXPOSE 8080

# Entry point
CMD ["python3", "-m", "src.processing.video_processor"]
```

**Build Command**:
```bash
docker build -t railsafenet-lidar:latest .
```

### Step 3: Tag and Push to Registry

```bash
# Tag for version
docker tag railsafenet-lidar:latest registry.example.com/railsafenet-lidar:v1.0.0

# Push to registry
docker push registry.example.com/railsafenet-lidar:v1.0.0
```

---

## Deployment on Nvidia AGX Orin

### Prerequisites

1. **JetPack SDK Installed**:
   ```bash
   # Verify JetPack version
   sudo apt-cache show nvidia-jetpack
   ```

2. **Docker Runtime with NVIDIA Support**:
   ```bash
   # Install nvidia-container-runtime
   sudo apt-get install nvidia-container-runtime

   # Verify NVIDIA Docker runtime
   docker run --rm --runtime nvidia nvcr.io/nvidia/l4t-base:r35.1.0 nvidia-smi
   ```

3. **Sufficient Storage**:
   - Minimum 8GB free space for Docker image
   - Minimum 4GB free space for logs and temporary files

### Deployment Steps

#### 1. Pull Docker Image

```bash
docker pull registry.example.com/railsafenet-lidar:v1.0.0
```

#### 2. Create Configuration

Create `/opt/railsafenet/config/production.yaml`:
```yaml
segmentation_engine_path: /app/assets/models_pretrained/segformer/segformer_b3_orin.engine
detection_engine_path: /app/assets/models_pretrained/yolo/yolov8s_orin.engine

enable_tracking: true
enable_vp_filtering: true
roi_height_fraction: 0.5

tracking_process_noise: 0.1
tracking_measurement_noise: 1.0

vp_cache_frames: 10
vp_angle_threshold: 10.0

track_width_mm: 1435
danger_distances_mm: [100, 400, 1000]

target_fps: 25.0
max_variance_percent: 20.0

enable_visualization: false  # Disable for production
save_visualizations: false
```

#### 3. Run Container

```bash
docker run -d \
  --name railsafenet-lidar \
  --runtime nvidia \
  --restart unless-stopped \
  -v /opt/railsafenet/config:/app/config:ro \
  -v /opt/railsafenet/logs:/app/logs:rw \
  -v /opt/railsafenet/data:/app/data:ro \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  --ipc=host \
  --network host \
  registry.example.com/railsafenet-lidar:v1.0.0
```

#### 4. Monitor Container

```bash
# Check container status
docker ps -a | grep railsafenet

# View logs
docker logs -f railsafenet-lidar

# Check resource usage
docker stats railsafenet-lidar

# Monitor GPU utilization
sudo tegrastats
```

---

## Performance Tuning for AGX Orin

### Power Mode Configuration

AGX Orin supports multiple power modes. For optimal performance:

```bash
# Set to maximum performance mode (60W)
sudo nvpmodel -m 0

# Maximize GPU/CPU clocks
sudo jetson_clocks

# Verify current mode
sudo nvpmodel -q
```

**Recommended Modes**:
- **Development/Testing**: Mode 0 (MAXN - 60W)
- **Production**: Mode 2 (30W) or Mode 3 (15W) depending on requirements

### Memory Optimization

```bash
# Increase swap space if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### TensorRT Optimization

For best inference performance on AGX Orin:
- Use **FP16 precision** (INT8 if accuracy permits)
- Enable **DLA (Deep Learning Accelerator)** if supported
- Optimize batch size (typically 1 for real-time)

---

## Monitoring and Health Checks

### System Metrics

Monitor these metrics for system health:

```bash
# GPU utilization
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu \
           --format=csv -l 1

# Jetson stats (AGX Orin specific)
sudo tegrastats

# Container resource usage
docker stats railsafenet-lidar --no-stream
```

### Application Metrics

The application logs performance metrics to `/app/logs/railsafenet.log`:
- Frame processing time
- FPS (frames per second)
- Segmentation/detection inference times
- GPU memory usage

Expected values:
- FPS: ≥ 25 (target: 25-30)
- Total frame time: < 40ms
- GPU utilization: 60-80%
- GPU memory: < 4GB

---

## Troubleshooting

### Common Issues

#### 1. Low FPS / High Latency

**Symptoms**: FPS < 20, frame time > 50ms

**Solutions**:
- Check power mode: `sudo nvpmodel -q`
- Maximize clocks: `sudo jetson_clocks`
- Verify TensorRT engines are built for correct architecture
- Reduce input resolution if acceptable
- Disable visualization in production

#### 2. GPU Out of Memory

**Symptoms**: CUDA out of memory errors

**Solutions**:
- Reduce batch size to 1
- Use FP16 precision instead of FP32
- Close other GPU applications
- Increase swap space

#### 3. Container Fails to Start

**Symptoms**: Container exits immediately

**Solutions**:
- Check docker logs: `docker logs railsafenet-lidar`
- Verify NVIDIA runtime: `docker run --rm --runtime nvidia ubuntu nvidia-smi`
- Check file permissions for mounted volumes
- Verify TensorRT engine files exist and are readable

#### 4. Inconsistent Performance

**Symptoms**: FPS varies significantly (>20% variance)

**Solutions**:
- Enable power governor: `sudo nvpmodel -m 0`
- Disable DVFS: `sudo jetson_clocks`
- Check for thermal throttling: `sudo tegrastats`
- Verify no background processes consuming GPU

---

## Update and Maintenance

### Rolling Update

To update the application without downtime:

```bash
# Pull new image
docker pull registry.example.com/railsafenet-lidar:v1.1.0

# Stop old container
docker stop railsafenet-lidar

# Remove old container
docker rm railsafenet-lidar

# Start new container with updated image
docker run -d \
  --name railsafenet-lidar \
  --runtime nvidia \
  --restart unless-stopped \
  -v /opt/railsafenet/config:/app/config:ro \
  -v /opt/railsafenet/logs:/app/logs:rw \
  -v /opt/railsafenet/data:/app/data:ro \
  -e NVIDIA_VISIBLE_DEVICES=all \
  registry.example.com/railsafenet-lidar:v1.1.0
```

### Backup and Recovery

**Configuration Backup**:
```bash
tar -czf railsafenet-config-$(date +%Y%m%d).tar.gz /opt/railsafenet/config
```

**Log Rotation**:
Configure logrotate for `/opt/railsafenet/logs/`:
```bash
/opt/railsafenet/logs/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
```

---

## Security Considerations

1. **Container Isolation**: Use read-only volumes where possible
2. **Network Security**: Use `--network host` only if necessary; prefer bridge networking
3. **Secrets Management**: Store sensitive data in Docker secrets or environment files
4. **Updates**: Regularly update base images for security patches
5. **Access Control**: Restrict SSH access to AGX Orin platform

---

## Performance Benchmarks

### Expected Performance on AGX Orin

| Metric | Target | Typical | Notes |
|--------|--------|---------|-------|
| **FPS** | ≥ 25 | 27-30 | With MAXN power mode |
| **Frame Time** | < 40ms | 33-37ms | Total pipeline |
| **Segmentation** | < 40ms | 28-35ms | SegFormer B3 |
| **Detection** | < 25ms | 18-22ms | YOLO v8s |
| **GPU Utilization** | 60-80% | 70-75% | Optimal range |
| **GPU Memory** | < 4GB | 2-3GB | Peak usage |
| **Power Consumption** | < 40W | 35-38W | Mode 0 (MAXN) |

### Comparison: Titan RTX vs AGX Orin

| Platform | FPS | Power | Notes |
|----------|-----|-------|-------|
| **Nvidia Titan RTX** | 35-40 | 250W | Development |
| **Nvidia AGX Orin** | 25-30 | 35-40W | Production (target) |

---

## Next Steps

This deployment guide will be fully implemented in **Phase 7: Polish & Cross-Cutting Concerns**.

Tasks include:
- Create production Dockerfile (Task T061)
- Build and test Docker image on AGX Orin
- Create deployment scripts and automation
- Document CI/CD pipeline for automated deployment
- Performance profiling and optimization for AGX Orin

---

## References

- [NVIDIA Jetson AGX Orin Documentation](https://developer.nvidia.com/embedded/jetson-agx-orin)
- [JetPack SDK](https://developer.nvidia.com/embedded/jetpack)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [L4T (Linux for Tegra) Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-tensorrt)
