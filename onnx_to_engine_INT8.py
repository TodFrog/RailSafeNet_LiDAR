#!/usr/bin/env python3
"""
Convert ONNX models to TensorRT INT8 engines with calibration.
### Titan RTX 24GB 환경에 최적화된 INT8 Quantization ###

INT8 Quantization Benefits:
- FP16 대비 30-50% 추가 성능 향상
- 메모리 사용량 감소
- Accuracy 손실: 보통 1-3% (acceptable for most cases)

Memory-Efficient Design:
- Batch size 1 for calibration (minimal VRAM usage)
- Small calibration dataset (100-200 images)
- Stream processing (no full dataset loading)

Usage:
    1. 먼저 calibration 데이터셋 준비 (자동으로 video에서 추출)
    2. python3 onnx_to_engine_INT8.py
    3. INT8 engine 생성 완료!
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os
import numpy as np
import cv2
import glob
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ============================================================================
# Configuration
# ============================================================================
CALIBRATION_DATASET_SIZE = 100  # Calibration에 사용할 이미지 수 (100-200 추천)
CALIBRATION_BATCH_SIZE = 1      # Batch size (Titan RTX에서는 1 추천)

# Video directory for calibration data extraction
VIDEO_DIR = "/home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets/crop"
CALIBRATION_CACHE_DIR = "/home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets/calibration_cache"

# ============================================================================
# INT8 Calibrator Implementation
# ============================================================================
class ImageBatcher:
    """
    메모리 효율적인 이미지 배치 생성기
    """
    def __init__(self, calibration_data, batch_size, shape, preprocessor=None):
        """
        Args:
            calibration_data: List of image file paths
            batch_size: Batch size for calibration
            shape: Input shape (C, H, W)
            preprocessor: Preprocessing function
        """
        self.images = calibration_data
        self.batch_size = batch_size
        self.shape = shape
        self.preprocessor = preprocessor
        self.current_index = 0

    def reset(self):
        """Reset to beginning of dataset"""
        self.current_index = 0

    def next_batch(self):
        """Get next batch of preprocessed images"""
        if self.current_index >= len(self.images):
            return None

        batch = []
        for _ in range(self.batch_size):
            if self.current_index >= len(self.images):
                break

            image_path = self.images[self.current_index]
            self.current_index += 1

            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠ Failed to load: {image_path}")
                continue

            if self.preprocessor is not None:
                image = self.preprocessor(image)

            batch.append(image)

        if len(batch) == 0:
            return None

        # Stack batch
        batch_array = np.stack(batch, axis=0)
        return batch_array

    def get_batch_size(self):
        """Return batch size"""
        return self.batch_size


class Int8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 Entropy Calibrator v2
    메모리 효율적으로 calibration 수행
    """
    def __init__(self, image_batcher, cache_file):
        """
        Args:
            image_batcher: ImageBatcher instance
            cache_file: Path to calibration cache file
        """
        super().__init__()
        self.image_batcher = image_batcher
        self.cache_file = cache_file

        # Allocate device memory for batch
        self.device_input = None

    def get_batch_size(self):
        """Return batch size"""
        return self.image_batcher.get_batch_size()

    def get_batch(self, names):
        """
        Get next batch for calibration

        Returns:
            List of device memory pointers
        """
        try:
            batch = self.image_batcher.next_batch()
            if batch is None:
                return None

            # Allocate device memory if not done
            if self.device_input is None:
                self.device_input = cuda.mem_alloc(batch.nbytes)

            # Copy batch to device
            cuda.memcpy_htod(self.device_input, np.ascontiguousarray(batch))

            return [int(self.device_input)]

        except Exception as e:
            print(f"❌ Error in get_batch: {e}")
            return None

    def read_calibration_cache(self):
        """
        Read calibration cache if exists
        """
        if os.path.exists(self.cache_file):
            print(f"✓ Reading calibration cache from: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """
        Write calibration cache to file
        """
        print(f"✓ Writing calibration cache to: {self.cache_file}")
        with open(self.cache_file, 'wb') as f:
            f.write(cache)


# ============================================================================
# Calibration Dataset Preparation
# ============================================================================
def extract_calibration_frames(video_dir, output_dir, num_frames=100):
    """
    비디오 파일들에서 calibration 프레임 추출

    Args:
        video_dir: 비디오 디렉토리
        output_dir: 출력 디렉토리
        num_frames: 추출할 프레임 수

    Returns:
        List of extracted frame paths
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check if already extracted
    existing_frames = glob.glob(os.path.join(output_dir, "*.jpg"))
    if len(existing_frames) >= num_frames:
        print(f"✓ Calibration frames already exist: {len(existing_frames)} frames")
        return sorted(existing_frames)[:num_frames]

    print(f"📹 Extracting calibration frames from videos...")
    print(f"   Target: {num_frames} frames")

    # Get video files
    video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if len(video_files) == 0:
        raise RuntimeError(f"No video files found in {video_dir}")

    print(f"   Found {len(video_files)} video files")

    # Calculate frames per video
    frames_per_video = max(1, num_frames // len(video_files))

    extracted = []
    for video_idx, video_path in enumerate(video_files):
        if len(extracted) >= num_frames:
            break

        print(f"   Processing: {os.path.basename(video_path)}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"   ⚠ Failed to open: {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)

        for frame_idx in frame_indices:
            if len(extracted) >= num_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            # Save frame
            output_path = os.path.join(output_dir, f"calib_{len(extracted):04d}.jpg")
            cv2.imwrite(output_path, frame)
            extracted.append(output_path)

        cap.release()

    print(f"✅ Extracted {len(extracted)} calibration frames")
    return extracted


def create_segformer_preprocessor(height=512, width=896):
    """
    SegFormer 전처리 함수 생성
    """
    transform = A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def preprocessor(image):
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Apply transforms
        transformed = transform(image=image)['image']
        # HWC to CHW
        transformed = transformed.transpose(2, 0, 1)
        return transformed.astype(np.float32)

    return preprocessor


def create_yolo_preprocessor(height=512, width=896):
    """
    YOLO 전처리 함수 생성
    """
    def preprocessor(image):
        # Resize
        image = cv2.resize(image, (width, height))
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        # HWC to CHW
        image = image.transpose(2, 0, 1)
        return image

    return preprocessor


# ============================================================================
# TensorRT Engine Creation with INT8
# ============================================================================
def create_tensorrt_engine_int8(onnx_path, engine_path, model_name, input_shape,
                                 calibration_data, preprocessor):
    """
    ONNX 모델을 INT8 TensorRT 엔진으로 변환

    Args:
        onnx_path: ONNX 모델 경로
        engine_path: 출력 엔진 경로
        model_name: 모델 이름
        input_shape: 입력 shape (B, C, H, W)
        calibration_data: Calibration 이미지 경로 리스트
        preprocessor: 전처리 함수
    """
    print("-" * 70)
    print(f"🚀 Converting {model_name} to INT8 TensorRT Engine")
    print(f"📁 ONNX Input: {onnx_path}")
    print(f"📁 Engine Output: {engine_path}")
    print(f"📊 Calibration dataset: {len(calibration_data)} images")

    if not os.path.exists(onnx_path):
        print(f"❌ ONNX model not found: {onnx_path}")
        return False

    try:
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        config = builder.create_builder_config()

        # Parse ONNX
        print("📦 Parsing ONNX model...")
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                print(f"❌ Failed to parse ONNX model for {model_name}")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return False

        print("✅ ONNX model parsed successfully.")

        # Optimization profile
        print("🔧 Configuring optimization profile...")
        profile = builder.create_optimization_profile()
        input_tensor_name = network.get_input(0).name
        profile.set_shape(input_tensor_name, min=input_shape, opt=input_shape, max=input_shape)
        config.add_optimization_profile(profile)

        # Workspace
        max_workspace_gb = 6  # INT8 uses less memory than FP16
        print(f"🛠️ Setting workspace to {max_workspace_gb}GB")
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_gb * (1 << 30))

        # INT8 Configuration
        if builder.platform_has_fast_int8:
            print("🛠️ Enabling INT8 mode")
            config.set_flag(trt.BuilderFlag.INT8)

            # Create calibrator
            cache_file = engine_path.replace('.engine', '_calibration.cache')

            # Create image batcher
            batch_size = CALIBRATION_BATCH_SIZE
            shape = input_shape[1:]  # (C, H, W)
            batcher = ImageBatcher(calibration_data, batch_size, shape, preprocessor)

            # Create calibrator
            calibrator = Int8EntropyCalibrator(batcher, cache_file)
            config.int8_calibrator = calibrator

            print(f"✅ INT8 calibrator configured")
            print(f"   - Batch size: {batch_size}")
            print(f"   - Calibration cache: {cache_file}")
        else:
            print("⚠️ INT8 not supported on this platform, falling back to FP16")
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

        # Build engine
        print("🧠 Building INT8 TensorRT engine... (This may take 5-15 minutes)")
        print("   ⏳ Calibration in progress - please wait...")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            print(f"❌ Failed to build TensorRT engine for {model_name}")
            return False

        print("✅ Engine built successfully.")

        # Save engine
        print(f"💾 Saving INT8 engine to {engine_path}...")
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)

        engine_size_mb = os.path.getsize(engine_path) / (1024*1024)
        onnx_size_mb = os.path.getsize(onnx_path) / (1024*1024)

        print(f"✅ INT8 TensorRT engine for '{model_name}' created successfully!")
        print(f"📊 ONNX size: {onnx_size_mb:.1f}MB → INT8 Engine size: {engine_size_mb:.1f}MB")
        print(f"📊 Size reduction: {(1 - engine_size_mb/onnx_size_mb)*100:.1f}%")
        return True

    except Exception as e:
        print(f"❌ Engine creation failed for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Main Function
# ============================================================================
def main():
    print("🚀 ONNX to INT8 TensorRT Engine Conversion Pipeline")
    print("=" * 70)
    print(f"Target: Titan RTX 24GB")
    print(f"Calibration dataset size: {CALIBRATION_DATASET_SIZE} images")
    print(f"Calibration batch size: {CALIBRATION_BATCH_SIZE}")
    print("=" * 70)

    # Step 1: Extract calibration frames
    print("\n📹 Step 1: Preparing calibration dataset...")
    calibration_frames = extract_calibration_frames(
        VIDEO_DIR,
        CALIBRATION_CACHE_DIR,
        CALIBRATION_DATASET_SIZE
    )

    if len(calibration_frames) < CALIBRATION_DATASET_SIZE:
        print(f"⚠️ Warning: Only {len(calibration_frames)} frames available (requested {CALIBRATION_DATASET_SIZE})")

    # Step 2: Convert models
    print("\n🔧 Step 2: Converting models to INT8...")

    models_to_convert = [
        {
            'name': 'SegFormer B3 (896x512) INT8',
            'onnx_path': '/home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets/models_pretrained/segformer/optimized/segformer_b3_transfer_best_0.7961_896x512.onnx',
            'engine_path': '/home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets/models_pretrained/segformer/optimized/segformer_b3_transfer_best_0.7961_896x512_int8.engine',
            'input_shape': (1, 3, 512, 896),
            'preprocessor': create_segformer_preprocessor(512, 896)
        },
        {
            'name': 'YOLOv8s (896x512) INT8',
            'onnx_path': '/home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets/models_pretrained/yolo/yolov8s_896x512.onnx',
            'engine_path': '/home/mmc-server4/Server/Users/minkyu/RailSafeNet_LiDAR/assets/models_pretrained/yolo/yolov8s_896x512_int8.engine',
            'input_shape': (1, 3, 512, 896),
            'preprocessor': create_yolo_preprocessor(512, 896)
        }
    ]

    results = []
    print(f"✅ TensorRT version: {trt.__version__}")

    for model_info in models_to_convert:
        print(f"\n{'='*70}")
        success = create_tensorrt_engine_int8(
            model_info['onnx_path'],
            model_info['engine_path'],
            model_info['name'],
            model_info['input_shape'],
            calibration_frames,
            model_info['preprocessor']
        )
        results.append((model_info['name'], success))

    # Final summary
    print("\n" + "=" * 70)
    print("📋 INT8 Engine Conversion Summary:")
    print("-" * 70)
    for name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status}: {name}")
    print("=" * 70)

    # Next steps
    print("\n🎯 Next Steps:")
    print("1. Test INT8 engines with videoAssessor_phase3_cache.py")
    print("2. Compare FPS: FP16 vs INT8")
    print("3. Validate accuracy (should be within 1-3% of FP16)")
    print("\nTo use INT8 engines, update engine paths in your code:")
    for model_info in models_to_convert:
        print(f"  - {model_info['engine_path']}")


if __name__ == '__main__':
    main()
