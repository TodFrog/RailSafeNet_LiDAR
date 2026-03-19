#!/usr/bin/env python3
"""ONNX Runtime 기반 SegFormer 프로덕션 보조 추론 모듈.

공식 재현 경로는 PyTorch 버전이지만, ONNX 배포 경로를 검토하거나 경량 추론을 시험할 때
참고할 수 있도록 유지하는 보조 구현이다.
"""

import numpy as np
import cv2
import time
import torch
import torch.nn.functional as F

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("⚠️  ONNX Runtime not available. Install with: pip install onnxruntime-gpu")
    ONNX_AVAILABLE = False

class SegFormerONNXProduction:
    """SegFormer ONNX 모델의 전처리-추론-후처리를 묶는 래퍼."""

    def __init__(self, model_path="/home/mmc-server4/RailSafeNet/models/segformer_b3_transfer_enhanced_0.7791.onnx"):
        """모델 경로를 받아 ONNX Runtime 세션을 준비한다."""
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None

        if ONNX_AVAILABLE:
            self._load_onnx_model()
        else:
            raise ImportError("ONNX Runtime is required for production inference")

    def _load_onnx_model(self):
        """최적화 옵션을 켠 ONNX Runtime 세션을 생성한다."""
        print(f"🚀 Loading ONNX model: {self.model_path}")

        # CUDA provider를 우선 시도하되, 환경에 따라 CPU fallback을 허용한다.
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Create session
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"✅ ONNX model loaded successfully")
        print(f"   Input: {self.input_name}")
        print(f"   Output: {self.output_name}")
        print(f"   Provider: {self.session.get_providers()}")

    def preprocess(self, image):
        """입력 이미지를 SegFormer ONNX 입력 텐서로 변환한다.

        현재 구현은 1024x1024 고정 입력과 ImageNet 정규화를 가정한다.
        export 모델의 입력 규격이 다르면 함께 수정해야 한다.
        """
        # Resize to model input size
        image_resized = cv2.resize(image, (1024, 1024))

        # Normalize for SegFormer (ImageNet normalization)
        image_norm = image_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        # Apply normalization
        image_norm = (image_norm - mean) / std

        # Convert to NCHW format (batch, channels, height, width)
        image_tensor = np.transpose(image_norm, (2, 0, 1))
        image_batch = np.expand_dims(image_tensor, axis=0)

        return image_batch.astype(np.float32)

    def inference(self, input_tensor):
        """전처리된 입력 텐서에 대해 ONNX 추론을 수행한다."""
        start_time = time.time()

        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )

        inference_time = time.time() - start_time
        print(f"⚡ ONNX inference time: {inference_time:.3f}s")

        return outputs[0]  # Return logits

    def postprocess(self, output_tensor, target_size=(1080, 1920)):
        """출력 logits를 원본 해상도 기준 세그멘테이션 마스크로 복원한다."""
        # Convert to torch tensor for interpolation
        logits = torch.from_numpy(output_tensor)

        # Resize to target size using bilinear interpolation
        logits_resized = F.interpolate(
            logits,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        # Get class predictions
        predictions = torch.argmax(logits_resized, dim=1)

        return predictions.squeeze().cpu().numpy()

    def predict(self, image):
        """단일 이미지 기준 전체 전처리-추론-후처리 흐름을 실행한다."""
        # Preprocess
        input_tensor = self.preprocess(image)

        # Inference
        output_tensor = self.inference(input_tensor)

        # Postprocess
        prediction = self.postprocess(output_tensor, target_size=(image.shape[0], image.shape[1]))

        return prediction

    def predict_batch(self, images):
        """여러 이미지를 배치로 묶어 예측한다."""
        batch_inputs = []
        original_sizes = []

        for image in images:
            original_sizes.append((image.shape[0], image.shape[1]))
            batch_inputs.append(self.preprocess(image))

        # Stack batch
        batch_tensor = np.vstack(batch_inputs)

        # Inference
        start_time = time.time()
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: batch_tensor}
        )
        inference_time = time.time() - start_time
        print(f"⚡ Batch ONNX inference time: {inference_time:.3f}s for {len(images)} images")

        # Postprocess each image
        predictions = []
        for i, (output, original_size) in enumerate(zip(outputs[0], original_sizes)):
            logits = torch.from_numpy(output).unsqueeze(0)
            logits_resized = F.interpolate(
                logits,
                size=original_size,
                mode='bilinear',
                align_corners=False
            )
            prediction = torch.argmax(logits_resized, dim=1).squeeze().cpu().numpy()
            predictions.append(prediction)

        return predictions

# Factory function for easy loading
def load_production_segformer(model_path="/home/mmc-server4/RailSafeNet/models/segformer_b3_transfer_enhanced_0.7791.onnx"):
    """배포용 SegFormer ONNX 래퍼를 생성한다."""
    return SegFormerONNXProduction(model_path)

# Performance benchmarking
def benchmark_model(model, num_iterations=50):
    """임의 입력을 반복 추론해 참고용 성능 분포를 측정한다."""
    print(f"\n🔬 Benchmarking ONNX model ({num_iterations} iterations)")

    # Create test data
    test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    times = []
    for i in range(num_iterations):
        start_time = time.time()
        prediction = model.predict(test_image)
        end_time = time.time()
        times.append(end_time - start_time)

        if i == 0:
            print(f"✅ First prediction shape: {prediction.shape}")
            print(f"✅ Unique classes: {np.unique(prediction)}")

    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1.0 / avg_time

    print(f"\n📊 Performance Results:")
    print(f"   Average time: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"   Min time: {min_time:.3f}s")
    print(f"   Max time: {max_time:.3f}s")
    print(f"   FPS: {fps:.1f}")

    return avg_time, fps

if __name__ == "__main__":
    print("🚀 RailSafeNet Production SegFormer ONNX Model")
    print("=" * 50)

    try:
        # Load model
        model = load_production_segformer()

        # Test single prediction
        test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        prediction = model.predict(test_image)

        print(f"✅ Single prediction completed")
        print(f"   Input shape: {test_image.shape}")
        print(f"   Output shape: {prediction.shape}")
        print(f"   Classes detected: {np.unique(prediction)}")

        # Benchmark performance
        avg_time, fps = benchmark_model(model)

        print(f"\n🎯 Model ready for production deployment!")
        print(f"   Expected FPS: {fps:.1f}")
        print(f"   Suitable for real-time inference: {'✅' if fps >= 10 else '❌'}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
