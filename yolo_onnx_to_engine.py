#!/usr/bin/env python3
"""
Convert YOLOv8 ONNX model to a TensorRT engine.
### Titan RTX 24GB 환경에 최적화된 버전 ###
"""

import tensorrt as trt
import os

def create_tensorrt_engine(onnx_path, engine_path, model_name):
    """
    Convert ONNX model to a platform-specific TensorRT engine for YOLOv8.
    """
    print(f"🚀 Converting {model_name} to TensorRT Engine")
    print(f"📁 ONNX Input: {onnx_path}")
    print(f"📁 Engine Output: {engine_path}")

    if not os.path.exists(onnx_path):
        print(f"❌ ONNX model not found: {onnx_path}")
        return False

    try:
        # --- 1. 로거 및 빌더 설정 ---
        # INFO 레벨로 로깅을 높여 더 자세한 정보 확인
        logger = trt.Logger(trt.Logger.INFO)
        print(f"✅ TensorRT version: {trt.__version__}")
        builder = trt.Builder(logger)
        print("✅ Builder created.")

        # --- 2. 네트워크 정의 및 ONNX 파싱 ---
        # EXPLICIT_BATCH 플래그는 ONNX 모델의 배치 차원을 그대로 사용하도록 설정
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        print("✅ Network and Parser created.")

        print("📦 Parsing ONNX model...")
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                print("❌ Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return False
        
        print("✅ ONNX model parsed successfully.")
        print(f"  - Network inputs: {network.num_inputs}")
        print(f"  - Network outputs: {network.num_outputs}")

        # --- 3. 빌더 환경설정 (핵심 수정) ---
        print("🔧 Configuring TensorRT builder for Titan RTX 24GB...")
        config = builder.create_builder_config()

        # 🚀 Titan RTX 24GB GPU에 맞춰 작업 공간을 넉넉하게 8GB로 설정
        max_workspace_gb = 8
        print(f"🛠️ Setting MemoryPool limit for WORKSPACE to {max_workspace_gb}GB")
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_gb * (1 << 30))
        
        # Titan RTX는 FP16 연산을 위한 Tensor Core를 지원하므로 FP16 모드 활성화
        if builder.platform_has_fast_fp16:
            print("🛠️ Enabling FP16 mode for maximum performance.")
            config.set_flag(trt.BuilderFlag.FP16)
        
        # 사용 가능한 모든 Tactic Source를 활성화하여 최적화 옵션 확장
        config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS) | 1 << int(trt.TacticSource.CUBLAS_LT) | 1 << int(trt.TacticSource.CUDNN))
        print("🛠️ Enabled all available tactic sources (CUBLAS, CUBLAS_LT, CUDNN).")
        
        print("✅ Builder configured.")

        # --- 4. 엔진 빌드 ---
        print("🧠 Building TensorRT engine... (This can take several minutes)")
        plan = builder.build_serialized_network(network, config)

        if plan is None:
            print("❌ Failed to build TensorRT engine")
            return False

        print("✅ Engine built successfully.")

        # --- 5. 엔진 저장 ---
        print("💾 Saving TensorRT engine...")
        with open(engine_path, 'wb') as f:
            f.write(plan)

        engine_size_mb = os.path.getsize(engine_path) / (1024*1024)
        onnx_size_mb = os.path.getsize(onnx_path) / (1024*1024)

        print(f"✅ TensorRT engine created successfully!")
        print(f"📊 ONNX size: {onnx_size_mb:.1f}MB → Engine size: {engine_size_mb:.1f}MB")
        print(f"🚀 Ready for deployment!")
        return True

    except Exception as e:
        print(f"❌ Engine creation failed for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("🚀 YOLOv8 ONNX to TensorRT Engine Conversion Pipeline (Titan RTX Profile)")
    print("=" * 70)
    
    # --- Configuration ---
    model_dir = "/home/mmc-server4/RailSafeNet/assets/models_pretrained/yolo"
    onnx_filename = "yolov8s_896x512.onnx"
    engine_filename = "yolov8s_896x512.engine"
    # --------------------

    onnx_path = os.path.join(model_dir, onnx_filename)
    engine_path = os.path.join(model_dir, engine_filename)

    create_tensorrt_engine(onnx_path, engine_path, onnx_filename)


if __name__ == '__main__':
    main()