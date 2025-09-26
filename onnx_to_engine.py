#!/usr/bin/env python3
"""
Convert ONNX models to TensorRT engines optimized for the target NVIDIA platform.
### 중요 ###
이 스크립트는 최종 배포 대상인 AGX Orin 또는 대상 GPU가 장착된 장비에서 직접 실행해야 합니다.
TensorRT 엔진은 실행 환경의 GPU 아키텍처에 맞춰 생성되므로 교차 컴파일이 불가능합니다.
"""

import tensorrt as trt
import os

def create_tensorrt_engine(onnx_path, engine_path, model_name):
    """
    Convert ONNX model to a platform-specific TensorRT engine.
    """
    print(f"🚀 Converting {model_name} to TensorRT Engine")
    print(f"📁 ONNX Input: {onnx_path}")
    print(f"📁 Engine Output: {engine_path}")

    if not os.path.exists(onnx_path):
        print(f"❌ ONNX model not found: {onnx_path}")
        return False

    try:
        print("🔧 Setting up TensorRT builder...")
        logger = trt.Logger(trt.Logger.WARNING) # INFO는 너무 장황하므로 WARNING으로 변경
        builder = trt.Builder(logger)

        print("📦 Creating network definition...")
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        print("📦 Parsing ONNX model...")
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                print("❌ Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    print(f"   Error {error}: {parser.get_error(error).desc()}")
                return False
        print("✅ ONNX model parsed successfully")

        print("⚙️  Configuring TensorRT builder...")
        config = builder.create_builder_config()

        # Set memory pool limit (Workspace) - 필요 시 4GB 등으로 늘릴 수 있음
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB

        # Enable FP16 precision for better performance on modern NVIDIA GPUs
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✅ FP16 precision enabled")

        # === 동적 입력 크기 처리를 위한 Optimization Profile 설정 ===
        profile = builder.create_optimization_profile()
        # 모델의 입력 이름 ('pixel_values')과 동일해야 함
        input_tensor = network.get_input(0) 
        input_name = input_tensor.name
        
        # 최소(min), 최적(opt), 최대(max) 입력 shape을 지정
        # 예: 배치 크기 1~4, 고정된 3x1024x1024 이미지
        profile.set_shape(input_name, min=(1, 3, 512, 896), opt=(1, 3, 512, 896), max=(1, 3, 512, 896))
        config.add_optimization_profile(profile)
        print(f"✅ Optimization profile created for dynamic shapes (min/opt/max batch size: 1/1/4)")
        # =========================================================

        print("🔨 Building TensorRT engine...")
        print("   This process may take several minutes...")
        
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            print("❌ Failed to build TensorRT engine")
            return False

        print("💾 Saving TensorRT engine...")
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)

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

# main 함수는 기존과 동일하게 유지...
def main():
    """Convert ONNX models to TensorRT engines"""

    print("🚀 ONNX to TensorRT Engine Conversion Pipeline")
    print("=" * 60)
    
    # ... (기존 main 함수 내용) ...
    # Define ONNX models to convert
    onnx_models = [
        {
            'name': 'Original SegFormer B3 (13 classes)',
            'onnx_path': '/home/mmc-server4/RailSafeNet/assets/models_pretrained/segformer/optimized/segformer_b3_original_13class_896x512.onnx',
            'engine_path': '/home/mmc-server4/RailSafeNet/assets/models_pretrained/segformer/optimized/segformer_b3_original_13class.engine'
        },
        {
            'name': 'Transfer Learning Best Model (Rail IoU: 0.7961)',
            'onnx_path': '/home/mmc-server4/RailSafeNet/assets/models_pretrained/segformer/optimized/segformer_b3_transfer_best_0.7961_896x512.onnx',
            'engine_path': '/home/mmc-server4/RailSafeNet/assets/models_pretrained/segformer/optimized/segformer_b3_transfer_best_0.7961.engine'
        }
    ]

    results = []

    # Check if TensorRT is available
    try:
        import tensorrt as trt
        print(f"✅ TensorRT version: {trt.__version__}")
    except ImportError:
        print("❌ TensorRT not available. Please install TensorRT for engine conversion.")
        return

    for model_info in onnx_models:
        success = create_tensorrt_engine(
            model_info['onnx_path'],
            model_info['engine_path'],
            model_info['name']
        )
        results.append((model_info['name'], success))

    # Summary
    print("📋 Engine Conversion Summary:")
    print("=" * 50)
    for name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status}: {name}")

    successful_conversions = sum(1 for _, success in results if success)
    print(f"\n🎯 {successful_conversions}/{len(results)} engines created successfully!")


if __name__ == "__main__":
    main()