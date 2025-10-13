#!/usr/bin/env python3
"""
Convert multiple ONNX models to TensorRT engines.
### Titan RTX 24GB 환경에 최적화된 최종 버전 (Dynamic Shape 지원) ###
"""

import tensorrt as trt
import os

def create_tensorrt_engine(onnx_path, engine_path, model_name, input_shape):
    """
    ONNX 모델을 Titan RTX에 최적화된 TensorRT 엔진으로 변환합니다.
    """
    print("-" * 60)
    print(f"🚀 Converting {model_name} to TensorRT Engine")
    print(f"📁 ONNX Input: {onnx_path}")
    print(f"📁 Engine Output: {engine_path}")

    if not os.path.exists(onnx_path):
        print(f"❌ ONNX model not found: {onnx_path}")
        return False

    try:
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        config = builder.create_builder_config()

        print("📦 Parsing ONNX model...")
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                print(f"❌ Failed to parse ONNX model for {model_name}")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return False
        
        print("✅ ONNX model parsed successfully.")

        # --- 🚀 Dynamic Shape 문제 해결을 위한 핵심 수정 ---
        print("🔧 Configuring TensorRT builder with Optimization Profile...")
        profile = builder.create_optimization_profile()
        
        # 네트워크의 첫 번째 입력 이름을 가져옵니다.
        input_tensor_name = network.get_input(0).name
        
        # 고정된 입력 크기를 사용할 것이므로 min, opt, max를 모두 동일하게 설정합니다.
        print(f"🛠️ Setting Optimization Profile for input '{input_tensor_name}' with shape {input_shape}")
        profile.set_shape(input_tensor_name, min=input_shape, opt=input_shape, max=input_shape)
        
        # 생성된 프로파일을 빌더 설정에 추가합니다.
        config.add_optimization_profile(profile)
        # --- 수정 끝 ---

        # 작업 공간 및 FP16 설정은 그대로 유지합니다.
        max_workspace_gb = 8
        print(f"🛠️ Setting MemoryPool limit for WORKSPACE to {max_workspace_gb}GB")
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_gb * (1 << 30))
        
        if builder.platform_has_fast_fp16:
            print("🛠️ Enabling FP16 mode for maximum performance.")
            config.set_flag(trt.BuilderFlag.FP16)
        
        print("✅ Builder configured.")

        print("🧠 Building TensorRT engine... (This can take several minutes)")
        plan = builder.build_serialized_network(network, config)

        if plan is None:
            print(f"❌ Failed to build TensorRT engine for {model_name}")
            return False

        print("✅ Engine built successfully.")
        
        print(f"💾 Saving TensorRT engine to {engine_path}...")
        with open(engine_path, 'wb') as f:
            f.write(plan)

        engine_size_mb = os.path.getsize(engine_path) / (1024*1024)
        onnx_size_mb = os.path.getsize(onnx_path) / (1024*1024)

        print(f"✅ TensorRT engine for '{model_name}' created successfully!")
        print(f"📊 ONNX size: {onnx_size_mb:.1f}MB → Engine size: {engine_size_mb:.1f}MB")
        return True

    except Exception as e:
        print(f"❌ Engine creation failed for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("🚀 ONNX to TensorRT Engine Conversion Pipeline (Titan RTX Profile)")
    print("=" * 70)

    # --- 변환할 모델 목록 (입력 shape 정보 추가) ---
    models_to_convert = [
        {
            'name': 'SegFormer (896x512)',
            'onnx_path': '/home/mmc-server4/RailSafeNet/assets/models_pretrained/segformer/optimized/segformer_b3_transfer_best_0.7961_896x512.onnx',
            'engine_path': '/home/mmc-server4/RailSafeNet/assets/models_pretrained/segformer/optimized/segformer_b3_transfer_best_0.7961.engine',
            'input_shape': (1, 3, 512, 896) # (batch, channels, height, width)
        },
        # {
        #     'name': 'YOLOv8s (896x512)',
        #     'onnx_path': '/home/mmc-server4/RailSafeNet/assets/models_pretrained/yolo/yolov8s_896x512.onnx',
        #     'engine_path': '/home/mmc-server4/RailSafeNet/assets/models_pretrained/yolo/yolov8s_896x512.engine',
        #     'input_shape': (1, 3, 512, 896)
        # }
    ]

    results = []
    print(f"✅ TensorRT version: {trt.__version__}")

    for model_info in models_to_convert:
        success = create_tensorrt_engine(
            model_info['onnx_path'],
            model_info['engine_path'],
            model_info['name'],
            model_info['input_shape'] # input_shape 인자 전달
        )
        results.append((model_info['name'], success))

    # --- 최종 요약 ---
    print("\n" + "=" * 70)
    print("📋 Engine Conversion Summary:")
    print("-" * 70)
    for name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status}: {name}")
    print("=" * 70)


if __name__ == '__main__':
    main()