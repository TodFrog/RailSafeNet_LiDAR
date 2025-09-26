#!/usr/bin/env python3
"""
Convert YOLOv8 ONNX model to a TensorRT engine.
### 중요 ###
이 스크립트는 최종 배포 대상인 AGX Orin 또는 대상 GPU가 장착된 장비에서 직접 실행해야 합니다.
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
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
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

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB, YOLO can be memory intensive

        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✅ FP16 precision enabled")
        
        # For YOLOv8, batch size is fixed during ONNX export, so no optimization profile is needed.
        # If you exported with dynamic=True, you would need to add an optimization profile here.
        print("ℹ️  Using fixed batch size defined in ONNX model (batch=1).")

        print("🔨 Building TensorRT engine... (This may take several minutes)")
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

def main():
    print("🚀 YOLOv8 ONNX to TensorRT Engine Conversion Pipeline")
    print("=" * 60)
    
    # --- Configuration ---
    model_dir = "/home/mmc-server4/RailSafeNet"
    onnx_filename = "yolov8n_896x512.onnx"
    engine_filename = "yolov8n_896x512.engine"
    # ---------------------

    onnx_path = os.path.join(model_dir, onnx_filename)
    engine_path = os.path.join(model_dir, engine_filename)

    try:
        import tensorrt as trt
        print(f"✅ TensorRT version: {trt.__version__}")
    except ImportError:
        print("❌ TensorRT not available. Please install for engine conversion.")
        return

    create_tensorrt_engine(onnx_path, engine_path, "YOLOv8n")

if __name__ == "__main__":
    main()