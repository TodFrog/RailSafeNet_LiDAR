#!/usr/bin/env python3
"""
Convert transfer learning model to optimized production format (PyTorch → ONNX → TensorRT)

This script takes a transfer learning model and converts it through the optimization pipeline:
1. PyTorch .pth → ONNX format
2. ONNX → TensorRT engine (optimized)
3. Production format compatible with TheDistanceAssessor.py
"""

import torch
import os
import numpy as np
from transformers import SegformerForSemanticSegmentation, SegformerConfig

# ONNX and TensorRT imports
try:
    import onnx
    import tensorrt as trt
    TRT_AVAILABLE = True
    print("✅ TensorRT and ONNX available")
except ImportError as e:
    TRT_AVAILABLE = False
    print(f"⚠️ TensorRT/ONNX not available: {e}")

def convert_to_onnx(model, onnx_path, input_shape=(1, 3, 1024, 1024)):
    """Convert PyTorch model to ONNX format"""
    print(f"🔄 Converting model to ONNX: {onnx_path}")

    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape).to(device)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['logits'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        verbose=False
    )

    # Verify ONNX model
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX model verified successfully")
        return onnx_path
    except Exception as e:
        print(f"❌ ONNX verification failed: {e}")
        return None

def convert_onnx_to_tensorrt(onnx_path, trt_path, input_shape=(1, 3, 1024, 1024)):
    """Convert ONNX model to TensorRT engine"""
    if not TRT_AVAILABLE:
        print("❌ TensorRT not available, skipping optimization")
        return None

    print(f"🚀 Converting ONNX to TensorRT: {trt_path}")

    # Create TensorRT builder and network
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX model
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print("❌ Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB

    # Enable FP16 precision if available
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✅ FP16 optimization enabled")

    # Build engine
    print("🔨 Building TensorRT engine... (this may take several minutes)")
    engine = builder.build_engine(network, config)

    if engine is None:
        print("❌ Failed to build TensorRT engine")
        return None

    # Serialize and save engine
    with open(trt_path, 'wb') as f:
        f.write(engine.serialize())

    print(f"✅ TensorRT engine saved successfully")
    return trt_path

def convert_transfer_to_production(transfer_model_path, output_path, enable_optimization=True):
    """
    Convert transfer learning model to production format with optional ONNX/TensorRT optimization

    Args:
        transfer_model_path: Path to the transfer learning .pth file
        output_path: Path where to save the production model
        enable_optimization: Whether to perform ONNX/TensorRT optimization
    """

    print(f"🔄 Converting {transfer_model_path} to production format...")
    if enable_optimization:
        print("🚀 Optimization enabled: PyTorch → ONNX → TensorRT")
    else:
        print("⚡ Standard PyTorch conversion (no optimization)")

    # Use GPU if available for optimization, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() and enable_optimization else 'cpu')
    print(f"📱 Using device: {device}")

    # Load transfer learning model
    print("📦 Loading transfer learning model...")
    checkpoint = torch.load(transfer_model_path, map_location=device, weights_only=False)

    # Create SegFormer model with same config as original
    print("🏗️  Creating SegFormer model structure...")
    config = SegformerConfig.from_pretrained("nvidia/mit-b3")
    config.num_labels = 13  # Match original model

    model = SegformerForSemanticSegmentation(config)
    model.to(device)

    # Load the trained weights
    print("⚡ Loading trained weights...")
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Load state dict with strict=False to handle any missing keys
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"⚠️  Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")

    # Set to evaluation mode
    model.eval()

    # Optional optimization pipeline
    onnx_path = None
    trt_path = None

    if enable_optimization and TRT_AVAILABLE:
        try:
            print("\n🚀 Starting optimization pipeline...")

            # Step 1: Convert to ONNX
            base_name = os.path.splitext(output_path)[0]
            onnx_path = f"{base_name}.onnx"
            onnx_result = convert_to_onnx(model, onnx_path)

            if onnx_result:
                # Step 2: Convert ONNX to TensorRT
                trt_path = f"{base_name}.trt"
                trt_result = convert_onnx_to_tensorrt(onnx_path, trt_path)

                if trt_result:
                    print("✅ Full optimization pipeline completed!")
                else:
                    print("⚠️ TensorRT conversion failed, keeping ONNX version")
            else:
                print("⚠️ ONNX conversion failed, using original PyTorch model")

        except Exception as e:
            print(f"⚠️ Optimization failed: {e}")
            print("📌 Falling back to standard PyTorch model")
            onnx_path = None
            trt_path = None

    # Save in the same format as original model
    print(f"💾 Saving production model to {output_path}...")

    # Create the production state dict in the same format as original
    production_state = {
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'num_classes': 13,
        'model_type': 'segformer_b3_production_optimized' if (onnx_path or trt_path) else 'segformer_b3_production',
        'original_transfer_model': transfer_model_path,
        'conversion_info': {
            'rail_iou_score': extract_score_from_filename(transfer_model_path),
            'conversion_device': str(device),
            'missing_keys': len(missing_keys),
            'unexpected_keys': len(unexpected_keys),
            'optimization_enabled': enable_optimization,
            'onnx_path': onnx_path,
            'tensorrt_path': trt_path,
            'optimization_available': TRT_AVAILABLE
        }
    }

    torch.save(production_state, output_path)
    print(f"✅ Production model saved successfully!")
    print(f"📊 Rail IoU Score: {extract_score_from_filename(transfer_model_path)}")

    return output_path

def extract_score_from_filename(filename):
    """Extract the performance score from filename"""
    import re
    match = re.search(r'_(\d+\.\d+)\.pth$', filename)
    if match:
        return float(match.group(1))
    return None

def main():
    # Define paths
    transfer_model_path = "/home/mmc-server4/RailSafeNet/models/segformer_b3_transfer_best_rail_0.7500.pth"
    output_path = "/home/mmc-server4/RailSafeNet/models/segformer_b3_production_optimized_rail_0.7500.pth"

    print("🚀 SegFormer B3 Model Optimization Pipeline")
    print("=" * 60)
    print(f"📁 Input: {transfer_model_path}")
    print(f"📁 Output: {output_path}")

    # Check if transfer model exists
    if not os.path.exists(transfer_model_path):
        print(f"❌ Transfer model not found: {transfer_model_path}")
        return

    # Convert the model with optimization
    try:
        production_path = convert_transfer_to_production(
            transfer_model_path,
            output_path,
            enable_optimization=True  # Enable ONNX→TensorRT optimization
        )

        print(f"\n🎉 Optimization pipeline completed!")
        print(f"📍 Optimized model: {production_path}")

        # List generated files
        base_name = os.path.splitext(output_path)[0]
        generated_files = [output_path]

        onnx_file = f"{base_name}.onnx"
        if os.path.exists(onnx_file):
            generated_files.append(onnx_file)

        trt_file = f"{base_name}.trt"
        if os.path.exists(trt_file):
            generated_files.append(trt_file)

        print("\n📂 Generated files:")
        for file_path in generated_files:
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024*1024)
                print(f"  • {file_path} ({size_mb:.1f}MB)")

        print(f"\n🔧 Ready for use with TheDistanceAssessor.py")

    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()