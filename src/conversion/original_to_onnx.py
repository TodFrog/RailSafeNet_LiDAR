#!/usr/bin/env python3
"""SegFormer PyTorch 가중치를 ONNX로 내보내는 변환 스크립트."""

import torch
import os
import sys
from transformers import SegformerForSemanticSegmentation, SegformerConfig

def convert_model_to_onnx(model_path, output_path, model_name):
    """PyTorch SegFormer 가중치를 ONNX 형식으로 내보낸다.

    더미 입력 해상도는 현재 배포 경로와 맞춘 512x896을 사용한다.
    이후 TensorRT 변환과 추론 코드가 같은 입력 규격을 가정하므로 함께 관리해야 한다.
    """

    print(f"🔄 Converting {model_name}")
    print(f"📁 Input: {model_path}")
    print(f"📁 Output: {output_path}")

    if not os.path.exists(model_path):
        print(f"❌ Input model not found: {model_path}")
        return False

    try:
        # Load model
        print("📦 Loading model...")

        if "SegFormer_B3_1024_finetuned.pth" in model_path:
            # Original complete model
            model = torch.load(model_path, map_location='cpu', weights_only=False)
            print(f"✅ Original complete model loaded")
            print(f"📊 Model classes: {model.config.num_labels}")
        else:
            # Our trained model - need to reconstruct
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            # Create model structure
            config = SegformerConfig.from_pretrained("nvidia/mit-b3")
            config.num_labels = 13  # Our models use 13 classes
            model = SegformerForSemanticSegmentation(config)

            # Load weights
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict, strict=False)
            print(f"✅ Trained model reconstructed and loaded")
            print(f"📊 Model classes: {config.num_labels}")

        model.eval()

        # 현재 배포 파이프라인의 입력 규격과 맞추기 위해 512x896을 사용한다.
        dummy_input = torch.randn(1, 3, 512, 896)
        print(f"🔧 Created dummy input: {dummy_input.shape}")

        # Test forward pass
        print("🧪 Testing forward pass...")
        with torch.no_grad():
            output = model(dummy_input)
            if hasattr(output, 'logits'):
                print(f"✅ Forward pass successful! Logits shape: {output.logits.shape}")
            else:
                print(f"✅ Forward pass successful! Output type: {type(output)}")

        # Export to ONNX
        print("🚀 Exporting to ONNX...")
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,  # Good compatibility with TensorRT
            do_constant_folding=True,
            input_names=['pixel_values'],
            output_names=['logits'],
            dynamic_axes={
                'pixel_values': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            },
            verbose=False
        )

        print(f"✅ ONNX export successful!")

        # Verify ONNX model
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX model verification successful!")
        except ImportError:
            print("⚠️  ONNX package not available for verification")
        except Exception as e:
            print(f"⚠️  ONNX verification warning: {e}")

        # Check file size
        size_mb = os.path.getsize(output_path) / (1024*1024)
        print(f"📊 ONNX model size: {size_mb:.1f}MB")
        print(f"✅ {model_name} conversion completed!\n")

        return True

    except Exception as e:
        print(f"❌ Conversion failed for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """원본 모델과 전이학습 모델을 차례로 ONNX로 변환한다."""

    print("🚀 Model to ONNX Conversion Pipeline")
    print("=" * 60)

    # Define models to convert
    models_to_convert = [
        {
            'name': 'Original SegFormer B3 (13 classes)',
            'input_path': '/home/mmc-server4/RailSafeNet/assets/models_pretrained/segformer/SegFormer_B3_1024_finetuned.pth',
            'output_path': '/home/mmc-server4/RailSafeNet/assets/models_pretrained/segformer/optimized/segformer_b3_original_13class_896x512.onnx'
        },
        {
            'name': 'Transfer Learning Best Model (Rail IoU: 0.7961)',
            'input_path': '/home/mmc-server4/RailSafeNet/models/segformer_b3_transfer_best_rail_0.7961.pth',
            'output_path': '/home/mmc-server4/RailSafeNet/assets/models_pretrained/segformer/optimized/segformer_b3_transfer_best_0.7961_896x512.onnx'
        }
    ]

    results = []

    for model_info in models_to_convert:
        success = convert_model_to_onnx(
            model_info['input_path'],
            model_info['output_path'],
            model_info['name']
        )
        results.append((model_info['name'], success))

    # Summary
    print("📋 Conversion Summary:")
    print("=" * 40)
    for name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status}: {name}")

    successful_conversions = sum(1 for _, success in results if success)
    print(f"\n🎯 {successful_conversions}/{len(results)} models converted successfully!")

    if successful_conversions > 0:
        print("\n📂 Generated ONNX models:")
        optimized_dir = "/home/mmc-server4/RailSafeNet/assets/models_pretrained/segformer/optimized"
        if os.path.exists(optimized_dir):
            for file in os.listdir(optimized_dir):
                if file.endswith('.onnx'):
                    file_path = os.path.join(optimized_dir, file)
                    size_mb = os.path.getsize(file_path) / (1024*1024)
                    print(f"  • {file} ({size_mb:.1f}MB)")

if __name__ == "__main__":
    main()
