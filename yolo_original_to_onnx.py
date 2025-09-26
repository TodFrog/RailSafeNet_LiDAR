#!/usr/bin/env python3
"""
Convert YOLOv8 .pt model to ONNX format with a specific resolution.
"""

import os
from ultralytics import YOLO

def convert_yolo_to_onnx(pt_path, output_name, height=512, width=896):
    """
    Convert YOLOv8 .pt model to ONNX format.

    Args:
        pt_path (str): Path to the input .pt model file.
        output_name (str): Filename for the output .onnx model.
        height (int): The height of the model input.
        width (int): The width of the model input.
    """
    print(f"🚀 Converting {os.path.basename(pt_path)} to ONNX")
    
    if not os.path.exists(pt_path):
        print(f"❌ Input model not found: {pt_path}")
        return

    try:
        # 1. Load the YOLOv8 model
        print("📦 Loading YOLOv8 model...")
        model = YOLO(pt_path)
        print("✅ Model loaded successfully.")

        # 2. Export the model to ONNX format
        # The ultralytics library provides a convenient export function.
        print(f"🚀 Exporting to ONNX with resolution {width}x{height}...")
        model.export(
            format='onnx',
            imgsz=[height, width],  # Set image size [height, width]
            opset=12,               # Opset version, 11 or 12 is good for compatibility
            simplify=True,          # Applies onnx-simplifier for optimization
            dynamic=False,           # Set to False for a fixed batch size of 1
            batch=1                 # Explicitly set batch size to 1
        )
        
        # The output file will be named automatically (e.g., yolov8n.onnx)
        # We will rename it for clarity.
        default_onnx_path = pt_path.replace('.pt', '.onnx')
        final_onnx_path = os.path.join(os.path.dirname(pt_path), output_name)
        
        if os.path.exists(default_onnx_path):
            os.rename(default_onnx_path, final_onnx_path)
            print(f"✅ ONNX export successful!")
            size_mb = os.path.getsize(final_onnx_path) / (1024*1024)
            print(f"📁 Output saved to: {final_onnx_path}")
            print(f"📊 ONNX model size: {size_mb:.1f}MB")
        else:
            print(f"❌ ONNX export failed. Default file not found.")

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🚀 YOLOv8 .pt to ONNX Conversion Pipeline")
    print("=" * 60)
    
    # --- Configuration ---
    model_dir = "/home/mmc-server4/RailSafeNet/assets/models_pretrained/yolo" # .pt 파일이 저장된 디렉토리
    pt_filename = "yolov8s.pt"
    onnx_filename = "yolov8s_896x512.onnx"
    # ---------------------

    os.makedirs(model_dir, exist_ok=True)
    
    pt_path = os.path.join(model_dir, pt_filename)
    
    # Check if yolov8n.pt exists, if not, it will be downloaded automatically by YOLO()
    if not os.path.exists(pt_path):
        print(f"ℹ️  {pt_filename} not found. It will be downloaded automatically.")

    convert_yolo_to_onnx(pt_path, onnx_filename)

if __name__ == "__main__":
    main()