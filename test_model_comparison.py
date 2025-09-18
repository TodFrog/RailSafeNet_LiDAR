#!/usr/bin/env python3
"""
Model Comparison Test for TheDistanceAssessor

This script compares the original model with the new transfer learning model
by processing 5 test images and saving the results to ~/comparison_results/
"""

import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
import shutil

# Import the production model loader
from production_segformer_pytorch import load_pytorch_model

def load_new_model(model_path):
    """Load the new transfer learning model"""
    from transformers import SegformerForSemanticSegmentation, SegformerConfig

    print(f"🔄 Loading new model from {model_path}...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Use available GPU

    print(f"💡 Using device: {device}")

    # Load the production model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Create model with the saved config
    if 'config' in checkpoint:
        config = SegformerConfig.from_dict(checkpoint['config'])
    else:
        config = SegformerConfig.from_pretrained("nvidia/mit-b3")
        config.num_labels = 13

    model = SegformerForSemanticSegmentation(config)

    # Load weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)

    print("✅ New model loaded successfully")
    return model

def load_and_preprocess_image(image_path, image_size=(1024, 1024), device=None):
    """Load and preprocess image for both models"""
    print(f"📷 Loading image: {os.path.basename(image_path)}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return None, None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()

    # Resize to model input size
    image = cv2.resize(image, image_size)

    # Normalize for SegFormer
    image_norm = image.astype(np.float32) / 255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for i in range(3):
        image_norm[:, :, i] = (image_norm[:, :, i] - mean[i]) / std[i]

    # Convert to tensor and move to device
    image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0)
    if device is not None:
        image_tensor = image_tensor.to(device)

    return image_tensor, original_image

def process_with_original_model(model, image_tensor):
    """Process image with original model"""
    print("🔧 Processing with original model...")

    with torch.no_grad():
        outputs = model(image_tensor)

        # Get logits and apply softmax
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs

        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1)

    return prediction.squeeze().cpu().numpy(), probabilities.squeeze().cpu().numpy()

def process_with_new_model(model, image_tensor):
    """Process image with new transfer learning model"""
    print("🆕 Processing with new model...")

    with torch.no_grad():
        outputs = model(image_tensor)

        # Get logits and apply softmax
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs

        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1)

    return prediction.squeeze().cpu().numpy(), probabilities.squeeze().cpu().numpy()

def create_segmentation_overlay(original_image, prediction, alpha=0.6):
    """Create segmentation overlay visualization"""
    # Define colors for different classes (13 classes)
    colors = [
        [0, 0, 0],       # 0: background
        [128, 64, 128],  # 1: platform
        [244, 35, 232],  # 2: person
        [70, 70, 70],    # 3: car
        [102, 102, 156], # 4: main track
        [190, 153, 153], # 5: secondary track
        [153, 153, 153], # 6: road
        [250, 170, 30],  # 7: building
        [220, 220, 0],   # 8: vegetation
        [107, 142, 35],  # 9: sky
        [152, 251, 152], # 10: fence
        [70, 130, 180],  # 11: pole
        [220, 20, 60],   # 12: sign
    ]

    # Resize prediction to match original image
    h, w = original_image.shape[:2]
    prediction_resized = cv2.resize(prediction.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    # Create colored segmentation mask
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        mask = prediction_resized == class_id
        colored_mask[mask] = color

    # Blend with original image
    overlay = cv2.addWeighted(original_image, 1-alpha, colored_mask, alpha, 0)

    return overlay, colored_mask

def save_comparison_results(original_image, pred_original, pred_new, image_name, output_dir):
    """Save comparison results as images"""
    print(f"💾 Saving comparison results for {image_name}...")

    # Create overlays
    overlay_original, mask_original = create_segmentation_overlay(original_image, pred_original)
    overlay_new, mask_new = create_segmentation_overlay(original_image, pred_new)

    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Model Comparison: {image_name}', fontsize=16, fontweight='bold')

    # Row 1: Original model results
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(overlay_original)
    axes[0, 1].set_title('Original Model (Segmentation Overlay)', fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(mask_original)
    axes[0, 2].set_title('Original Model (Segmentation Mask)', fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2: New model results
    axes[1, 0].imshow(original_image)
    axes[1, 0].set_title('Original Image', fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(overlay_new)
    axes[1, 1].set_title('New Model - Rail IoU 75% (Segmentation Overlay)', fontweight='bold', color='green')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(mask_new)
    axes[1, 2].set_title('New Model - Rail IoU 75% (Segmentation Mask)', fontweight='bold', color='green')
    axes[1, 2].axis('off')

    # Save the comparison
    output_path = os.path.join(output_dir, f'{image_name}_comparison.jpg')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # Verify file was saved
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"✅ Saved: {output_path} ({file_size // 1024}KB)")
    else:
        print(f"❌ Failed to save: {output_path}")

def find_test_images():
    """Find test images for comparison"""
    test_image_paths = []

    # Try different potential image directories
    potential_dirs = [
        "/home/mmc-server4/Server/Datasets_hdd/rs19_val/jpgs/rs19_val",
        "/home/mmc-server4/RailSafeNet/assets/crop",
        "/home/mmc-server4/RailSafeNet/assets/test_images"
    ]

    for dir_path in potential_dirs:
        if os.path.exists(dir_path):
            print(f"📁 Checking directory: {dir_path}")
            files = os.listdir(dir_path)
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if image_files:
                # Take first 5 images
                selected_files = image_files[:5]
                for img_file in selected_files:
                    test_image_paths.append(os.path.join(dir_path, img_file))
                print(f"✅ Found {len(selected_files)} images in {dir_path}")
                break

    if not test_image_paths:
        print("❌ No test images found!")
        return []

    print(f"🎯 Selected {len(test_image_paths)} test images")
    return test_image_paths

def main():
    """Main comparison function"""
    print("🚀 Starting Model Comparison Test")
    print("=" * 60)

    # Setup output directory
    output_dir = "/home/mmc-server4/RailSafeNet/comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    # Set device - when CUDA_VISIBLE_DEVICES=3, GPU 3 becomes device 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"💡 Using device: {device}")

    # Load models
    print("\n📦 Loading models...")
    print("-" * 40)

    # Load original model
    try:
        original_model = load_pytorch_model()
        # Move original model to GPU 3 if available
        if device.type == 'cuda':
            original_model.model = original_model.model.to(device)
            original_model.device = device
        print("✅ Original model loaded")
    except Exception as e:
        print(f"❌ Failed to load original model: {e}")
        return

    # Load new model
    new_model_path = "/home/mmc-server4/RailSafeNet/models/segformer_b3_production_rail_0.7500.pth"
    try:
        new_model = load_new_model(new_model_path)
        print("✅ New model loaded (Rail IoU: 75%)")
    except Exception as e:
        print(f"❌ Failed to load new model: {e}")
        return

    # Find test images
    print("\n🔍 Finding test images...")
    print("-" * 40)
    test_images = find_test_images()

    if not test_images:
        print("❌ No test images found. Exiting.")
        return

    # Process each image
    print(f"\n🔄 Processing {len(test_images)} images...")
    print("-" * 40)

    for i, image_path in enumerate(test_images):
        print(f"\n[{i+1}/{len(test_images)}] Processing: {os.path.basename(image_path)}")

        # Load and preprocess image
        image_tensor, original_image = load_and_preprocess_image(image_path, device=device)
        if image_tensor is None:
            continue

        try:
            # Process with original model
            pred_original, prob_original = process_with_original_model(original_model, image_tensor)

            # Process with new model
            pred_new, prob_new = process_with_new_model(new_model, image_tensor)

            # Save comparison results
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            save_comparison_results(original_image, pred_original, pred_new, image_name, output_dir)

        except Exception as e:
            print(f"❌ Error processing image {image_path}: {e}")
            continue

    # Create summary report
    print(f"\n📊 Creating summary report...")
    print("-" * 40)

    summary_path = os.path.join(output_dir, "comparison_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Model Comparison Test Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Original Model: production_segformer_pytorch.py\n")
        f.write(f"New Model: segformer_b3_production_rail_0.7500.pth (Rail IoU: 75%)\n")
        f.write(f"Test Images: {len(test_images)}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write("\nTest Images:\n")
        for i, img_path in enumerate(test_images):
            f.write(f"  {i+1}. {os.path.basename(img_path)}\n")
        f.write(f"\nComparison images saved with suffix '_comparison.jpg'\n")
        f.write(f"Device used: {device}\n")

    print(f"✅ Summary saved: {summary_path}")

    print(f"\n🎉 Model comparison completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"🔍 Check the comparison images to evaluate the new model performance")
    print(f"📊 New model achieved Rail IoU: 75% (vs original model)")

if __name__ == "__main__":
    main()