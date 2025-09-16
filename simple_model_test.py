import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from ultralyticsplus import YOLO

def load_segformer_model(model_path):
    """Load the trained SegFormer model for inference"""
    print(f"Loading SegFormer model from: {model_path}")
    
    # Initialize model
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b3", 
        num_labels=19,
        ignore_mismatched_sizes=True
    )
    
    # Load trained weights
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    return model

def preprocess_for_segformer(image_path, size=1024):
    """Preprocess image for SegFormer inference"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape[:2]
    
    # Resize
    image = cv2.resize(image, (size, size))
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    for i in range(3):
        image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
    
    # Convert to tensor and add batch dimension
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    return image, original_shape

def segformer_inference(model, image_tensor, target_size=(1080, 1920)):
    """Run SegFormer inference and return segmentation map"""
    with torch.no_grad():
        outputs = model(image_tensor)
        logits = outputs.logits
        
        # Resize to target size (1080x1920 for full HD)
        logits = F.interpolate(
            logits, size=target_size, 
            mode='bilinear', align_corners=False
        )
        
        # Get predictions
        predictions = torch.argmax(logits, dim=1)
        
    return predictions.squeeze().cpu().numpy()

def load_yolo_model(model_path):
    """Load YOLO model"""
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    return model

def yolo_inference(model, image_path):
    """Run YOLO inference"""
    image = cv2.imread(image_path)
    results = model.predict(image)
    return results, image

def visualize_combined_results(image_path, segmentation, yolo_results, yolo_model):
    """Visualize both segmentation and detection results"""
    # Load original image
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, (1920, 1080))  # Resize to match segmentation
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Original image
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Segmentation
    axes[1].imshow(segmentation, cmap='tab20')
    axes[1].set_title('Segmentation (SegFormer B3 Improved)')
    axes[1].axis('off')
    
    # Combined view
    axes[2].imshow(original)
    
    # Overlay detection boxes
    if yolo_results and len(yolo_results[0].boxes) > 0:
        boxes = yolo_results[0].boxes.xywh.cpu().numpy()
        classes = yolo_results[0].boxes.cls.cpu().numpy()
        
        # Scale boxes to match image size
        img_h, img_w = original.shape[:2]
        
        for box, cls in zip(boxes, classes):
            x_center, y_center, width, height = box
            x1 = int((x_center - width/2) * img_w / yolo_results[0].orig_img.shape[1])
            y1 = int((y_center - height/2) * img_h / yolo_results[0].orig_img.shape[0])
            x2 = int((x_center + width/2) * img_w / yolo_results[0].orig_img.shape[1])
            y2 = int((y_center + height/2) * img_h / yolo_results[0].orig_img.shape[0])
            
            # Draw bounding box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                               edgecolor='red', facecolor='none')
            axes[2].add_patch(rect)
            
            # Add label
            class_name = yolo_model.names[int(cls)]
            axes[2].text(x1, y1-5, class_name, color='red', fontsize=8, 
                        bbox=dict(facecolor='white', alpha=0.7))
    
    axes[2].set_title('Combined: Segmentation + Detection')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    unique_classes, counts = np.unique(segmentation, return_counts=True)
    print("\nSegmentation classes detected:")
    rail_classes = [3, 9, 10, 12, 17, 18]
    detected_rails = []
    
    for cls, count in zip(unique_classes, counts):
        percentage = (count / segmentation.size) * 100
        print(f"Class {cls}: {count} pixels ({percentage:.2f}%)")
        if cls in rail_classes:
            detected_rails.append(cls)
    
    print(f"\nRail classes detected: {detected_rails}")
    
    if yolo_results and len(yolo_results[0].boxes) > 0:
        print(f"YOLO detections: {len(yolo_results[0].boxes)} objects")
        detected_classes = yolo_results[0].boxes.cls.cpu().numpy()
        for cls in np.unique(detected_classes):
            class_name = yolo_model.names[int(cls)]
            count = np.sum(detected_classes == cls)
            print(f"  {class_name}: {count} detections")
    else:
        print("No YOLO detections found")

def main():
    # Paths
    segformer_model_path = "/home/mmc-server4/RailSafeNet/models/segformer_b3_improved_best_0.7424.pth"
    yolo_model_path = "/home/mmc-server4/RailSafeNet_mini_DT/assets/models_pretrained/yolo/yolov8s.pt"
    test_image_path = "/home/mmc-server4/Server/Datasets_hdd/rs19_val/jpgs/rs19_val/rs07650.jpg"
    
    # Check if files exist
    for path, name in [(segformer_model_path, "SegFormer model"), 
                       (yolo_model_path, "YOLO model"), 
                       (test_image_path, "Test image")]:
        if not os.path.exists(path):
            print(f"❌ {name} not found: {path}")
            return
    
    try:
        # Load models
        print("🔄 Loading models...")
        segformer_model = load_segformer_model(segformer_model_path)
        yolo_model = load_yolo_model(yolo_model_path)
        print("✅ Models loaded successfully!")
        
        # Run SegFormer inference
        print("🔄 Running SegFormer inference...")
        image_tensor, original_shape = preprocess_for_segformer(test_image_path)
        segmentation = segformer_inference(segformer_model, image_tensor)
        print("✅ SegFormer inference completed!")
        
        # Run YOLO inference
        print("🔄 Running YOLO inference...")
        yolo_results, _ = yolo_inference(yolo_model, test_image_path)
        print("✅ YOLO inference completed!")
        
        # Visualize results
        print("🔄 Visualizing combined results...")
        visualize_combined_results(test_image_path, segmentation, yolo_results, yolo_model)
        
        print("\n🎉 Combined model test completed successfully!")
        print("✅ Both SegFormer B3 Improved and YOLO models are working correctly!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()