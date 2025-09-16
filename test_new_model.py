import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_model(model_path):
    """Load the trained SegFormer model"""
    print(f"Loading model from: {model_path}")
    
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

def preprocess_image(image_path, size=1024):
    """Preprocess image for inference"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
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
    
    return image

def inference(model, image_tensor):
    """Run inference on preprocessed image"""
    with torch.no_grad():
        outputs = model(image_tensor)
        logits = outputs.logits
        
        # Resize to match input size
        logits = F.interpolate(
            logits, size=(1024, 1024), 
            mode='bilinear', align_corners=False
        )
        
        # Get predictions
        predictions = torch.argmax(logits, dim=1)
        
    return predictions.squeeze().cpu().numpy()

def visualize_results(original_image_path, prediction):
    """Visualize original image and prediction"""
    # Load original image
    original = cv2.imread(original_image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, (1024, 1024))
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(prediction, cmap='tab20')
    axes[1].set_title('Segmentation Prediction')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print class statistics
    unique_classes, counts = np.unique(prediction, return_counts=True)
    print("\nDetected classes:")
    for cls, count in zip(unique_classes, counts):
        percentage = (count / prediction.size) * 100
        print(f"Class {cls}: {count} pixels ({percentage:.2f}%)")

def main():
    # Paths
    model_path = "/home/mmc-server4/RailSafeNet/models/segformer_b3_improved_best_0.7424.pth"
    test_image_path = "/home/mmc-server4/Server/Datasets_hdd/rs19_val/jpgs/rs19_val/rs07650.jpg"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return
    
    try:
        # Load model
        print("🔄 Loading model...")
        model = load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Preprocess image
        print("🔄 Preprocessing image...")
        image_tensor = preprocess_image(test_image_path)
        print("✅ Image preprocessed!")
        
        # Run inference
        print("🔄 Running inference...")
        prediction = inference(model, image_tensor)
        print("✅ Inference completed!")
        
        # Visualize results
        print("🔄 Visualizing results...")
        visualize_results(test_image_path, prediction)
        
        # Check for rail classes
        rail_classes = [3, 9, 10, 12, 17, 18]  # Common rail-related classes
        detected_rails = [cls for cls in rail_classes if cls in prediction]
        
        if detected_rails:
            print(f"✅ Rail classes detected: {detected_rails}")
        else:
            print("⚠️  No rail classes detected in this image")
            
        print("\n🎉 Model test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()