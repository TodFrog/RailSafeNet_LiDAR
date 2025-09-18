#!/usr/bin/env python3
"""
Simple script to use transfer learning model in TheDistanceAssessor
Just change the model path and it should work!
"""

import os
import shutil

def update_distance_assessor_for_transfer_model(transfer_model_path):
    """Update TheDistanceAssessor.py to use transfer learning model"""
    
    print("🔧 Updating TheDistanceAssessor.py for transfer learning model...")
    
    # Read the original file
    original_file = "/home/mmc-server4/RailSafeNet/TheDistanceAssessor.py" 
    
    if not os.path.exists(original_file):
        print(f"❌ Original file not found: {original_file}")
        return False
    
    # Create backup
    backup_file = original_file + ".backup"
    if not os.path.exists(backup_file):
        shutil.copy2(original_file, backup_file)
        print(f"💾 Backup created: {backup_file}")
    
    # Read content
    with open(original_file, 'r') as f:
        content = f.read()
    
    # Update paths - make them absolute and point to our directory structure
    replacements = {
        "PATH_jpgs = 'RailNet_DT/assets/rs19val/jpgs/test'": 
        "PATH_jpgs = '/home/mmc-server4/Server/Datasets_hdd/rs19_val/jpgs/rs19_val'",
        
        "PATH_model_seg = 'RailNet_DT/assets/models_pretrained/segformer/SegFormer_B3_1024_finetuned.pth'":
        f"PATH_model_seg = '{transfer_model_path}'",
        
        "PATH_model_det = 'RailNet_DT/assets/models_pretrained/ultralyticsplus/yolov8s'":
        "PATH_model_det = '/home/mmc-server4/RailSafeNet_mini_DT/assets/models_pretrained/yolo/yolov8s.pt'",
        
        "PATH_base = 'RailNet_DT/assets/pilsen_railway_dataset/'":
        "PATH_base = '/home/mmc-server4/RailSafeNet_mini_DT/assets/pilsen_railway_dataset/'",
        
        'eda_path = "RailNet_DT/assets/pilsen_railway_dataset/eda_table.table.json"':
        'eda_path = "/home/mmc-server4/RailSafeNet_mini_DT/assets/pilsen_railway_dataset/eda_table.table.json"'
    }
    
    # Apply replacements
    updated_content = content
    for old, new in replacements.items():
        updated_content = updated_content.replace(old, new)
    
    # Also need to fix the import issue
    if "from scripts.test_filtered_cls import load, load_model, process" in updated_content:
        # Add our own load_model function that handles state_dict
        load_model_function = '''
def load_model(model_path):
    """Load the trained SegFormer model - supports both complete model and state_dict"""
    import torch
    from transformers import SegformerForSemanticSegmentation
    
    print(f"Loading SegFormer model from: {model_path}")
    
    try:
        # Try loading as complete model first (original format)
        model_data = torch.load(model_path, map_location='cpu')
        
        if hasattr(model_data, 'state_dict') and hasattr(model_data, 'config'):
            # Original complete model
            print("✅ Loading complete model directly")
            model_data.eval()
            return model_data
        else:
            # State dict only - create model first
            print("✅ Loading state dict model")
            model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-b3", 
                num_labels=13,  # Transfer learning uses 13 classes
                ignore_mismatched_sizes=True
            )
            model.load_state_dict(model_data)
            model.eval()
            return model
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e

def load(filename, PATH_jpgs, image_size, dataset_type, item=None):
    """Load and preprocess image for segmentation"""
    import cv2
    import numpy as np
    import torch
    
    image_path = os.path.join(PATH_jpgs, filename)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    
    # Resize to model input size
    image = cv2.resize(image, (image_size[0], image_size[1]))
    
    # Normalize for SegFormer
    image_norm = image.astype(np.float32) / 255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    for i in range(3):
        image_norm[:, :, i] = (image_norm[:, :, i] - mean[i]) / std[i]
    
    # Convert to tensor
    image_norm = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0)
    
    return image_norm, None, original_image, None, None

def process(model, image_norm, mask, model_type):
    """Process image through SegFormer model"""
    import torch
    import torch.nn.functional as F
    
    with torch.no_grad():
        outputs = model(image_norm)
        logits = outputs.logits
        
        # Resize to target size (1080x1920 for full HD)
        logits = F.interpolate(
            logits, size=(1080, 1920), 
            mode='bilinear', align_corners=False
        )
        
        # Get predictions
        predictions = torch.argmax(logits, dim=1)
        
    return predictions.squeeze().cpu().numpy()

'''
        
        # Replace the import with our functions
        updated_content = updated_content.replace(
            "from scripts.test_filtered_cls import load, load_model, process",
            load_model_function
        )
    
    # Write updated content
    with open(original_file, 'w') as f:
        f.write(updated_content)
    
    print("✅ TheDistanceAssessor.py updated successfully!")
    print(f"📁 Model path set to: {transfer_model_path}")
    print("🚀 Ready to use transfer learning model!")
    
    return True

def main():
    """Main function to update TheDistanceAssessor for transfer learning"""
    
    print("🚀 TheDistanceAssessor Transfer Learning Setup")
    print("="*60)
    
    # Look for the best transfer learning model
    models_dir = "/home/mmc-server4/RailSafeNet/models"
    
    # Find best model
    best_models = []
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.startswith("segformer_b3_transfer_best_") and file.endswith(".pth"):
                best_models.append(os.path.join(models_dir, file))
    
    if best_models:
        # Use the most recent best model
        best_model = sorted(best_models)[-1]
        print(f"🎯 Found best transfer model: {best_model}")
        
        if update_distance_assessor_for_transfer_model(best_model):
            print("\n✅ Setup completed! You can now run:")
            print("   python TheDistanceAssessor.py")
        else:
            print("\n❌ Setup failed!")
    else:
        print("❌ No transfer learning models found!")
        print(f"   Looking in: {models_dir}")
        print("   Please complete training first.")

if __name__ == "__main__":
    main()