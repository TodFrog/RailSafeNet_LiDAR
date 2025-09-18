#!/usr/bin/env python3
"""
TheDistanceAssessor modified for Transfer Learning models
Compatible with both original and new transfer learning models
"""

import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.path as mplPath
import matplotlib.patches as patches
from ultralyticsplus import YOLO
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

PATH_jpgs = '/home/mmc-server4/Server/Datasets_hdd/rs19_val/jpgs/rs19_val'

# Model paths - you can switch between different models
MODEL_PATHS = {
    'original': '/home/mmc-server4/RailSafeNet_mini_DT/assets/models_pretrained/segformer/SegFormer_B3_1024_finetuned.pth',
    'improved': '/home/mmc-server4/RailSafeNet/models/segformer_b3_improved_best_0.7424.pth', 
    'transfer_best': '/home/mmc-server4/RailSafeNet/models/segformer_b3_transfer_best_*.pth',  # Use latest best
    'transfer_final': '/home/mmc-server4/RailSafeNet/models/segformer_b3_transfer_final.pth'
}

PATH_model_det = '/home/mmc-server4/RailSafeNet_mini_DT/assets/models_pretrained/yolo/yolov8s.pt'

def detect_model_type(model_path):
    """Detect if model is original complete model or state_dict only"""
    try:
        model_data = torch.load(model_path, map_location='cpu')
        
        if hasattr(model_data, 'state_dict') and hasattr(model_data, 'config'):
            # Original complete model
            num_classes = model_data.config.num_labels
            return 'complete', num_classes
        elif isinstance(model_data, dict) and 'decode_head.classifier.weight' in model_data:
            # State dict - determine classes from classifier
            classifier_weight = model_data['decode_head.classifier.weight']
            num_classes = classifier_weight.shape[0]
            return 'state_dict', num_classes
        else:
            print(f"⚠️  Unknown model format, defaulting to 19 classes")
            return 'unknown', 19
            
    except Exception as e:
        print(f"❌ Error detecting model type: {e}")
        return 'unknown', 19

def load_model_smart(model_path):
    """Smart model loader that handles both original and transfer learning models"""
    print(f"🔍 Detecting model type: {model_path}")
    
    model_type, num_classes = detect_model_type(model_path)
    
    print(f"📊 Detected: {model_type} model with {num_classes} classes")
    
    if model_type == 'complete':
        # Original complete model - load directly
        print("✅ Loading complete model directly")
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        return model
        
    elif model_type == 'state_dict':
        # State dict only - need to create model first
        print(f"✅ Loading state dict with {num_classes} classes")
        
        # Create model with correct number of classes
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b3", 
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Load state dict
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
        
    else:
        # Unknown format - try default approach
        print("⚠️  Unknown format, trying default loader")
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b3", 
            num_labels=19,
            ignore_mismatched_sizes=True
        )
        
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        except:
            print("❌ Failed to load model with default approach")
            
        model.eval()
        return model

def load(filename, PATH_jpgs, image_size, dataset_type, item=None):
    """Load and preprocess image for segmentation"""
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

def load_yolo(PATH_model):
    model = YOLO(PATH_model)
    model.overrides['conf'] = 0.25  # NMS confidence threshold
    model.overrides['iou'] = 0.45  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 1000  # maximum number of detections per image
    return model

def find_extreme_y_values(arr, values=[1, 5, 9, 10, 12]):
    """Find the lowest and highest y-values where rail classes appear"""
    # Handle both 13-class and 19-class systems
    if arr.max() <= 12:
        # 13-class system - map to appropriate classes
        values = [1, 5, 9, 10, 12] if max(values) <= 12 else [1, 2, 3, 4, 5]
    
    mask = np.isin(arr, values)
    rows_with_values = np.any(mask, axis=1)
    
    y_indices = np.nonzero(rows_with_values)[0]
    
    if y_indices.size == 0:
        return None, None
    
    return y_indices[0], y_indices[-1]

def find_edges(image, y_levels, values=[1, 5, 9, 10, 12], min_width=5):
    """Find rail edges at specified y-levels"""
    # Handle both 13-class and 19-class systems
    if image.max() <= 12:
        # 13-class system - adjust rail classes
        values = [1, 5, 9, 10, 12] if max(values) <= 12 else [1, 2, 3, 4, 5]
    
    edges_dict = {}
    
    for y in y_levels:
        row = image[y, :]
        mask = np.isin(row, values).astype(int)
        padded_mask = np.pad(mask, (1, 1), 'constant', constant_values=0)
        diff = np.diff(padded_mask)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1

        # Filter sequences based on the minimum width criteria
        filtered_edges = [(start, end) for start, end in zip(starts, ends) if end - start + 1 >= min_width]
        filtered_edges = [(start, end) for start, end in filtered_edges if 0 not in (start, end) and 1919 not in (start, end)]
        
        if filtered_edges:
            edges_dict[y] = filtered_edges
    
    return edges_dict

def get_clues(segmentation_mask, number_of_clues):
    lowest, highest = find_extreme_y_values(segmentation_mask)
    if lowest is not None and highest is not None:
        clue_step = int((highest - lowest) / number_of_clues+1)
        clues = []
        for i in range(number_of_clues):
            clues.append(highest - (i*clue_step))
        clues.append(lowest+int(0.5*clue_step))
                
        return clues
    else:
        return []

def segment(model_seg, image_size, filename, PATH_jpgs, dataset_type, model_type, item=None):
    image_norm, _, image, mask, _ = load(filename, PATH_jpgs, image_size, dataset_type=dataset_type, item=item)
    id_map = process(model_seg, image_norm, mask, model_type)
    id_map = cv2.resize(id_map, [1920,1080], interpolation=cv2.INTER_NEAREST)
    return id_map, image

def detect(model_det, filename_img, PATH_jpgs):
    image = cv2.imread(os.path.join(PATH_jpgs, filename_img))
    results = model_det.predict(image)
    return results, model_det, image

def analyze_rail_detection(segmentation_mask, filename, model_info=""):
    """Analyze rail detection results with model info"""
    
    # Determine rail classes based on model
    if segmentation_mask.max() <= 12:
        # 13-class system
        rail_classes = [1, 2, 3, 4, 5]  # Adjust based on actual mapping
        print(f"📊 Using 13-class system for analysis")
    else:
        # 19-class system
        rail_classes = [1, 5, 9, 10, 12]
        print(f"📊 Using 19-class system for analysis")
    
    # Count rail pixels
    rail_pixels = {}
    total_pixels = segmentation_mask.size
    
    for rail_class in rail_classes:
        count = np.sum(segmentation_mask == rail_class)
        percentage = (count / total_pixels) * 100
        rail_pixels[rail_class] = {
            'count': count,
            'percentage': percentage
        }
    
    print(f"\n🔍 Rail Detection Analysis for {filename} {model_info}:")
    print("-" * 60)
    
    for rail_class, stats in rail_pixels.items():
        if stats['count'] > 0:  # Only show classes that exist
            print(f"🚂 Class {rail_class}: {stats['count']} pixels ({stats['percentage']:.2f}%)")
    
    total_rail_pixels = sum(stats['count'] for stats in rail_pixels.values())
    total_rail_percentage = (total_rail_pixels / total_pixels) * 100
    print(f"📊 Total Rail Coverage: {total_rail_pixels} pixels ({total_rail_percentage:.2f}%)")
    
    return total_rail_pixels > 0

def run_distance_assessment(model_path, test_images=None, model_name=""):
    """Run distance assessment with specified model"""
    
    print(f"\n🚀 RailSafeNet Distance Assessment - {model_name}")
    print("="*80)
    
    # Load models
    print("🔄 Loading models...")
    model_seg = load_model_smart(model_path)
    model_det = load_yolo(PATH_model_det)
    print("✅ Models loaded successfully!")
    
    # Test configuration
    image_size = [1024, 1024]
    data_type = 'railsem19'
    model_type = "segformer"
    num_ys = 15
    
    if test_images is None:
        test_images = ["rs00000.jpg", "rs00001.jpg", "rs00010.jpg", "rs07650.jpg"]
    
    for image_name in test_images:
        print(f"\n{'='*50}")
        print(f"🔍 PROCESSING: {image_name}")
        print(f"{'='*50}")
        
        try:
            # Run segmentation
            segmentation_mask, image = segment(
                model_seg, image_size, image_name, PATH_jpgs, data_type, model_type
            )
            
            # Analyze rail detection
            rails_detected = analyze_rail_detection(segmentation_mask, image_name, f"({model_name})")
            
            if rails_detected:
                # Find rail edges
                clues = get_clues(segmentation_mask, num_ys)
                if clues:
                    edges = find_edges(segmentation_mask, clues)
                    total_segments = sum(len(edge_list) for edge_list in edges.values())
                    print(f"✅ Rail edge detection: {len(edges)} levels, {total_segments} segments")
                else:
                    print("❌ No clues found for rail detection!")
            else:
                print("❌ No rails detected in this image!")
                
        except Exception as e:
            print(f"❌ Error processing {image_name}: {e}")
    
    print(f"\n🎉 Assessment completed with {model_name}!")

def main():
    """Main function to test different models"""
    
    print("🚀 TheDistanceAssessor Transfer Learning Compatibility Test")
    print("="*80)
    
    # Test different models
    models_to_test = {
        'Original Model': MODEL_PATHS['original'],
        # 'Transfer Learning Best': MODEL_PATHS['transfer_best'],  # Enable when available
        # 'Transfer Learning Final': MODEL_PATHS['transfer_final']  # Enable when available
    }
    
    test_images = ["rs00000.jpg", "rs00001.jpg"]  # Quick test
    
    for model_name, model_path in models_to_test.items():
        if os.path.exists(model_path):
            run_distance_assessment(model_path, test_images, model_name)
        else:
            print(f"⚠️  Model not found: {model_path}")

if __name__ == "__main__":
    main()