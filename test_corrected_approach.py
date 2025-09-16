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

# Use the corrected multi-class approach
RAIL_CLASSES = [1, 5, 9, 10, 12]  # ALL rail-related classes
PATH_jpgs = '/home/mmc-server4/Server/Datasets_hdd/rs19_val/jpgs/rs19_val'
PATH_model_seg = '/home/mmc-server4/RailSafeNet/models/segformer_b3_improved_best_0.7424.pth'

def load_model(model_path):
    """Load the trained SegFormer model"""
    print(f"Loading SegFormer model from: {model_path}")
    
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b3", 
        num_labels=19,
        ignore_mismatched_sizes=True
    )
    
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
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

def segment(model_seg, image_size, filename, PATH_jpgs, dataset_type, model_type, item=None):
    image_norm, _, image, mask, _ = load(filename, PATH_jpgs, image_size, dataset_type=dataset_type, item=item)
    id_map = process(model_seg, image_norm, mask, model_type)
    id_map = cv2.resize(id_map, [1920,1080], interpolation=cv2.INTER_NEAREST)
    return id_map, image

def find_extreme_y_values(arr, values=RAIL_CLASSES):
    """Find the lowest and highest y-values where rail classes appear"""
    mask = np.isin(arr, values)
    rows_with_values = np.any(mask, axis=1)
    
    y_indices = np.nonzero(rows_with_values)[0]
    
    if y_indices.size == 0:
        return None, None
    
    return y_indices[0], y_indices[-1]

def find_edges(image, y_levels, values=RAIL_CLASSES, min_width=5):
    """Find rail edges at specified y-levels"""
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

def test_corrected_approach():
    """Test the corrected multi-class approach"""
    
    print("🚂 Testing Corrected Multi-Class Rail Detection")
    print("=" * 60)
    print(f"📋 Rail Classes: {RAIL_CLASSES}")
    print(f"   - Class 1: flat-track")
    print(f"   - Class 5: main-track") 
    print(f"   - Class 9: rail-track")
    print(f"   - Class 10: rail-raised")
    print(f"   - Class 12: rail-occluded")
    print("=" * 60)
    
    # Load model
    model_seg = load_model(PATH_model_seg)
    
    # Test images
    test_images = ["rs00000.jpg", "rs00001.jpg", "rs00010.jpg", "rs07650.jpg"]
    
    for image_name in test_images:
        print(f"\n{'='*50}")
        print(f"🔍 TESTING: {image_name}")
        print(f"{'='*50}")
        
        # Run segmentation
        segmentation_mask, image = segment(model_seg, [1024,1024], image_name, PATH_jpgs, 'railsem19', 'segformer')
        
        # Analyze rail coverage
        total_pixels = segmentation_mask.size
        rail_coverage = {}
        
        for rail_class in RAIL_CLASSES:
            count = np.sum(segmentation_mask == rail_class)
            percentage = (count / total_pixels) * 100
            rail_coverage[rail_class] = {'count': count, 'percentage': percentage}
        
        print("🚂 Rail Class Coverage:")
        total_rail_pixels = 0
        for rail_class, stats in rail_coverage.items():
            class_names = {1: 'flat-track', 5: 'main-track', 9: 'rail-track', 10: 'rail-raised', 12: 'rail-occluded'}
            class_name = class_names.get(rail_class, f'class-{rail_class}')
            print(f"   Class {rail_class:2d} ({class_name:12s}): {stats['count']:6d} pixels ({stats['percentage']:5.2f}%)")
            total_rail_pixels += stats['count']
        
        total_rail_percentage = (total_rail_pixels / total_pixels) * 100
        print(f"📊 Total Rail Coverage: {total_rail_pixels:6d} pixels ({total_rail_percentage:5.2f}%)")
        
        # Test edge detection
        clues = get_clues(segmentation_mask, 15)
        if clues:
            edges = find_edges(segmentation_mask, clues)
            total_segments = sum(len(edge_list) for edge_list in edges.values())
            print(f"🔍 Edge Detection: {len(edges)} y-levels, {total_segments} total segments")
            
            if len(edges) > 0:
                print("✅ Rail detection SUCCESSFUL!")
                # Show some example edge coordinates
                example_levels = list(edges.keys())[:3]
                for y in example_levels:
                    segments = edges[y]
                    print(f"   Y={y}: {len(segments)} segments")
            else:
                print("❌ No rail edges detected!")
        else:
            print("❌ No clues found for rail detection!")
        
        # Visualize
        visualize_corrected_detection(image, segmentation_mask, image_name)

def visualize_corrected_detection(image, segmentation_mask, image_name):
    """Visualize the corrected rail detection"""
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (segmentation_mask.shape[1], segmentation_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Create combined rail mask
    combined_rail_mask = np.zeros_like(segmentation_mask)
    colors = {1: 50, 5: 100, 9: 150, 10: 200, 12: 255}
    
    for rail_class in RAIL_CLASSES:
        combined_rail_mask[segmentation_mask == rail_class] = colors[rail_class]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Original image
    axes[0,0].imshow(image)
    axes[0,0].set_title(f'Original Image - {image_name}')
    axes[0,0].axis('off')
    
    # Combined rail mask
    axes[0,1].imshow(combined_rail_mask, cmap='gray')
    axes[0,1].set_title('All Rail Classes Combined')
    axes[0,1].axis('off')
    
    # Individual rail classes
    axes[1,0].imshow(segmentation_mask, cmap='tab20', vmin=0, vmax=18)
    axes[1,0].set_title('Full Segmentation (All Classes)')
    axes[1,0].axis('off')
    
    # Statistics
    axes[1,1].axis('off')
    
    # Calculate coverage statistics
    total_pixels = segmentation_mask.size
    stats_text = f"Rail Detection Statistics:\n\n"
    
    for rail_class in RAIL_CLASSES:
        count = np.sum(segmentation_mask == rail_class)
        percentage = (count / total_pixels) * 100
        class_names = {1: 'flat-track', 5: 'main-track', 9: 'rail-track', 10: 'rail-raised', 12: 'rail-occluded'}
        class_name = class_names.get(rail_class, f'class-{rail_class}')
        stats_text += f"Class {rail_class} ({class_name}):\n"
        stats_text += f"  {count:,} pixels ({percentage:.2f}%)\n\n"
    
    total_rail_pixels = sum(np.sum(segmentation_mask == cls) for cls in RAIL_CLASSES)
    total_rail_percentage = (total_rail_pixels / total_pixels) * 100
    stats_text += f"Total Rail Coverage:\n{total_rail_pixels:,} pixels ({total_rail_percentage:.2f}%)"
    
    axes[1,1].text(0.1, 0.9, stats_text, transform=axes[1,1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_corrected_approach()