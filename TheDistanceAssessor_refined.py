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
PATH_model_seg = '/home/mmc-server4/RailSafeNet/models/segformer_b3_improved_best_0.7424.pth'
PATH_model_det = '/home/mmc-server4/RailSafeNet_mini_DT/assets/models_pretrained/yolo/yolov8s.pt'

# Configuration
USE_EGO_TRACK_ONLY = False  # Set to True for ego track only, False for all rails
RAIL_CLASSES = [9]  # Only Class 9 (rail-track)
MIN_RAIL_WIDTH = 3  # Minimum width for rail detection

def load_model(model_path):
    """Load the trained SegFormer model"""
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

def find_extreme_y_values(arr, values=RAIL_CLASSES):
        """Find the lowest and highest y-values where rail classes appear"""
        mask = np.isin(arr, values)
        rows_with_values = np.any(mask, axis=1)
        
        y_indices = np.nonzero(rows_with_values)[0]
        
        if y_indices.size == 0:
                return None, None
        
        return y_indices[0], y_indices[-1]

def find_edges(image, y_levels, values=RAIL_CLASSES, min_width=MIN_RAIL_WIDTH):
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

def identify_ego_track(edges_dict, image_width):
    """Identify ego track (center-most track)"""
    ego_edges_dict = {}
    last_ego_track_center = None
    
    sorted_y_levels = sorted(edges_dict.keys(), reverse=True)
    image_center_x = image_width / 2
    
    if sorted_y_levels:
        first_y = sorted_y_levels[0]
        tracks_at_first_y = edges_dict[first_y]
        
        if tracks_at_first_y:
            closest_track = min(
                tracks_at_first_y,
                key=lambda track: abs(((track[0] + track[1]) / 2) - image_center_x)
            )
            ego_edges_dict[first_y] = [closest_track]
            last_ego_track_center = (closest_track[0] + closest_track[1]) / 2

    for y in sorted_y_levels[1:]:
        if last_ego_track_center is None:
            break
            
        tracks_at_y = edges_dict[y]
        if tracks_at_y:
            closest_track = min(
                tracks_at_y,
                key=lambda track: abs(((track[0] + track[1]) / 2) - last_ego_track_center)
            )
            ego_edges_dict[y] = [closest_track]
            last_ego_track_center = (closest_track[0] + closest_track[1]) / 2
            
    return ego_edges_dict

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

def analyze_rail_detection(segmentation_mask, filename):
    """Analyze rail detection results"""
    
    # Count rail pixels
    rail_pixels = {}
    total_pixels = segmentation_mask.size
    
    for rail_class in RAIL_CLASSES:
        count = np.sum(segmentation_mask == rail_class)
        percentage = (count / total_pixels) * 100
        rail_pixels[rail_class] = {
            'count': count,
            'percentage': percentage
        }
    
    print(f"\n🔍 Rail Detection Analysis for {filename}:")
    print("-" * 50)
    
    for rail_class, stats in rail_pixels.items():
        class_name = "rail-track" if rail_class == 9 else f"rail-class-{rail_class}"
        print(f"🚂 Class {rail_class} ({class_name}): {stats['count']} pixels ({stats['percentage']:.2f}%)")
    
    total_rail_pixels = sum(stats['count'] for stats in rail_pixels.values())
    total_rail_percentage = (total_rail_pixels / total_pixels) * 100
    print(f"📊 Total Rail Coverage: {total_rail_pixels} pixels ({total_rail_percentage:.2f}%)")
    
    return total_rail_pixels > 0

def visualize_rail_detection(image, segmentation_mask, edges_dict, filename):
    """Visualize rail detection results"""
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (segmentation_mask.shape[1], segmentation_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Create rail-only mask
    rail_mask = np.zeros_like(segmentation_mask)
    for rail_class in RAIL_CLASSES:
        rail_mask[segmentation_mask == rail_class] = rail_class
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Original image
    axes[0,0].imshow(image)
    axes[0,0].set_title(f'Original Image - {filename}')
    axes[0,0].axis('off')
    
    # Rail mask only
    axes[0,1].imshow(rail_mask, cmap='tab20', vmin=0, vmax=18)
    axes[0,1].set_title('Rail Detection (Class 9 only)')
    axes[0,1].axis('off')
    
    # Detected edges overlay
    axes[1,0].imshow(image)
    if edges_dict:
        for y, edges in edges_dict.items():
            for start, end in edges:
                axes[1,0].plot([start, end], [y, y], 'r-', linewidth=3)
                axes[1,0].plot([start, start], [y-5, y+5], 'r-', linewidth=2)
                axes[1,0].plot([end, end], [y-5, y+5], 'r-', linewidth=2)
    
    track_type = "Ego Track" if USE_EGO_TRACK_ONLY else "All Rails"
    axes[1,0].set_title(f'Detected Rail Edges ({track_type})')
    axes[1,0].axis('off')
    
    # Statistics
    axes[1,1].axis('off')
    stats_text = f"""
    Rail Detection Statistics:
    
    🎯 Configuration:
    - Rail Classes: {RAIL_CLASSES}
    - Track Mode: {track_type}
    - Min Width: {MIN_RAIL_WIDTH} pixels
    
    📊 Detection Results:
    - Edge Levels Found: {len(edges_dict)}
    - Total Rail Segments: {sum(len(edges) for edges in edges_dict.values())}
    
    🔍 Coverage:
    """
    
    # Add rail class statistics
    total_pixels = segmentation_mask.size
    for rail_class in RAIL_CLASSES:
        count = np.sum(segmentation_mask == rail_class)
        percentage = (count / total_pixels) * 100
        stats_text += f"\n    - Class {rail_class}: {percentage:.2f}%"
    
    axes[1,1].text(0.1, 0.9, stats_text, transform=axes[1,1].transAxes, 
                   fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()

def run_refined_analysis(model_seg, model_det, image_size, filepath_img, PATH_jpgs, dataset_type, model_type, num_ys=15):
    """Run refined rail analysis"""
    
    segmentation_mask, image = segment(model_seg, image_size, filepath_img, PATH_jpgs, dataset_type, model_type)
    
    print(f'\n{"="*60}')
    print(f'🔍 REFINED RAIL ANALYSIS: {filepath_img}')
    print(f'{"="*60}')
    
    # Analyze rail detection
    rails_detected = analyze_rail_detection(segmentation_mask, filepath_img)
    
    if not rails_detected:
        print("❌ No rails detected in this image!")
        return False
    
    # Find rail edges
    clues = get_clues(segmentation_mask, num_ys)
    if not clues:
        print("❌ No clues found for rail detection!")
        return False
    
    print(f"📏 Scanning at {len(clues)} y-levels: {clues[:5]}{'...' if len(clues) > 5 else ''}")
    
    # Get all rail edges
    all_edges = find_edges(segmentation_mask, clues)
    
    # Choose between ego track or all rails
    if USE_EGO_TRACK_ONLY:
        image_width = segmentation_mask.shape[1]
        final_edges = identify_ego_track(all_edges, image_width)
        print(f"🎯 Ego Track Mode: {len(final_edges)} levels selected from {len(all_edges)} total")
    else:
        final_edges = all_edges
        print(f"🚂 All Rails Mode: Using all {len(final_edges)} detected rail levels")
    
    # Print edge details
    if final_edges:
        total_segments = sum(len(edges) for edges in final_edges.values())
        print(f"✅ Rail edges found: {total_segments} segments across {len(final_edges)} y-levels")
        
        # Show some examples
        example_levels = list(final_edges.keys())[:3]
        for y in example_levels:
            segments = final_edges[y]
            print(f"   Y={y}: {len(segments)} segments - {segments}")
    else:
        print("❌ No rail edges detected!")
        return False
    
    # Visualize results
    visualize_rail_detection(image, segmentation_mask, final_edges, filepath_img)
    
    return True

if __name__ == "__main__":
    # Configuration
    data_type = 'railsem19'
    model_type = "segformer"
    image_size = [1024,1024]
    num_ys = 15
    
    print("🚂 RailSafeNet Refined Rail Detection")
    print("=" * 60)
    print(f"🎯 Configuration:")
    print(f"   - Rail Classes: {RAIL_CLASSES}")
    print(f"   - Use Ego Track Only: {USE_EGO_TRACK_ONLY}")
    print(f"   - Min Rail Width: {MIN_RAIL_WIDTH} pixels")
    print("=" * 60)
    
    # Load models
    print("🔄 Loading models...")
    model_seg = load_model(PATH_model_seg)
    model_det = load_yolo(PATH_model_det)
    print("✅ Models loaded!")
    
    # Test on specific images
    test_images = ["rs00000.jpg", "rs00001.jpg", "rs00010.jpg", "rs07650.jpg"]
    
    for image_name in test_images:
        success = run_refined_analysis(model_seg, model_det, image_size, image_name, PATH_jpgs, data_type, model_type, num_ys)
        
        if not success:
            print(f"⚠️  Skipping {image_name} - no rails detected")
        
        print(f"\nPress Enter to continue to next image...")
        try:
            input()
        except:
            break
    
    print("\n🎉 Refined rail analysis completed!")