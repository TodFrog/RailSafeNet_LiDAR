import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
import cv2
import numpy as np
import os
from tqdm import tqdm
import json
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

class RailSafeNetEvaluator:
    def __init__(self, model_path, images_path, masks_path):
        self.model_path = model_path
        self.images_path = images_path
        self.masks_path = masks_path
        self.model = None
        
        # RailSem19 class mapping
        self.class_names = {
            0: 'void', 1: 'flat-track', 2: 'flat-terrain', 3: 'human', 4: 'vehicle',
            5: 'main-track', 6: 'side-track', 7: 'raised-track', 8: 'sky', 
            9: 'rail-track', 10: 'rail-raised', 11: 'rail-embedded', 12: 'rail-occluded',
            13: 'buffer-stop', 14: 'signal', 15: 'bridge', 16: 'building', 
            17: 'fence', 18: 'vegetation'
        }
        
        # Focus on rail-related classes
        self.rail_classes = [1, 5, 6, 7, 9, 10, 11, 12]
        self.important_classes = [1, 3, 4, 5, 9, 10, 12]  # rails + humans + vehicles
        
        # Statistics
        self.results = {
            'total_images': 0,
            'processed_images': 0,
            'failed_images': [],
            'class_ious': {},
            'rail_detection_rate': 0,
            'per_image_results': [],
            'confusion_matrices': {},
            'class_precision': {},
            'class_recall': {},
            'class_f1': {}
        }
    
    def load_model(self):
        """Load the trained SegFormer model"""
        print(f"🔄 Loading SegFormer model from: {self.model_path}")
        
        # Initialize model
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b3", 
            num_labels=19,
            ignore_mismatched_sizes=True
        )
        
        # Load trained weights
        state_dict = torch.load(self.model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print("✅ Model loaded successfully!")
    
    def preprocess_image(self, image_path, size=1024):
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
    
    def inference(self, image_tensor, target_size=(1080, 1920)):
        """Run inference on preprocessed image"""
        with torch.no_grad():
            outputs = self.model(image_tensor)
            logits = outputs.logits
            
            # Resize to target size
            logits = F.interpolate(
                logits, size=target_size, 
                mode='bilinear', align_corners=False
            )
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            
        return predictions.squeeze().cpu().numpy()
    
    def load_ground_truth(self, image_name):
        """Load ground truth mask"""
        mask_file = image_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.masks_path, mask_file)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (1920, 1080), interpolation=cv2.INTER_NEAREST)
            return mask
        else:
            return None
    
    def calculate_iou(self, pred, gt, class_id):
        """Calculate IoU for a specific class"""
        pred_mask = (pred == class_id)
        gt_mask = (gt == class_id)
        
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union == 0:
            return float('nan')  # No pixels of this class
        
        return intersection / union
    
    def calculate_metrics_per_class(self, pred, gt):
        """Calculate precision, recall, F1 for each class"""
        # Flatten arrays
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()
        
        # Remove ignore class (255) if present
        valid_mask = gt_flat != 255
        pred_flat = pred_flat[valid_mask]
        gt_flat = gt_flat[valid_mask]
        
        # Calculate confusion matrix
        cm = confusion_matrix(gt_flat, pred_flat, labels=list(range(19)))
        
        # Calculate metrics per class
        metrics = {}
        for class_id in range(19):
            tp = cm[class_id, class_id]
            fp = cm[:, class_id].sum() - tp
            fn = cm[class_id, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[class_id] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': cm[class_id, :].sum()
            }
        
        return metrics, cm
    
    def evaluate_single_image(self, image_name):
        """Evaluate a single image"""
        image_path = os.path.join(self.images_path, image_name)
        
        if not os.path.exists(image_path):
            return None
        
        try:
            # Load and process image
            image_tensor = self.preprocess_image(image_path)
            prediction = self.inference(image_tensor)
            
            # Load ground truth
            ground_truth = self.load_ground_truth(image_name)
            if ground_truth is None:
                return None
            
            # Calculate IoU for each class
            class_ious = {}
            for class_id in range(19):
                iou = self.calculate_iou(prediction, ground_truth, class_id)
                if not np.isnan(iou):
                    class_ious[class_id] = iou
            
            # Check rail detection
            pred_rails = [cls for cls in self.rail_classes if cls in np.unique(prediction)]
            gt_rails = [cls for cls in self.rail_classes if cls in np.unique(ground_truth)]
            rail_detected = len(pred_rails) > 0
            rail_present = len(gt_rails) > 0
            
            # Calculate per-class metrics
            class_metrics, cm = self.calculate_metrics_per_class(prediction, ground_truth)
            
            result = {
                'image_name': image_name,
                'class_ious': class_ious,
                'rail_detected': rail_detected,
                'rail_present': rail_present,
                'pred_rails': pred_rails,
                'gt_rails': gt_rails,
                'class_metrics': class_metrics,
                'confusion_matrix': cm
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing {image_name}: {e}")
            return {'image_name': image_name, 'error': str(e)}
    
    def evaluate_dataset(self, max_images=None, save_results=True):
        """Evaluate the entire dataset"""
        if self.model is None:
            self.load_model()
        
        # Get all image files
        image_files = sorted([f for f in os.listdir(self.images_path) if f.endswith('.jpg')])
        
        if max_images:
            image_files = image_files[:max_images]
        
        self.results['total_images'] = len(image_files)
        
        print(f"🚀 Starting evaluation on {len(image_files)} images...")
        print("📊 This may take a while. Progress will be shown below.\n")
        
        # Initialize aggregated metrics
        aggregated_ious = defaultdict(list)
        aggregated_metrics = defaultdict(lambda: defaultdict(list))
        total_cm = np.zeros((19, 19), dtype=np.int64)
        
        rail_detection_stats = {'detected': 0, 'present': 0, 'total': 0}
        
        # Process images with progress bar
        for i, image_name in enumerate(tqdm(image_files, desc="Processing images")):
            result = self.evaluate_single_image(image_name)
            
            if result is None or 'error' in result:
                self.results['failed_images'].append(image_name)
                continue
            
            self.results['per_image_results'].append(result)
            self.results['processed_images'] += 1
            
            # Aggregate IoUs
            for class_id, iou in result['class_ious'].items():
                aggregated_ious[class_id].append(iou)
            
            # Aggregate class metrics
            for class_id, metrics in result['class_metrics'].items():
                for metric_name, value in metrics.items():
                    if metric_name != 'support':
                        aggregated_metrics[class_id][metric_name].append(value)
            
            # Aggregate confusion matrix
            total_cm += result['confusion_matrix']
            
            # Rail detection stats
            rail_detection_stats['total'] += 1
            if result['rail_present']:
                rail_detection_stats['present'] += 1
                if result['rail_detected']:
                    rail_detection_stats['detected'] += 1
            
            # Print progress every 100 images
            if (i + 1) % 100 == 0:
                processed = self.results['processed_images']
                failed = len(self.results['failed_images'])
                print(f"✅ Processed: {processed}, ❌ Failed: {failed}")
        
        # Calculate final metrics
        print("\n🔄 Calculating final metrics...")
        
        # Average IoUs
        for class_id, ious in aggregated_ious.items():
            self.results['class_ious'][class_id] = {
                'mean_iou': np.mean(ious),
                'std_iou': np.std(ious),
                'count': len(ious)
            }
        
        # Average class metrics
        for class_id, metrics_dict in aggregated_metrics.items():
            self.results['class_precision'][class_id] = np.mean(metrics_dict['precision'])
            self.results['class_recall'][class_id] = np.mean(metrics_dict['recall'])
            self.results['class_f1'][class_id] = np.mean(metrics_dict['f1'])
        
        # Rail detection rate
        if rail_detection_stats['present'] > 0:
            self.results['rail_detection_rate'] = rail_detection_stats['detected'] / rail_detection_stats['present']
        
        # Store overall confusion matrix
        self.results['confusion_matrices']['overall'] = total_cm
        
        # Save results
        if save_results:
            self.save_results()
        
        print("✅ Evaluation completed!")
        return self.results
    
    def save_results(self, output_dir="evaluation_results"):
        """Save evaluation results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results as JSON
        results_copy = self.results.copy()
        # Convert numpy arrays to lists for JSON serialization
        if 'confusion_matrices' in results_copy:
            for key, cm in results_copy['confusion_matrices'].items():
                results_copy['confusion_matrices'][key] = cm.tolist()
        
        # Remove per-image results for summary (too large)
        detailed_results = results_copy.copy()
        summary_results = results_copy.copy()
        summary_results.pop('per_image_results', None)
        
        with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w') as f:
            json.dump(summary_results, f, indent=2, default=str)
        
        print(f"📁 Results saved to {output_dir}/")
    
    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*80)
        print("🎯 RAILSAFENET SEGFORMER B3 EVALUATION SUMMARY")
        print("="*80)
        
        print(f"📊 Total Images: {self.results['total_images']}")
        print(f"✅ Processed: {self.results['processed_images']}")
        print(f"❌ Failed: {len(self.results['failed_images'])}")
        print(f"📈 Success Rate: {self.results['processed_images']/self.results['total_images']*100:.1f}%")
        
        print(f"\n🚂 Rail Detection Rate: {self.results['rail_detection_rate']*100:.1f}%")
        
        print(f"\n🎯 CLASS-WISE IoU RESULTS:")
        print("-" * 60)
        
        # Sort by IoU for better readability
        sorted_classes = sorted(self.results['class_ious'].items(), 
                              key=lambda x: x[1]['mean_iou'], reverse=True)
        
        for class_id, metrics in sorted_classes:
            class_name = self.class_names.get(class_id, f"unknown-{class_id}")
            mean_iou = metrics['mean_iou']
            count = metrics['count']
            marker = "🚂" if class_id in self.rail_classes else "  "
            
            print(f"{marker} Class {class_id:2d} ({class_name:15s}): IoU {mean_iou:.4f} ({count:4d} images)")
        
        print(f"\n🎯 RAIL CLASSES PERFORMANCE:")
        print("-" * 40)
        rail_ious = [self.results['class_ious'][cls]['mean_iou'] 
                    for cls in self.rail_classes 
                    if cls in self.results['class_ious']]
        
        if rail_ious:
            avg_rail_iou = np.mean(rail_ious)
            print(f"🚂 Average Rail IoU: {avg_rail_iou:.4f}")
            print(f"🎯 Target IoU ≥ 0.6: {'✅ ACHIEVED!' if avg_rail_iou >= 0.6 else f'❌ Need {0.6-avg_rail_iou:.4f} more'}")
        
        print(f"\n🔍 IMPORTANT CLASSES SUMMARY:")
        print("-" * 50)
        for class_id in self.important_classes:
            if class_id in self.results['class_ious']:
                class_name = self.class_names[class_id]
                metrics = self.results['class_ious'][class_id]
                precision = self.results['class_precision'].get(class_id, 0)
                recall = self.results['class_recall'].get(class_id, 0)
                f1 = self.results['class_f1'].get(class_id, 0)
                
                print(f"Class {class_id:2d} ({class_name:12s}): IoU {metrics['mean_iou']:.3f}, "
                      f"P {precision:.3f}, R {recall:.3f}, F1 {f1:.3f}")

def main():
    # Paths
    model_path = "/home/mmc-server4/RailSafeNet/models/segformer_b3_improved_best_0.7424.pth"
    images_path = "/home/mmc-server4/Server/Datasets_hdd/rs19_val/jpgs/rs19_val"
    masks_path = "/home/mmc-server4/Server/Datasets_hdd/rs19_val/uint8/rs19_val"
    
    # Initialize evaluator
    evaluator = RailSafeNetEvaluator(model_path, images_path, masks_path)
    
    print("🎯 RailSafeNet Comprehensive Evaluation")
    print("=" * 60)
    
    # Default to quick test for CLI environment
    print("🔄 Running quick evaluation on 100 images...")
    print("💡 To run full evaluation, modify max_images in the script")
    max_images = 100
    
    # Run evaluation
    results = evaluator.evaluate_dataset(max_images=max_images)
    
    # Print summary
    evaluator.print_summary()
    
    print(f"\n📁 Detailed results saved in: evaluation_results/")
    print("🎉 Evaluation completed!")

if __name__ == "__main__":
    main()