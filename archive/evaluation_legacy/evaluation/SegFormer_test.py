"""SegFormer 세그멘테이션 모델의 수동/오프라인 평가 스크립트."""

import numpy as np
import pandas as pd
import torch
import cv2
import os
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
import json
from pathlib import Path
from tqdm import tqdm

# Import custom modules
from src.common.metrics_filtered_cls import compute_map_cls, compute_IoU, image_morpho

# Configuration
PATH_jpgs = '/home/mmc-server4/Server/Datasets_hdd/rs19_val/jpgs/rs19_val'
PATH_masks = '/home/mmc-server4/Server/Datasets_hdd/rs19_val/uint8/rs19_val'
PATH_model = '/home/mmc-server4/RailSafeNet/assets/models_pretrained/segformer/production/segformer_b3_transfer_best_rail_0.7791.pth'

# Output directories
OUTPUT_DIR = 'evaluation_results'
GRAPHS_DIR = os.path.join(OUTPUT_DIR, 'graphs')
SAMPLES_DIR = os.path.join(OUTPUT_DIR, 'sample_images')
GOOD_SAMPLES_DIR = os.path.join(SAMPLES_DIR, 'good_predictions')
BAD_SAMPLES_DIR = os.path.join(SAMPLES_DIR, 'bad_predictions')

# Create directories
for dir_path in [OUTPUT_DIR, GRAPHS_DIR, GOOD_SAMPLES_DIR, BAD_SAMPLES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 13-class system color mapping for rail segmentation
rs19_label2bgr = {
    0: (128, 64, 128),    # class 0
    1: (244, 35, 232),    # rail road/platform
    2: (70, 70, 70),      # class 2
    3: (102, 102, 156),   # class 3
    4: (0, 255, 0),       # rail track (bright green)
    5: (153, 153, 153),   # class 5
    6: (107, 142, 35),    # vegetation/sky
    7: (220, 220, 0),     # class 7
    8: (190, 153, 153),   # class 8
    9: (255, 255, 0),     # rail road (bright yellow)
    10: (70, 130, 180),   # class 10
    11: (220, 20, 60),    # class 11
    12: (255, 255, 255),  # background (white)
}

class RailSegmentationEvaluator:
    """세그멘테이션 예측, 정량 지표 계산, 샘플 시각화를 묶어 제공한다."""

    def __init__(self, model_path, image_size=[1024, 1024]):
        self.model_path = model_path
        self.image_size = image_size
        self.model = self.load_model()
        self.results = {
            'per_image_metrics': [],
            'class_metrics': {},
            'overall_metrics': {},
            'good_samples': [],
            'bad_samples': []
        }
        
    def load_model(self):
        """체크포인트 형식에 맞춰 평가용 SegFormer 모델을 로드한다."""

        from transformers import SegformerForSemanticSegmentation
        
        checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
        
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
                num_labels=13,
                ignore_mismatched_sizes=True
            )
            model.load_state_dict(state_dict, strict=False)
        else:
            model = checkpoint
        
        model = model.cpu()
        model.eval()
        return model
    
    def load_and_preprocess(self, filename):
        """이미지와 GT 마스크를 읽고 평가 입력 형식으로 전처리한다.

        학습 시 사용한 클래스 매핑과 동일한 변환을 적용해 평가와 학습 조건을 맞춘다.
        """

        # Image preprocessing
        transform_img = A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1], interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ])
        
        # Mask preprocessing
        transform_mask = A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1], interpolation=cv2.INTER_NEAREST),
            ToTensorV2(p=1.0),
        ])
        
        # Load image and mask
        image_path = os.path.join(PATH_jpgs, filename)
        mask_path = os.path.join(PATH_masks, filename.replace('.jpg', '.png'))
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        image_tensor = transform_img(image=image)['image'].unsqueeze(0)
        mask_tensor = transform_mask(image=mask)['image']
        
        # Apply the same class mapping as used in training
        mask_tensor = self.apply_class_mapping(mask_tensor)
        
        return image_tensor, mask_tensor, image, mask
    
    def apply_class_mapping(self, mask):
        """학습 스크립트와 동일한 19->13 클래스 매핑을 적용한다.

        rail 관련 클래스는 1, 4, 9 축으로 모으고, 나머지 다수 클래스는 배경 12로 보낸다.
        정확한 도메인 근거는 추가 문서화가 필요하다.
        """
        # Convert to numpy for easier manipulation
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy().astype(np.uint8)
        else:
            mask_np = mask.astype(np.uint8)
        
        # Apply the same class mapping as in training
        class_mapping = {
            # Rails mapping - most important
            3: 4,    # wall -> main rail track
            12: 9,   # rider -> secondary rail track
            17: 4,   # motorcycle -> main rail track
            18: 9,   # bicycle -> secondary rail track

            # Infrastructure
            0: 1,    # road -> rail road/platform
            1: 1,    # sidewalk -> rail road/platform
            9: 1,    # terrain -> rail road/platform

            # Environment
            8: 6,    # vegetation -> vegetation/sky
            10: 6,   # sky -> vegetation/sky

            # Objects/Background
            2: 12, 4: 12, 5: 12, 6: 12, 7: 12, 11: 12,
            13: 12, 14: 12, 15: 12, 16: 12,  # all -> background
        }

        # Apply mapping
        mapped_mask = np.full_like(mask_np, 255, dtype=np.uint8)
        for src_class, tgt_class in class_mapping.items():
            mapped_mask[mask_np == src_class] = tgt_class

        # Set unmapped classes to background
        mapped_mask[mapped_mask == 255] = 12
        
        # Convert back to tensor
        return torch.from_numpy(mapped_mask).long()
    
    def predict(self, image_tensor):
        """모델 추론 후 업샘플링과 morphology 후처리를 적용한다."""

        with torch.no_grad():
            outputs = self.model(image_tensor)
            logits = outputs.logits
            
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=self.image_size,
                mode="bilinear",
                align_corners=False
            )
            
            confidence_scores = F.softmax(upsampled_logits, dim=1).cpu().numpy().squeeze()
            predicted_mask = np.argmax(confidence_scores, axis=0).astype(np.uint8)
            predicted_mask = image_morpho(predicted_mask)
            
            return predicted_mask, confidence_scores
    
    def compute_metrics(self, gt_mask, pred_mask):
        """Compute comprehensive metrics focusing on rail classes (4 and 9)"""
        gt_mask = np.array(gt_mask.cpu().numpy(), dtype=np.uint8)
        gt_mask = np.clip(gt_mask, 0, 12)
        
        # Overall metrics
        overall_acc = np.mean(gt_mask == pred_mask)
        
        # Class-specific metrics
        class_metrics = {}
        rail_classes = [4, 9]  # Focus on rail track and rail road
        
        for class_id in range(13):
            # Binary masks for current class
            gt_binary = (gt_mask == class_id).astype(int)
            pred_binary = (pred_mask == class_id).astype(int)
            
            # Compute metrics
            tp = np.sum((gt_binary == 1) & (pred_binary == 1))
            tn = np.sum((gt_binary == 0) & (pred_binary == 0))
            fp = np.sum((gt_binary == 0) & (pred_binary == 1))
            fn = np.sum((gt_binary == 1) & (pred_binary == 0))
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            
            class_metrics[class_id] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'iou': iou,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn
            }
        
        # Rail-specific metrics (classes 4 and 9)
        rail_metrics = {}
        for rail_class in rail_classes:
            rail_metrics[f'rail_class_{rail_class}'] = class_metrics[rail_class]
        
        return {
            'overall_accuracy': overall_acc,
            'class_metrics': class_metrics,
            'rail_metrics': rail_metrics
        }
    
    def evaluate_sample_quality(self, metrics, rail_threshold=0.7):
        """Determine if a sample is good or bad based on rail class performance"""
        rail_4_iou = metrics['rail_metrics']['rail_class_4']['iou']
        rail_9_iou = metrics['rail_metrics']['rail_class_9']['iou']
        
        # Consider sample good if both rail classes perform well
        avg_rail_iou = (rail_4_iou + rail_9_iou) / 2
        return avg_rail_iou >= rail_threshold
    
    def save_sample_visualization(self, filename, original_img, gt_mask, pred_mask, metrics, is_good_sample):
        """Save visualization of prediction results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Ground truth
        gt_colored = self.colorize_mask(gt_mask)
        axes[0, 1].imshow(gt_colored)
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        # Prediction
        pred_colored = self.colorize_mask(pred_mask)
        axes[1, 0].imshow(pred_colored)
        axes[1, 0].set_title('Prediction')
        axes[1, 0].axis('off')
        
        # Metrics text
        rail_4_metrics = metrics['rail_metrics']['rail_class_4']
        rail_9_metrics = metrics['rail_metrics']['rail_class_9']
        
        metrics_text = f"""
        Overall Accuracy: {metrics['overall_accuracy']:.3f}
        
        Rail Track (Class 4):
        IoU: {rail_4_metrics['iou']:.3f}
        Precision: {rail_4_metrics['precision']:.3f}
        Recall: {rail_4_metrics['recall']:.3f}
        F1: {rail_4_metrics['f1']:.3f}
        
        Rail Road (Class 9):
        IoU: {rail_9_metrics['iou']:.3f}
        Precision: {rail_9_metrics['precision']:.3f}
        Recall: {rail_9_metrics['recall']:.3f}
        F1: {rail_9_metrics['f1']:.3f}
        """
        
        axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes,
                        verticalalignment='top', fontsize=10, fontfamily='monospace')
        axes[1, 1].axis('off')
        
        # Save
        sample_type = 'good' if is_good_sample else 'bad'
        save_dir = GOOD_SAMPLES_DIR if is_good_sample else BAD_SAMPLES_DIR
        save_path = os.path.join(save_dir, f'{sample_type}_{filename}')
        
        plt.suptitle(f'{sample_type.title()} Sample: {filename}', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def colorize_mask(self, mask):
        """Convert mask to colored visualization"""
        mask = np.array(mask)
        colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        for class_id in range(13):
            class_pixels = mask == class_id
            color = rs19_label2bgr[class_id]
            colored[class_pixels] = color
            
        return colored
    
    def run_evaluation(self, max_samples=None):
        """Run comprehensive evaluation"""
        print("Starting SegFormer Rail Segmentation Evaluation...")
        print(f"Model: {self.model_path}")
        print(f"Image size: {self.image_size}")
        print("-" * 60)
        
        image_files = [f for f in os.listdir(PATH_jpgs) if f.endswith('.jpg')]
        if max_samples:
            image_files = image_files[:max_samples]
        
        good_samples_count = 0
        bad_samples_count = 0
        
        for filename in tqdm(image_files, desc="Processing images"):
            try:
                # Load and preprocess
                image_tensor, gt_mask_tensor, original_img, gt_mask = self.load_and_preprocess(filename)
                
                # Predict
                pred_mask, confidence_scores = self.predict(image_tensor)
                
                # Compute metrics
                metrics = self.compute_metrics(gt_mask_tensor, pred_mask)
                
                # Store per-image results
                self.results['per_image_metrics'].append({
                    'filename': filename,
                    'metrics': metrics
                })
                
                # Evaluate sample quality
                is_good_sample = self.evaluate_sample_quality(metrics)
                
                # Save sample visualizations (limit to 5 each)
                if is_good_sample and good_samples_count < 5:
                    save_path = self.save_sample_visualization(filename, original_img, gt_mask, pred_mask, metrics, True)
                    self.results['good_samples'].append(save_path)
                    good_samples_count += 1
                elif not is_good_sample and bad_samples_count < 5:
                    save_path = self.save_sample_visualization(filename, original_img, gt_mask, pred_mask, metrics, False)
                    self.results['bad_samples'].append(save_path)
                    bad_samples_count += 1
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        # Aggregate results
        self.aggregate_results()
        
        # Generate reports
        self.generate_reports()
        
        print(f"\nEvaluation completed!")
        print(f"Total samples processed: {len(self.results['per_image_metrics'])}")
        print(f"Good samples saved: {len(self.results['good_samples'])}")
        print(f"Bad samples saved: {len(self.results['bad_samples'])}")
        print(f"Results saved in: {OUTPUT_DIR}")
    
    def aggregate_results(self):
        """Aggregate metrics across all images"""
        all_metrics = [item['metrics'] for item in self.results['per_image_metrics']]
        
        # Overall metrics
        overall_accuracies = [m['overall_accuracy'] for m in all_metrics]
        self.results['overall_metrics']['mean_accuracy'] = np.mean(overall_accuracies)
        self.results['overall_metrics']['std_accuracy'] = np.std(overall_accuracies)
        
        # Class-wise aggregation
        for class_id in range(13):
            class_ious = [m['class_metrics'][class_id]['iou'] for m in all_metrics]
            class_f1s = [m['class_metrics'][class_id]['f1'] for m in all_metrics]
            class_precisions = [m['class_metrics'][class_id]['precision'] for m in all_metrics]
            class_recalls = [m['class_metrics'][class_id]['recall'] for m in all_metrics]
            
            self.results['class_metrics'][class_id] = {
                'mean_iou': np.mean(class_ious),
                'std_iou': np.std(class_ious),
                'mean_f1': np.mean(class_f1s),
                'std_f1': np.std(class_f1s),
                'mean_precision': np.mean(class_precisions),
                'std_precision': np.std(class_precisions),
                'mean_recall': np.mean(class_recalls),
                'std_recall': np.std(class_recalls)
            }
    
    def generate_reports(self):
        """Generate comprehensive reports and visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Performance summary graph
        self.plot_class_performance()
        
        # 2. Rail classes focus graph
        self.plot_rail_performance()
        
        # 3. Confusion matrix for rail classes
        self.plot_rail_confusion_matrix()
        
        # 4. JSON report
        self.save_json_report(timestamp)
        
        # 5. Text summary
        self.generate_text_summary(timestamp)
    
    def plot_class_performance(self):
        """Plot performance metrics for all classes"""
        classes = list(range(13))
        class_names = [f"Class {i}" for i in classes]
        
        # Highlight rail classes
        colors = ['red' if i in [4, 9] else 'skyblue' for i in classes]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # IoU
        ious = [self.results['class_metrics'][i]['mean_iou'] for i in classes]
        iou_stds = [self.results['class_metrics'][i]['std_iou'] for i in classes]
        axes[0, 0].bar(class_names, ious, yerr=iou_stds, color=colors, alpha=0.7)
        axes[0, 0].set_title('IoU by Class')
        axes[0, 0].set_ylabel('IoU')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1 Score
        f1s = [self.results['class_metrics'][i]['mean_f1'] for i in classes]
        f1_stds = [self.results['class_metrics'][i]['std_f1'] for i in classes]
        axes[0, 1].bar(class_names, f1s, yerr=f1_stds, color=colors, alpha=0.7)
        axes[0, 1].set_title('F1 Score by Class')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Precision
        precisions = [self.results['class_metrics'][i]['mean_precision'] for i in classes]
        precision_stds = [self.results['class_metrics'][i]['std_precision'] for i in classes]
        axes[1, 0].bar(class_names, precisions, yerr=precision_stds, color=colors, alpha=0.7)
        axes[1, 0].set_title('Precision by Class')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Recall
        recalls = [self.results['class_metrics'][i]['mean_recall'] for i in classes]
        recall_stds = [self.results['class_metrics'][i]['std_recall'] for i in classes]
        axes[1, 1].bar(class_names, recalls, yerr=recall_stds, color=colors, alpha=0.7)
        axes[1, 1].set_title('Recall by Class')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.suptitle('SegFormer Performance by Class (Rail Classes in Red)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPHS_DIR, 'class_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_rail_performance(self):
        """Focus plot on rail classes performance"""
        rail_classes = [4, 9]
        rail_names = ['Rail Track (Class 4)', 'Rail Road (Class 9)']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics = ['mean_iou', 'mean_f1', 'mean_precision', 'mean_recall']
        titles = ['IoU', 'F1 Score', 'Precision', 'Recall']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            values = [self.results['class_metrics'][i][metric] for i in rail_classes]
            std_values = [self.results['class_metrics'][i][metric.replace('mean_', 'std_')] for i in rail_classes]
            
            bars = ax.bar(rail_names, values, yerr=std_values, color=['green', 'gold'], alpha=0.8)
            ax.set_title(f'Rail Classes {title}')
            ax.set_ylabel(title)
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle('Rail Classes Performance Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPHS_DIR, 'rail_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_rail_confusion_matrix(self):
        """Plot confusion matrix focusing on rail classes"""
        all_gt = []
        all_pred = []
        
        # Collect all predictions for rail classes
        for item in self.results['per_image_metrics']:
            filename = item['filename']
            
            # Reload masks for confusion matrix
            try:
                _, gt_mask_tensor, _, _ = self.load_and_preprocess(filename)
                image_tensor, _, _, _ = self.load_and_preprocess(filename)
                pred_mask, _ = self.predict(image_tensor)
                
                gt_mask = np.array(gt_mask_tensor.cpu().numpy(), dtype=np.uint8)
                gt_mask = np.clip(gt_mask, 0, 12)
                
                # Focus on rail pixels only
                rail_pixels_gt = np.isin(gt_mask, [4, 9])
                rail_pixels_pred = np.isin(pred_mask, [4, 9])
                
                if np.any(rail_pixels_gt) or np.any(rail_pixels_pred):
                    # Map to binary: 0=non-rail, 1=rail_track(4), 2=rail_road(9)
                    gt_rail = np.zeros_like(gt_mask)
                    pred_rail = np.zeros_like(pred_mask)
                    
                    gt_rail[gt_mask == 4] = 1
                    gt_rail[gt_mask == 9] = 2
                    pred_rail[pred_mask == 4] = 1
                    pred_rail[pred_mask == 9] = 2
                    
                    all_gt.extend(gt_rail.flatten())
                    all_pred.extend(pred_rail.flatten())
                    
            except Exception as e:
                continue
        
        if all_gt and all_pred:
            # Generate confusion matrix
            cm = confusion_matrix(all_gt, all_pred, labels=[0, 1, 2])
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Background', 'Rail Track (4)', 'Rail Road (9)'],
                       yticklabels=['Background', 'Rail Track (4)', 'Rail Road (9)'])
            plt.title('Confusion Matrix: Rail Classes')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(GRAPHS_DIR, 'rail_confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_json_report(self, timestamp):
        """Save detailed JSON report"""
        report = {
            'timestamp': timestamp,
            'model_path': self.model_path,
            'image_size': self.image_size,
            'total_samples': len(self.results['per_image_metrics']),
            'overall_metrics': self.results['overall_metrics'],
            'class_metrics': self.results['class_metrics'],
            'rail_specific_summary': {
                'rail_track_4': self.results['class_metrics'][4],
                'rail_road_9': self.results['class_metrics'][9]
            },
            'sample_files': {
                'good_samples': self.results['good_samples'],
                'bad_samples': self.results['bad_samples']
            }
        }
        
        with open(os.path.join(OUTPUT_DIR, f'evaluation_report_{timestamp}.json'), 'w') as f:
            json.dump(report, f, indent=2)
    
    def generate_text_summary(self, timestamp):
        """Generate human-readable summary report"""
        rail_4_metrics = self.results['class_metrics'][4]
        rail_9_metrics = self.results['class_metrics'][9]
        
        summary = f"""
SegFormer Rail Segmentation Model Evaluation Report
==================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {self.model_path}

OVERALL PERFORMANCE
-------------------
Mean Accuracy: {self.results['overall_metrics']['mean_accuracy']:.3f} ± {self.results['overall_metrics']['std_accuracy']:.3f}
Total Samples: {len(self.results['per_image_metrics'])}

RAIL CLASS PERFORMANCE (PRIMARY FOCUS)
--------------------------------------
Rail Track (Class 4):
  - IoU: {rail_4_metrics['mean_iou']:.3f} ± {rail_4_metrics['std_iou']:.3f}
  - F1 Score: {rail_4_metrics['mean_f1']:.3f} ± {rail_4_metrics['std_f1']:.3f}
  - Precision: {rail_4_metrics['mean_precision']:.3f} ± {rail_4_metrics['std_precision']:.3f}
  - Recall: {rail_4_metrics['mean_recall']:.3f} ± {rail_4_metrics['std_recall']:.3f}

Rail Road (Class 9):
  - IoU: {rail_9_metrics['mean_iou']:.3f} ± {rail_9_metrics['std_iou']:.3f}
  - F1 Score: {rail_9_metrics['mean_f1']:.3f} ± {rail_9_metrics['std_f1']:.3f}
  - Precision: {rail_9_metrics['mean_precision']:.3f} ± {rail_9_metrics['std_precision']:.3f}
  - Recall: {rail_9_metrics['mean_recall']:.3f} ± {rail_9_metrics['std_recall']:.3f}

SAMPLE ANALYSIS
---------------
Good Prediction Samples: {len(self.results['good_samples'])}
Poor Prediction Samples: {len(self.results['bad_samples'])}

FILES GENERATED
---------------
- Graphs: {GRAPHS_DIR}/
  - class_performance.png
  - rail_performance.png
  - rail_confusion_matrix.png
- Sample Images: {SAMPLES_DIR}/
  - Good samples: {GOOD_SAMPLES_DIR}/
  - Bad samples: {BAD_SAMPLES_DIR}/
- Reports: {OUTPUT_DIR}/
  - evaluation_report_{timestamp}.json

CONCLUSION
----------
This evaluation focuses on the model's ability to segment rail infrastructure,
specifically Rail Track (Class 4) and Rail Road (Class 9). The results show
the model's performance in critical rail safety applications.
        """
        
        with open(os.path.join(OUTPUT_DIR, f'evaluation_summary_{timestamp}.txt'), 'w') as f:
            f.write(summary)
        
        print(summary)


def main():
    # Initialize evaluator
    evaluator = RailSegmentationEvaluator(
        model_path=PATH_model,
        image_size=[512, 512]
    )
    
    # Run evaluation (limit to reasonable number for testing)
    evaluator.run_evaluation(max_samples=100)  # Remove or increase for full evaluation


if __name__ == "__main__":
    main()
