from scripts.dataloader_SegFormer import CustomDataset
from scripts.metrics_filtered_cls import compute_map_cls, compute_IoU
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import os
import wandb
from tqdm import tqdm
import time
import copy

# Dataset paths
PATH_JPGS = "/home/mmc-server4/Server/Datasets_hdd/rs19_val/jpgs/rs19_val"
PATH_MASKS = "/home/mmc-server4/Server/Datasets_hdd/rs19_val/uint8/rs19_val"
PATH_MODELS = "/home/mmc-server4/RailSafeNet/models"

os.makedirs(PATH_MODELS, exist_ok=True)

def create_model_b0(num_classes=19):
    """Create SegFormer B0 model - lighter than B3"""
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    model.train()
    return model

def train_single_config():
    """Train with a single, tested configuration"""
    
    # Fixed, tested configuration
    config = {
        'epochs': 30,
        'learning_rate': 0.0001,
        'batch_size': 2,
        'image_size': 512,
        'num_classes': 19,  # RailSem19 has classes 0-18
        'optimizer': 'adamw',
        'weight_decay': 0.01
    }
    
    # Initialize WandB
    wandb.init(
        project="RailSafeNet-SegFormer-B3",
        name="segformer-b0-test",
        config=config
    )
    
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    
    print(f"Creating SegFormer B0 model with {config['num_classes']} classes...")
    model = create_model_b0(config['num_classes'])
    model = model.to(device)
    
    print(f"Model created successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Loss function
    loss_function = nn.CrossEntropyLoss(ignore_index=255)
    
    best_model = None
    best_rail_iou = 0.0
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 50)
        
        epoch_metrics = {'train_loss': 0, 'val_loss': 0}
        
        for phase in ['Train', 'Valid']:
            if phase == 'Train':
                model.train()
            else:
                model.eval()
            
            # Create dataset and dataloader
            image_processor = SegformerImageProcessor(do_reduce_labels=False)
            dataset = CustomDataset(
                PATH_JPGS, PATH_MASKS, image_processor, 
                [config['image_size'], config['image_size']], 
                subset=phase, val_fraction=0.2
            )
            dataloader = DataLoader(
                dataset, batch_size=config['batch_size'], 
                shuffle=(phase == 'Train'), drop_last=True, num_workers=2
            )
            
            running_loss = 0.0
            classes_IoU = {}
            
            with torch.set_grad_enabled(phase == 'Train'):
                for i, (inputs, masks) in enumerate(tqdm(dataloader, desc=f"{phase} Epoch {epoch+1}")):
                    inputs = inputs.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    
                    if phase == 'Train':
                        optimizer.zero_grad()
                    
                    outputs = model(inputs)
                    logits = outputs.logits
                    
                    # Upsample to match mask size
                    upsampled_logits = F.interpolate(
                        logits, size=masks.shape[-2:], 
                        mode="bilinear", align_corners=False
                    )
                    
                    loss = loss_function(upsampled_logits, masks)
                    
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
                    
                    running_loss += loss.item()
                    
                    # Compute IoU for validation
                    if phase == 'Valid':
                        predicted_masks = upsampled_logits
                        gt_masks = masks.cpu().detach().numpy()
                        
                        for prediction, gt in zip(predicted_masks, gt_masks):
                            prediction = F.softmax(prediction, dim=0).cpu().detach().numpy()
                            prediction = np.argmax(prediction, axis=0).astype(np.uint8)
                            
                            # Compute IoU
                            try:
                                _, _, _, _, classes_IoU = compute_IoU(gt, prediction, classes_IoU)
                            except:
                                pass
                    
                    # Clear cache periodically
                    if i % 10 == 0:
                        torch.cuda.empty_cache()
            
            # Compute epoch metrics
            epoch_loss = running_loss / len(dataloader)
            epoch_metrics[f'{phase.lower()}_loss'] = epoch_loss
            
            if phase == 'Valid':
                # Compute class-wise IoU
                rail_ious = []
                if classes_IoU:
                    for cls, value in classes_IoU.items():
                        classes_IoU[cls] = np.divide(value[0], value[1])
                    
                    # Get rail-related IoUs (classes 3, 12, 17, 18)
                    for rail_class in [3, 12, 17, 18]:  # tram-track, rail-track, rail-raised, rail-embedded
                        if rail_class in classes_IoU:
                            iou_val = classes_IoU[rail_class][0]
                            if iou_val > 0:
                                rail_ious.append(iou_val)
                                print(f"  Class {rail_class} IoU: {iou_val:.4f}")
                
                avg_rail_iou = np.mean(rail_ious) if rail_ious else 0.0
                epoch_metrics['val_rail_iou'] = avg_rail_iou
                
                # Save best model
                if avg_rail_iou > best_rail_iou:
                    best_rail_iou = avg_rail_iou
                    best_model = copy.deepcopy(model.state_dict())
                    print(f"  New best rail IoU: {best_rail_iou:.4f}")
                
                # Update scheduler
                scheduler.step(avg_rail_iou)
        
        # Log to WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_metrics['train_loss'],
            "val_loss": epoch_metrics['val_loss'],
            "val_rail_iou": epoch_metrics.get('val_rail_iou', 0),
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        print(f"Train Loss: {epoch_metrics['train_loss']:.4f} | "
              f"Val Loss: {epoch_metrics['val_loss']:.4f} | "
              f"Rail IoU: {epoch_metrics.get('val_rail_iou', 0):.4f}")
    
    # Save final model
    if best_model is not None:
        model.load_state_dict(best_model)
        torch.save(model, os.path.join(PATH_MODELS, f'segformer_b0_best_rail_{best_rail_iou:.4f}.pth'))
        print(f"Model saved with best rail IoU: {best_rail_iou:.4f}")
    
    wandb.finish()
    return best_rail_iou

if __name__ == "__main__":
    print("Starting SegFormer B0 Training for Rail Detection")
    print(f"Dataset - Images: {PATH_JPGS}")
    print(f"Dataset - Masks: {PATH_MASKS}")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("CUDA not available!")
        exit(1)
    
    try:
        best_score = train_single_config()
        print(f"\nTraining completed! Best rail IoU: {best_score:.4f}")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()