import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
from pathlib import Path
import wandb
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import itertools

# Dataset paths
PATH_JPGS = "/home/mmc-server4/Server/Datasets_hdd/rs19_val/jpgs/rs19_val"
PATH_MASKS = "/home/mmc-server4/Server/Datasets_hdd/rs19_val/uint8/rs19_val"
PATH_MODELS = "/home/mmc-server4/RailSafeNet/models"

os.makedirs(PATH_MODELS, exist_ok=True)

class SimpleDataset(Dataset):
    def __init__(self, image_folder, mask_folder, image_size, subset, val_fraction=0.2):
        self.image_folder = Path(image_folder)
        self.mask_folder = Path(mask_folder)
        self.image_size = image_size
        
        # Get all files
        self.image_list = sorted(list(self.image_folder.glob("*.jpg")))
        self.mask_list = sorted(list(self.mask_folder.glob("*.png")))
        
        # Split train/val
        split_idx = int(len(self.image_list) * (1 - val_fraction))
        if subset == 'Train':
            self.image_list = self.image_list[:split_idx]
            self.mask_list = self.mask_list[:split_idx]
        else:  # Valid
            self.image_list = self.image_list[split_idx:]
            self.mask_list = self.mask_list[split_idx:]
        
        # Optimized transforms for B3
        if subset == 'Train':
            self.transform = A.Compose([
                A.Resize(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        self.mask_transform = A.Compose([
            A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_NEAREST),
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # Load image and mask
        image = cv2.imread(str(self.image_list[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(self.mask_list[idx]), cv2.IMREAD_GRAYSCALE)
        
        # RailSem19 class mapping
        class_mapping = {
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 
            11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 255: 255
        }
        
        mapped_mask = np.full_like(mask, 255, dtype=np.uint8)
        for original_class, new_class in class_mapping.items():
            if original_class != 255:
                mapped_mask[mask == original_class] = new_class
        
        # Apply transforms
        mask_transformed = self.mask_transform(image=mask, mask=mapped_mask)['mask']
        img_transformed = self.transform(image=image)['image']
        
        return img_transformed, torch.tensor(mask_transformed, dtype=torch.long)

def calculate_rail_iou(pred_mask, true_mask, rail_classes=[3, 12, 17, 18]):
    """Calculate IoU for rail classes"""
    rail_ious = []
    for rail_class in rail_classes:
        pred_binary = (pred_mask == rail_class).float()
        true_binary = (true_mask == rail_class).float()
        
        intersection = (pred_binary * true_binary).sum()
        union = pred_binary.sum() + true_binary.sum() - intersection
        
        if union > 0:
            iou = intersection / union
            rail_ious.append(iou.item())
    
    return np.mean(rail_ious) if rail_ious else 0.0

def train_with_config(config, run_name):
    """Train SegFormer B3 with given configuration"""
    
    print(f"🚀 SegFormer B3 Manual Sweep - {run_name}")
    print(f"📊 Config: {config}")
    
    # Initialize WandB
    wandb.init(
        project="RailSafeNet-SegFormer-B3-Sweep", 
        name=run_name,
        config=config,
        tags=['b3', 'manual_sweep', 'gpu2'],
        reinit=True
    )
    
    # Set device
    device = torch.device('cuda:0')  # GPU 2 becomes index 0 with CUDA_VISIBLE_DEVICES=2
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    
    print(f"🎯 Using device: {device}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Create model
    try:
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
            num_labels=19,
            ignore_mismatched_sizes=True
        )
        model = model.to(device)
        print(f"✅ Model created! Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        wandb.finish()
        return None
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    if config['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'] * 1700
        )
    elif config['scheduler'] == 'cosine_warm':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config['epochs']//4, T_mult=1
        )
    elif config['scheduler'] == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=config['learning_rate'],
            epochs=config['epochs'],
            steps_per_epoch=1700
        )
    else:  # step
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config['epochs']//3, gamma=0.5
        )
    
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    best_rail_iou = 0.0
    
    for epoch in range(config['epochs']):
        print(f"\n📊 Epoch {epoch+1}/{config['epochs']}")
        print("-" * 60)
        
        for phase in ['Train', 'Valid']:
            if phase == 'Train':
                model.train()
                print("🏋️  Training phase...")
            else:
                model.eval()
                print("🔍 Validation phase...")
            
            dataset = SimpleDataset(PATH_JPGS, PATH_MASKS, config['image_size'], phase)
            dataloader = DataLoader(
                dataset, 
                batch_size=config['batch_size'], 
                shuffle=(phase=='Train'), 
                num_workers=4,
                pin_memory=True
            )
            
            running_loss = 0.0
            all_rail_ious = []
            
            with torch.set_grad_enabled(phase == 'Train'):
                for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc=f"{phase}")):
                    try:
                        images = images.to(device, non_blocking=True)
                        masks = masks.to(device, non_blocking=True)
                        
                        if phase == 'Train':
                            optimizer.zero_grad()
                        
                        # Forward pass
                        outputs = model(images)
                        logits = outputs.logits
                        
                        logits = F.interpolate(
                            logits, size=masks.shape[-2:], 
                            mode='bilinear', align_corners=False
                        )
                        
                        loss = criterion(logits, masks)
                        
                        if phase == 'Train':
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            if config['scheduler'] in ['cosine', 'cosine_warm', 'onecycle']:
                                scheduler.step()
                        
                        running_loss += loss.item()
                        
                        # Calculate rail IoU
                        if phase == 'Valid':
                            pred_masks = torch.argmax(logits, dim=1)
                            for pred_mask, true_mask in zip(pred_masks, masks):
                                rail_iou = calculate_rail_iou(pred_mask.cpu(), true_mask.cpu())
                                if rail_iou > 0:
                                    all_rail_ious.append(rail_iou)
                        
                        # Memory management
                        if batch_idx % 30 == 0:
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        print(f"❌ Error at batch {batch_idx}: {e}")
                        torch.cuda.empty_cache()
                        continue
            
            # Step scheduler for step-based schedulers
            if phase == 'Train' and config['scheduler'] == 'step':
                scheduler.step()
            
            # Calculate metrics
            epoch_loss = running_loss / len(dataloader)
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
            
            if phase == 'Valid':
                avg_rail_iou = np.mean(all_rail_ious) if all_rail_ious else 0.0
                
                if avg_rail_iou > best_rail_iou:
                    best_rail_iou = avg_rail_iou
                    model_path = os.path.join(PATH_MODELS, f'segformer_b3_manual_sweep_{run_name}_best_{best_rail_iou:.4f}.pth')
                    torch.save(model.state_dict(), model_path)
                    print(f"💾 New best model: {best_rail_iou:.4f}")
                
                # Log to WandB
                wandb.log({
                    "epoch": epoch,
                    f"{phase.lower()}_loss": epoch_loss,
                    "val_rail_iou": avg_rail_iou,
                    "learning_rate": current_lr,
                    "best_rail_iou": best_rail_iou
                })
                
                print(f"📈 {phase} - Loss: {epoch_loss:.4f}, Rail IoU: {avg_rail_iou:.4f} (Best: {best_rail_iou:.4f})")
                print(f"🎯 Target Progress: {avg_rail_iou/0.6*100:.1f}%")
                
            else:
                wandb.log({
                    f"{phase.lower()}_loss": epoch_loss,
                    "learning_rate": current_lr
                })
                print(f"📈 {phase} - Loss: {epoch_loss:.4f}, LR: {current_lr:.2e}")
        
        print(f"✅ Epoch {epoch+1} completed!")
        
        # Early stopping if target achieved
        if best_rail_iou >= 0.6:
            print(f"🎯 Target Rail IoU ≥ 0.6 achieved! Stopping early.")
            break
    
    # Log final result to wandb
    wandb.log({"final_rail_iou": best_rail_iou})
    wandb.finish()
    print(f"🎉 {run_name} Completed! Final Rail IoU: {best_rail_iou:.4f}")
    return best_rail_iou

def run_manual_sweep():
    """Run manual hyperparameter sweep"""
    
    # Define parameter combinations
    learning_rates = [0.00005, 0.00007, 0.00008, 0.00009, 0.0001]
    schedulers = ['cosine_warm', 'cosine', 'onecycle']
    batch_sizes = [2, 4]
    
    base_config = {
        'epochs': 30,
        'image_size': 512,
        'num_classes': 19,
        'device': 'cuda:0',
        'weight_decay': 0.01
    }
    
    results = []
    best_overall = 0.0
    
    print("🚀 Starting Manual SegFormer B3 Hyperparameter Sweep")
    print(f"📊 Total combinations: {len(learning_rates) * len(schedulers) * len(batch_sizes)}")
    
    for lr, scheduler, batch_size in itertools.product(learning_rates, schedulers, batch_sizes):
        config = base_config.copy()
        config.update({
            'learning_rate': lr,
            'scheduler': scheduler,
            'batch_size': batch_size
        })
        
        run_name = f"b3_lr{lr:.5f}_sch{scheduler}_bs{batch_size}"
        
        try:
            result = train_with_config(config, run_name)
            results.append((config, result))
            
            if result and result > best_overall:
                best_overall = result
                print(f"🏆 New best overall: {best_overall:.4f} with {run_name}")
                
            # Clear memory between runs
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Failed {run_name}: {e}")
            torch.cuda.empty_cache()
            continue
    
    # Summary
    print("\n" + "="*80)
    print("🎉 Manual Sweep Completed!")
    print(f"🏆 Best Overall Rail IoU: {best_overall:.4f}")
    print("="*80)
    
    return results

if __name__ == "__main__":
    print("="*80)
    print("🚂 RailSafeNet SegFormer B3 Manual Hyperparameter Sweep")
    print("="*80)
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    try:
        results = run_manual_sweep()
        
    except Exception as e:
        print(f"💥 Sweep crashed: {e}")
        import traceback
        traceback.print_exc()