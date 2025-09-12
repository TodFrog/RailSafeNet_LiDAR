import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
from pathlib import Path
import wandb
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


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
                A.HorizontalFlip(p=0.5),  # Balanced augmentation
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
        
        # RailSem19 class mapping to sequential 0-18
        class_mapping = {
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 
            11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 255: 255
        }
        
        # Apply class mapping
        mapped_mask = np.full_like(mask, 255, dtype=np.uint8)
        for original_class, new_class in class_mapping.items():
            if original_class != 255:
                mapped_mask[mask == original_class] = new_class
        
        # Apply transforms
        mask_transformed = self.mask_transform(image=mask, mask=mapped_mask)['mask']
        img_transformed = self.transform(image=image)['image']
        
        return img_transformed, torch.tensor(mask_transformed, dtype=torch.long)

def calculate_rail_iou(pred_mask, true_mask, rail_classes=[3, 12, 17, 18]):
    """Calculate IoU specifically for rail-related classes"""
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

def train_segformer_b3_improved():
    """SegFormer B3 개선된 훈련 (WandB Sweep 연동)"""
    
    # 1. WandB 초기화 (sweep이 여기서 config를 주입합니다)
    wandb.init()
    
    # 2. 하드코딩된 config 대신 wandb.config 사용
    config = wandb.config
    
    print(f"🚀 SegFormer B3 Training with Sweep")
    print(f"📊 Config from WandB: {config}")
    
    # wandb.run.name = f"opt-{config.optimizer}_lr-{config.learning_rate:.1e}" # 필요시 실행 이름 설정
    
    # Set device
    device = torch.device('cuda:0') # GPU는 고정
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    
    print(f"🎯 Using device: {device}")
    
    # Create SegFormer B3 model
    print("🧠 Creating SegFormer B3 model...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
        num_labels=19, # num_classes
        ignore_mismatched_sizes=True
    ).to(device)
    
    # 3. Optimizer 선택 로직 추가
    if config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    print(f"⚙️ Optimizer: {config.optimizer}, LR: {config.learning_rate}, WD: {config.weight_decay}")

    # Learning rate scheduler
    # 참고: 데이터셋 크기에 따라 total_steps, warmup_steps를 config 값으로 조정하는 것이 더 정확합니다.
    # 여기서는 기존 로직을 유지합니다.
    num_train_samples = 6800 
    total_steps = config.epochs * (num_train_samples // config.batch_size)
    warmup_steps = config.warmup_epochs * (num_train_samples // config.batch_size)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=total_steps//4, T_mult=1
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
            
            # Create dataset
            dataset = SimpleDataset(PATH_JPGS, PATH_MASKS, config.image_size, phase)
            dataloader = DataLoader(
                dataset, 
                batch_size=config['batch_size'], 
                shuffle=(phase=='Train'), 
                num_workers=3,  # Slightly more workers
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
                        
                        # Resize logits to match mask size
                        logits = F.interpolate(
                            logits, size=masks.shape[-2:], 
                            mode='bilinear', align_corners=False
                        )
                        
                        loss = criterion(logits, masks)
                        
                        if phase == 'Train':
                            loss.backward()
                            # Gradient clipping for stability
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            scheduler.step()
                        
                        running_loss += loss.item()
                        
                        # Calculate rail IoU for validation
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
                        print(f"❌ Batch {batch_idx} failed: {e}")
                        torch.cuda.empty_cache()
                        continue
            
            # Calculate epoch metrics
            epoch_loss = running_loss / len(dataloader)
            current_lr = scheduler.get_last_lr()[0]
            
            if phase == 'Valid':
                avg_rail_iou = np.mean(all_rail_ious) if all_rail_ious else 0.0
                
                # Save best model
                if avg_rail_iou > best_rail_iou:
                    best_rail_iou = avg_rail_iou
                    model_path = os.path.join(PATH_MODELS, f'segformer_b3_improved_best_{best_rail_iou:.4f}.pth')
                    torch.save(model.state_dict(), model_path)
                    print(f"💾 New best model saved: {best_rail_iou:.4f}")
                
                # Log to WandB
                wandb.log({
                    "epoch": epoch,
                    f"{phase.lower()}_loss": epoch_loss,
                    "val_rail_iou": avg_rail_iou,
                    "learning_rate": current_lr,
                    "best_rail_iou": best_rail_iou
                })
                
                print(f"📈 {phase} - Loss: {epoch_loss:.4f}, Rail IoU: {avg_rail_iou:.4f} (Best: {best_rail_iou:.4f})")
                print(f"🎯 Target Rail IoU ≥ 0.6: {'✅ ACHIEVED!' if avg_rail_iou >= 0.6 else f'Progress: {avg_rail_iou/0.6*100:.1f}%'}")
                
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
    
    wandb.finish()
    print(f"\n🎉 SegFormer B3 Improved Training Completed!")
    print(f"🏆 Best Rail IoU: {best_rail_iou:.4f}")
    
    return best_rail_iou

if __name__ == "__main__":
    print("=" * 80)
    print("🚂 RailSafeNet SegFormer B3 Improved Training")
    print("=" * 80)
    
    if torch.cuda.is_available():
        print(f"🎮 GPU (Physical GPU 2): {torch.cuda.get_device_name(0)}")
        print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("❌ CUDA not available!")
        exit(1)
    
    try:
        best_score = train_segformer_b3_improved()
        
        if best_score is not None:
            print(f"\n🎯 Final Results:")
            print(f"🏆 Best Rail IoU: {best_score:.4f}")
            
            if best_score >= 0.6:
                print("✅ Target achieved! Ready for deployment testing.")
                print("🚀 Next: Proceed with hyperparameter sweep for optimization.")
            elif best_score >= 0.4:
                print("⚡ Good progress! Consider batch_size=4 test.")
            else:
                print("⚠️  Needs further optimization. Check batch size and learning rate.")
        else:
            print("❌ Training failed. Check configuration.")
            
    except Exception as e:
        print(f"💥 Training crashed: {e}")
        import traceback
        traceback.print_exc()