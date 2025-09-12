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
        
        # Simple transforms
        if subset == 'Train':
            self.transform = A.Compose([
                A.Resize(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
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
        
        # Critical: Map classes to sequential 0-N range
        # RailSem19 has sparse classes, we need to map them to dense classes
        class_mapping = {
            0: 0,    # road
            1: 1,    # sidewalk  
            2: 2,    # construction
            3: 3,    # tram-track (IMPORTANT for rail detection)
            4: 4,    # fence
            5: 5,    # pole
            6: 6,    # traffic-light
            7: 7,    # traffic-sign
            8: 8,    # vegetation
            9: 9,    # terrain
            10: 10,  # sky
            11: 11,  # human
            12: 12,  # rail-track (MOST IMPORTANT for rail detection)
            13: 13,  # car
            14: 14,  # truck
            15: 15,  # trackbed
            16: 16,  # on-rails
            17: 17,  # rail-raised (IMPORTANT for rail detection)
            18: 18,  # rail-embedded (IMPORTANT for rail detection)
            255: 255 # ignore
        }
        
        # Apply class mapping
        mapped_mask = np.full_like(mask, 255, dtype=np.uint8)  # Default to ignore
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
        # Binary masks for this class
        pred_binary = (pred_mask == rail_class).float()
        true_binary = (true_mask == rail_class).float()
        
        # Calculate IoU
        intersection = (pred_binary * true_binary).sum()
        union = pred_binary.sum() + true_binary.sum() - intersection
        
        if union > 0:
            iou = intersection / union
            rail_ious.append(iou.item())
    
    return np.mean(rail_ious) if rail_ious else 0.0

def train_simple_model():
    # Configuration
    config = {
        'epochs': 25,
        'learning_rate': 0.0001,
        'batch_size': 2,
        'image_size': 384,  # Smaller size to avoid memory issues
        'num_classes': 19,  # Classes 0-18
        'device': 'cuda:0'
    }
    
    print(f"Starting training with config: {config}")
    
    # Initialize WandB
    wandb.init(project="RailSafeNet-SegFormer-B3", name="segformer-b0-fixed", config=config)
    
    # Set device
    device = torch.device(config['device'])
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    
    # Create model - using B0 for memory efficiency
    print("Creating SegFormer B0 model...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        num_labels=config['num_classes'],
        ignore_mismatched_sizes=True
    )
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    best_rail_iou = 0.0
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 50)
        
        for phase in ['Train', 'Valid']:
            if phase == 'Train':
                model.train()
            else:
                model.eval()
            
            # Create dataset
            dataset = SimpleDataset(PATH_JPGS, PATH_MASKS, config['image_size'], phase)
            dataloader = DataLoader(dataset, batch_size=config['batch_size'], 
                                  shuffle=(phase=='Train'), num_workers=2)
            
            running_loss = 0.0
            all_rail_ious = []
            
            with torch.set_grad_enabled(phase == 'Train'):
                for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc=f"{phase}")):
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    
                    if phase == 'Train':
                        optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(images)
                    logits = outputs.logits
                    
                    # Resize logits to match mask size
                    logits = F.interpolate(logits, size=masks.shape[-2:], 
                                         mode='bilinear', align_corners=False)
                    
                    loss = criterion(logits, masks)
                    
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
                    
                    running_loss += loss.item()
                    
                    # Calculate rail IoU for validation
                    if phase == 'Valid':
                        pred_masks = torch.argmax(logits, dim=1)
                        for pred_mask, true_mask in zip(pred_masks, masks):
                            rail_iou = calculate_rail_iou(pred_mask.cpu(), true_mask.cpu())
                            if rail_iou > 0:
                                all_rail_ious.append(rail_iou)
                    
                    # Clear cache periodically
                    if batch_idx % 20 == 0:
                        torch.cuda.empty_cache()
            
            # Calculate epoch metrics
            epoch_loss = running_loss / len(dataloader)
            
            if phase == 'Valid':
                avg_rail_iou = np.mean(all_rail_ious) if all_rail_ious else 0.0
                
                # Save best model
                if avg_rail_iou > best_rail_iou:
                    best_rail_iou = avg_rail_iou
                    torch.save(model.state_dict(), 
                             os.path.join(PATH_MODELS, f'segformer_best_rail_{best_rail_iou:.4f}.pth'))
                    print(f"New best rail IoU: {best_rail_iou:.4f}")
                
                # Log metrics
                wandb.log({
                    "epoch": epoch,
                    f"{phase.lower()}_loss": epoch_loss,
                    "val_rail_iou": avg_rail_iou,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
                
                print(f"{phase} Loss: {epoch_loss:.4f}, Rail IoU: {avg_rail_iou:.4f}")
            else:
                wandb.log({f"{phase.lower()}_loss": epoch_loss})
                print(f"{phase} Loss: {epoch_loss:.4f}")
    
    wandb.finish()
    return best_rail_iou

if __name__ == "__main__":
    print("=== SegFormer Rail Detection Training ===")
    print(f"Images: {PATH_JPGS}")
    print(f"Masks: {PATH_MASKS}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    try:
        best_score = train_simple_model()
        print(f"\n🎉 Training completed! Best rail IoU: {best_score:.4f}")
        
        if best_score >= 0.6:
            print("✅ Target rail IoU (≥0.6) achieved!")
        else:
            print(f"⚠️  Target rail IoU not reached. Best: {best_score:.4f}")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()