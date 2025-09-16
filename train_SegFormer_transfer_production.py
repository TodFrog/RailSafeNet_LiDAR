#!/usr/bin/env python3
"""
Production Transfer Learning Training Script for SegFormer B3
Optimized settings based on successful simple version

Author: Claude Code Assistant
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation
import numpy as np
import cv2
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import argparse

class ProductionRailDataset(Dataset):
    """Production dataset with configurable parameters"""
    
    def __init__(self, images_path, masks_path, image_size=512, max_samples=None, augment=False):
        self.images_path = images_path
        self.masks_path = masks_path
        self.image_size = image_size
        self.augment = augment
        
        # Get all image files
        all_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
        all_files.sort()
        
        if max_samples:
            self.image_files = all_files[:max_samples]
        else:
            self.image_files = all_files
        
        # ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        print(f"📦 Dataset initialized with {len(self.image_files)} samples")
        print(f"📏 Image size: {self.image_size}x{self.image_size}")
        print(f"🎨 Augmentation: {'Enabled' if augment else 'Disabled'}")
        
    def __len__(self):
        return len(self.image_files)
    
    def apply_augmentation(self, image, mask):
        """Apply simple augmentations"""
        if not self.augment:
            return image, mask
            
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        
        # Random brightness adjustment
        if np.random.rand() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        return image, mask
    
    def __getitem__(self, idx):
        try:
            # Load image
            img_name = self.image_files[idx]
            img_path = os.path.join(self.images_path, img_name)
            image = cv2.imread(img_path)
            
            if image is None:
                # Return a dummy sample
                image = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
                mask = np.zeros((self.image_size, self.image_size), dtype=np.int64)
                return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(mask)
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load mask
            mask_name = img_name.replace('.jpg', '.png')
            mask_path = os.path.join(self.masks_path, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                mask = np.zeros((image.shape[1], image.shape[0]), dtype=np.uint8)
            
            # Apply augmentation before resize
            image, mask = self.apply_augmentation(image, mask)
            
            # Resize
            image = cv2.resize(image, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            
            # Normalize image
            image = image.astype(np.float32) / 255.0
            for i in range(3):
                image[:, :, i] = (image[:, :, i] - self.mean[i]) / self.std[i]
            
            # Convert to tensors
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
            mask = torch.from_numpy(mask).long()
            
            # Clamp mask values for 13-class system
            mask = torch.clamp(mask, 0, 12)
            
            return image, mask
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return dummy data
            image = torch.zeros(3, self.image_size, self.image_size)
            mask = torch.zeros(self.image_size, self.image_size, dtype=torch.long)
            return image, mask

def load_original_model_production(model_path, num_classes=13):
    """Production model loading with better error handling"""
    print(f"Loading original model from: {model_path}")
    
    try:
        # Load the original complete model
        original_model = torch.load(model_path, map_location='cpu')
        print(f"✅ Original model loaded successfully")
        print(f"📊 Original model classes: {original_model.config.num_labels}")
        
        # Create new model with target number of classes
        new_model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b3",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Load compatible weights
        original_state_dict = original_model.state_dict()
        new_state_dict = new_model.state_dict()
        
        # Transfer compatible parameters
        compatible_dict = {}
        incompatible_keys = []
        
        for key, value in original_state_dict.items():
            if key in new_state_dict and new_state_dict[key].shape == value.shape:
                compatible_dict[key] = value
            else:
                incompatible_keys.append(key)
        
        # Load the compatible parameters
        missing_keys, unexpected_keys = new_model.load_state_dict(compatible_dict, strict=False)
        
        print(f"✅ Transferred {len(compatible_dict)} compatible parameters")
        print(f"⚠️  {len(missing_keys)} parameters will be randomly initialized")
        print(f"⚠️  {len(incompatible_keys)} parameters were incompatible")
        
        return new_model, original_model.config
        
    except Exception as e:
        print(f"❌ Error loading original model: {e}")
        raise e

def create_optimizer_production(model, config):
    """Create optimizer with differential learning rates"""
    
    # Group parameters by component
    backbone_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'segformer.encoder' in name:
            backbone_params.append(param)
        elif 'decode_head' in name:
            decoder_params.append(param)
    
    # Create parameter groups with different learning rates
    param_groups = [
        {'params': backbone_params, 'lr': config['lr_backbone'], 'name': 'backbone'},
        {'params': decoder_params, 'lr': config['lr_decoder'], 'name': 'decoder'}
    ]
    
    optimizer = optim.AdamW(param_groups, weight_decay=config['weight_decay'])
    
    print(f"Optimizer groups:")
    print(f"  - Backbone: {len(backbone_params)} params, lr={config['lr_backbone']}")
    print(f"  - Decoder: {len(decoder_params)} params, lr={config['lr_decoder']}")
    
    return optimizer

def calculate_iou(outputs, targets, num_classes=13):
    """Calculate IoU metrics"""
    outputs = F.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)
    preds = torch.argmax(outputs, dim=1)
    
    # Convert to numpy for calculation
    preds_np = preds.cpu().numpy().flatten()
    targets_np = targets.cpu().numpy().flatten()
    
    # Calculate IoU per class
    iou_per_class = []
    valid_classes = []
    
    for cls in range(num_classes):
        if cls in targets_np:  # Only calculate if class exists
            pred_mask = (preds_np == cls)
            target_mask = (targets_np == cls)
            
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()
            
            if union > 0:
                iou = intersection / union
                iou_per_class.append(iou)
                valid_classes.append(cls)
    
    mean_iou = np.mean(iou_per_class) if iou_per_class else 0.0
    
    return mean_iou, iou_per_class, valid_classes

def train_epoch_production(model, dataloader, optimizer, device, config, epoch):
    """Production training epoch with better metrics"""
    model.train()
    
    total_loss = 0
    total_iou = 0
    total_accuracy = 0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        logits = outputs.logits
        
        # Resize logits to match mask size
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
        
        # Calculate loss
        loss = F.cross_entropy(logits, masks, ignore_index=255)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            # IoU calculation
            mean_iou, _, _ = calculate_iou(logits, masks, config['num_classes'])
            
            # Pixel accuracy
            preds = torch.argmax(logits, dim=1)
            correct = (preds == masks).sum().float()
            total_pixels = masks.numel()
            accuracy = correct / total_pixels
            
            total_loss += loss.item()
            total_iou += mean_iou
            total_accuracy += accuracy.item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'IoU': f'{mean_iou:.4f}',
            'Acc': f'{accuracy.item():.4f}'
        })
        
        # Log to wandb every 50 batches
        if batch_idx % 50 == 0:
            wandb.log({
                'batch_loss': loss.item(),
                'batch_iou': mean_iou,
                'batch_accuracy': accuracy.item(),
                'epoch': epoch,
                'batch': batch_idx
            })
    
    # Calculate epoch averages
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_iou, avg_accuracy

def validate_epoch_production(model, dataloader, device, config):
    """Production validation epoch"""
    model.eval()
    
    total_loss = 0
    total_iou = 0
    total_accuracy = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            logits = outputs.logits
            
            # Resize logits to match mask size
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            
            # Calculate loss
            loss = F.cross_entropy(logits, masks, ignore_index=255)
            
            # Calculate metrics
            mean_iou, _, _ = calculate_iou(logits, masks, config['num_classes'])
            
            preds = torch.argmax(logits, dim=1)
            correct = (preds == masks).sum().float()
            total_pixels = masks.numel()
            accuracy = correct / total_pixels
            
            total_loss += loss.item()
            total_iou += mean_iou
            total_accuracy += accuracy.item()
            
            pbar.set_postfix({
                'Val Loss': f'{loss.item():.4f}',
                'Val IoU': f'{mean_iou:.4f}',
                'Val Acc': f'{accuracy.item():.4f}'
            })
    
    # Calculate averages
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_iou, avg_accuracy

def main():
    parser = argparse.ArgumentParser(description='Production Transfer Learning Training for SegFormer B3')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--image_size', type=int, default=512, help='Input image size')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples (for testing)')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='Learning rate for backbone')
    parser.add_argument('--lr_decoder', type=float, default=1e-4, help='Learning rate for decoder')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--save_every', type=int, default=5, help='Save model every N epochs')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'original_model_path': "/home/mmc-server4/RailSafeNet_mini_DT/assets/models_pretrained/segformer/SegFormer_B3_1024_finetuned.pth",
        'images_path': "/home/mmc-server4/Server/Datasets_hdd/rs19_val/jpgs/rs19_val",
        'masks_path': "/home/mmc-server4/Server/Datasets_hdd/rs19_val/uint8/rs19_val",
        'save_dir': "/home/mmc-server4/RailSafeNet/models",
        'num_classes': 13,
        
        # From arguments
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'num_epochs': args.num_epochs,
        'max_samples': args.max_samples,
        'lr_backbone': args.lr_backbone,
        'lr_decoder': args.lr_decoder,
        'weight_decay': args.weight_decay,
        'augment': args.augment,
        'num_workers': args.num_workers,
        'save_every': args.save_every
    }
    
    print("🚀 Production Transfer Learning Training")
    print("="*80)
    print(f"📋 Configuration:")
    for key, value in config.items():
        if key not in ['images_path', 'masks_path', 'original_model_path', 'save_dir']:
            print(f"   {key}: {value}")
    print("="*80)
    
    # Device setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"🔥 GPU Info: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project="RailSafeNet-TransferLearning-Production",
            name=f"SegFormer_B3_Transfer_bs{args.batch_size}_sz{args.image_size}",
            config=config
        )
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Load model
    print("🔄 Loading original model for transfer learning...")
    model, original_config = load_original_model_production(config['original_model_path'], config['num_classes'])
    model = model.to(device)
    
    # Create datasets
    print("📦 Creating datasets...")
    
    full_dataset = ProductionRailDataset(
        config['images_path'],
        config['masks_path'],
        config['image_size'],
        config['max_samples'],
        config['augment']
    )
    
    # Train/Val split (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"📊 Dataset Statistics:")
    print(f"   Total samples: {len(full_dataset)}")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer_production(model, config)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    # Training loop
    print("🏋️  Starting production training...")
    
    best_iou = 0.0
    training_history = {
        'train_loss': [], 'train_iou': [], 'train_acc': [],
        'val_loss': [], 'val_iou': [], 'val_acc': []
    }
    
    for epoch in range(config['num_epochs']):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"{'='*80}")
        
        # Training
        train_loss, train_iou, train_acc = train_epoch_production(
            model, train_loader, optimizer, device, config, epoch
        )
        
        # Validation
        val_loss, val_iou, val_acc = validate_epoch_production(
            model, val_loader, device, config
        )
        
        # Update scheduler
        scheduler.step()
        
        # Save metrics
        training_history['train_loss'].append(train_loss)
        training_history['train_iou'].append(train_iou)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_iou'].append(val_iou)
        training_history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"   Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Acc: {train_acc:.4f}")
        print(f"   Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Acc: {val_acc:.4f}")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_iou': train_iou,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_iou': val_iou,
                'val_accuracy': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            best_path = os.path.join(config['save_dir'], f'segformer_b3_transfer_best_{val_iou:.4f}.pth')
            torch.save(model.state_dict(), best_path)
            print(f"💾 Saved best model: {best_path}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % config['save_every'] == 0:
            checkpoint_path = os.path.join(config['save_dir'], f'segformer_b3_transfer_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    # Final model save
    final_path = os.path.join(config['save_dir'], 'segformer_b3_transfer_final.pth')
    torch.save(model.state_dict(), final_path)
    
    print(f"\n🎉 Training completed!")
    print(f"📊 Final Results:")
    print(f"   Best Validation IoU: {best_iou:.4f}")
    print(f"   Final Train IoU: {train_iou:.4f}")
    print(f"   Final Val IoU: {val_iou:.4f}")
    print(f"💾 Final model saved: {final_path}")
    
    # Plot training curves
    if len(training_history['train_loss']) > 1:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(training_history['train_loss'], label='Train Loss')
        plt.plot(training_history['val_loss'], label='Val Loss')
        plt.title('Training/Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(training_history['train_iou'], label='Train IoU')
        plt.plot(training_history['val_iou'], label='Val IoU')
        plt.title('Training/Validation IoU')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(training_history['train_acc'], label='Train Accuracy')
        plt.plot(training_history['val_acc'], label='Val Accuracy')
        plt.title('Training/Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(config['save_dir'], 'transfer_learning_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Training curves saved: {plot_path}")
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()