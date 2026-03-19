#!/usr/bin/env python3
"""SegFormer B3 전이학습 스크립트.

활성 추론/평가 코드와 같은 13클래스 체계를 목표로 하며, RailSem19 라벨을 축약된 클래스
체계로 재매핑하는 가정이 학습·평가·추론 전반에 공유된다.
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
import json
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import argparse
from pathlib import Path
from torch.cuda.amp import GradScaler, autocast

def parse_args():
    """전이학습 실험용 CLI 인자를 정의한다."""
    parser = argparse.ArgumentParser(description='Transfer Learning Training for SegFormer B3')

    # Model settings
    parser.add_argument('--model-name', type=str, default='nvidia/mit-b3', help='Model name')
    parser.add_argument('--original-model-path', type=str,
                       default='/home/mmc-server4/RailSafeNet_mini_DT/assets/models_pretrained/segformer/SegFormer_B3_1024_finetuned.pth',
                       help='Path to original pretrained model')

    # Dataset paths
    parser.add_argument('--train-images-path', type=str,
                       default='/home/mmc-server4/Server/Datasets_hdd/rs19_val/jpgs/rs19_val',
                       help='Path to training images')
    parser.add_argument('--train-masks-path', type=str,
                       default='/home/mmc-server4/Server/Datasets_hdd/rs19_val/uint8/rs19_val',
                       help='Path to training masks')

    # Training settings
    parser.add_argument('--num-epochs', '--num_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', '--batch_size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--num-workers', '--num_workers', type=int, default=8, help='Number of data loader workers')
    parser.add_argument('--image-size', '--image_size', type=int, default=768, help='Input image size')

    # Learning rates
    parser.add_argument('--lr-backbone', '--lr_backbone', type=float, default=1e-5, help='Learning rate for backbone')
    parser.add_argument('--lr-decoder', '--lr_decoder', type=float, default=1e-4, help='Learning rate for decoder')
    parser.add_argument('--lr-new-layers', '--lr_new_layers', type=float, default=1e-3, help='Learning rate for new layers')
    parser.add_argument('--weight-decay', '--weight_decay', type=float, default=1e-4, help='Weight decay')

    # Training phases
    parser.add_argument('--freeze-epochs', '--freeze_epochs', type=int, default=5, help='Epochs to keep backbone frozen')
    parser.add_argument('--warmup-epochs', type=int, default=3, help='Epochs for learning rate warmup')

    # Model saving
    parser.add_argument('--save-dir', type=str, default='/home/mmc-server4/RailSafeNet/models',
                       help='Directory to save models')
    parser.add_argument('--save-every-n-epochs', type=int, default=5, help='Save checkpoint every N epochs')

    # WandB settings
    parser.add_argument('--project-name', type=str, default='RailSafeNet-TransferLearning',
                       help='WandB project name')
    parser.add_argument('--experiment-name', type=str, default='SegFormer_B3_Transfer_v1',
                       help='WandB experiment name')
    parser.add_argument('--use-wandb', action='store_true', default=True, help='Use Weights & Biases logging')

    # Class settings
    parser.add_argument('--use-original-classes', action='store_true', default=True,
                       help='Use original 13 classes instead of 19')
    parser.add_argument('--num-classes', type=int, default=13, help='Number of classes')

    return parser.parse_args()

class RailDataset(Dataset):
    """철도 세그멘테이션 학습용 데이터셋.

    필요 시 19클래스 GT를 13클래스 체계로 재매핑한다. 이 매핑은 평가/추론 코드와
    공유되는 핵심 가정이므로 변경 시 전체 파이프라인 검증이 다시 필요하다.
    """
    
    def __init__(self, images_path, masks_path, image_size=1024, num_classes=19, transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform
        
        # Get all image files
        self.image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
        self.image_files.sort()
        
        # ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_path, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.masks_path, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
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
        
        # Handle class mapping if using original 13-class system
        if self.num_classes == 13:
            # RailSem19 일부 클래스를 rail/infrastructure/background 중심 13클래스로 압축한다.
            # 어떤 클래스가 4 또는 9로 들어가는지에 따라 이후 선로 추정 로직이 달라진다.
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
            mapped_mask = torch.full_like(mask, 255, dtype=torch.long)
            for src_class, tgt_class in class_mapping.items():
                mapped_mask[mask == src_class] = tgt_class

            # Set unmapped classes to background
            mapped_mask[mapped_mask == 255] = 12
            mask = mapped_mask
        
        return image, mask

def load_original_model(model_path, num_classes):
    """전이학습 초기값으로 사용할 원본 모델을 불러온다."""
    print(f"Loading original model from: {model_path}")
    
    # Load the original complete model
    original_model = torch.load(model_path, map_location='cpu')
    
    print(f"Original model classes: {original_model.config.num_labels}")
    print(f"Target classes: {num_classes}")
    
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
    print(f"⚠️ {len(missing_keys)} parameters will be randomly initialized")
    print(f"⚠️ {len(incompatible_keys)} parameters were incompatible")
    
    return new_model, original_model.config

def create_optimizer(model, args):
    """Create optimizer with differential learning rates"""

    # Group parameters by component
    backbone_params = []
    decoder_params = []
    new_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'segformer.encoder' in name:
            backbone_params.append(param)
        elif 'decode_head' in name:
            decoder_params.append(param)
        else:
            new_params.append(param)

    # Create parameter groups with different learning rates
    param_groups = [
        {'params': backbone_params, 'lr': args.lr_backbone, 'name': 'backbone'},
        {'params': decoder_params, 'lr': args.lr_decoder, 'name': 'decoder'},
        {'params': new_params, 'lr': args.lr_new_layers, 'name': 'new_layers'}
    ]

    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)

    print(f"Optimizer groups:")
    print(f"  - Backbone: {len(backbone_params)} params, lr={args.lr_backbone}")
    print(f"  - Decoder: {len(decoder_params)} params, lr={args.lr_decoder}")
    print(f"  - New layers: {len(new_params)} params, lr={args.lr_new_layers}")

    return optimizer

def create_loss_function(args):
    """Create combined loss function with rail class focus"""

    # Create class weights - higher weight for rail classes
    class_weights = torch.ones(args.num_classes)

    # Give higher weights to rail classes
    if hasattr(args, 'rail_classes'):
        for rail_class in args.rail_classes:
            if rail_class < args.num_classes:
                class_weights[rail_class] = 3.0  # 3x weight for rail classes

    # Cross entropy loss with class weights (will be moved to device in combined_loss)
    ce_loss = nn.CrossEntropyLoss(ignore_index=255)
    
    def combined_loss(outputs, targets):
        # Resize outputs to match targets
        outputs = F.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)
        
        # Cross entropy loss with class weights
        device_class_weights = class_weights.to(outputs.device)
        ce = F.cross_entropy(outputs, targets, weight=device_class_weights, ignore_index=255)
        
        # Dice loss for rail classes (optional)
        dice_loss = 0.0
        if hasattr(args, 'rail_classes') and args.rail_classes:
            probs = F.softmax(outputs, dim=1)
            for cls in args.rail_classes:
                if cls < outputs.shape[1]:  # Ensure class exists
                    pred_mask = probs[:, cls]
                    true_mask = (targets == cls).float()
                    intersection = (pred_mask * true_mask).sum()
                    dice = (2. * intersection) / (pred_mask.sum() + true_mask.sum() + 1e-8)
                    dice_loss += (1 - dice)
            dice_loss /= len(args.rail_classes)
        
        # Combined loss
        total_loss = ce + 0.3 * dice_loss
        return total_loss, ce, dice_loss
    
    return combined_loss

def calculate_metrics(outputs, targets, num_classes, rail_classes=None):
    """Calculate IoU and other metrics including rail-specific metrics"""
    outputs = F.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)
    preds = torch.argmax(outputs, dim=1)

    # Convert to numpy for sklearn metrics
    preds_np = preds.cpu().numpy().flatten()
    targets_np = targets.cpu().numpy().flatten()

    # Calculate IoU per class
    iou_per_class = []
    for cls in range(num_classes):
        if cls in targets_np:  # Only calculate if class exists
            iou = jaccard_score(targets_np == cls, preds_np == cls, average='binary', zero_division=0)
            iou_per_class.append(iou)
        else:
            iou_per_class.append(0.0)

    mean_iou = np.mean(iou_per_class)

    # Calculate rail-specific metrics
    rail_iou = 0.0
    rail_ious = []
    if rail_classes:
        for rail_cls in rail_classes:
            if rail_cls < len(iou_per_class):
                rail_ious.append(iou_per_class[rail_cls])
        rail_iou = np.mean(rail_ious) if rail_ious else 0.0

    return mean_iou, iou_per_class, rail_iou, rail_ious

def freeze_backbone(model):
    """초기 학습 안정화를 위해 encoder(backbone) 파라미터를 잠근다."""
    for name, param in model.named_parameters():
        if 'segformer.encoder' in name:
            param.requires_grad = False
    print("🧊 Backbone frozen")

def unfreeze_backbone(model):
    """후반 미세조정을 위해 encoder(backbone) 파라미터를 다시 연다."""
    for name, param in model.named_parameters():
        if 'segformer.encoder' in name:
            param.requires_grad = True
    print("🔥 Backbone unfrozen")

def train_epoch(model, dataloader, optimizer, loss_fn, device, config, epoch, scaler=None):
    """Train for one epoch with mixed precision"""
    model.train()

    total_loss = 0
    total_ce_loss = 0
    total_dice_loss = 0
    total_iou = 0
    total_rail_iou = 0
    num_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # Clear gradients
        optimizer.zero_grad()
        
        if scaler is not None:
            # Mixed precision forward pass
            with autocast():
                outputs = model(images)
                logits = outputs.logits
                loss, ce_loss, dice_loss = loss_fn(logits, masks)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward pass
            outputs = model(images)
            logits = outputs.logits
            loss, ce_loss, dice_loss = loss_fn(logits, masks)
            
            # Standard backward pass
            loss.backward()
            optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            mean_iou, _, rail_iou, _ = calculate_metrics(logits, masks, config.num_classes, config.rail_classes)
        
        # Update running averages
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_dice_loss += dice_loss
        total_iou += mean_iou
        total_rail_iou += rail_iou
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'IoU': f'{mean_iou:.4f}',
            'Rail_IoU': f'{rail_iou:.4f}'
        })
        
        # Log to wandb
        if batch_idx % 50 == 0 and config.use_wandb:
            wandb.log({
                'batch_loss': loss.item(),
                'batch_iou': mean_iou,
                'batch_rail_iou': rail_iou,
                'epoch': epoch,
                'batch': batch_idx
            })
    
    # Calculate epoch averages
    avg_loss = total_loss / num_batches
    avg_ce_loss = total_ce_loss / num_batches
    avg_dice_loss = total_dice_loss / num_batches
    avg_iou = total_iou / num_batches
    avg_rail_iou = total_rail_iou / num_batches

    return avg_loss, avg_ce_loss, avg_dice_loss, avg_iou, avg_rail_iou

def validate_epoch(model, dataloader, loss_fn, device, config):
    """Validate for one epoch"""
    model.eval()

    total_loss = 0
    total_iou = 0
    total_rail_iou = 0
    all_class_ious = []
    all_rail_ious = []
    num_batches = len(dataloader)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            logits = outputs.logits
            
            # Calculate loss
            loss, _, _ = loss_fn(logits, masks)
            
            # Calculate metrics
            mean_iou, class_ious, rail_iou, rail_ious = calculate_metrics(logits, masks, config.num_classes, config.rail_classes)
            
            total_loss += loss.item()
            total_iou += mean_iou
            total_rail_iou += rail_iou
            all_class_ious.append(class_ious)
            all_rail_ious.append(rail_ious)

            pbar.set_postfix({
                'Val Loss': f'{loss.item():.4f}',
                'Val IoU': f'{mean_iou:.4f}',
                'Val Rail IoU': f'{rail_iou:.4f}'
            })
    
    # Calculate averages
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    avg_rail_iou = total_rail_iou / num_batches

    # Calculate average class IoUs
    avg_class_ious = np.mean(all_class_ious, axis=0)

    return avg_loss, avg_iou, avg_rail_iou, avg_class_ious

def main():
    # Parse arguments
    args = parse_args()

    # Set rail classes based on number of classes
    if args.use_original_classes:
        args.rail_classes = [1, 4, 9]  # platform, main track, secondary track
    else:
        args.rail_classes = list(range(1, 6))
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Initialize wandb if requested
    if args.use_wandb:
        # Check if running as part of sweep
        if wandb.run is None:
            wandb.init(
                project=args.project_name,
                name=args.experiment_name,
                config=vars(args)
            )
        else:
            # Running as part of sweep - update args with sweep config
            config = wandb.config
            for key, value in config.items():
                # WandB config keys should already use underscores
                if hasattr(args, key):
                    setattr(args, key, value)
                    print(f"Updated {key} = {value} from sweep config")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load model
    print("🚀 Loading original model for transfer learning...")
    model, original_config = load_original_model(args.original_model_path, args.num_classes)
    model = model.to(device)
    
    # Multi-GPU setup with memory management
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        # Clear cache before multi-GPU setup
        torch.cuda.empty_cache()
        model = nn.DataParallel(model)
    
    # Enable mixed precision training to reduce memory usage
    scaler = GradScaler()
    
    # Create datasets
    print("📦 Creating datasets...")
    
    # For now, use the same data for train/val (would normally split)
    train_dataset = RailDataset(
        args.train_images_path,
        args.train_masks_path,
        args.image_size,
        args.num_classes
    )
    
    # Simple train/val split (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create optimizer and loss function
    optimizer = create_optimizer(model, args)
    loss_fn = create_loss_function(args)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # Training loop
    print("🏋️ Starting training...")
    
    best_iou = 0.0
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    
    for epoch in range(args.num_epochs):
        print(f"\\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"{'='*60}")
        
        # Freeze/unfreeze backbone based on epoch
        if epoch == 0:
            freeze_backbone(model)
        elif epoch == args.freeze_epochs:
            unfreeze_backbone(model)
            # Recreate optimizer with new parameter requirements
            optimizer = create_optimizer(model, args)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs-epoch)

        # Train
        train_loss, train_ce_loss, train_dice_loss, train_iou, train_rail_iou = train_epoch(
            model, train_loader, optimizer, loss_fn, device, args, epoch, scaler
        )

        # Validate
        val_loss, val_iou, val_rail_iou, val_class_ious = validate_epoch(
            model, val_loader, loss_fn, device, args
        )
        
        # Update scheduler
        scheduler.step()
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)
        
        # Print epoch results
        print(f"\\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f} (CE: {train_ce_loss:.4f}, Dice: {train_dice_loss:.4f})")
        print(f"  Train IoU:  {train_iou:.4f}")
        print(f"  Train Rail IoU: {train_rail_iou:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val IoU:    {val_iou:.4f}")
        print(f"  Val Rail IoU: {val_rail_iou:.4f}")
        
        # Print rail class IoUs
        if len(val_class_ious) > 0:
            rail_ious = [val_class_ious[cls] for cls in args.rail_classes if cls < len(val_class_ious)]
            if rail_ious:
                print(f"  Rail IoUs:  {[f'{iou:.3f}' for iou in rail_ious]}")
        
        # Log to wandb
        if args.use_wandb:
            log_dict = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_iou': train_iou,
                'train_rail_iou': train_rail_iou,
                'val_loss': val_loss,
                'val_iou': val_iou,
                'val_rail_iou': val_rail_iou,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            
            # Add class-specific IoUs
            for i, iou in enumerate(val_class_ious):
                log_dict[f'val_iou_class_{i}'] = iou
            
            wandb.log(log_dict)
        
        # Save model if best rail IoU
        if val_rail_iou > best_iou:
            best_iou = val_rail_iou
            save_path = os.path.join(args.save_dir, f'segformer_b3_transfer_best_rail_{val_rail_iou:.4f}.pth')
            
            # Save state dict only (for compatibility)
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            
            print(f"💾 Saved best model: {save_path}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every_n_epochs == 0:
            checkpoint_path = os.path.join(args.save_dir, f'segformer_b3_transfer_epoch_{epoch+1}.pth')
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), checkpoint_path)
            else:
                torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    print(f"\\n🎉 Training completed!")
    print(f"Best validation Rail IoU: {best_iou:.4f}")
    
    # Final model save
    final_path = os.path.join(args.save_dir, 'segformer_b3_transfer_final.pth')
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), final_path)
    else:
        torch.save(model.state_dict(), final_path)
    print(f"💾 Saved final model: {final_path}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training/Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_ious, label='Train IoU')
    plt.plot(val_ious, label='Val IoU')
    plt.title('Training/Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'transfer_learning_curves.png'))
    print(f"📊 Saved training curves")
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
