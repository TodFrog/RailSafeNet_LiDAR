#!/usr/bin/env python3
"""
Simplified Transfer Learning Training Script for SegFormer B3
Fix data loading bottleneck issues

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

class SimpleRailDataset(Dataset):
    """Simplified dataset for debugging"""
    
    def __init__(self, images_path, masks_path, image_size=256, max_samples=1000):
        self.images_path = images_path
        self.masks_path = masks_path
        self.image_size = image_size
        
        # Get limited number of files for testing
        all_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
        all_files.sort()
        self.image_files = all_files[:max_samples]  # Limit dataset size
        
        # ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        print(f"📦 Dataset initialized with {len(self.image_files)} samples")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        try:
            # Load image
            img_name = self.image_files[idx]
            img_path = os.path.join(self.images_path, img_name)
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"Failed to load image: {img_path}")
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
                print(f"Failed to load mask: {mask_path}")
                mask = np.zeros((self.image_size, self.image_size), dtype=np.int64)
            
            # Resize (smaller size to reduce memory)
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

def load_original_model_simple(model_path, num_classes=13):
    """Simplified model loading"""
    print(f"Loading original model from: {model_path}")
    
    # Load the original complete model
    original_model = torch.load(model_path, map_location='cpu')
    print(f"Original model classes: {original_model.config.num_labels}")
    
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
    for key, value in original_state_dict.items():
        if key in new_state_dict and new_state_dict[key].shape == value.shape:
            compatible_dict[key] = value
    
    # Load the compatible parameters
    missing_keys, unexpected_keys = new_model.load_state_dict(compatible_dict, strict=False)
    
    print(f"✅ Transferred {len(compatible_dict)} compatible parameters")
    print(f"⚠️ {len(missing_keys)} parameters will be randomly initialized")
    
    return new_model

def simple_train_step(model, batch, optimizer, device):
    """Single training step"""
    images, masks = batch
    images = images.to(device)
    masks = masks.to(device)
    
    # Forward pass
    outputs = model(images)
    logits = outputs.logits
    
    # Resize logits to match mask size
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
    
    # Calculate loss
    loss = F.cross_entropy(logits, masks, ignore_index=255)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Calculate IoU
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        intersection = ((preds == masks) & (masks != 255)).sum().float()
        total = (masks != 255).sum().float()
        accuracy = intersection / (total + 1e-8)
    
    return loss.item(), accuracy.item()

def main():
    print("🚀 Simplified Transfer Learning Training")
    print("="*60)
    
    # Configuration
    config = {
        'original_model_path': "/home/mmc-server4/RailSafeNet_mini_DT/assets/models_pretrained/segformer/SegFormer_B3_1024_finetuned.pth",
        'images_path': "/home/mmc-server4/Server/Datasets_hdd/rs19_val/jpgs/rs19_val",
        'masks_path': "/home/mmc-server4/Server/Datasets_hdd/rs19_val/uint8/rs19_val",
        'batch_size': 2,  # Very small batch size
        'image_size': 256,  # Much smaller image size
        'num_classes': 13,
        'max_samples': 100,  # Limit to 100 samples for testing
        'num_epochs': 3,  # Short test run
        'learning_rate': 1e-4,
    }
    
    # Device setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Use single GPU
    print(f"Using device: {device}")
    
    # Initialize wandb
    wandb.init(
        project="RailSafeNet-TransferLearning-Debug",
        name="SegFormer_B3_Transfer_Simple",
        config=config
    )
    
    # Load model
    model = load_original_model_simple(config['original_model_path'], config['num_classes'])
    model = model.to(device)
    
    # Create dataset and dataloader
    dataset = SimpleRailDataset(
        config['images_path'],
        config['masks_path'],
        config['image_size'],
        config['max_samples']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # No multiprocessing to avoid deadlocks
        pin_memory=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataloader batches: {len(dataloader)}")
    
    # Test data loading
    print("🔍 Testing data loading...")
    try:
        sample_batch = next(iter(dataloader))
        print(f"✅ Data loading successful!")
        print(f"   Batch images shape: {sample_batch[0].shape}")
        print(f"   Batch masks shape: {sample_batch[1].shape}")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    print("🏋️ Starting simplified training...")
    
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        
        print(f"\\nEpoch {epoch+1}/{config['num_epochs']}")
        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                loss, acc = simple_train_step(model, batch, optimizer, device)
                epoch_loss += loss
                epoch_acc += acc
                
                pbar.set_postfix({
                    'Loss': f'{loss:.4f}',
                    'Acc': f'{acc:.4f}'
                })
                
                # Log to wandb
                wandb.log({
                    'batch_loss': loss,
                    'batch_accuracy': acc,
                    'epoch': epoch,
                    'batch': batch_idx
                })
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Epoch results
        avg_loss = epoch_loss / len(dataloader)
        avg_acc = epoch_acc / len(dataloader)
        
        print(f"Epoch {epoch+1} Results:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Accuracy: {avg_acc:.4f}")
        
        # Log epoch results
        wandb.log({
            'epoch_loss': avg_loss,
            'epoch_accuracy': avg_acc,
            'epoch': epoch + 1
        })
    
    # Save model
    save_path = "/home/mmc-server4/RailSafeNet/models/segformer_b3_transfer_simple.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model saved: {save_path}")
    
    wandb.finish()
    print("🎉 Simple training completed!")

if __name__ == "__main__":
    main()