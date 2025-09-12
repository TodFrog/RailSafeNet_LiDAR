from scripts.dataloader_SegFormer import CustomDataset
from scripts.metrics_filtered_cls import compute_map_cls, compute_IoU
from transformers import SegformerModel, SegformerConfig, SegformerForSemanticSegmentation, SegformerImageProcessor
from torch.optim import SGD, Adam, Adagrad, AdamW
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
import albumentations as A
import torch
import numpy as np
import os
import cv2
import wandb
from tqdm import tqdm
import time
import copy

# Updated dataset paths
PATH_JPGS = "/home/mmc-server4/Server/Datasets_hdd/rs19_val/jpgs/rs19_val"
PATH_MASKS = "/home/mmc-server4/Server/Datasets_hdd/rs19_val/uint8/rs19_val"
PATH_MODELS = "/home/mmc-server4/RailSafeNet/models"
PATH_LOGS = "/home/mmc-server4/RailSafeNet/logs"

# Ensure directories exist
os.makedirs(PATH_MODELS, exist_ok=True)
os.makedirs(PATH_LOGS, exist_ok=True)

def create_model(output_channels=1):
    # Use SegFormer B3 instead of B0 for better performance
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
                                                            num_labels=output_channels,
                                                            ignore_mismatched_sizes=True)
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

def train(model, num_epochs, batch_size, image_size, optimizer, criterion, scheduler, config):
    start = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    best_rail_iou = 0.0  # Track best rail IoU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        print('-' * 20)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        
        train_loss = 0
        val_loss = 0
        val_MmAP, val_mAP, val_IoU, val_MIoU = list(), list(), list(), list()
        classes_MAP, classes_AP, classes_IoU, classes_MIoU = {},{},{},{}
        
        dl_lentrain = 0
        dl_lenval = 0
        
        for phase in ['Train', 'Valid']:
            image_processor = SegformerImageProcessor(reduce_labels=False)
            dataset = CustomDataset(PATH_JPGS, PATH_MASKS, image_processor, image_size, subset=phase, val_fraction=0.2)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=False)
            
            if phase == 'Train':
                model.train()
                dl_lentrain = len(dataloader)
                accumulation_steps = max(1, 4 // batch_size)  # Effective batch size of 4
                
                for i, (inputs, masks) in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch+1}")):
                    inputs = inputs.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    
                    outputs = model(inputs)
                    logits = outputs.logits
                    
                    upsampled_logits = nn.functional.interpolate(
                        logits,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )

                    upsampled_logits = upsampled_logits.float()
                    loss = criterion(upsampled_logits, masks) / accumulation_steps

                    loss.backward()
                    
                    if (i + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    train_loss += loss.item() * accumulation_steps
                    
                    # Clear cache periodically
                    if i % 50 == 0:
                        torch.cuda.empty_cache()
                    
            elif phase == 'Valid':
                model.eval()
                dl_lenval = len(dataloader)
                with torch.no_grad():
                    for inputs, masks in tqdm(dataloader, desc=f"Validation Epoch {epoch+1}"):
                        inputs = inputs.to(device)
                        masks = masks.to(device)

                        outputs = model(inputs)
                        logits = outputs.logits
                        
                        upsampled_logits = nn.functional.interpolate(
                            logits,
                            size=masks.shape[-2:],
                            mode="bilinear",
                            align_corners=False
                        )

                        upsampled_logits = upsampled_logits.float()
                        loss = criterion(upsampled_logits, masks)
                        val_loss += loss.item()
                        
                        predicted_masks = upsampled_logits
                        gt_masks = masks.cpu().detach().numpy()
                        
                        for prediction, gt in zip(predicted_masks, gt_masks):
                            prediction = F.softmax(prediction, dim=0).cpu().detach().numpy()
                            prediction = np.argmax(prediction, axis=0).astype(np.uint8)
                            
                            mAP, classes_AP = compute_map_cls(gt, prediction, classes_AP)
                            Mmap, classes_MAP = compute_map_cls(gt, prediction, classes_MAP, major=True)
                            IoU, _, _, _, classes_IoU = compute_IoU(gt, prediction, classes_IoU)
                            MIoU, _, _, _, classes_MIoU = compute_IoU(gt, prediction, classes_MIoU, major=True)
                            val_mAP.append(mAP)
                            val_MmAP.append(Mmap)
                            val_IoU.append(IoU)
                            val_MIoU.append(MIoU)

        # Compute metrics
        val_MmAP, val_mAP = np.nanmean(val_MmAP), np.nanmean(val_mAP)
        val_MIoU, val_IoU = np.nanmean(val_MIoU), np.nanmean(val_IoU)
        
        # Compute class-wise metrics
        for cls, value in classes_IoU.items():
            classes_IoU[cls] = np.divide(value[0], value[1])
        
        # Get rail-track IoU (class 12)
        rail_track_iou = classes_IoU.get(12, [0, 0, 0, 0])[0] if 12 in classes_IoU else 0
        
        # Get all rail-related IoUs
        tram_track_iou = classes_IoU.get(3, [0, 0, 0, 0])[0] if 3 in classes_IoU else 0
        rail_raised_iou = classes_IoU.get(17, [0, 0, 0, 0])[0] if 17 in classes_IoU else 0
        rail_embedded_iou = classes_IoU.get(18, [0, 0, 0, 0])[0] if 18 in classes_IoU else 0
        
        # Average rail IoU for optimization target
        rail_ious = [rail_track_iou, tram_track_iou, rail_raised_iou, rail_embedded_iou]
        valid_rail_ious = [iou for iou in rail_ious if iou > 0]
        avg_rail_iou = np.mean(valid_rail_ious) if valid_rail_ious else 0
        
        classes_IoU_all = np.mean(np.array(list(classes_IoU.values()))[:, :4], axis=0) if classes_IoU else [0, 0, 0, 0]
        
        # Learning rate scheduling
        if config.scheduler == 'LinearLR':
            if epoch > 20:
                scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        elif config.scheduler == 'ReduceLROnPlateau':
            scheduler.step(avg_rail_iou)  # Use rail IoU for scheduling
            current_lr = scheduler._last_lr[0]
        
        # Print epoch summary
        print('Epoch {}/{}: Train loss: {:.4f} | Val loss: {:.4f} | lr: {:.6f}'.format(
            epoch + 1, num_epochs, train_loss/dl_lentrain, val_loss/dl_lenval, current_lr))
        print('Rail IoUs - Track: {:.4f} | Tram: {:.4f} | Raised: {:.4f} | Embedded: {:.4f} | Avg: {:.4f}'.format(
            rail_track_iou, tram_track_iou, rail_raised_iou, rail_embedded_iou, avg_rail_iou))
        
        # Save best model based on rail IoU
        if phase == 'Valid' and avg_rail_iou > best_rail_iou:
            best_rail_iou = avg_rail_iou
            best_model = copy.deepcopy(model.state_dict())
            print('New best rail IoU: {:.4f} - Saving model'.format(best_rail_iou))
            
        # Log to WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss / dl_lentrain,
            "val_loss": val_loss / dl_lenval,
            "learning_rate": current_lr,
            "val_iou/rail-track": rail_track_iou,
            "val_iou/tram-track": tram_track_iou,
            "val_iou/rail-raised": rail_raised_iou,
            "val_iou/rail-embedded": rail_embedded_iou,
            "val_iou/avg_rail": avg_rail_iou,
            "val_miou": classes_IoU_all[0] if len(classes_IoU_all) > 0 else 0,
        })

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Rail IoU: {:.4f}'.format(best_rail_iou))

    final_model = model
    model.load_state_dict(best_model)
    return final_model, model

# Sweep configuration optimized for rail detection
sweep_config = {
    'method': 'bayes',  # Use Bayesian optimization
    'metric': {
        'name': 'val_iou/avg_rail',  # Maximize average rail IoU
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {
            'value': 50  # Reasonable number for testing
        },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-2
        },
        'optimizer': {
            'values': ['adamw', 'adam']  # Focus on best optimizers
        },
        'scheduler': {
            'values': ['ReduceLROnPlateau', 'LinearLR']
        },
        'batch_size': {
            'values': [1, 2, 3, 4]  # Smaller batch sizes for B3 model at 1024x1024
        },
        'image_size': {
            'values': [512, 768]  # Test smaller image sizes first
        },
        'num_classes': {
            'values': [13, 19]  # Test both 13-class and 19-class
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-2
        }
    }
}

def sweep_train():
    with wandb.init(project="RailSafeNet-SegFormer-B3") as run:
        config = wandb.config
        
        # Set device to GPU 0
        torch.cuda.set_device(0)
        
        model = create_model(config.num_classes)
        
        # Define optimizer with weight decay
        if config.optimizer == 'adam':
            optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer == 'adamw':
            optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        # Define scheduler
        if config.scheduler == 'ReduceLROnPlateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=8, verbose=True, 
                threshold=0.01, threshold_mode='abs'
            )
        elif config.scheduler == 'LinearLR':
            scheduler = lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.1, total_iters=20
            )
        
        # Weighted loss for rail classes
        class_weights = torch.ones(config.num_classes)
        if config.num_classes == 19:
            # Increase weights for rail-related classes
            class_weights[3] = 2.0   # tram-track
            class_weights[12] = 3.0  # rail-track (most important)
            class_weights[17] = 2.0  # rail-raised
            class_weights[18] = 2.0  # rail-embedded
        elif config.num_classes == 13:
            # For 13-class model, assuming rail classes are compressed
            class_weights[3] = 2.5   # Combined rail classes
            
        class_weights = class_weights.to('cuda:0')
        loss_function = nn.CrossEntropyLoss(ignore_index=255, weight=class_weights)
        
        model_final, best_model = train(
            model, config.epochs, config.batch_size, [config.image_size, config.image_size], 
            optimizer, loss_function, scheduler, config
        )
        
        # Save models with run name
        torch.save(model_final, os.path.join(PATH_MODELS, f'model_final_{wandb.run.name}.pth'))
        torch.save(best_model, os.path.join(PATH_MODELS, f'model_best_{wandb.run.name}.pth'))
        print(f'Models saved as: model_final_{wandb.run.name}.pth and model_best_{wandb.run.name}.pth')

if __name__ == "__main__":
    print("Starting SegFormer B3 Hyperparameter Sweep for Rail Detection")
    print(f"Dataset paths:")
    print(f"Images: {PATH_JPGS}")
    print(f"Masks: {PATH_MASKS}")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="RailSafeNet-SegFormer-B3")
    print(f"Sweep ID: {sweep_id}")
    
    # Run sweep on GPU 0
    wandb.agent(sweep_id, sweep_train, count=15)  # Run 15 trials initially