#!/usr/bin/env python3
"""
WandB Sweep Configuration for SegFormer Transfer Learning
Hyperparameter optimization for rail detection model
"""

import wandb
import subprocess
import os

# Sweep configuration
sweep_config = {
    'method': 'bayes',  # Bayesian optimization for efficient search
    'program': 'train_SegFormer_transfer_learning.py',  # Specify the training script
    'metric': {
        'name': 'val_iou',
        'goal': 'maximize'
    },
    'parameters': {
        # Training hyperparameters
        'batch-size': {
            'values': [4, 6, 8, 12, 16]  # Optimized for TITAN RTX memory
        },
        'image-size': {
            'values': [768, 1024, 1280]  # Different input resolutions
        },
        'num-epochs': {
            'values': [20, 30, 50]
        },

        # Learning rates - differential for transfer learning
        'lr-backbone': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-4
        },
        'lr-decoder': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3
        },
        'lr-new-layers': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 1e-2
        },

        # Regularization
        'weight-decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-2
        },

        # Training phases
        'freeze-epochs': {
            'values': [3, 5, 8, 10]
        }
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 10
    }
}

def run_training():
    """Run training with wandb configuration"""

    # Initialize wandb run for sweep
    wandb.init()

    # Get hyperparameters from wandb config
    config = wandb.config

    # Build command with hyperparameters
    cmd = [
        'python', 'train_SegFormer_transfer_learning.py',
        '--use-wandb',
        '--batch-size', str(config.batch_size),
        '--image-size', str(config.image_size),
        '--num-epochs', str(config.num_epochs),
        '--lr-backbone', str(config.lr_backbone),
        '--lr-decoder', str(config.lr_decoder),
        '--lr-new-layers', str(config.lr_new_layers),
        '--weight-decay', str(config.weight_decay),
        '--freeze-epochs', str(config.freeze_epochs),
        '--experiment-name', f'Transfer_Sweep_{wandb.run.name}'
    ]

    print(f"🚀 Running command: {' '.join(cmd)}")

    # Set CUDA device
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '2'  # Use GPU 2 as requested

    # Run training
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)  # 2 hour timeout

        if result.returncode != 0:
            print(f"❌ Training failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            wandb.log({"training_status": "failed"})
        else:
            print(f"✅ Training completed successfully")
            wandb.log({"training_status": "success"})

    except subprocess.TimeoutExpired:
        print(f"⏰ Training timed out after 2 hours")
        wandb.log({"training_status": "timeout"})
    except Exception as e:
        print(f"💥 Error running training: {e}")
        wandb.log({"training_status": "error", "error_message": str(e)})

def create_sweep():
    """Create wandb sweep"""

    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep_config,
        project="RailSafeNet-TransferLearning",
        entity=None  # Use default entity
    )

    print(f"🔄 Created sweep with ID: {sweep_id}")
    print(f"🎯 Sweep URL: https://wandb.ai/wandb/{sweep_config.get('project', 'RailSafeNet-TransferLearning')}/sweeps/{sweep_id}")

    return sweep_id

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "create":
        # Create sweep
        sweep_id = create_sweep()
        print(f"\n🚀 To run the sweep agent, execute:")
        print(f"CUDA_VISIBLE_DEVICES=2 wandb agent {sweep_id}")

    elif len(sys.argv) > 1 and sys.argv[1] == "run":
        # Run training (called by wandb agent)
        run_training()

    else:
        print("Usage:")
        print("  python sweep_transfer.py create  # Create sweep")
        print("  python sweep_transfer.py run     # Run training (used by wandb agent)")