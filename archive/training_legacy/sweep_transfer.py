#!/usr/bin/env python3
"""
SegFormer 전이학습용 WandB sweep 설정 스크립트.

이 파일은 여러 하이퍼파라미터 조합을 자동 탐색하기 위한 보조 도구다.
최종 납품 기준의 공식 실행 경로는 아니며, GPU 번호와 실행 시간 제한 같은
실험실 환경 가정이 포함되어 있으므로 그대로 범용 배포용으로 쓰기보다는
참고용 실험 스크립트로 보는 것이 적절하다.
"""

import wandb
import subprocess
import os

# Sweep 구성은 원본 실험 가정을 보존하기 위해 하드코딩으로 유지한다.
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
    """WandB agent가 전달한 설정으로 학습 스크립트를 실행한다."""

    # 현재 run을 먼저 초기화해야 wandb.config를 통해 sweep 값을 읽을 수 있다.
    wandb.init()

    # Get hyperparameters from wandb config
    config = wandb.config

    # 하이퍼파라미터를 개별 CLI 인자로 풀어 넘겨 재현 가능한 실행 문자열을 만든다.
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

    # 기존 실험 환경에서는 GPU 2를 고정 사용했다.
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '2'  # Use GPU 2 as requested

    # subprocess 경로는 스크립트 실패/timeout을 sweep 로그에 반영하기 쉽다.
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
    """WandB sweep을 생성하고 식별자를 출력한다."""

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
