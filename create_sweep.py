import wandb
import os

def create_optimized_sweep():
    """Create WandB sweep based on successful B3 configuration"""
    
    # Sweep configuration based on successful B3 settings
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization for efficient search
        'metric': {
            'name': 'best_rail_iou',
            'goal': 'maximize'
        },
        'parameters': {
            'epochs': {'value': 30},  # Fixed 30 epochs as requested
            'batch_size': {'values': [2, 4]},  # Fixed options as requested  
            'image_size': {'value': 512},  # Fixed as requested
            
            # Learning rate variations around successful value (8e-5)
            'learning_rate': {
                'values': [0.00005, 0.00006, 0.00007, 0.00008, 0.00009, 0.0001, 0.00012, 0.00015]
            },
            
            # Weight decay variations
            'weight_decay': {
                'values': [0.005, 0.01, 0.015, 0.02]
            },
            
            # Scheduler variations
            'scheduler': {
                'values': ['cosine_warm', 'cosine', 'onecycle', 'step']
            },
            
            # Fixed parameters from successful config
            'num_classes': {'value': 19},
            'device': {'value': 'cuda:0'}
        },
        
        # Early termination for poor performing runs
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 5,
            'max_iter': 30,
            's': 2
        }
    }
    
    print("🚀 Creating WandB Sweep for SegFormer B3 Optimization")
    print("📊 Sweep Configuration:")
    print(f"- Method: {sweep_config['method']}")
    print(f"- Epochs: {sweep_config['parameters']['epochs']['value']}")
    print(f"- Batch sizes: {sweep_config['parameters']['batch_size']['values']}")
    print(f"- Image size: {sweep_config['parameters']['image_size']['value']}")
    print(f"- Learning rates: {len(sweep_config['parameters']['learning_rate']['values'])} values")
    print(f"- Schedulers: {sweep_config['parameters']['scheduler']['values']}")
    
    # Create sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        project="RailSafeNet-SegFormer-B3-Sweep"
    )
    
    print(f"✅ Sweep created successfully!")
    print(f"🆔 Sweep ID: {sweep_id}")
    print(f"🌐 Sweep URL: https://wandb.ai/{wandb.api.default_entity}/RailSafeNet-SegFormer-B3-Sweep/sweeps/{sweep_id}")
    
    return sweep_id

if __name__ == "__main__":
    # Login to wandb if not already logged in
    if not os.path.exists(os.path.expanduser("~/.netrc")):
        print("Please login to WandB first:")
        wandb.login()
    
    sweep_id = create_optimized_sweep()
    
    print("\n🎯 Next steps:")
    print(f"1. Run: wandb agent {sweep_id}")
    print("2. Or use the agent script that will be created")
    
    # Create agent runner script
    agent_script = f"""#!/bin/bash
# SegFormer B3 Sweep Agent Runner
echo "🚀 Starting SegFormer B3 Sweep Agent"
echo "🆔 Sweep ID: {sweep_id}"

# Set GPU environment
export CUDA_VISIBLE_DEVICES=2

# Run agent with auto-restart
while true; do
    echo "🔄 Starting/Restarting sweep agent..."
    wandb agent {sweep_id} --count 1
    
    # Check if we should continue
    sleep 5
    echo "✅ Run completed. Starting next run..."
done
"""
    
    with open("/home/mmc-server4/RailSafeNet/run_sweep.sh", "w") as f:
        f.write(agent_script)
    
    # Make it executable
    os.chmod("/home/mmc-server4/RailSafeNet/run_sweep.sh", 0o755)
    
    print("📝 Created run_sweep.sh script")
    print("🚀 To start sweep: bash run_sweep.sh")