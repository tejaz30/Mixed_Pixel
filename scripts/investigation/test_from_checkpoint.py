"""
Test training continuation from checkpoint epoch 20.

This will help us see if the NaN is reproducible and what values
lead to it.
"""

import sys
from pathlib import Path
import torch
import yaml

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from training.trainer import Trainer
from diagnose_nan import add_diagnostics_to_trainer


def main():
    # Load config
    config_path = project_root / "configs/experiment1/kl_autoencoder_kaggle_2gpu.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("Testing with diagnostics enabled...")
    print(f"Config learning rate: {config['training']['learning_rate']}")
    print(f"Config clip_grad_norm: {config['training'].get('clip_grad_norm', 'NOT SET - using hardcoded 1.0')}")
    
    # Add diagnostics
    add_diagnostics_to_trainer()
    
    # Create trainer
    trainer = Trainer(
        config=config,
        model_type='kl',
        enable_diagnostics=True
    )
    
    # Check if checkpoint exists
    checkpoint_path = Path("checkpoints/experiment1/kl_no_mixed/checkpoint_epoch_20.pth")
    if checkpoint_path.exists():
        print(f"\n✓ Found checkpoint: {checkpoint_path}")
        print("Loading checkpoint...")
        trainer.load_checkpoint(checkpoint_path)
        print(f"Resuming from epoch {trainer.current_epoch}")
    else:
        print(f"\n✗ No checkpoint found at {checkpoint_path}")
        print("Starting from scratch (will run a few epochs with diagnostics)")
        # Reduce max epochs for testing
        config['training']['max_epochs'] = 5
    
    # Train
    try:
        results = trainer.train()
        print("\n✅ Training completed successfully!")
        print(f"Final results: {results}")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
