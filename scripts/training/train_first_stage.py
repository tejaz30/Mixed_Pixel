"""
Training script for first-stage autoencoders (KL and VQ).

Usage:
    python scripts/training/train_first_stage.py <config_path>
    
Example:
    python scripts/training/train_first_stage.py configs/experiment1/kl_autoencoder.yaml
    python scripts/training/train_first_stage.py configs/experiment1/kl_autoencoder.yaml --max-epochs 5
"""

import argparse
import sys
from pathlib import Path
import yaml
import logging

# Add src to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train first-stage autoencoder')
    parser.add_argument('config', type=Path, help='Path to config file')
    parser.add_argument('--model-type', type=str, choices=['kl_autoencoder', 'vq_autoencoder'],
                       help='Model type (if not specified in config)')
    parser.add_argument('--max-epochs', type=int, help='Override max epochs')
    parser.add_argument('--device', type=str, help='Override device (auto/cuda/cpu)')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    args = parser.parse_args()
    
    # Validate config exists
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        return 1
    
    # Load config
    print("="*70)
    print("FIRST-STAGE AUTOENCODER TRAINING")
    print("="*70)
    print(f"Config: {args.config}")
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    if args.max_epochs:
        config['training']['max_epochs'] = args.max_epochs
        print(f"Override: max_epochs = {args.max_epochs}")
    
    if args.device:
        config['hardware']['device'] = args.device
        print(f"Override: device = {args.device}")
    
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
        print(f"Override: batch_size = {args.batch_size}")
    
    # Determine model type
    model_type = args.model_type or config['model']['type']
    print(f"Model type: {model_type}")
    print("="*70)
    
    # Create trainer
    print("\nInitializing trainer...")
    try:
        trainer = Trainer(
            config=config,
            model_type=model_type,
            enable_diagnostics=True
        )
    except Exception as e:
        print(f"Error creating trainer: {e}")
        return 1
    
    print(f"\nConfiguration:")
    print(f"  Max epochs: {config['training']['max_epochs']}")
    print(f"  Batch size: {config['data']['batch_size']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Device: {trainer.device}")
    print(f"  Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"  Dataset: {config['data']['dataset_root']}")
    print(f"  Train samples: {len(trainer.train_loader.dataset)}")
    print(f"  Val samples: {len(trainer.val_loader.dataset)}")
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)
    
    # Train
    try:
        results = trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best validation loss: {results['best_val_loss']:.6f}")
    print(f"Best epoch: {results['best_epoch']}")
    print(f"Total epochs: {results['epochs_completed']}")
    print(f"\nOutputs:")
    print(f"  Best model: {config['paths']['checkpoint_dir']}/best_model.pth")
    print(f"  Checkpoints: {config['paths']['checkpoint_dir']}/")
    print(f"  Logs: {config['paths']['log_dir']}/")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
