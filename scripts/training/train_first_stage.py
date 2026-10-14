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
import shutil
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
    parser.add_argument('--resume', type=Path, help='Resume from checkpoint file')
    parser.add_argument('--diagnose', action='store_true', help='Enable detailed NaN diagnostics')
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
    
    # Enable diagnostics if requested
    if args.diagnose:
        print("\n🔍 DIAGNOSTIC MODE ENABLED")
        print("Detailed logging of logvar, gradients, and losses every 50 batches")
        print("="*70)
        # Import from scripts.investigation
        investigation_path = repo_root / 'scripts' / 'investigation'
        sys.path.insert(0, str(investigation_path))
        from diagnose_nan import add_diagnostics_to_trainer
        add_diagnostics_to_trainer()
    
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
    
    # Auto-resume from latest checkpoint (check both Dataset and local)
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    latest_checkpoint = None
    
    # If on Kaggle, try to download checkpoints from Dataset first
    if Path('/kaggle/working').exists():
        print(f"\n{'='*70}")
        print(f"🔍 CHECKING KAGGLE DATASET FOR CHECKPOINTS")
        print(f"{'='*70}")
        
        import os
        import subprocess
        
        kaggle_username = os.environ.get('KAGGLE_USERNAME', 'unknown')
        model_type = config['model']['type'].replace('_', '-')
        dataset_name = f"{kaggle_username}/mixed-pixels-{model_type}-checkpoints"
        
        print(f"Dataset: {dataset_name}")
        
        # Try to download dataset
        try:
            download_dir = Path('/kaggle/temp_download')
            download_dir.mkdir(exist_ok=True)
            
            cmd = f"kaggle datasets download -d {dataset_name} -p {download_dir} --unzip"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Dataset downloaded - copy checkpoints to checkpoint dir
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                downloaded_checkpoints = list(download_dir.glob('*.pth'))
                if downloaded_checkpoints:
                    for ckpt in downloaded_checkpoints:
                        dest = checkpoint_dir / ckpt.name
                        shutil.copy(ckpt, dest)
                        print(f"  ✓ Downloaded: {ckpt.name}")
                    
                    # Clean up download dir
                    shutil.rmtree(download_dir, ignore_errors=True)
                else:
                    print(f"  ⚠️  Dataset exists but no checkpoints found")
            else:
                print(f"  ℹ️  No existing dataset found (starting fresh)")
        except Exception as e:
            print(f"  ⚠️  Could not download dataset: {e}")
            print(f"  ℹ️  Starting fresh training")
    
    # Now check local checkpoint directory for latest checkpoint
    if args.resume:
        # Explicit resume path provided
        if not args.resume.exists():
            print(f"\nError: Checkpoint not found: {args.resume}")
            return 1
        latest_checkpoint = args.resume
    else:
        # Auto-detect latest checkpoint
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
            if checkpoints:
                latest_checkpoint = checkpoints[-1]  # Most recent epoch
    
    if latest_checkpoint:
        print(f"\n{'='*70}")
        print(f"🔄 RESUMING FROM CHECKPOINT")
        print(f"{'='*70}")
        print(f"Checkpoint: {latest_checkpoint}")
        try:
            trainer.load_checkpoint(latest_checkpoint)
            print(f"✓ Resumed from epoch {trainer.current_epoch}")
            print(f"✓ Best val loss so far: {trainer.best_val_loss:.6f}")
            print(f"✓ Will continue training from epoch {trainer.current_epoch + 1}")
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            print(f"⚠️  Starting from scratch instead...")
    else:
        print(f"\n{'='*70}")
        print(f"Starting fresh training (no checkpoint found)")
        print(f"{'='*70}")
    
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
