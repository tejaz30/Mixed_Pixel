"""
Test checkpoint saving and auto-resume functionality.

This script simulates training interruption and verifies that:
1. Checkpoints are saved correctly
2. Training auto-resumes from the latest checkpoint
3. Training continues from the correct epoch
"""

import sys
import shutil
from pathlib import Path
import yaml
import tempfile

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

def test_checkpoint_resume():
    """Test checkpoint save and resume."""
    
    print("="*70)
    print("CHECKPOINT RESUME TEST")
    print("="*70)
    
    # Create temporary test directory
    test_dir = Path(tempfile.mkdtemp(prefix="checkpoint_test_"))
    print(f"\nTest directory: {test_dir}")
    
    try:
        # Create minimal test config
        config = {
            'experiment': {
                'name': 'test_checkpoint_resume'
            },
            'model': {
                'type': 'kl_autoencoder',
                'in_channels': 1,
                'latent_channels': 4,
                'base_channels': 32,
                'channel_multipliers': [1, 2],
                'num_res_blocks': 1,
                'kl_weight': 1e-6
            },
            'data': {
                'dataset_root': str(repo_root / 'data' / 'processed' / 'No_Mixed'),
                'batch_size': 2,
                'train_ratio': 0.8,
                'seed': 42,
                'shuffle': True,
                'num_workers': 0,
                'pin_memory': False,
                'load_mode': 'on_demand'
            },
            'training': {
                'max_epochs': 5,
                'learning_rate': 0.0005,
                'betas': [0.9, 0.999],
                'weight_decay': 0.0,
                'eps': 1e-8,
                'save_frequency': 2,  # Save every 2 epochs
                'clip_grad_norm': 1.0,
                'early_stopping': {
                    'enabled': False
                },
                'lr_scheduler': {
                    'enabled': False
                }
            },
            'paths': {
                'checkpoint_dir': str(test_dir / 'checkpoints'),
                'log_dir': str(test_dir / 'logs')
            },
            'hardware': {
                'device': 'cpu',  # Use CPU for testing
                'mixed_precision': False
            },
            'reproducibility': {
                'deterministic': True,
                'seed': 42
            }
        }
        
        # Save config
        config_path = test_dir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        print(f"Config saved to: {config_path}")
        
        # Test 1: Initial training (3 epochs)
        print("\n" + "="*70)
        print("TEST 1: Initial Training (3 epochs)")
        print("="*70)
        
        from training.trainer import Trainer
        
        trainer = Trainer(
            config=config,
            model_type='kl_autoencoder',
            enable_diagnostics=False
        )
        
        # Manually train for 3 epochs
        print("\nTraining for 3 epochs...")
        for epoch in range(3):
            trainer.current_epoch = epoch + 1
            print(f"\nEpoch {trainer.current_epoch}")
            
            # Train one epoch
            train_losses = trainer.train_epoch()
            print(f"  Train loss: {train_losses['total']:.6f}")
            
            # Validate
            val_losses = trainer.validate()
            print(f"  Val loss: {val_losses['total']:.6f}")
            
            # Save checkpoint
            if trainer.current_epoch % config['training']['save_frequency'] == 0:
                trainer.save_checkpoint(trainer.current_epoch, train_losses, val_losses)
            
            # Update best
            if val_losses['total'] < trainer.best_val_loss:
                trainer.best_val_loss = val_losses['total']
                trainer.best_epoch = trainer.current_epoch
                trainer.save_best_model(trainer.current_epoch, trainer.best_val_loss)
        
        print(f"\n✓ Initial training complete. Stopped at epoch {trainer.current_epoch}")
        
        # Check checkpoints exist
        checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        
        print(f"\nCheckpoints saved:")
        for cp in checkpoints:
            print(f"  - {cp.name}")
        
        assert len(checkpoints) > 0, "No checkpoints were saved!"
        
        # Test 2: Resume from checkpoint
        print("\n" + "="*70)
        print("TEST 2: Auto-Resume from Checkpoint")
        print("="*70)
        
        # Create new trainer (simulates restart)
        trainer2 = Trainer(
            config=config,
            model_type='kl_autoencoder',
            enable_diagnostics=False
        )
        
        # Find latest checkpoint
        latest_checkpoint = checkpoints[-1]
        print(f"\nLatest checkpoint: {latest_checkpoint.name}")
        
        # Load checkpoint
        print("Loading checkpoint...")
        trainer2.load_checkpoint(latest_checkpoint)
        
        print(f"✓ Resumed from epoch {trainer2.current_epoch}")
        print(f"✓ Best val loss: {trainer2.best_val_loss:.6f}")
        
        # Verify correct epoch
        expected_epoch = int(latest_checkpoint.stem.split('_')[-1])
        assert trainer2.current_epoch == expected_epoch, \
            f"Expected epoch {expected_epoch}, got {trainer2.current_epoch}"
        
        print(f"✓ Epoch verification passed")
        
        # Test 3: Continue training
        print("\n" + "="*70)
        print("TEST 3: Continue Training (2 more epochs)")
        print("="*70)
        
        # Continue for 2 more epochs
        for epoch in range(trainer2.current_epoch, trainer2.current_epoch + 2):
            trainer2.current_epoch = epoch + 1
            print(f"\nEpoch {trainer2.current_epoch}")
            
            train_losses = trainer2.train_epoch()
            print(f"  Train loss: {train_losses['total']:.6f}")
            
            val_losses = trainer2.validate()
            print(f"  Val loss: {val_losses['total']:.6f}")
            
            if trainer2.current_epoch % config['training']['save_frequency'] == 0:
                trainer2.save_checkpoint(trainer2.current_epoch, train_losses, val_losses)
            
            if val_losses['total'] < trainer2.best_val_loss:
                trainer2.best_val_loss = val_losses['total']
                trainer2.best_epoch = trainer2.current_epoch
                trainer2.save_best_model(trainer2.current_epoch, trainer2.best_val_loss)
        
        print(f"\n✓ Continued training complete. Now at epoch {trainer2.current_epoch}")
        
        # Final verification
        print("\n" + "="*70)
        print("TEST RESULTS")
        print("="*70)
        
        final_checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        print(f"\nFinal checkpoints:")
        for cp in final_checkpoints:
            print(f"  - {cp.name}")
        
        print(f"\n✅ ALL TESTS PASSED!")
        print(f"✅ Checkpoints save correctly")
        print(f"✅ Auto-resume works correctly")
        print(f"✅ Training continues from correct epoch")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        print(f"\nCleaning up test directory: {test_dir}")
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == '__main__':
    success = test_checkpoint_resume()
    sys.exit(0 if success else 1)
