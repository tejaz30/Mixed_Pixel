"""
Smoke test for training pipeline - quick validation with real dataset subset.
"""

import torch
import yaml
from pathlib import Path
from typing import Optional
import copy

from .trainer import Trainer


def create_smoke_config(base_config_path: Path) -> dict:
    """
    Create smoke test configuration from base config.
    
    Args:
        base_config_path: Path to base configuration file
    
    Returns:
        Smoke test configuration dictionary
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify for smoke test
    smoke_config = copy.deepcopy(config)
    
    # Hardware: Use GPU if available (auto)
    smoke_config['hardware']['device'] = 'auto'
    smoke_config['hardware']['mixed_precision'] = False
    
    # Model: Keep approved architecture (base_channels=32, no changes)
    
    # Training: Minimal epochs for quick test
    smoke_config['training']['max_epochs'] = 2
    smoke_config['training']['save_frequency'] = 1
    smoke_config['training']['early_stopping']['enabled'] = False
    
    # Data: Small batch size
    smoke_config['data']['batch_size'] = 2
    smoke_config['data']['num_workers'] = 0
    smoke_config['data']['pin_memory'] = False
    
    # Paths: Use temp directory
    smoke_dir = Path('temp_smoke_test')
    smoke_config['paths'] = {
        'checkpoint_dir': str(smoke_dir / 'checkpoints'),
        'log_dir': str(smoke_dir / 'logs'),
        'tensorboard_dir': str(smoke_dir / 'tensorboard'),
        'output_dir': str(smoke_dir / 'outputs'),
    }
    
    return smoke_config


def run_smoke_test(
    config_path: Path,
    model_type: str,
    max_samples: int = 20,
    dataset_root: Optional[str] = None
) -> bool:
    """
    Run smoke test on training pipeline.
    
    Args:
        config_path: Path to configuration file
        model_type: 'kl_autoencoder' or 'vq_autoencoder'
        max_samples: Maximum number of samples to use (default: 20)
        dataset_root: Override dataset path (default: use No_Mixed from config)
    
    Returns:
        True if smoke test passed, False otherwise
    """
    print("="*70)
    print("SMOKE TEST - Training Pipeline Validation")
    print("="*70)
    print(f"Config: {config_path}")
    print(f"Model: {model_type}")
    print(f"Max samples: {max_samples}")
    print(f"Device: auto (GPU if available)")
    print("="*70)
    
    try:
        # Create smoke config
        config = create_smoke_config(config_path)
        
        # Use real dataset (No_Mixed by default, or override)
        if dataset_root is not None:
            config['data']['dataset_root'] = dataset_root
        else:
            # Use No_Mixed from processed datasets
            config['data']['dataset_root'] = 'src/datasets/processed/No_Mixed'
        
        print(f"\n0. Using dataset: {config['data']['dataset_root']}")
        
        # Verify dataset exists
        dataset_path = Path(config['data']['dataset_root'])
        if not dataset_path.exists():
            print(f"✗ Dataset not found: {dataset_path}")
            print(f"   Please ensure the dataset exists or provide --dataset-root")
            return False
        print(f"   Dataset OK")
        
        # Create trainer
        print("\n1. Initializing trainer...")
        trainer = Trainer(
            config=config,
            model_type=model_type,
            enable_diagnostics=False
        )
        
        print("✓ Trainer initialized")
        
        # Check model can be created
        print("\n2. Checking model...")
        print(f"   Model type: {trainer.model.__class__.__name__}")
        print(f"   Parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
        print("✓ Model OK")
        
        # Check dataloaders
        print("\n3. Checking dataloaders...")
        print(f"   Train batches: {len(trainer.train_loader)}")
        print(f"   Val batches: {len(trainer.val_loader)}")
        
        # Get a batch
        batch = next(iter(trainer.train_loader))
        print(f"   Batch shape: {batch['image'].shape}")
        print("✓ Dataloaders OK")
        
        # Check forward pass
        print("\n4. Testing forward pass...")
        trainer.model.eval()
        with torch.no_grad():
            images = batch['image'].to(trainer.device)
            
            if model_type == 'kl_autoencoder':
                reconstruction, posterior = trainer.model(images)
                print(f"   Input: {images.shape}")
                print(f"   Reconstruction: {reconstruction.shape}")
                print(f"   Posterior mean: {posterior.mean.shape}")
            else:  # VQ
                reconstruction, vq_outputs = trainer.model(images)
                print(f"   Input: {images.shape}")
                print(f"   Reconstruction: {reconstruction.shape}")
                # VQ outputs keys: z_quantized, z_continuous, etc.
                if 'z_quantized' in vq_outputs:
                    print(f"   Quantized: {vq_outputs['z_quantized'].shape}")
                elif 'quantized' in vq_outputs:
                    print(f"   Quantized: {vq_outputs['quantized'].shape}")
        
        print("✓ Forward pass OK")
        
        # Check loss computation
        print("\n5. Testing loss computation...")
        if model_type == 'kl_autoencoder':
            losses = trainer.criterion(reconstruction, images, posterior)
        else:
            losses = trainer.criterion(reconstruction, images, vq_outputs)
        
        print(f"   Total loss: {losses['total'].item():.6f}")
        for key, value in losses.items():
            if key != 'total':
                val = value.item() if torch.is_tensor(value) else value
                print(f"   {key}: {val:.6f}")
        
        print("✓ Loss computation OK")
        
        # Check backward pass
        print("\n6. Testing backward pass...")
        trainer.model.train()
        trainer.optimizer.zero_grad()
        
        if model_type == 'kl_autoencoder':
            reconstruction, posterior = trainer.model(images)
            losses = trainer.criterion(reconstruction, images, posterior)
        else:
            reconstruction, vq_outputs = trainer.model(images)
            losses = trainer.criterion(reconstruction, images, vq_outputs)
        
        losses['total'].backward()
        trainer.optimizer.step()
        
        # Check gradients were computed
        has_grads = any(p.grad is not None for p in trainer.model.parameters())
        if not has_grads:
            print("✗ No gradients computed!")
            return False
        
        print("✓ Backward pass OK")
        
        # Run mini training loop
        print("\n7. Running mini training (2 epochs)...")
        
        # Limit dataset size
        if len(trainer.train_loader.dataset) > max_samples:
            # Create subset
            indices = list(range(max_samples))
            subset = torch.utils.data.Subset(trainer.train_loader.dataset, indices)
            trainer.train_loader = torch.utils.data.DataLoader(
                subset,
                batch_size=config['data']['batch_size'],
                shuffle=False,
                num_workers=0
            )
            
            val_size = max(2, max_samples // 5)
            val_indices = list(range(val_size))
            val_subset = torch.utils.data.Subset(trainer.val_loader.dataset, val_indices)
            trainer.val_loader = torch.utils.data.DataLoader(
                val_subset,
                batch_size=config['data']['batch_size'],
                shuffle=False,
                num_workers=0
            )
            
            print(f"   Using subset: {max_samples} train, {val_size} val samples")
        
        # Train for 2 epochs
        for epoch in range(2):
            print(f"\n   Epoch {epoch + 1}/2")
            
            # Train
            train_losses = trainer.train_epoch()
            print(f"   Train loss: {train_losses['total']:.6f}")
            
            # Validate
            val_losses = trainer.validate()
            print(f"   Val loss: {val_losses['total']:.6f}")
        
        print("\n✓ Mini training OK")
        
        # Check checkpoint saving
        print("\n8. Testing checkpoint saving...")
        trainer.save_checkpoint(1, train_losses, val_losses)
        trainer.save_best_model(1, val_losses['total'])
        
        checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        checkpoint_exists = (checkpoint_dir / 'checkpoint_epoch_1.pth').exists()
        best_exists = (checkpoint_dir / 'best_model.pth').exists()
        
        if not checkpoint_exists:
            print("✗ Checkpoint not saved!")
            return False
        if not best_exists:
            print("✗ Best model not saved!")
            return False
        
        print("✓ Checkpoint saving OK")
        
        # Cleanup
        print("\n9. Cleaning up...")
        import shutil
        smoke_dir = Path('temp_smoke_test')
        if smoke_dir.exists():
            shutil.rmtree(smoke_dir)
        print("✓ Cleanup OK")
        
        print("\n" + "="*70)
        print("✓ SMOKE TEST PASSED!")
        print("="*70)
        
        return True
    
    except Exception as e:
        print(f"\n✗ SMOKE TEST FAILED!")
        print(f"Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python -m src.training.smoke_test <config_path> <model_type> [--dataset-root PATH]")
        print("Example: python -m src.training.smoke_test configs/experiment1/kl_autoencoder.yaml kl_autoencoder")
        print("         python -m src.training.smoke_test configs/experiment1/kl_autoencoder.yaml kl_autoencoder --dataset-root datasets/processed/Chilli_Leaves")
        sys.exit(1)
    
    config_path = Path(sys.argv[1])
    model_type = sys.argv[2]
    
    # Parse optional dataset override
    dataset_root = None
    if '--dataset-root' in sys.argv:
        idx = sys.argv.index('--dataset-root')
        if idx + 1 < len(sys.argv):
            dataset_root = sys.argv[idx + 1]
    
    success = run_smoke_test(config_path, model_type, dataset_root=dataset_root)
    sys.exit(0 if success else 1)
