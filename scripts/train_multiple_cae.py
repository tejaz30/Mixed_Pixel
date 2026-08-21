"""
Train multiple CAE models, one for each dataset.

This script trains separate CAE models for each thermal image dataset:
- Chilli_Leaves
- Okra  
- Paddy_Leaves
- No_Mixed

Each model is trained independently with dataset-specific configurations.

Usage:
    python scripts/train_multiple_cae.py
    python scripts/train_multiple_cae.py --datasets Chilli_Leaves Okra
    python scripts/train_multiple_cae.py --skip-completed
"""

import sys
import argparse
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader, Subset

from models.cae import (
    ConvolutionalAutoencoder,
    CAETrainer,
    get_device,
    load_config,
    setup_logging,
    count_parameters,
    seed_everything,
    create_directories
)
from data import ThermalImageDataset, get_train_transforms


# Dataset configurations
DATASET_CONFIGS = {
    'Chilli_Leaves': {
        'name': 'Chilli_Leaves',
        'classes': ['Bacterial Spot', 'Healthy Leaves', 'Leaf Curl', 'Powdery Mildew'],
        'expected_count': 1454
    },
    'Okra': {
        'name': 'Okra',
        'classes': ['adequate_matured_Okra', 'over_matured_Okra'],
        'expected_count': 501
    },
    'Paddy_Leaves': {
        'name': 'Paddy_Leaves',
        'classes': ['Blast', 'BLB', 'healthy', 'hispa', 'leaf folder', 'leaf spot'],
        'expected_count': 636
    },
    'No_Mixed': {
        'name': 'No_mixed',
        'classes': ['No_mixed'],
        'expected_count': 2068
    }
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train multiple CAE models for different thermal image datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/cae.yaml',
        help='Base configuration file'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=list(DATASET_CONFIGS.keys()),
        choices=list(DATASET_CONFIGS.keys()),
        help='Datasets to train on'
    )
    
    parser.add_argument(
        '--skip-completed',
        action='store_true',
        help='Skip training if best_model.pth already exists'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to train on (cuda, cpu). If not specified, uses config.'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config for all datasets)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Maximum epochs (overrides config for all datasets)'
    )
    
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Train sequentially (one at a time) instead of showing progress'
    )
    
    return parser.parse_args()


def create_dataset_loader(dataset_name: str, config: dict, split: str = 'train') -> DataLoader:
    """
    Create data loader for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'Chilli_Leaves')
        config: Configuration dictionary
        split: 'train' or 'val'
    
    Returns:
        DataLoader for the specified dataset and split
    """
    data_config = config['data']
    training_config = config['training']
    
    # Get dataset root
    dataset_root = Path(data_config['dataset_root']) / dataset_name
    
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")
    
    # Create transforms
    transforms = get_train_transforms()
    
    # Create dataset for this specific dataset
    dataset = ThermalImageDataset(
        root_dir=dataset_root,
        transform=transforms,
        return_metadata=False
    )
    
    # Split into train and validation
    train_ratio = data_config['train_ratio']
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    # Use fixed seed for reproducible split
    generator = torch.Generator().manual_seed(data_config['seed'])
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )
    
    # Select the appropriate split
    split_dataset = train_dataset if split == 'train' else val_dataset
    
    # Create data loader
    loader = DataLoader(
        split_dataset,
        batch_size=training_config['batch_size'],
        shuffle=(split == 'train' and data_config['shuffle']),
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory']
    )
    
    return loader, len(dataset), dataset.class_names


def train_single_model(
    dataset_name: str,
    config: dict,
    device: torch.device,
    skip_completed: bool = False
) -> dict:
    """
    Train CAE model for a single dataset.
    
    Args:
        dataset_name: Name of the dataset
        config: Configuration dictionary
        device: Device to train on
        skip_completed: Skip if model already exists
    
    Returns:
        Training results dictionary
    """
    print("\n" + "=" * 80)
    print(f"TRAINING CAE FOR: {dataset_name}")
    print("=" * 80)
    
    # Setup paths
    checkpoint_dir = Path(config['paths']['checkpoint_dir']) / dataset_name
    log_dir = Path(config['paths']['log_dir']) / dataset_name
    tensorboard_dir = Path(config['paths']['tensorboard_dir']) / dataset_name
    
    # Check if already completed
    best_model_path = checkpoint_dir / "best_model.pth"
    if skip_completed and best_model_path.exists():
        print(f"✓ Model already trained: {best_model_path}")
        print(f"  Skipping training for {dataset_name}")
        return {'status': 'skipped', 'reason': 'already_completed'}
    
    # Create directories
    create_directories([checkpoint_dir, log_dir, tensorboard_dir])
    
    # Setup logging for this dataset
    log_file = log_dir / "train.log"
    setup_logging(
        log_level=config['logging']['level'],
        log_file=log_file
    )
    
    print(f"\nDataset: {dataset_name}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"TensorBoard dir: {tensorboard_dir}")
    print("")
    
    # Create data loaders for this dataset
    try:
        train_loader, total_size, class_names = create_dataset_loader(
            dataset_name, config, split='train'
        )
        val_loader, _, _ = create_dataset_loader(
            dataset_name, config, split='val'
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return {'status': 'failed', 'reason': str(e)}
    
    print(f"Total dataset size: {total_size} images")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print("")
    
    # Create model
    print("Creating model...")
    model_config = config['model']
    model = ConvolutionalAutoencoder(
        input_channels=model_config['input_channels'],
        input_height=model_config['input_height'],
        input_width=model_config['input_width'],
        encoder_filters=tuple(model_config['encoder_filters']),
        kernel_size=model_config['kernel_size'],
        pool_size=model_config['pool_size'],
        padding=model_config['padding']
    )
    
    print(model)
    print("")
    params = count_parameters(model)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print("")
    
    # Create trainer
    print("Initializing trainer...")
    training_config = config['training']
    trainer = CAETrainer(
        model=model,
        device=device,
        checkpoint_dir=checkpoint_dir,
        tensorboard_dir=tensorboard_dir if config['logging']['tensorboard']['enabled'] else None,
        learning_rate=training_config['learning_rate'],
        max_epochs=training_config['max_epochs'],
        patience=training_config['early_stopping']['patience'],
        save_frequency=training_config['save_frequency']
    )
    print("✓ Trainer initialized")
    print("")
    
    # Train model
    start_time = datetime.now()
    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        # Print summary
        print("")
        print("=" * 80)
        print(f"TRAINING COMPLETE: {dataset_name}")
        print("=" * 80)
        print(f"Epochs trained: {history['epochs_trained']}")
        print(f"Best validation loss: {history['best_val_loss']:.6f}")
        print(f"Total training time: {training_duration:.2f}s ({training_duration/60:.2f}m)")
        print(f"Average time per epoch: {training_duration / history['epochs_trained']:.2f}s")
        print(f"✓ Best model saved to: {checkpoint_dir}/best_model.pth")
        print("=" * 80)
        
        return {
            'status': 'success',
            'dataset': dataset_name,
            'epochs_trained': history['epochs_trained'],
            'best_val_loss': history['best_val_loss'],
            'training_time': training_duration,
            'checkpoint_path': str(best_model_path),
            'total_images': total_size,
            'num_classes': len(class_names)
        }
    
    except Exception as e:
        print(f"\n❌ Training failed for {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'failed',
            'dataset': dataset_name,
            'error': str(e)
        }


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(Path(args.config))
    
    # Override config with command line arguments
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['max_epochs'] = args.epochs
    
    # Set random seeds for reproducibility
    if config['reproducibility']['deterministic']:
        seed_everything(config['reproducibility']['seed'])
    
    # Get device
    if args.device:
        device = get_device(args.device)
    else:
        device_config = config['hardware']['device']
        device = get_device(device_config if device_config != 'auto' else None)
    
    print("\n" + "=" * 80)
    print("MULTIPLE CAE MODEL TRAINING")
    print("=" * 80)
    print(f"Datasets to train: {', '.join(args.datasets)}")
    print(f"Device: {device}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Max epochs: {config['training']['max_epochs']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Early stopping patience: {config['training']['early_stopping']['patience']}")
    if args.skip_completed:
        print(f"✓ Skip completed: Enabled")
    print("=" * 80)
    
    # Train models for each dataset
    results = []
    for i, dataset_name in enumerate(args.datasets, 1):
        print(f"\n[{i}/{len(args.datasets)}] Training on {dataset_name}...")
        
        result = train_single_model(
            dataset_name=dataset_name,
            config=config,
            device=device,
            skip_completed=args.skip_completed
        )
        results.append(result)
    
    # Save overall results
    results_dir = Path('results/training')
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'config': {
                'batch_size': config['training']['batch_size'],
                'max_epochs': config['training']['max_epochs'],
                'learning_rate': config['training']['learning_rate'],
                'patience': config['training']['early_stopping']['patience']
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("OVERALL TRAINING SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    skipped = [r for r in results if r['status'] == 'skipped']
    
    print(f"Total datasets: {len(results)}")
    print(f"✓ Successful: {len(successful)}")
    if skipped:
        print(f"⊘ Skipped: {len(skipped)}")
    if failed:
        print(f"✗ Failed: {len(failed)}")
    
    print("\nPer-Dataset Results:")
    for result in successful:
        print(f"  • {result['dataset']:<20} - Loss: {result['best_val_loss']:.6f}, "
              f"Epochs: {result['epochs_trained']:3d}, "
              f"Time: {result['training_time']/60:.1f}m")
    
    for result in skipped:
        print(f"  ⊘ {result['dataset']:<20} - Skipped ({result['reason']})")
    
    for result in failed:
        print(f"  ✗ {result['dataset']:<20} - Failed: {result.get('error', 'Unknown error')}")
    
    if successful:
        total_time = sum(r['training_time'] for r in successful)
        print(f"\nTotal training time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
    
    print("=" * 80)
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
