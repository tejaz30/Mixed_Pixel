"""
Training script for Convolutional Autoencoder (CAE).

Usage:
    python scripts/train_cae.py
    python scripts/train_cae.py --config configs/cae.yaml
    python scripts/train_cae.py --resume checkpoints/cae/checkpoint_epoch_50.pth
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Convolutional Autoencoder for thermal image reconstruction"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/cae.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
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
        help='Batch size (overrides config)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Maximum epochs (overrides config)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Specific dataset to train on (e.g., Chilli_Leaves, Okra, Paddy_Leaves, No_Mixed)'
    )
    
    return parser.parse_args()


def create_data_loaders(config: dict, dataset_name: str = None) -> tuple:
    """
    Create train and validation data loaders.
    
    Args:
        config: Configuration dictionary.
        dataset_name: Optional specific dataset name to use. If None, uses all datasets.
    
    Returns:
        Tuple of (train_loader, val_loader).
    """
    data_config = config['data']
    training_config = config['training']
    
    # Get dataset root
    dataset_root = Path(data_config['dataset_root'])
    
    # If specific dataset requested, use that subdirectory
    if dataset_name:
        dataset_root = dataset_root / dataset_name
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_root}")
    
    # Create transforms
    transforms = get_train_transforms()
    
    # Create full dataset
    full_dataset = ThermalImageDataset(
        root_dir=dataset_root,
        transform=transforms,
        return_path=False
    )
    
    print(f"Total dataset size: {len(full_dataset)} images")
    print(f"Number of classes: {len(full_dataset.classes)}")
    print(f"Classes: {full_dataset.classes}")
    
    # Split into train and validation (paper: 80/20)
    train_ratio = data_config['train_ratio']
    val_ratio = data_config['val_ratio']
    
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Use fixed seed for reproducible split
    generator = torch.Generator().manual_seed(data_config['seed'])
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=data_config['shuffle'],
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory']
    )
    
    return train_loader, val_loader


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(Path(args.config))
    
    # Override config with command line arguments
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['max_epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # Adjust paths if training specific dataset
    dataset_suffix = f"/{args.dataset}" if args.dataset else ""
    
    # Create directories
    paths_config = config['paths']
    checkpoint_dir = Path(paths_config['checkpoint_dir'] + dataset_suffix)
    log_dir = Path(paths_config['log_dir'] + dataset_suffix)
    tensorboard_dir = Path(paths_config['tensorboard_dir'] + dataset_suffix)
    
    create_directories([
        checkpoint_dir,
        log_dir,
        tensorboard_dir
    ])
    
    # Setup logging
    log_config = config['logging']
    log_file = None
    if log_config['log_to_file']:
        log_file = log_dir / "train.log"
    
    setup_logging(
        log_level=log_config['level'],
        log_file=log_file
    )
    
    print("=" * 70)
    print("CONVOLUTIONAL AUTOENCODER TRAINING")
    print("=" * 70)
    print(f"Experiment: {config['experiment']['name']}")
    if args.dataset:
        print(f"Dataset: {args.dataset}")
    else:
        print(f"Dataset: All datasets combined")
    print(f"Description: {config['experiment']['description']}")
    print("")
    
    # Set random seeds for reproducibility
    if config['reproducibility']['deterministic']:
        seed_everything(config['reproducibility']['seed'])
        print(f"✓ Random seed set to: {config['reproducibility']['seed']}")
    
    # Get device
    if args.device:
        device = get_device(args.device)
    else:
        device_config = config['hardware']['device']
        device = get_device(device_config if device_config != 'auto' else None)
    
    print(f"✓ Device: {device}")
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
    
    # Print model info
    print(model)
    print("")
    params = count_parameters(model)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print("")
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader = create_data_loaders(config, args.dataset)
    print("")
    
    # Create trainer
    print("Initializing trainer...")
    training_config = config['training']
    trainer = CAETrainer(
        model=model,
        device=device,
        checkpoint_dir=checkpoint_dir,
        tensorboard_dir=tensorboard_dir if log_config['tensorboard']['enabled'] else None,
        learning_rate=training_config['learning_rate'],
        max_epochs=training_config['max_epochs'],
        patience=training_config['early_stopping']['patience'],
        save_frequency=training_config['save_frequency']
    )
    print("✓ Trainer initialized")
    print("")
    
    # Resume from checkpoint if specified
    resume_path = Path(args.resume) if args.resume else None
    
    # Train model
    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            resume_from=resume_path
        )
        
        # Print summary
        print("")
        print("=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        print(f"Epochs trained: {history['epochs_trained']}")
        print(f"Best validation loss: {history['best_val_loss']:.6f}")
        print(f"Total training time: {history['training_time']:.2f}s")
        print(f"Average time per epoch: {history['training_time'] / history['epochs_trained']:.2f}s")
        print("")
        print(f"✓ Best model saved to: {checkpoint_dir}/best_model.pth")
        print("=" * 70)
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
