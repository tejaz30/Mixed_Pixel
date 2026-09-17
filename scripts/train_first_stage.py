"""
Training script for first-stage autoencoders (KL and VQ).

Usage:
    python scripts/train_first_stage.py --config configs/experiment1/kl_autoencoder.yaml
    python scripts/train_first_stage.py --config configs/experiment1/vq_autoencoder.yaml
"""

import sys
import argparse
from pathlib import Path
import time
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import numpy as np

from models.ldm.first_stage.kl import KLAutoencoder, KLAutoencoderLoss
from models.ldm.first_stage.vq import VQAutoencoder, VQAutoencoderLoss
from data import ThermalImageDataset, get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_directories(paths: dict):
    """Create necessary directories."""
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)


def get_device(device_name: str = 'auto') -> torch.device:
    """Get torch device."""
    if device_name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_name)


def create_model(config: dict) -> torch.nn.Module:
    """Create model based on configuration."""
    model_config = config['model']
    model_type = model_config['type']
    
    if model_type == 'kl_autoencoder':
        model = KLAutoencoder(
            in_channels=model_config['in_channels'],
            latent_channels=model_config['latent_channels'],
            base_channels=model_config['base_channels'],
            channel_multipliers=tuple(model_config['channel_multipliers']),
            num_res_blocks=model_config['num_res_blocks']
        )
    elif model_type == 'vq_autoencoder':
        model = VQAutoencoder(
            in_channels=model_config['in_channels'],
            latent_channels=model_config['latent_channels'],
            base_channels=model_config['base_channels'],
            channel_multipliers=tuple(model_config['channel_multipliers']),
            num_res_blocks=model_config['num_res_blocks'],
            num_embeddings=model_config['num_embeddings'],
            commitment_cost=model_config['commitment_cost']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def create_criterion(config: dict):
    """Create loss criterion based on model type."""
    model_type = config['model']['type']
    
    if model_type == 'kl_autoencoder':
        kl_weight = config['model'].get('kl_weight', 1e-6)
        return KLAutoencoderLoss(kl_weight=kl_weight)
    elif model_type == 'vq_autoencoder':
        return VQAutoencoderLoss()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_dataloaders(config: dict) -> tuple:
    """Create train and validation dataloaders."""
    data_config = config['data']
    
    # Create transforms
    train_transform = get_train_transforms(augment=False, normalize=False)
    val_transform = get_val_transforms(normalize=False)
    
    # Create full dataset
    full_dataset = ThermalImageDataset(
        root_dir=data_config['dataset_root'],
        transform=None,  # Apply later
        load_mode=data_config['load_mode']
    )
    
    print(f"Total dataset size: {len(full_dataset)} images")
    print(f"Number of classes: {len(full_dataset.classes)}")
    
    # Split dataset
    train_ratio = data_config['train_ratio']
    val_ratio = data_config['val_ratio']
    
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    generator = torch.Generator().manual_seed(data_config['seed'])
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )
    
    # Wrap with transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=data_config['shuffle'],
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory']
    )
    
    return train_loader, val_loader


def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer = None,
    log_frequency: int = 50
) -> dict:
    """Train for one epoch."""
    model.train()
    
    epoch_losses = {}
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx == 0:
            print(f"Loading first batch...")
        
        images = batch['image'].to(device)
        
        if batch_idx == 0:
            print(f"First batch loaded. Shape: {images.shape}")
            print(f"Running forward pass...")
        
        # Forward pass
        if isinstance(model, KLAutoencoder):
            reconstruction, posterior = model(images, sample=True)
            losses = criterion(reconstruction, images, posterior)
        elif isinstance(model, VQAutoencoder):
            reconstruction, vq_outputs = model(images)
            losses = criterion(reconstruction, images, vq_outputs)
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
        
        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
        
        if batch_idx == 0:
            print(f"First batch complete! Total loss: {losses['total'].item():.6f}")
        
        # Accumulate losses
        for key, value in losses.items():
            if torch.is_tensor(value):
                value = value.item()
            if key not in epoch_losses:
                epoch_losses[key] = 0.0
            epoch_losses[key] += value
        
        num_batches += 1
        
        # TensorBoard logging
        if writer and batch_idx % log_frequency == 0:
            global_step = epoch * len(train_loader) + batch_idx
            for key, value in losses.items():
                if torch.is_tensor(value):
                    value = value.item()
                writer.add_scalar(f'Train_batch/{key}', value, global_step)
    
    # Average losses
    epoch_losses_avg = {k: v / num_batches for k, v in epoch_losses.items()}
    
    return epoch_losses_avg


def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion,
    device: torch.device
) -> dict:
    """Validate model."""
    model.eval()
    
    val_losses = {}
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            
            # Forward pass
            if isinstance(model, KLAutoencoder):
                reconstruction, posterior = model(images, sample=False)  # Use mean
                losses = criterion(reconstruction, images, posterior)
            elif isinstance(model, VQAutoencoder):
                reconstruction, vq_outputs = model(images)
                losses = criterion(reconstruction, images, vq_outputs)
            else:
                raise ValueError(f"Unknown model type: {type(model)}")
            
            # Accumulate losses
            for key, value in losses.items():
                if torch.is_tensor(value):
                    value = value.item()
                if key not in val_losses:
                    val_losses[key] = 0.0
                val_losses[key] += value
            
            num_batches += 1
    
    # Average losses
    val_losses_avg = {k: v / num_batches for k, v in val_losses.items()}
    
    return val_losses_avg


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop


def main():
    parser = argparse.ArgumentParser(description="Train first-stage autoencoder")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    args = parser.parse_args()
    
    # Load config
    config = load_config(Path(args.config))
    print("="*70)
    print(f"EXPERIMENT: {config['experiment']['name']}")
    print(f"Description: {config['experiment']['description']}")
    print("="*70)
    
    # Set seed
    if config['reproducibility']['deterministic']:
        set_seed(config['reproducibility']['seed'])
        print(f"Random seed set to: {config['reproducibility']['seed']}")
    
    # Create directories
    create_directories(config['paths'])
    
    # Setup logging
    log_file = Path(config['paths']['log_dir']) / 'train.log'
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Get device
    device = get_device(config['hardware']['device'])
    print(f"Device: {device}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    
    model = model.to(device)
    print(model)
    print()
    
    params = model.count_parameters()
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print()
    
    # Create criterion
    criterion = create_criterion(config)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=config['training']['betas'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    print()
    
    # TensorBoard
    writer = None
    if config['logging']['tensorboard']['enabled']:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(config['paths']['tensorboard_dir'])
    
    # Early stopping
    early_stopping = None
    if config['training']['early_stopping']['enabled']:
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta'],
            mode='min'
        )
    
    # Training loop
    print("Starting training...")
    print("="*70)
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")
    print(f"Batch size: {config['data']['batch_size']}")
    print("="*70)
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(config['training']['max_epochs']):
        epoch_start = time.time()
        
        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, writer, config['logging']['tensorboard']['log_frequency']
        )
        
        # Validate
        val_losses = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        # Log
        log_str = f"Epoch {epoch+1}/{config['training']['max_epochs']} | "
        log_str += f"Train Loss: {train_losses['total']:.6f} | "
        log_str += f"Val Loss: {val_losses['total']:.6f} | "
        
        if 'reconstruction' in val_losses:
            log_str += f"Val Recon: {val_losses['reconstruction']:.6f} | "
        
        log_str += f"Time: {epoch_time:.2f}s"
        print(log_str)
        logger.info(log_str)
        
        # TensorBoard
        if writer:
            for key, value in train_losses.items():
                writer.add_scalar(f'Train_epoch/{key}', value, epoch)
            for key, value in val_losses.items():
                writer.add_scalar(f'Val_epoch/{key}', value, epoch)
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_frequency'] == 0:
            checkpoint_path = Path(config['paths']['checkpoint_dir']) / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': config
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_path = Path(config['paths']['checkpoint_dir']) / 'best_model.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'config': config
            }, best_path)
            print(f"  [BEST] Model saved: {best_path} (val_loss: {best_val_loss:.6f})")
        
        # Early stopping
        if early_stopping:
            if early_stopping(val_losses['total']):
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    total_time = time.time() - start_time
    print("="*70)
    print(f"Training complete! Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("="*70)
    
    if writer:
        writer.close()


if __name__ == '__main__':
    main()
