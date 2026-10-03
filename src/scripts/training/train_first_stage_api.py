"""
API wrapper for training first-stage autoencoders from notebooks.

This module provides a programmatic interface to train_first_stage.py
for use in Jupyter notebooks and other Python scripts.
"""

import sys
from pathlib import Path
import time
import logging
from typing import Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from models.ldm.first_stage.kl import KLAutoencoder, KLAutoencoderLoss
from models.ldm.first_stage.vq import VQAutoencoder, VQAutoencoderLoss
from data import ThermalImageDataset, get_train_transforms, get_val_transforms


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_directories(paths: dict):
    """Create necessary directories."""
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)


def get_device(device_name: str = 'auto') -> torch.device:
    """Get torch device."""
    if device_name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_name)


def create_model(config: dict, device: torch.device) -> torch.nn.Module:
    """Create model based on configuration with multi-GPU support."""
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
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs with DataParallel!")
        print(f"   GPU devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
        model = torch.nn.DataParallel(model)
    elif torch.cuda.is_available():
        print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    model = model.to(device)
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
        transform=None,
        load_mode=data_config['load_mode']
    )
    
    print(f"Total dataset size: {len(full_dataset)} images")
    
    # Split dataset
    train_ratio = data_config['train_ratio']
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
    
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    
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


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    epoch_losses = {}
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        
        # Forward pass
        if isinstance(model, KLAutoencoder) or (isinstance(model, torch.nn.DataParallel) and isinstance(model.module, KLAutoencoder)):
            reconstruction, posterior = model(images, sample=True)
            losses = criterion(reconstruction, images, posterior)
        else:  # VQ
            reconstruction, vq_outputs = model(images)
            losses = criterion(reconstruction, images, vq_outputs)
        
        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
        
        # Accumulate losses
        for key, value in losses.items():
            if torch.is_tensor(value):
                value = value.item()
            if key not in epoch_losses:
                epoch_losses[key] = 0.0
            epoch_losses[key] += value
        
        num_batches += 1
        
        # Progress
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {losses['total'].item():.6f}", end='\r')
    
    print()  # New line after progress
    
    # Average losses
    return {k: v / num_batches for k, v in epoch_losses.items()}


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    val_losses = {}
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            
            # Forward pass
            if isinstance(model, KLAutoencoder) or (isinstance(model, torch.nn.DataParallel) and isinstance(model.module, KLAutoencoder)):
                reconstruction, posterior = model(images, sample=False)
                losses = criterion(reconstruction, images, posterior)
            else:  # VQ
                reconstruction, vq_outputs = model(images)
                losses = criterion(reconstruction, images, vq_outputs)
            
            # Accumulate losses
            for key, value in losses.items():
                if torch.is_tensor(value):
                    value = value.item()
                if key not in val_losses:
                    val_losses[key] = 0.0
                val_losses[key] += value
            
            num_batches += 1
    
    # Average losses
    return {k: v / num_batches for k, v in val_losses.items()}


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


def train_first_stage(config: dict, model_type: str) -> Dict:
    """
    Train a first-stage autoencoder (KL or VQ).
    
    Args:
        config: Configuration dictionary
        model_type: Either 'kl' or 'vq'
    
    Returns:
        Dictionary with training results
    """
    print("="*70)
    print(f"EXPERIMENT: {config['experiment']['name']}")
    print(f"Model Type: {model_type.upper()}")
    print("="*70)
    
    # Set seed
    if config['reproducibility']['deterministic']:
        set_seed(config['reproducibility']['seed'])
        print(f"Random seed: {config['reproducibility']['seed']}")
    
    # Create directories
    create_directories(config['paths'])
    
    # Get device
    device = get_device(config['hardware']['device'])
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA devices available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
    
    # Create model
    print("\nCreating model...")
    model = create_model(config, device)
    
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
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    
    # Early stopping
    early_stopping = None
    if config['training']['early_stopping']['enabled']:
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta'],
            mode='min'
        )
    
    # Training loop
    print("\nStarting training...")
    print(f"Epochs: {config['training']['max_epochs']}")
    print(f"Batch size: {config['data']['batch_size']}")
    print(f"Batches per epoch: {len(train_loader)}")
    print("="*70)
    
    best_val_loss = float('inf')
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(config['training']['max_epochs']):
        epoch_start = time.time()
        
        # Train
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_losses = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        # Log
        log_str = f"Epoch {epoch+1:3d}/{config['training']['max_epochs']} | "
        log_str += f"Train: {train_losses['total']:.6f} | "
        log_str += f"Val: {val_losses['total']:.6f} | "
        log_str += f"Time: {epoch_time:.1f}s"
        print(log_str)
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_frequency'] == 0:
            checkpoint_path = Path(config['paths']['checkpoint_dir']) / f'checkpoint_epoch_{epoch+1}.pth'
            
            # Handle DataParallel
            model_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': config
            }, checkpoint_path)
            print(f"  → Checkpoint saved")
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_epoch = epoch + 1
            best_path = Path(config['paths']['checkpoint_dir']) / 'best_model.pth'
            
            # Handle DataParallel
            model_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_state,
                'val_loss': best_val_loss,
                'config': config
            }, best_path)
            print(f"  ★ BEST model saved (val_loss: {best_val_loss:.6f})")
        
        # Early stopping
        if early_stopping and early_stopping(val_losses['total']):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print("="*70)
    print(f"Training complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print("="*70)
    
    return {
        'final_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'total_time': total_time,
        'num_epochs_trained': epoch + 1
    }
