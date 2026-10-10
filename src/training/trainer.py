"""
Trainer class for first-stage autoencoders.
"""

import sys
from pathlib import Path

# Ensure src is in path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Tuple

from models.ldm.first_stage.kl import KLAutoencoder, KLAutoencoderLoss
from models.ldm.first_stage.vq import VQAutoencoder, VQAutoencoderLoss
from data import ThermalImageDataset, get_train_transforms, get_val_transforms

from .utils import set_seed, get_device, create_directories, count_parameters
from .diagnostics import TrainingDiagnostics


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


class Trainer:
    """Trainer for first-stage autoencoders (KL or VQ)."""
    
    def __init__(
        self,
        config: dict,
        model_type: str,
        enable_diagnostics: bool = False,
        diagnostic_log_path: Optional[Path] = None
    ):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
            model_type: 'kl_autoencoder' or 'vq_autoencoder'
            enable_diagnostics: Enable memory diagnostics
            diagnostic_log_path: Path for diagnostic log
        """
        self.config = config
        self.model_type = model_type
        
        # Setup
        self._setup_reproducibility()
        self._setup_device()
        self._setup_directories()
        
        # Components
        self.model = self._create_model()
        self.criterion = self._create_criterion()
        self.optimizer = self._create_optimizer()
        self.train_loader, self.val_loader = self._create_dataloaders()
        self.early_stopping = self._create_early_stopping()
        
        # Mixed precision training
        self.use_amp = self.config['hardware'].get('mixed_precision', False) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print("✨ Mixed precision training ENABLED (faster training, less memory)")
        
        # Diagnostics
        self.diagnostics = TrainingDiagnostics(
            log_path=diagnostic_log_path,
            enabled=enable_diagnostics
        )
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.current_epoch = 0
    
    def _setup_reproducibility(self) -> None:
        """Setup random seeds."""
        if self.config['reproducibility']['deterministic']:
            set_seed(self.config['reproducibility']['seed'])
            print(f"Random seed: {self.config['reproducibility']['seed']}")
    
    def _setup_device(self) -> None:
        """Setup compute device."""
        self.device = get_device(self.config['hardware']['device'])
        print(f"Device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"CUDA devices available: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
    
    def _setup_directories(self) -> None:
        """Create output directories."""
        create_directories(self.config['paths'])
    
    def _create_model(self) -> torch.nn.Module:
        """Create model."""
        print("\nCreating model...")
        model_config = self.config['model']
        
        if model_config['type'] == 'kl_autoencoder':
            model = KLAutoencoder(
                in_channels=model_config['in_channels'],
                latent_channels=model_config['latent_channels'],
                base_channels=model_config['base_channels'],
                channel_multipliers=tuple(model_config['channel_multipliers']),
                num_res_blocks=model_config['num_res_blocks']
            )
        elif model_config['type'] == 'vq_autoencoder':
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
            raise ValueError(f"Unknown model type: {model_config['type']}")
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"🚀 Using {torch.cuda.device_count()} GPUs with DataParallel!")
            model = torch.nn.DataParallel(model)
        elif torch.cuda.is_available():
            print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")
        
        model = model.to(self.device)
        
        param_count = count_parameters(model)
        print(f"Model parameters: {param_count:,}")
        
        return model
    
    def _create_criterion(self):
        """Create loss criterion."""
        model_type = self.config['model']['type']
        
        if model_type == 'kl_autoencoder':
            kl_weight = self.config['model'].get('kl_weight', 1e-6)
            return KLAutoencoderLoss(kl_weight=kl_weight)
        elif model_type == 'vq_autoencoder':
            return VQAutoencoderLoss()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        return optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            betas=self.config['training']['betas'],
            weight_decay=self.config['training']['weight_decay']
        )
    
    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders."""
        print("\nCreating dataloaders...")
        data_config = self.config['data']
        
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
    
    def _create_early_stopping(self) -> Optional[EarlyStopping]:
        """Create early stopping handler."""
        if self.config['training']['early_stopping']['enabled']:
            return EarlyStopping(
                patience=self.config['training']['early_stopping']['patience'],
                min_delta=self.config['training']['early_stopping']['min_delta'],
                mode='min'
            )
        return None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {}
        num_batches = 0
        
        self.diagnostics.epoch_start(self.current_epoch)
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            
            # Check batch data
            self.diagnostics.check_batch_data(images, batch_idx)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                if isinstance(self.model, KLAutoencoder) or (
                    isinstance(self.model, torch.nn.DataParallel) and 
                    isinstance(self.model.module, KLAutoencoder)
                ):
                    reconstruction, posterior = self.model(images, sample=True)
                    losses = self.criterion(reconstruction, images, posterior)
                else:  # VQ
                    reconstruction, vq_outputs = self.model(images)
                    losses = self.criterion(reconstruction, images, vq_outputs)
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(losses['total']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total'].backward()
                self.optimizer.step()
            
            # Accumulate losses
            for key, value in losses.items():
                if torch.is_tensor(value):
                    value = value.item()
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value
            
            num_batches += 1
            
            # Diagnostics
            self.diagnostics.batch_checkpoint(
                self.current_epoch, batch_idx, losses['total'].item()
            )
            self.diagnostics.critical_batch_check(
                self.current_epoch, batch_idx, images, losses['total'].item()
            )
            
            # Progress
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)} - "
                      f"Loss: {losses['total'].item():.6f}", end='\r')
        
        print()  # New line after progress
        
        # Average losses
        return {k: v / num_batches for k, v in epoch_losses.items()}
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        val_losses = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                
                # Forward pass with autocast
                with autocast(enabled=self.use_amp):
                    if isinstance(self.model, KLAutoencoder) or (
                        isinstance(self.model, torch.nn.DataParallel) and 
                        isinstance(self.model.module, KLAutoencoder)
                    ):
                        reconstruction, posterior = self.model(images, sample=False)
                        losses = self.criterion(reconstruction, images, posterior)
                    else:  # VQ
                        reconstruction, vq_outputs = self.model(images)
                        losses = self.criterion(reconstruction, images, vq_outputs)
                
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
    
    def save_checkpoint(self, epoch: int, train_losses: dict, val_losses: dict) -> None:
        """Save training checkpoint."""
        checkpoint_path = Path(self.config['paths']['checkpoint_dir']) / \
                         f'checkpoint_epoch_{epoch}.pth'
        
        # Handle DataParallel
        model_state = self.model.module.state_dict() if \
                     isinstance(self.model, torch.nn.DataParallel) else \
                     self.model.state_dict()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': self.config
        }, checkpoint_path)
        
        print(f"  → Checkpoint saved")
    
    def save_best_model(self, epoch: int, val_loss: float) -> None:
        """Save best model."""
        best_path = Path(self.config['paths']['checkpoint_dir']) / 'best_model.pth'
        
        # Handle DataParallel
        model_state = self.model.module.state_dict() if \
                     isinstance(self.model, torch.nn.DataParallel) else \
                     self.model.state_dict()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'val_loss': val_loss,
            'config': self.config
        }, best_path)
        
        print(f"  ★ BEST model saved (val_loss: {val_loss:.6f})")
    
    def train(self) -> Dict:
        """
        Run full training loop.
        
        Returns:
            Dictionary with training results
        """
        print("\n" + "="*70)
        print(f"EXPERIMENT: {self.config['experiment']['name']}")
        print(f"Model Type: {self.model_type.upper()}")
        print("="*70)
        print(f"\nStarting training...")
        print(f"Epochs: {self.config['training']['max_epochs']}")
        print(f"Batch size: {self.config['data']['batch_size']}")
        print(f"Batches per epoch: {len(self.train_loader)}")
        print("="*70)
        
        start_time = time.time()
        
        try:
            for epoch in range(self.config['training']['max_epochs']):
                self.current_epoch = epoch + 1
                epoch_start = time.time()
                
                # Train
                train_losses = self.train_epoch()
                
                # Validate
                val_losses = self.validate()
                
                epoch_time = time.time() - epoch_start
                
                # Log epoch end
                self.diagnostics.epoch_end(
                    self.current_epoch,
                    train_losses['total'],
                    val_losses['total']
                )
                
                # Log
                log_str = f"Epoch {self.current_epoch:3d}/{self.config['training']['max_epochs']} | "
                log_str += f"Train: {train_losses['total']:.6f} | "
                log_str += f"Val: {val_losses['total']:.6f} | "
                log_str += f"Time: {epoch_time:.1f}s"
                print(log_str)
                
                # Save checkpoint
                if self.current_epoch % self.config['training']['save_frequency'] == 0:
                    self.save_checkpoint(self.current_epoch, train_losses, val_losses)
                
                # Save best model
                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self.best_epoch = self.current_epoch
                    self.save_best_model(self.current_epoch, self.best_val_loss)
                
                # Early stopping
                if self.early_stopping and self.early_stopping(val_losses['total']):
                    print(f"\nEarly stopping at epoch {self.current_epoch}")
                    break
        
        except Exception as e:
            self.diagnostics.log(f"❌ Training failed: {type(e).__name__}: {str(e)}")
            self.diagnostics.save_report()
            raise
        
        finally:
            self.diagnostics.save_report()
        
        total_time = time.time() - start_time
        print("="*70)
        print(f"Training complete!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}")
        print("="*70)
        
        return {
            'final_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'total_time': total_time,
            'num_epochs_trained': self.current_epoch
        }
