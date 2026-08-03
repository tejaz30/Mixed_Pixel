"""
Training module for Convolutional Autoencoder.

Implements training loop with:
- Adam optimizer (lr=0.001)
- Early stopping (patience=5)
- Checkpoint saving
- TensorBoard logging
- Resume training support
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from .model import ConvolutionalAutoencoder
from .loss import CAELoss, compute_reconstruction_metrics

logger = logging.getLogger("models.cae.trainer")


class EarlyStopping:
    """
    Early stopping handler.
    
    Stops training when validation loss doesn't improve for `patience` epochs.
    Restores best model weights after stopping.
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as improvement.
            mode: 'min' for loss, 'max' for accuracy.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        self._compare = np.less if mode == 'min' else np.greater
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score (loss or metric).
            epoch: Current epoch number.
        
        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        # Check if improved
        if self._compare(score, self.best_score - self.min_delta):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                f"EarlyStopping counter: {self.counter}/{self.patience} "
                f"(best: {self.best_score:.6f} at epoch {self.best_epoch})"
            )
            
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(
                    f"Early stopping triggered. Best epoch: {self.best_epoch}, "
                    f"best score: {self.best_score:.6f}"
                )
                return True
        
        return False


class CAETrainer:
    """
    Trainer for Convolutional Autoencoder.
    
    Implements complete training pipeline as specified in Section 4.1.3 of the paper:
    - Adam optimizer with learning rate 0.001
    - Maximum 500 epochs
    - Early stopping with patience 5
    - MSE reconstruction loss
    - 80/20 train/val split
    """
    
    def __init__(
        self,
        model: ConvolutionalAutoencoder,
        device: torch.device,
        checkpoint_dir: Path,
        tensorboard_dir: Optional[Path] = None,
        learning_rate: float = 0.001,
        max_epochs: int = 500,
        patience: int = 5,
        save_frequency: int = 10
    ):
        """
        Initialize trainer.
        
        Args:
            model: CAE model to train.
            device: Device to train on (CPU/GPU).
            checkpoint_dir: Directory to save checkpoints.
            tensorboard_dir: Directory for TensorBoard logs.
            learning_rate: Initial learning rate (paper: 0.001).
            max_epochs: Maximum training epochs (paper: 500).
            patience: Early stopping patience (paper: 5).
            save_frequency: Save checkpoint every N epochs.
        """
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.save_frequency = save_frequency
        
        # Loss function (MSE)
        self.criterion = CAELoss()
        
        # Optimizer (Adam with lr=0.001)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=patience,
            mode='min'
        )
        
        # TensorBoard
        if tensorboard_dir:
            self.writer = SummaryWriter(log_dir=str(tensorboard_dir))
        else:
            self.writer = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.train_losses = []
        self.val_losses = []
        
        logger.info(
            f"Trainer initialized: lr={learning_rate}, max_epochs={max_epochs}, "
            f"patience={patience}, device={device}"
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader.
        
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Get data
            images = batch['image'].to(self.device)
            
            # Forward pass
            reconstructed = self.model(images)
            
            # Compute loss
            losses = self.criterion(reconstructed, images)
            loss = losses['total']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            epoch_recon_loss += losses['reconstruction'].item()
            num_batches += 1
        
        # Average metrics
        metrics = {
            'loss': epoch_loss / num_batches,
            'reconstruction_loss': epoch_recon_loss / num_batches
        }
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader.
        
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_mse = 0.0
        epoch_mae = 0.0
        epoch_psnr = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Get data
                images = batch['image'].to(self.device)
                
                # Forward pass
                reconstructed = self.model(images)
                
                # Compute loss
                losses = self.criterion(reconstructed, images)
                loss = losses['total']
                
                # Compute metrics
                metrics = compute_reconstruction_metrics(reconstructed, images)
                
                # Track metrics
                epoch_loss += loss.item()
                epoch_recon_loss += losses['reconstruction'].item()
                epoch_mse += metrics['mse']
                epoch_mae += metrics['mae']
                epoch_psnr += metrics['psnr']
                num_batches += 1
        
        # Average metrics
        metrics = {
            'loss': epoch_loss / num_batches,
            'reconstruction_loss': epoch_recon_loss / num_batches,
            'mse': epoch_mse / num_batches,
            'mae': epoch_mae / num_batches,
            'psnr': epoch_psnr / num_batches
        }
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_from: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Train model with early stopping.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            resume_from: Optional checkpoint path to resume from.
        
        Returns:
            Training history and final metrics.
        """
        # Resume if requested
        if resume_from:
            self.load_checkpoint(resume_from)
            logger.info(f"Resumed training from epoch {self.current_epoch}")
        
        logger.info("=" * 70)
        logger.info("STARTING TRAINING")
        logger.info("=" * 70)
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Max epochs: {self.max_epochs}")
        logger.info(f"Early stopping patience: {self.patience}")
        logger.info(f"Training batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        logger.info("")
        
        start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, self.max_epochs):
                epoch_start_time = time.time()
                
                # Train
                train_metrics = self.train_epoch(train_loader)
                
                # Validate
                val_metrics = self.validate(val_loader)
                
                # Update state
                self.current_epoch = epoch + 1
                self.train_losses.append(train_metrics['loss'])
                self.val_losses.append(val_metrics['loss'])
                
                # Log metrics
                epoch_time = time.time() - epoch_start_time
                logger.info(
                    f"Epoch {epoch+1}/{self.max_epochs} | "
                    f"Train Loss: {train_metrics['loss']:.6f} | "
                    f"Val Loss: {val_metrics['loss']:.6f} | "
                    f"Val PSNR: {val_metrics['psnr']:.2f} dB | "
                    f"Time: {epoch_time:.2f}s"
                )
                
                # TensorBoard logging
                if self.writer:
                    self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch + 1)
                    self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch + 1)
                    self.writer.add_scalar('Metrics/PSNR', val_metrics['psnr'], epoch + 1)
                    self.writer.add_scalar('Metrics/MAE', val_metrics['mae'], epoch + 1)
                
                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    logger.info(f"✓ New best model (val_loss: {self.best_val_loss:.6f})")
                
                # Save checkpoint periodically
                if (epoch + 1) % self.save_frequency == 0:
                    checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
                    self.save_checkpoint(checkpoint_path)
                
                # Early stopping check
                if self.early_stopping(val_metrics['loss'], epoch + 1):
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
        
        # Training complete
        total_time = time.time() - start_time
        logger.info("")
        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total epochs: {self.current_epoch}")
        logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("✓ Restored best model weights")
        
        # Save final model
        final_path = self.checkpoint_dir / "best_model.pth"
        self.save_checkpoint(final_path, save_training_state=False)
        logger.info(f"✓ Saved best model to: {final_path}")
        
        # Close TensorBoard
        if self.writer:
            self.writer.close()
        
        return {
            'epochs_trained': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'training_time': total_time
        }
    
    def save_checkpoint(
        self,
        path: Path,
        save_training_state: bool = True
    ):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint.
            save_training_state: If True, save optimizer and training state.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_channels': self.model.input_channels,
                'input_height': self.model.input_height,
                'input_width': self.model.input_width,
                'encoder_filters': self.model.encoder_filters
            }
        }
        
        if save_training_state:
            checkpoint.update({
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': self.current_epoch,
                'best_val_loss': self.best_val_loss,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses
            })
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: Path):
        """
        Load model checkpoint and resume training.
        
        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
        
        logger.info(f"Checkpoint loaded: {path}")
