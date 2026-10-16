"""
Diagnostic script to investigate NaN loss at epoch 25.

Run this to add detailed logging to identify where NaN first appears.
"""

import sys
from pathlib import Path
import torch
import yaml

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from models.ldm.first_stage.kl import KLAutoencoder
from training.trainer import Trainer


def add_diagnostics_to_trainer():
    """
    Monkey-patch the trainer to add detailed diagnostics.
    """
    original_train_epoch = Trainer.train_epoch
    
    def train_epoch_with_diagnostics(self):
        """Train epoch with extensive logging."""
        self.model.train()
        epoch_losses = {}
        num_batches = 0
        
        print(f"\n{'='*70}")
        print(f"EPOCH {self.current_epoch} DIAGNOSTICS")
        print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6e}")
        print(f"{'='*70}")
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            
            # Forward pass
            if isinstance(self.model, KLAutoencoder) or (
                isinstance(self.model, torch.nn.DataParallel) and 
                isinstance(self.model.module, KLAutoencoder)
            ):
                reconstruction, mean, logvar = self.model(images, sample=True)
                
                # DIAGNOSTIC: Check all values
                checks = {
                    'images': torch.isfinite(images).all().item(),
                    'reconstruction': torch.isfinite(reconstruction).all().item(),
                    'mean': torch.isfinite(mean).all().item(),
                    'logvar': torch.isfinite(logvar).all().item(),
                    'exp_logvar': torch.isfinite(logvar.exp()).all().item(),
                }
                
                # Log detailed stats every 50 batches or if problems detected
                if batch_idx % 50 == 0 or not all(checks.values()):
                    max_logvar = logvar.max().item()
                    min_logvar = logvar.min().item()
                    max_exp_logvar = logvar.exp().max().item()
                    
                    print(f"\nBatch {batch_idx}:")
                    print(f"  Finite checks: {checks}")
                    print(f"  logvar range: [{min_logvar:.2f}, {max_logvar:.2f}]")
                    print(f"  max exp(logvar): {max_exp_logvar:.2e}")
                    print(f"  mean range: [{mean.min().item():.4f}, {mean.max().item():.4f}]")
                    
                    # If exp(logvar) is getting large, warn
                    if max_exp_logvar > 1000:
                        print(f"  ⚠️  WARNING: exp(logvar) = {max_exp_logvar:.2e} is very large!")
                    
                    if not all(checks.values()):
                        print(f"  ❌ NON-FINITE VALUE DETECTED!")
                        print(f"     First non-finite in: {[k for k, v in checks.items() if not v]}")
                        raise ValueError(f"Non-finite value at epoch {self.current_epoch}, batch {batch_idx}")
                
                losses = self.criterion(reconstruction, images, mean, logvar)
                
                # Check loss components
                loss_checks = {
                    'total': torch.isfinite(losses['total']).item(),
                    'reconstruction': torch.isfinite(losses['reconstruction']).item(),
                    'kl': torch.isfinite(losses['kl']).item(),
                }
                
                if batch_idx % 50 == 0 or not all(loss_checks.values()):
                    print(f"  Loss components:")
                    print(f"    reconstruction: {losses['reconstruction'].item():.6f}")
                    print(f"    kl: {losses['kl'].item():.6f}")
                    print(f"    weighted_kl: {losses['weighted_kl'].item():.9f}")
                    print(f"    total: {losses['total'].item():.6f}")
                    
                    if not all(loss_checks.values()):
                        print(f"  ❌ NON-FINITE LOSS DETECTED!")
                        print(f"     First non-finite in: {[k for k, v in loss_checks.items() if not v]}")
                        raise ValueError(f"Non-finite loss at epoch {self.current_epoch}, batch {batch_idx}")
            else:
                # VQ model diagnostics
                reconstruction, vq_outputs = self.model(images)
                losses = self.criterion(reconstruction, images, vq_outputs)
                
                # Ensure total loss is scalar
                if losses['total'].dim() > 0:
                    losses['total'] = losses['total'].mean()
                
                # Check loss components  
                if batch_idx % 50 == 0:
                    print(f"  Loss components:")
                    print(f"    reconstruction: {losses['reconstruction'].item():.6f}")
                    print(f"    vq_loss: {losses['vq_loss'].item():.6f}")
                    if 'codebook_loss' in losses:
                        print(f"    codebook_loss: {losses['codebook_loss'].item():.6f}")
                    if 'commitment_loss' in losses:
                        print(f"    commitment_loss: {losses['commitment_loss'].item():.6f}")
                    print(f"    total: {losses['total'].item():.6f}")
            
            # Ensure loss is scalar before backward
            if not losses['total'].dim() == 0:
                raise ValueError(f"Loss must be scalar, got shape {losses['total'].shape}")
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Check gradients before clipping
            total_norm_before = 0.0
            max_grad = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm_before += param_norm ** 2
                    max_grad = max(max_grad, p.grad.data.abs().max().item())
            total_norm_before = total_norm_before ** 0.5
            
            if batch_idx % 50 == 0:
                print(f"  Gradients before clip: norm={total_norm_before:.4f}, max={max_grad:.4f}")
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Check gradients after clipping
            total_norm_after = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm_after += param_norm ** 2
            total_norm_after = total_norm_after ** 0.5
            
            if batch_idx % 50 == 0:
                print(f"  Gradients after clip:  norm={total_norm_after:.4f}")
                if total_norm_before > 1.0:
                    print(f"    (clipped from {total_norm_before:.4f})")
            
            self.optimizer.step()
            
            # Check for NaN
            if torch.isnan(losses['total']):
                raise ValueError(f"NaN loss detected at epoch {self.current_epoch}, batch {batch_idx}")
            
            # Accumulate losses
            for key, value in losses.items():
                if torch.is_tensor(value):
                    value = value.item()
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value
            
            num_batches += 1
        
        # Return average losses
        return {k: v / num_batches for k, v in epoch_losses.items()}
    
    Trainer.train_epoch = train_epoch_with_diagnostics
    print("✅ Diagnostic logging enabled")


if __name__ == "__main__":
    print("This script adds diagnostic logging to the trainer.")
    print("Import and call add_diagnostics_to_trainer() before training.")
    print("\nExample:")
    print("  from scripts.investigation.diagnose_nan import add_diagnostics_to_trainer")
    print("  add_diagnostics_to_trainer()")
    print("  trainer = Trainer(...)")
    print("  trainer.train()")
