"""
Loss functions for KL-regularized autoencoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class KLAutoencoderLoss(nn.Module):
    """
    Combined loss for KL-regularized autoencoder.
    
    Total loss = reconstruction_loss + kl_weight * kl_loss
    
    Following Rombach et al., the KL loss is weighted to balance
    reconstruction quality with regularization.
    """
    
    def __init__(self, kl_weight: float = 1e-6):
        """
        Initialize loss.
        
        Args:
            kl_weight: Weight for KL divergence term.
                      Rombach et al. use a small weight (e.g., 1e-6) to avoid
                      posterior collapse while maintaining good reconstruction.
        """
        super().__init__()
        self.kl_weight = kl_weight
    
    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Args:
            reconstruction: Reconstructed image (B, 3, H, W).
            target: Target image (B, 3, H, W).
            mean: Posterior mean from encoder (B, latent_ch, H', W').
            logvar: Posterior log variance from encoder (B, latent_ch, H', W').
        
        Returns:
            Dictionary containing:
                - total: Total loss
                - reconstruction: Reconstruction loss (MSE)
                - kl: KL divergence loss
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstruction, target, reduction='mean')
        
        # KL divergence to standard normal prior
        # KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        kl_loss = kl_loss / mean.shape[0]  # Average over batch
        
        # Total loss
        total_loss = reconstruction_loss + self.kl_weight * kl_loss
        
        return {
            'total': total_loss,
            'reconstruction': reconstruction_loss,
            'kl': kl_loss,
            'weighted_kl': self.kl_weight * kl_loss
        }


def compute_reconstruction_metrics(
    reconstruction: torch.Tensor,
    target: torch.Tensor
) -> Dict[str, float]:
    """
    Compute additional reconstruction metrics for evaluation.
    
    Args:
        reconstruction: Reconstructed images (B, C, H, W).
        target: Target images (B, C, H, W).
    
    Returns:
        Dictionary of metrics.
    """
    with torch.no_grad():
        # MSE
        mse = F.mse_loss(reconstruction, target).item()
        
        # MAE
        mae = F.l1_loss(reconstruction, target).item()
        
        # PSNR
        psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
        psnr = psnr.item()
        
    return {
        'mse': mse,
        'mae': mae,
        'psnr': psnr
    }
