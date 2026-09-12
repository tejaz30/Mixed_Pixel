"""
Loss functions for VQ-regularized autoencoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class VQAutoencoderLoss(nn.Module):
    """
    Combined loss for VQ-regularized autoencoder.
    
    Total loss = reconstruction_loss + vq_loss
    
    The VQ loss (commitment + codebook) is already computed by the quantizer.
    """
    
    def __init__(self):
        """Initialize loss."""
        super().__init__()
    
    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        vq_losses: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Args:
            reconstruction: Reconstructed image (B, 3, H, W).
            target: Target image (B, 3, H, W).
            vq_losses: Dictionary containing VQ losses from quantizer.
        
        Returns:
            Dictionary containing:
                - total: Total loss
                - reconstruction: Reconstruction loss (MSE)
                - vq_loss: VQ loss (commitment + codebook)
                - codebook_loss: Codebook loss component
                - commitment_loss: Commitment loss component
                - perplexity: Codebook perplexity
                - codebook_utilization: Fraction of active codes
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstruction, target, reduction='mean')
        
        # VQ loss
        vq_loss = vq_losses['vq_loss']
        
        # Total loss
        total_loss = reconstruction_loss + vq_loss
        
        return {
            'total': total_loss,
            'reconstruction': reconstruction_loss,
            'vq_loss': vq_loss,
            'codebook_loss': vq_losses['codebook_loss'],
            'commitment_loss': vq_losses['commitment_loss'],
            'perplexity': vq_losses['perplexity'],
            'codebook_utilization': vq_losses['codebook_utilization']
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
