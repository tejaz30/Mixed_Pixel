"""
Loss functions for Convolutional Autoencoder training.

As specified in the paper, the CAE uses Mean Squared Error (MSE) loss
for reconstruction.
"""

import logging
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("models.cae.loss")


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for autoencoder training.
    
    Uses Mean Squared Error (MSE) as specified in the paper.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize reconstruction loss.
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super(ReconstructionLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.reduction = reduction
    
    def forward(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            reconstructed: Reconstructed images from decoder (B, C, H, W).
            target: Original input images (B, C, H, W).
        
        Returns:
            Scalar loss value (if reduction='mean' or 'sum').
        """
        loss = self.mse_loss(reconstructed, target)
        return loss
    
    def __repr__(self) -> str:
        return f"ReconstructionLoss(criterion=MSE, reduction={self.reduction})"


class CAELoss(nn.Module):
    """
    Complete loss function for CAE training.
    
    Currently only uses reconstruction loss (MSE).
    Future extensions can add regularization terms if needed.
    """
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        reduction: str = 'mean'
    ):
        """
        Initialize CAE loss.
        
        Args:
            reconstruction_weight: Weight for reconstruction loss (default: 1.0).
            reduction: Reduction method for losses.
        """
        super(CAELoss, self).__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.reconstruction_loss = ReconstructionLoss(reduction=reduction)
    
    def forward(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss and components.
        
        Args:
            reconstructed: Reconstructed images (B, C, H, W).
            target: Original images (B, C, H, W).
        
        Returns:
            Dictionary containing:
                - 'total': Total loss
                - 'reconstruction': Reconstruction loss
        """
        # Compute reconstruction loss
        recon_loss = self.reconstruction_loss(reconstructed, target)
        
        # Total loss (currently just reconstruction)
        total_loss = self.reconstruction_weight * recon_loss
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss
        }
    
    def __repr__(self) -> str:
        return (
            f"CAELoss(\n"
            f"  reconstruction_weight={self.reconstruction_weight},\n"
            f"  criterion=MSE\n"
            f")"
        )


def compute_reconstruction_metrics(
    reconstructed: torch.Tensor,
    target: torch.Tensor
) -> Dict[str, float]:
    """
    Compute reconstruction quality metrics.
    
    Args:
        reconstructed: Reconstructed images (B, C, H, W).
        target: Original images (B, C, H, W).
    
    Returns:
        Dictionary of metrics:
            - 'mse': Mean Squared Error
            - 'mae': Mean Absolute Error
            - 'psnr': Peak Signal-to-Noise Ratio
    """
    with torch.no_grad():
        # MSE
        mse = F.mse_loss(reconstructed, target).item()
        
        # MAE
        mae = F.l1_loss(reconstructed, target).item()
        
        # PSNR (assuming pixel values in [0, 1])
        if mse > 0:
            psnr = 10 * torch.log10(1.0 / torch.tensor(mse)).item()
        else:
            psnr = float('inf')
    
    return {
        'mse': mse,
        'mae': mae,
        'psnr': psnr
    }
