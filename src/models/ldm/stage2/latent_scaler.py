"""
Latent Scaler for LDM Stage 2.

Performs standardization/unstandardization of VQ latents using
training-set statistics measured from outputs/vq_latent_statistics.yaml.
"""

import torch
import torch.nn as nn
from typing import Optional


class LatentScaler(nn.Module):
    """
    Standardize and unstandardize latent embeddings.
    
    Uses global mean and std computed from training set.
    Transformation is linear and invertible.
    
    Statistics from outputs/vq_latent_statistics.yaml:
    - mean: 0.01289583370089531
    - std:  0.19909434020519257
    
    Forward:  z_scaled = (z - mean) / std
    Inverse:  z = z_scaled * std + mean
    """
    
    def __init__(
        self,
        mean: float = 0.01289583370089531,
        std: float = 0.19909434020519257,
        eps: float = 1e-8
    ):
        """
        Initialize latent scaler.
        
        Args:
            mean: Global mean from training latents
            std: Global std from training latents
            eps: Small constant for numerical stability
        """
        super().__init__()
        
        assert std > eps, f"Standard deviation {std} too small (< {eps})"
        
        # Register as buffers (non-trainable, moved with model)
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))
        self.eps = eps
    
    def scale(self, z: torch.Tensor) -> torch.Tensor:
        """
        Standardize latent (z -> z_scaled).
        
        Args:
            z: Latent tensor (B, C, H, W) in original VQ range
        
        Returns:
            z_scaled: Standardized latent, approximately N(0, 1)
        """
        return (z - self.mean) / self.std
    
    def unscale(self, z_scaled: torch.Tensor) -> torch.Tensor:
        """
        Inverse standardization (z_scaled -> z).
        
        Args:
            z_scaled: Standardized latent (B, C, H, W)
        
        Returns:
            z: Latent in original VQ range
        """
        return z_scaled * self.std + self.mean
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass applies scaling."""
        return self.scale(z)
    
    def test_invertibility(
        self,
        z: torch.Tensor,
        tol: float = 1e-5
    ) -> tuple[bool, float]:
        """
        Test that unscale(scale(z)) ≈ z.
        
        Args:
            z: Test latent tensor
            tol: Tolerance for max absolute error
        
        Returns:
            (passes, max_error): Whether test passes and the max error
        """
        z_reconstructed = self.unscale(self.scale(z))
        error = (z - z_reconstructed).abs().max().item()
        return error < tol, error
    
    def verify_statistics(
        self,
        z_batch: torch.Tensor,
        mean_tol: float = 0.1,
        std_tol: float = 0.1
    ) -> tuple[bool, float, float]:
        """
        Verify scaled latents have mean≈0, std≈1.
        
        Args:
            z_batch: Batch of latents to verify
            mean_tol: Tolerance for mean
            std_tol: Tolerance for std
        
        Returns:
            (passes, actual_mean, actual_std)
        """
        z_scaled = self.scale(z_batch)
        actual_mean = z_scaled.mean().item()
        actual_std = z_scaled.std().item()
        
        mean_ok = abs(actual_mean) < mean_tol
        std_ok = abs(actual_std - 1.0) < std_tol
        
        return mean_ok and std_ok, actual_mean, actual_std
    
    def extra_repr(self) -> str:
        """String representation."""
        return f'mean={self.mean.item():.8f}, std={self.std.item():.8f}'
