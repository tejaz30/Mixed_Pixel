"""
KL-Regularized Autoencoder (First-Stage for LDM).

Based on Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models"
Implements the KL-regularized continuous latent autoencoder as the first stage.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from encoder import Encoder
from decoder import Decoder


class DiagonalGaussianDistribution:
    """
    Diagonal Gaussian distribution for KL-regularized latent.
    
    Parameterizes a diagonal Gaussian posterior q(z|x) where the encoder
    outputs mean and log-variance.
    """
    
    def __init__(self, parameters: torch.Tensor):
        """
        Args:
            parameters: Tensor of shape (B, 2*C, H, W) containing
                       concatenated mean and log-variance.
        """
        self.parameters = parameters
        
        # Split into mean and log-variance
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        
        # Clamp log-variance for numerical stability
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
    
    def sample(self) -> torch.Tensor:
        """
        Sample from the distribution using reparameterization trick.
        
        Returns:
            Sampled latent z ~ q(z|x)
        """
        # Reparameterization: z = mean + std * epsilon
        epsilon = torch.randn_like(self.mean)
        return self.mean + self.std * epsilon
    
    def mode(self) -> torch.Tensor:
        """
        Return the mode (mean) of the distribution.
        
        Used during inference for deterministic encoding.
        """
        return self.mean
    
    def kl(self, other: 'DiagonalGaussianDistribution' = None) -> torch.Tensor:
        """
        Compute KL divergence KL(q(z|x) || p(z)).
        
        If other is None, assumes standard normal prior p(z) = N(0, I).
        
        Returns:
            KL divergence summed over all dimensions except batch.
        """
        if other is None:
            # KL divergence from standard normal
            # KL(q||p) = 0.5 * sum(1 + log(var) - mean^2 - var)
            kl = 0.5 * torch.sum(
                self.mean.pow(2) + self.var - 1.0 - self.logvar,
                dim=[1, 2, 3]
            )
        else:
            # KL divergence from another Gaussian
            kl = 0.5 * torch.sum(
                (self.mean - other.mean).pow(2) / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar,
                dim=[1, 2, 3]
            )
        
        return kl


class KLAutoencoder(nn.Module):
    """
    KL-Regularized Autoencoder for thermal image compression.
    
    This is the first-stage model from Rombach et al. LDM paper.
    
    Architecture:
        Input (B, 3, 480, 640)
          ↓
        Encoder
          ↓
        Latent parameters (B, 2*latent_channels, 60, 80)
          ↓
        Sample from q(z|x) using reparameterization
          ↓
        Latent z (B, latent_channels, 60, 80)
          ↓
        Decoder
          ↓
        Reconstruction (B, 3, 480, 640)
    
    The model learns a continuous latent space regularized by KL divergence
    to a standard normal prior.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2
    ):
        """
        Initialize KL-regularized autoencoder.
        
        Args:
            in_channels: Number of input channels (3 for RGB).
            latent_channels: Number of latent channels.
            base_channels: Base channel dimension for encoder/decoder.
            channel_multipliers: Channel multipliers for each resolution level.
            num_res_blocks: Number of residual blocks per level.
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        
        # Encoder outputs 2*latent_channels (mean and log-variance)
        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            latent_channels=2 * latent_channels  # Output mean and logvar
        )
        
        # Decoder
        self.decoder = Decoder(
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            out_channels=in_channels
        )
        
        # Calculate latent dimensions
        self.latent_height = 480 // (2 ** (len(channel_multipliers) - 1))
        self.latent_width = 640 // (2 ** (len(channel_multipliers) - 1))
    
    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        """
        Encode input to latent distribution.
        
        Args:
            x: Input tensor of shape (B, 3, 480, 640).
        
        Returns:
            Diagonal Gaussian distribution q(z|x).
        """
        h = self.encoder(x)
        posterior = DiagonalGaussianDistribution(h)
        return posterior
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to reconstruction.
        
        Args:
            z: Latent tensor of shape (B, latent_channels, 60, 80).
        
        Returns:
            Reconstructed image of shape (B, 3, 480, 640).
        """
        return self.decoder(z)
    
    def forward(
        self,
        x: torch.Tensor,
        sample: bool = True
    ) -> Tuple[torch.Tensor, DiagonalGaussianDistribution]:
        """
        Forward pass: encode, sample, decode.
        
        Args:
            x: Input tensor of shape (B, 3, 480, 640).
            sample: If True, sample from posterior. If False, use mean (deterministic).
        
        Returns:
            Tuple of (reconstruction, posterior_distribution)
        """
        # Encode to posterior distribution
        posterior = self.encode(x)
        
        # Sample or use mean
        if sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        
        # Decode
        reconstruction = self.decode(z)
        
        return reconstruction, posterior
    
    def get_latent_shape(self) -> Tuple[int, int, int]:
        """
        Get shape of latent representation.
        
        Returns:
            (channels, height, width) of latent space.
        """
        return (self.latent_channels, self.latent_height, self.latent_width)
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters in the model.
        
        Returns:
            Dictionary with parameter counts.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder = sum(p.numel() for p in self.encoder.parameters())
        decoder = sum(p.numel() for p in self.decoder.parameters())
        
        return {
            'total': total,
            'trainable': trainable,
            'encoder': encoder,
            'decoder': decoder
        }
    
    def __repr__(self) -> str:
        params = self.count_parameters()
        return (
            f"KLAutoencoder(\n"
            f"  input_shape=({self.in_channels}, 480, 640),\n"
            f"  latent_shape=({self.latent_channels}, {self.latent_height}, {self.latent_width}),\n"
            f"  base_channels={self.base_channels},\n"
            f"  total_parameters={params['total']:,},\n"
            f"  trainable_parameters={params['trainable']:,}\n"
            f")"
        )
