"""
VQ-Regularized Autoencoder (First-Stage for LDM).

Based on Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models"
Implements the VQ-regularized discrete latent autoencoder as the first stage.
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
from vq.vector_quantizer import VectorQuantizer


class VQAutoencoder(nn.Module):
    """
    VQ-Regularized Autoencoder for thermal image compression.
    
    This is the first-stage model from Rombach et al. LDM paper with VQ regularization.
    
    Architecture:
        Input (B, 3, 480, 640)
          ↓
        Encoder
          ↓
        Pre-quantization latent (B, latent_channels, 60, 80)
          ↓
        Vector Quantization (codebook lookup)
          ↓
        Quantized latent (B, latent_channels, 60, 80)
          ↓
        Decoder
          ↓
        Reconstruction (B, 3, 480, 640)
    
    The model learns a discrete latent space through vector quantization
    with a learned codebook.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        num_embeddings: int = 1024,
        commitment_cost: float = 0.25
    ):
        """
        Initialize VQ-regularized autoencoder.
        
        Args:
            in_channels: Number of input channels (3 for RGB).
            latent_channels: Number of latent channels.
            base_channels: Base channel dimension for encoder/decoder.
            channel_multipliers: Channel multipliers for each resolution level.
            num_res_blocks: Number of residual blocks per level.
            num_embeddings: Size of the codebook.
            commitment_cost: Weight for commitment loss in VQ.
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.num_embeddings = num_embeddings
        
        # Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            latent_channels=latent_channels
        )
        
        # Pre-quantization convolution (optional, for better alignment)
        self.quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        
        # Vector Quantizer
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_channels,
            commitment_cost=commitment_cost
        )
        
        # Post-quantization convolution
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        
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
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Encode and quantize input.
        
        Args:
            x: Input tensor of shape (B, 3, 480, 640).
        
        Returns:
            Tuple of (quantized_latent, pre_quantization_latent, vq_losses)
        """
        # Encode
        h = self.encoder(x)
        h = self.quant_conv(h)
        
        # Quantize
        z_quantized, vq_losses = self.quantizer(h, compute_loss=True)
        
        return z_quantized, h, vq_losses
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latent to reconstruction.
        
        Args:
            z: Quantized latent tensor of shape (B, latent_channels, 60, 80).
        
        Returns:
            Reconstructed image of shape (B, 3, 480, 640).
        """
        z = self.post_quant_conv(z)
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass: encode, quantize, decode.
        
        Args:
            x: Input tensor of shape (B, 3, 480, 640).
        
        Returns:
            Tuple of (reconstruction, loss_dict)
            loss_dict contains VQ-specific losses and statistics.
        """
        # Encode and quantize
        z_quantized, z_continuous, vq_losses = self.encode(x)
        
        # Decode
        reconstruction = self.decode(z_quantized)
        
        # Return reconstruction and VQ losses
        loss_dict = vq_losses
        loss_dict['z_continuous'] = z_continuous  # For analysis
        loss_dict['z_quantized'] = z_quantized     # For analysis
        
        return reconstruction, loss_dict
    
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
        quantizer = sum(p.numel() for p in self.quantizer.parameters())
        
        return {
            'total': total,
            'trainable': trainable,
            'encoder': encoder,
            'decoder': decoder,
            'quantizer': quantizer
        }
    
    def get_codebook_usage(self) -> Dict[str, float]:
        """
        Get codebook usage statistics.
        
        Returns:
            Dictionary with codebook statistics.
        """
        with torch.no_grad():
            cluster_size = self.quantizer.ema_cluster_size
            active = (cluster_size > 0).sum().item()
            utilization = active / self.num_embeddings
            
            return {
                'active_codes': active,
                'total_codes': self.num_embeddings,
                'utilization': utilization
            }
    
    def __repr__(self) -> str:
        params = self.count_parameters()
        return (
            f"VQAutoencoder(\n"
            f"  input_shape=({self.in_channels}, 480, 640),\n"
            f"  latent_shape=({self.latent_channels}, {self.latent_height}, {self.latent_width}),\n"
            f"  base_channels={self.base_channels},\n"
            f"  num_embeddings={self.num_embeddings},\n"
            f"  total_parameters={params['total']:,},\n"
            f"  trainable_parameters={params['trainable']:,}\n"
            f")"
        )
