"""
Convolutional Autoencoder (CAE) for thermal image mixed-pixel detection.

Implements the baseline CAE architecture as described in Section 4.1.2 of the paper.
This is the baseline model for the research project.
"""

import logging
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn

from .encoder import CAEEncoder
from .decoder import CAEDecoder

logger = logging.getLogger("models.cae.model")


class ConvolutionalAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for thermal image reconstruction.
    
    Architecture:
        Input (B, 3, 480, 640)
          ↓
        Encoder (Conv blocks with pooling)
          ↓
        Latent Representation (B, 8, 60, 80)
          ↓
        Decoder (Upsampling + Conv blocks)
          ↓
        Reconstructed Output (B, 3, 480, 640)
    
    The model learns to compress thermal images into a latent representation
    while preserving spatial information necessary for mixed-pixel detection.
    
    As specified in the paper:
    - Encoder filters: 1, 6, 8
    - Decoder mirrors encoder
    - Input: 640×480 thermal images (normalized to [0, 1])
    - Output: Reconstructed images in [0, 1] (Sigmoid activation)
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        input_height: int = 480,
        input_width: int = 640,
        encoder_filters: Tuple[int, int, int] = (1, 6, 8),
        kernel_size: int = 3,
        pool_size: int = 2,
        padding: int = 1
    ):
        """
        Initialize Convolutional Autoencoder.
        
        Args:
            input_channels: Number of input channels (3 for RGB).
            input_height: Height of input images (480 per paper).
            input_width: Width of input images (640 per paper).
            encoder_filters: Conv filters for encoder blocks (paper: 1, 6, 8).
            kernel_size: Convolution kernel size (default: 3x3).
            pool_size: Max pooling size (default: 2x2).
            padding: Convolution padding (default: 1).
        """
        super(ConvolutionalAutoencoder, self).__init__()
        
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.encoder_filters = encoder_filters
        
        # Initialize encoder
        self.encoder = CAEEncoder(
            input_channels=input_channels,
            conv_filters=encoder_filters,
            kernel_size=kernel_size,
            pool_size=pool_size,
            padding=padding
        )
        
        # Calculate latent dimensions
        self.latent_channels = encoder_filters[2]
        self.latent_height = input_height // (pool_size ** 3)
        self.latent_width = input_width // (pool_size ** 3)
        
        # Initialize decoder (mirror of encoder)
        decoder_filters = (
            encoder_filters[2],  # 8
            encoder_filters[1],  # 6
            encoder_filters[0]   # 1
        )
        
        self.decoder = CAEDecoder(
            latent_channels=self.latent_channels,
            conv_filters=decoder_filters,
            output_channels=input_channels,
            kernel_size=kernel_size,
            upsample_scale=pool_size,
            padding=padding
        )
        
        logger.info(
            f"CAE initialized: input=({input_channels}, {input_height}, {input_width}), "
            f"latent=({self.latent_channels}, {self.latent_height}, {self.latent_width}), "
            f"filters={encoder_filters}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode then decode.
        
        Args:
            x: Input tensor of shape (B, 3, 480, 640).
               Values should be in range [0, 1].
        
        Returns:
            Reconstructed image of shape (B, 3, 480, 640).
            Values are in range [0, 1] (Sigmoid activation).
        """
        # Encode to latent representation
        latent = self.encoder(x)
        
        # Decode back to image
        reconstructed = self.decoder(latent)
        
        return reconstructed
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor of shape (B, 3, 480, 640).
        
        Returns:
            Latent representation of shape (B, 8, 60, 80).
        """
        return self.encoder(x)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to image.
        
        Args:
            latent: Latent tensor of shape (B, 8, 60, 80).
        
        Returns:
            Reconstructed image of shape (B, 3, 480, 640).
        """
        return self.decoder(latent)
    
    def reconstruct(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct image and compute reconstruction error.
        
        Args:
            x: Input tensor of shape (B, 3, 480, 640).
        
        Returns:
            Tuple of (reconstructed_image, reconstruction_error_map).
            Both tensors have shape (B, 3, 480, 640).
        """
        # Reconstruct
        reconstructed = self.forward(x)
        
        # Compute per-pixel reconstruction error
        error_map = torch.abs(x - reconstructed)
        
        return reconstructed, error_map
    
    def get_latent_shape(self) -> Tuple[int, int, int]:
        """
        Get shape of latent representation.
        
        Returns:
            (channels, height, width) of latent space.
        """
        return (self.latent_channels, self.latent_height, self.latent_width)
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count trainable and total parameters.
        
        Returns:
            Dictionary with parameter counts.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'encoder': encoder_params,
            'decoder': decoder_params
        }
    
    def __repr__(self) -> str:
        """String representation of model."""
        params = self.count_parameters()
        return (
            f"ConvolutionalAutoencoder(\n"
            f"  input_shape=({self.input_channels}, {self.input_height}, {self.input_width}),\n"
            f"  latent_shape=({self.latent_channels}, {self.latent_height}, {self.latent_width}),\n"
            f"  encoder_filters={self.encoder_filters},\n"
            f"  total_parameters={params['total']:,},\n"
            f"  trainable_parameters={params['trainable']:,}\n"
            f")"
        )


def create_cae(config: Dict[str, Any] = None) -> ConvolutionalAutoencoder:
    """
    Factory function to create CAE from configuration.
    
    Args:
        config: Configuration dictionary. If None, uses paper defaults.
    
    Returns:
        Initialized ConvolutionalAutoencoder.
    """
    if config is None:
        config = {}
    
    # Paper defaults
    model = ConvolutionalAutoencoder(
        input_channels=config.get('input_channels', 3),
        input_height=config.get('input_height', 480),
        input_width=config.get('input_width', 640),
        encoder_filters=tuple(config.get('encoder_filters', [1, 6, 8])),
        kernel_size=config.get('kernel_size', 3),
        pool_size=config.get('pool_size', 2),
        padding=config.get('padding', 1)
    )
    
    logger.info("CAE model created from configuration")
    return model
