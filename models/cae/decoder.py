"""
Decoder module for Convolutional Autoencoder.

Implements the decoder architecture that mirrors the encoder.
The decoder reconstructs the original image from the latent representation.
"""

import logging
from typing import Tuple
import torch
import torch.nn as nn

logger = logging.getLogger("models.cae.decoder")


class CAEDecoder(nn.Module):
    """
    Decoder for Convolutional Autoencoder.
    
    Architecture:
    - Mirrors the encoder structure
    - Each block: Upsampling -> Conv2D -> ReLU
    - Final layer uses Sigmoid activation to constrain output to [0, 1]
    
    Reconstructs original image resolution from latent representation.
    """
    
    def __init__(
        self,
        latent_channels: int = 8,
        conv_filters: Tuple[int, int, int] = (8, 6, 1),
        output_channels: int = 3,
        kernel_size: int = 3,
        upsample_scale: int = 2,
        padding: int = 1
    ):
        """
        Initialize decoder.
        
        Args:
            latent_channels: Number of channels in latent representation.
            conv_filters: Number of filters for each conv block (reverse of encoder).
            output_channels: Number of output channels (3 for RGB).
            kernel_size: Kernel size for convolutions (default: 3x3).
            upsample_scale: Upsampling scale factor (default: 2).
            padding: Padding for convolutions (default: 1).
        """
        super(CAEDecoder, self).__init__()
        
        self.latent_channels = latent_channels
        self.conv_filters = conv_filters
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.upsample_scale = upsample_scale
        self.padding = padding
        
        # Build decoder blocks (mirror of encoder)
        # Block 1: Upsample from latent
        self.block1 = self._make_deconv_block(
            in_channels=latent_channels,
            out_channels=conv_filters[0]
        )
        
        # Block 2: Continue upsampling
        self.block2 = self._make_deconv_block(
            in_channels=conv_filters[0],
            out_channels=conv_filters[1]
        )
        
        # Block 3: Final upsampling
        self.block3 = self._make_deconv_block(
            in_channels=conv_filters[1],
            out_channels=conv_filters[2]
        )
        
        # Final reconstruction layer (no upsampling, just projection to RGB)
        self.final_conv = nn.Conv2d(
            in_channels=conv_filters[2],
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        
        self.final_activation = nn.Sigmoid()
        
        logger.info(
            f"Decoder initialized: filters={conv_filters}, "
            f"output_channels={output_channels}"
        )
    
    def _make_deconv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Create a decoder block: Upsampling -> Conv2D -> ReLU.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (filters).
            
        Returns:
            Sequential module containing the block.
        """
        return nn.Sequential(
            nn.Upsample(
                scale_factor=self.upsample_scale,
                mode='nearest'
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding
            ),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            x: Latent representation from encoder.
               Expected shape: (B, 8, 60, 80) after 3x pooling from (B, 3, 480, 640)
        
        Returns:
            Reconstructed image of shape (B, 3, 480, 640).
            Output is in range [0, 1] due to Sigmoid activation.
        """
        # Block 1: Upsample (60, 80) -> (120, 160)
        x = self.block1(x)  # (B, 8, 120, 160)
        
        # Block 2: Upsample (120, 160) -> (240, 320)
        x = self.block2(x)  # (B, 6, 240, 320)
        
        # Block 3: Upsample (240, 320) -> (480, 640)
        x = self.block3(x)  # (B, 1, 480, 640)
        
        # Final reconstruction (no upsampling, just conv + sigmoid)
        # Since block3 already reached target resolution
        x = self.final_conv(x)       # (B, 3, 480, 640)
        x = self.final_activation(x) # (B, 3, 480, 640) in range [0, 1]
        
        return x
    
    def __repr__(self) -> str:
        """String representation of decoder."""
        return (
            f"CAEDecoder(\n"
            f"  latent_channels={self.latent_channels},\n"
            f"  conv_filters={self.conv_filters},\n"
            f"  output_channels={self.output_channels},\n"
            f"  kernel_size={self.kernel_size},\n"
            f"  upsample_scale={self.upsample_scale}\n"
            f")"
        )
