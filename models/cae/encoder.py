"""
Encoder module for Convolutional Autoencoder.

Implements the encoder architecture as described in Section 4.1.2 of the paper.
The encoder progressively compresses spatial dimensions while extracting features.
"""

import logging
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn

logger = logging.getLogger("models.cae.encoder")


class CAEEncoder(nn.Module):
    """
    Encoder for Convolutional Autoencoder.
    
    Architecture:
    - Three convolutional blocks
    - Each block: Conv2D -> ReLU -> MaxPooling
    - Progressively extracts:
      * Low-level edges
      * Intermediate spatial features
      * Higher-level thermal features
    
    As specified in the paper, filters are: 1, 6, 8
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        conv_filters: Tuple[int, int, int] = (1, 6, 8),
        kernel_size: int = 3,
        pool_size: int = 2,
        padding: int = 1
    ):
        """
        Initialize encoder.
        
        Args:
            input_channels: Number of input image channels (3 for RGB).
            conv_filters: Number of filters for each conv block (paper: 1, 6, 8).
            kernel_size: Kernel size for convolutions (default: 3x3).
            pool_size: Pooling window size (default: 2x2).
            padding: Padding for convolutions (default: 1).
        """
        super(CAEEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.padding = padding
        
        # Build encoder blocks
        self.block1 = self._make_conv_block(
            in_channels=input_channels,
            out_channels=conv_filters[0]
        )
        
        self.block2 = self._make_conv_block(
            in_channels=conv_filters[0],
            out_channels=conv_filters[1]
        )
        
        self.block3 = self._make_conv_block(
            in_channels=conv_filters[1],
            out_channels=conv_filters[2]
        )
        
        logger.info(
            f"Encoder initialized: filters={conv_filters}, "
            f"kernel_size={kernel_size}, pool_size={pool_size}"
        )
    
    def _make_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Create a convolutional block: Conv2D -> ReLU -> MaxPooling.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (filters).
            
        Returns:
            Sequential module containing the block.
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=self.pool_size,
                stride=self.pool_size
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
               Expected: (B, 3, 480, 640) for RGB thermal images.
        
        Returns:
            Encoded latent representation.
        """
        # Block 1: Extract low-level edges
        x = self.block1(x)  # (B, 1, 240, 320)
        
        # Block 2: Extract intermediate spatial features
        x = self.block2(x)  # (B, 6, 120, 160)
        
        # Block 3: Extract higher-level thermal features
        x = self.block3(x)  # (B, 8, 60, 80)
        
        return x
    
    def get_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Calculate output shape given input shape.
        
        Args:
            input_shape: (C, H, W) of input.
            
        Returns:
            (C, H, W) of output after encoding.
        """
        c, h, w = input_shape
        
        # After 3 pooling operations (each divides by 2)
        h_out = h // (self.pool_size ** 3)
        w_out = w // (self.pool_size ** 3)
        c_out = self.conv_filters[2]
        
        return (c_out, h_out, w_out)
    
    def __repr__(self) -> str:
        """String representation of encoder."""
        return (
            f"CAEEncoder(\n"
            f"  input_channels={self.input_channels},\n"
            f"  conv_filters={self.conv_filters},\n"
            f"  kernel_size={self.kernel_size},\n"
            f"  pool_size={self.pool_size}\n"
            f")"
        )
