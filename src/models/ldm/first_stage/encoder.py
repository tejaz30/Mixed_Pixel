"""
Encoder for first-stage autoencoders (KL and VQ).

Based on Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models"
Implements the encoder architecture for the first-stage compression model.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ResBlock(nn.Module):
    """
    Residual block with two convolutions.
    """
    def __init__(self, in_channels: int, out_channels: int = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=self.out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        
        self.activation = nn.SiLU()
        
        # Skip connection
        if self.in_channels != self.out_channels:
            self.skip = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        
        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)
        
        return x + self.skip(residual)


class DownsampleBlock(nn.Module):
    """
    Downsampling block using stride-2 convolution.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad to ensure proper downsampling
        x = nn.functional.pad(x, (0, 1, 0, 1), mode='constant', value=0)
        return self.conv(x)


class Encoder(nn.Module):
    """
    Encoder for first-stage autoencoder.
    
    Architecture follows Rombach et al. LDM encoder design:
    - Initial convolution
    - Multiple resolution levels with ResBlocks
    - Downsampling between levels
    - Final ResBlocks and normalization
    
    Target: 8x spatial compression (480×640 -> 60×80)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        latent_channels: int = 4
    ):
        """
        Initialize encoder.
        
        Args:
            in_channels: Number of input channels (3 for RGB).
            base_channels: Base channel dimension.
            channel_multipliers: Channel multiplier for each resolution level.
            num_res_blocks: Number of ResBlocks per level.
            latent_channels: Number of output latent channels.
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_resolutions = len(channel_multipliers)
        self.num_res_blocks = num_res_blocks
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        
        # Downsampling path
        self.down = nn.ModuleList()
        in_ch = base_channels
        
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            
            # ResBlocks at this resolution
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(in_ch, out_ch))
                in_ch = out_ch
            
            # Downsampling (except last level)
            down_module = nn.Module()
            down_module.blocks = blocks
            if i < self.num_resolutions - 1:
                down_module.downsample = DownsampleBlock(in_ch)
            else:
                down_module.downsample = nn.Identity()
            
            self.down.append(down_module)
        
        # Middle blocks
        self.mid = nn.ModuleList([
            ResBlock(in_ch),
            ResBlock(in_ch)
        ])
        
        # Output normalization and convolution
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(in_ch, latent_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (B, 3, 480, 640)
        
        Returns:
            Encoded representation of shape (B, latent_channels, 60, 80)
        """
        # Initial convolution
        h = self.conv_in(x)
        
        # Downsampling path
        for down_module in self.down:
            for block in down_module.blocks:
                h = block(h)
            h = down_module.downsample(h)
        
        # Middle blocks
        for block in self.mid:
            h = block(h)
        
        # Output
        h = self.norm_out(h)
        h = self.activation(h)
        h = self.conv_out(h)
        
        return h
    
    def get_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Calculate output shape given input shape."""
        c, h, w = input_shape
        
        # After (num_resolutions - 1) downsamplings
        num_downsample = self.num_resolutions - 1
        h_out = h // (2 ** num_downsample)
        w_out = w // (2 ** num_downsample)
        
        return (self.conv_out.out_channels, h_out, w_out)
