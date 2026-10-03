"""
Decoder for first-stage autoencoders (KL and VQ).

Based on Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models"
Implements the decoder architecture for the first-stage compression model.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ResBlock(nn.Module):
    """
    Residual block with two convolutions (same as in encoder).
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


class UpsampleBlock(nn.Module):
    """
    Upsampling block using nearest neighbor interpolation + convolution.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


class Decoder(nn.Module):
    """
    Decoder for first-stage autoencoder.
    
    Architecture mirrors the encoder:
    - Initial convolution from latent space
    - Middle ResBlocks
    - Multiple resolution levels with ResBlocks
    - Upsampling between levels
    - Final output convolution
    
    Target: 8x spatial expansion (60×80 -> 480×640)
    """
    
    def __init__(
        self,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        out_channels: int = 3
    ):
        """
        Initialize decoder.
        
        Args:
            latent_channels: Number of input latent channels.
            base_channels: Base channel dimension.
            channel_multipliers: Channel multiplier for each resolution level (reversed from encoder).
            num_res_blocks: Number of ResBlocks per level.
            out_channels: Number of output channels (3 for RGB).
        """
        super().__init__()
        
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.num_resolutions = len(channel_multipliers)
        self.num_res_blocks = num_res_blocks
        
        # Initial convolution from latent
        in_ch = base_channels * channel_multipliers[-1]
        self.conv_in = nn.Conv2d(latent_channels, in_ch, kernel_size=3, stride=1, padding=1)
        
        # Middle blocks
        self.mid = nn.ModuleList([
            ResBlock(in_ch),
            ResBlock(in_ch)
        ])
        
        # Upsampling path (reverse order from encoder)
        self.up = nn.ModuleList()
        
        for i in reversed(range(self.num_resolutions)):
            out_ch = base_channels * channel_multipliers[i]
            
            # Upsampling (except first level which is already at lowest resolution)
            up_module = nn.Module()
            if i < self.num_resolutions - 1:
                up_module.upsample = UpsampleBlock(in_ch)
            else:
                up_module.upsample = nn.Identity()
            
            # ResBlocks at this resolution
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):  # +1 for the first block after upsampling
                blocks.append(ResBlock(in_ch, out_ch))
                in_ch = out_ch
            
            up_module.blocks = blocks
            self.up.append(up_module)
        
        # Output normalization and convolution
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(in_ch, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.SiLU()
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            z: Latent tensor of shape (B, latent_channels, 60, 80)
        
        Returns:
            Reconstructed image of shape (B, 3, 480, 640)
        """
        # Initial convolution
        h = self.conv_in(z)
        
        # Middle blocks
        for block in self.mid:
            h = block(h)
        
        # Upsampling path
        for up_module in self.up:
            h = up_module.upsample(h)
            for block in up_module.blocks:
                h = block(h)
        
        # Output
        h = self.norm_out(h)
        h = self.activation(h)
        h = self.conv_out(h)
        
        return h
