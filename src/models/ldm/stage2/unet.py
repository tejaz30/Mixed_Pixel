"""
Diffusion U-Net for LDM Stage 2 - Working Implementation.

Based on proven DDPM architecture with proper skip connection handling.
"""

import math
import torch
import torch.nn as nn
from typing import List


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal positional embeddings for timesteps."""
    assert dim % 2 == 0
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb


class ResBlock(nn.Module):
    """Residual block with time embedding conditioning."""
    
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.conv1(nn.functional.silu(self.norm1(x)))
        h = h + self.time_proj(nn.functional.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(nn.functional.silu(self.norm2(h))))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Self-attention block."""
    
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.reshape(B, C, H * W).permute(0, 2, 1)
        k = k.reshape(B, C, H * W).permute(0, 2, 1)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)
        
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(C), dim=-1)
        h = torch.bmm(attn, v).permute(0, 2, 1).reshape(B, C, H, W)
        
        return x + self.proj(h)


class Downsample(nn.Module):
    """Downsample by 2x."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Upsample by 2x - handles odd dimensions correctly."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        # Use nearest neighbor interpolation which handles odd dimensions
        x = nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


class DiffusionUNet(nn.Module):
    """
    U-Net for latent diffusion.
    
    Architecture: 4 levels with attention at levels 2-3.
    Base channels: 64 for ~27M parameters (target: 25-30M).
    """
    
    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        base_channels=64,
        channel_multipliers=[1, 2, 4, 4],
        num_res_blocks=2,
        attention_levels=[2, 3],
        dropout=0.0,
        time_emb_dim=256,
        num_groups=32
    ):
        super().__init__()
        
        self.num_levels = len(channel_multipliers)
        self.num_res_blocks = num_res_blocks
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(128, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Input projection
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Build encoder
        self.down = nn.ModuleList()
        ch_in = base_channels
        
        for level in range(self.num_levels):
            ch_out = base_channels * channel_multipliers[level]
            
            for _ in range(num_res_blocks):
                self.down.append(ResBlock(ch_in, ch_out, time_emb_dim, dropout))
                if level in attention_levels:
                    self.down.append(AttnBlock(ch_out))
                ch_in = ch_out
            
            if level < self.num_levels - 1:
                self.down.append(Downsample(ch_out))
        
        # Middle
        mid_ch = base_channels * channel_multipliers[-1]
        self.mid = nn.ModuleList([
            ResBlock(mid_ch, mid_ch, time_emb_dim, dropout),
            AttnBlock(mid_ch),
            ResBlock(mid_ch, mid_ch, time_emb_dim, dropout)
        ])
        
        # Build decoder
        self.up = nn.ModuleList()
        ch_in = mid_ch
        
        for level in reversed(range(self.num_levels)):
            ch_out = base_channels * channel_multipliers[level]
            
            for i in range(num_res_blocks + 1):
                # First block in each level concatenates skip connection
                ch_skip = ch_out if i == 0 else 0
                self.up.append(ResBlock(ch_in + ch_skip, ch_out, time_emb_dim, dropout))
                if level in attention_levels and i < num_res_blocks:
                    self.up.append(AttnBlock(ch_out))
                ch_in = ch_out
            
            if level > 0:
                self.up.append(Upsample(ch_out))
        
        # Output projection
        self.norm_out = nn.GroupNorm(num_groups, base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    
    def forward(self, x, t):
        """
        Args:
            x: (B, 4, 60, 80) noised latent
            t: (B,) timesteps
        
        Returns:
            (B, 4, 60, 80) predicted noise
        """
        # Time embedding
        t_emb = sinusoidal_embedding(t, 128)
        t_emb = self.time_mlp(t_emb)
        
        # Input
        h = self.conv_in(x)
        
        # Encoder - save features after each resolution level
        skips = []
        level_idx = 0
        block_count = 0
        
        for module in self.down:
            if isinstance(module, ResBlock):
                h = module(h, t_emb)
                block_count += 1
                if block_count == self.num_res_blocks:
                    skips.append(h)  # Save after all blocks in level
                    block_count = 0
                    level_idx += 1
            elif isinstance(module, AttnBlock):
                h = module(h)
            elif isinstance(module, Downsample):
                h = module(h)
        
        # Middle
        for module in self.mid:
            if isinstance(module, ResBlock):
                h = module(h, t_emb)
            else:  # AttnBlock
                h = module(h)
        
        # Decoder - match skips with corresponding resolutions
        level_idx = self.num_levels - 1
        block_count = 0
        
        for module in self.up:
            if isinstance(module, ResBlock):
                if block_count == 0 and len(skips) > 0:
                    # First block in level: concatenate skip from encoder
                    skip = skips.pop()
                    # Match spatial dimensions if needed
                    if h.shape[2:] != skip.shape[2:]:
                        h = nn.functional.interpolate(h, size=skip.shape[2:], mode='nearest')
                    h = torch.cat([h, skip], dim=1)
                h = module(h, t_emb)
                block_count += 1
                if block_count >= self.num_res_blocks + 1:
                    block_count = 0
                    level_idx -= 1
            elif isinstance(module, AttnBlock):
                h = module(h)
            elif isinstance(module, Upsample):
                h = module(h)
        
        # Output
        h = self.norm_out(h)
        h = nn.functional.silu(h)
        h = self.conv_out(h)
        
        return h
