"""
Simplified Diffusion U-Net for LDM Stage 2.

Simpler architecture that's easier to debug and matches the 25-30M parameter target.
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
    """Residual block with time embedding."""
    
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
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = h + self.time_proj(torch.nn.functional.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(torch.nn.functional.silu(self.norm2(h))))
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
        
        # Attention
        q = q.reshape(B, C, H * W).permute(0, 2, 1)
        k = k.reshape(B, C, H * W).permute(0, 2, 1)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)
        
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(C), dim=-1)
        h = torch.bmm(attn, v).permute(0, 2, 1).reshape(B, C, H, W)
        
        return x + self.proj(h)


class DiffusionUNet(nn.Module):
    """
    Simplified U-Net for diffusion.
    
    Reduces base_channels from 128 to 64 to hit 25-30M parameter target.
    """
    
    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        base_channels=64,
        channel_mults=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_levels=[2, 3],
        dropout=0.0,
        time_emb_dim=256
    ):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(128, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Input
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        ch = base_channels
        
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(ch, out_ch, time_emb_dim, dropout))
                if level in attn_levels:
                    self.down_blocks.append(AttnBlock(out_ch))
                ch = out_ch
            
            if level != len(channel_mults) - 1:
                self.down_samples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            else:
                self.down_samples.append(nn.Identity())
        
        # Middle
        mid_ch = base_channels * channel_mults[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, time_emb_dim, dropout)
        self.mid_attn = AttnBlock(mid_ch)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_emb_dim, dropout)
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            
            for i in range(num_res_blocks + 1):
                self.up_blocks.append(ResBlock(ch + out_ch, out_ch, time_emb_dim, dropout))
                if level in attn_levels and i < num_res_blocks:
                    self.up_blocks.append(AttnBlock(out_ch))
                ch = out_ch
            
            if level != 0:
                self.up_samples.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
            else:
                self.up_samples.append(nn.Identity())
        
        # Output
        self.norm_out = nn.GroupNorm(32, base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    
    def forward(self, x, t):
        # Time embedding
        t_emb = sinusoidal_embedding(t, 128)
        t_emb = self.time_mlp(t_emb)
        
        # Input
        h = self.conv_in(x)
        
        # Encoder
        hs = [h]
        block_idx = 0
        for level, downsample in enumerate(self.down_samples):
            num_blocks = 0
            while block_idx < len(self.down_blocks):
                block = self.down_blocks[block_idx]
                if isinstance(block, ResBlock):
                    h = block(h, t_emb)
                    num_blocks += 1
                    block_idx += 1
                else:  # AttnBlock
                    h = block(h)
                    block_idx += 1
                
                if num_blocks == 2:  # num_res_blocks
                    break
            
            hs.append(h)
            h = downsample(h)
        
        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # Decoder
        block_idx = 0
        for level, upsample in enumerate(self.up_samples):
            num_blocks = 0
            for i in range(3):  # num_res_blocks + 1
                h = torch.cat([h, hs.pop()], dim=1)
                
                block = self.up_blocks[block_idx]
                if isinstance(block, ResBlock):
                    h = block(h, t_emb)
                    num_blocks += 1
                    block_idx += 1
                else:  # AttnBlock
                    h = block(h)
                    block_idx += 1
                
                if i < 2:  # Check for attention after non-final blocks
                    if block_idx < len(self.up_blocks) and isinstance(self.up_blocks[block_idx], AttnBlock):
                        h = self.up_blocks[block_idx](h)
                        block_idx += 1
            
            h = upsample(h)
        
        # Output
        h = self.conv_out(torch.nn.functional.silu(self.norm_out(h)))
        return h
