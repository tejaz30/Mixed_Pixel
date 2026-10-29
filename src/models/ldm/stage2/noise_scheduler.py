"""
Noise Scheduler for LDM Stage 2.

Implements cosine beta schedule (Nichol & Dhariwal 2021) for DDPM.
Handles forward diffusion (adding noise) and provides precomputed values
for reverse diffusion (denoising).
"""

import math
import torch
import torch.nn as nn
from typing import Optional


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule from Nichol & Dhariwal 2021.
    
    Args:
        timesteps: Total number of diffusion timesteps (T=1000)
        s: Offset to prevent β_t from being too small at t=0
    
    Returns:
        beta: (T,) tensor of β_t values, clipped to [0.0001, 0.9999]
    
    Formula:
        f(t) = cos(((t/T) + s) / (1+s) * π/2)²
        ᾱ_t = f(t) / f(0)
        β_t = 1 - (ᾱ_t / ᾱ_{t-1})
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.0001, 0.9999)
    return betas


class NoiseScheduler(nn.Module):
    """
    DDPM noise scheduler with cosine beta schedule.
    
    Precomputes values needed for forward and reverse diffusion.
    All buffers are registered as non-trainable parameters that
    move with the model to the correct device.
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule: str = 'cosine',
        s: float = 0.008
    ):
        """
        Initialize noise scheduler.
        
        Args:
            num_timesteps: Total timesteps T (default: 1000)
            schedule: Schedule type, only 'cosine' supported
            s: Cosine schedule offset parameter
        """
        super().__init__()
        
        if schedule != 'cosine':
            raise ValueError(f"Only 'cosine' schedule supported, got: {schedule}")
        
        self.num_timesteps = num_timesteps
        
        # Compute schedule
        betas = cosine_beta_schedule(num_timesteps, s=s)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        
        # Register buffers (non-trainable, moved with model)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Precompute values for forward diffusion q(z_t | z_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # Precompute values for reverse diffusion p(z_{t-1} | z_t, z_0)
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Posterior variance for reverse process
        # β_t_tilde = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) * β_t
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
    
    def add_noise(
        self,
        z_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Add noise to latent at timestep t (forward diffusion).
        
        Implements: z_t = √(ᾱ_t) z_0 + √(1 - ᾱ_t) ε
        
        Args:
            z_0: Original latent (B, C, H, W)
            t: Timesteps (B,) integers in [0, num_timesteps)
            noise: Optional noise tensor (B, C, H, W). If None, samples N(0, I)
        
        Returns:
            z_t: Noised latent (B, C, H, W)
        """
        if noise is None:
            noise = torch.randn_like(z_0)
        
        # Extract coefficients for batch
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting: (B,) -> (B, 1, 1, 1)
        while len(sqrt_alpha_cumprod_t.shape) < len(z_0.shape):
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.unsqueeze(-1)
        
        z_t = sqrt_alpha_cumprod_t * z_0 + sqrt_one_minus_alpha_cumprod_t * noise
        return z_t
    
    def get_posterior_mean(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute posterior mean for reverse diffusion.
        
        μ_θ(z_t, t) = (1/√α_t) (z_t - (β_t/√(1-ᾱ_t)) ε_θ(z_t, t))
        
        Args:
            z_t: Noisy latent at timestep t (B, C, H, W)
            t: Timesteps (B,) integers
            noise_pred: Predicted noise from U-Net (B, C, H, W)
        
        Returns:
            posterior_mean: Mean of p(z_{t-1} | z_t) (B, C, H, W)
        """
        # Extract coefficients
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]
        betas_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        while len(sqrt_recip_alphas_t.shape) < len(z_t.shape):
            sqrt_recip_alphas_t = sqrt_recip_alphas_t.unsqueeze(-1)
            betas_t = betas_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        # Compute mean
        posterior_mean = sqrt_recip_alphas_t * (
            z_t - betas_t / sqrt_one_minus_alphas_cumprod_t * noise_pred
        )
        return posterior_mean
    
    def get_posterior_variance(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get posterior variance for reverse diffusion.
        
        Args:
            t: Timesteps (B,)
        
        Returns:
            variance: Posterior variance (B,)
        """
        return self.posterior_variance[t]
    
    def extra_repr(self) -> str:
        """String representation."""
        return f'num_timesteps={self.num_timesteps}, schedule=cosine'
