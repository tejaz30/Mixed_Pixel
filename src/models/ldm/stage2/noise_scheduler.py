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

    def get_respaced_posterior(
        self,
        z_t: torch.Tensor,
        t_current: torch.Tensor,
        t_next: torch.Tensor,
        noise_pred: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior mean and variance for non-adjacent timestep transition.
        
        For respaced DDPM with stride > 1 (e.g., t_current=980, t_next=960, stride=20),
        the posterior variance must be recomputed to account for the larger jump.
        
        Mathematical derivation:
        - Let s = t_next, t = t_current (where s < t)
        - ᾱ_t = product(α_i, i=0..t)
        - ᾱ_s = product(α_i, i=0..s)
        - Effective transition: α_eff = ᾱ_s / ᾱ_t, β_eff = 1 - α_eff
        
        Posterior variance (from DDPM paper, generalized):
        σ²_respaced = (1 - ᾱ_s) / (1 - ᾱ_t) * (1 - ᾱ_t / ᾱ_s)
                    = (1 - ᾱ_s) / (1 - ᾱ_t) * β_eff
        
        Posterior mean (ε-prediction formulation):
        Start from: z_t = √(ᾱ_t) z_0 + √(1 - ᾱ_t) ε
        Solve for: z_0 = (z_t - √(1 - ᾱ_t) ε) / √(ᾱ_t)
        
        Then: z_s = √(ᾱ_s) z_0 + √(1 - ᾱ_s - σ²_respaced) ε
        
        Combining:
        μ = √(ᾱ_s) (z_t - √(1 - ᾱ_t) ε_pred) / √(ᾱ_t)
            + √(1 - ᾱ_s - σ²_respaced) ε_pred
        
        Args:
            z_t: Noisy latent at current timestep (B, C, H, W)
            t_current: Current timesteps (B,) integers
            t_next: Next (earlier) timesteps (B,) integers, t_next < t_current
            noise_pred: Predicted noise from U-Net (B, C, H, W)
        
        Returns:
            posterior_mean: Mean of p(z_{t_next} | z_t) (B, C, H, W)
            posterior_variance: Variance (B,) scalar per batch element
        
        Special cases:
            - t_next = 0 (final transition): Use ᾱ_0 = 1.0, variance = 0
            - t_current = t_next + 1 (adjacent): Reduces to standard DDPM
        """
        # Handle final transition to clean latent (t_next = 0)
        # When t_next = 0, we want the clean latent without noise
        is_final_step = (t_next == 0)
        
        # Extract alpha_cumprod values
        # For t_next = 0, alpha_cumprod_next should be 1.0 (clean sample has no noise)
        alpha_cumprod_current = self.alphas_cumprod[t_current]
        alpha_cumprod_next = torch.where(
            is_final_step,
            torch.ones_like(self.alphas_cumprod[t_next]),
            self.alphas_cumprod[t_next]
        )
        
        # Compute posterior variance
        # σ²_respaced = (1 - ᾱ_next) / (1 - ᾱ_current) * (1 - ᾱ_current / ᾱ_next)
        # For final step (t_next=0): variance = 0
        beta_eff = 1.0 - alpha_cumprod_current / (alpha_cumprod_next + 1e-10)
        posterior_var = (1.0 - alpha_cumprod_next) / (1.0 - alpha_cumprod_current + 1e-10) * beta_eff
        posterior_var = torch.where(
            is_final_step,
            torch.zeros_like(posterior_var),
            torch.clamp(posterior_var, min=0.0, max=0.9999)
        )
        
        # Compute predicted z_0 from ε-prediction
        sqrt_alpha_cumprod_current = torch.sqrt(alpha_cumprod_current)
        sqrt_one_minus_alpha_cumprod_current = torch.sqrt(1.0 - alpha_cumprod_current)
        
        # Reshape for broadcasting
        while len(sqrt_alpha_cumprod_current.shape) < len(z_t.shape):
            sqrt_alpha_cumprod_current = sqrt_alpha_cumprod_current.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod_current = sqrt_one_minus_alpha_cumprod_current.unsqueeze(-1)
        
        # z_0_pred = (z_t - √(1 - ᾱ_t) ε) / √(ᾱ_t)
        z_0_pred = (z_t - sqrt_one_minus_alpha_cumprod_current * noise_pred) / sqrt_alpha_cumprod_current
        
        # Compute posterior mean
        # μ = √(ᾱ_next) z_0_pred + √(1 - ᾱ_next - σ²) ε_pred
        sqrt_alpha_cumprod_next = torch.sqrt(alpha_cumprod_next)
        sqrt_noise_component = torch.sqrt(torch.clamp(1.0 - alpha_cumprod_next - posterior_var, min=0.0))
        
        # Reshape for broadcasting
        while len(sqrt_alpha_cumprod_next.shape) < len(z_t.shape):
            sqrt_alpha_cumprod_next = sqrt_alpha_cumprod_next.unsqueeze(-1)
            sqrt_noise_component = sqrt_noise_component.unsqueeze(-1)
        
        posterior_mean = sqrt_alpha_cumprod_next * z_0_pred + sqrt_noise_component * noise_pred
        
        return posterior_mean, posterior_var
    
    def sample_respaced_ddpm(
        self,
        unet: nn.Module,
        shape: tuple,
        num_steps: int = 50,
        device: str = 'cuda',
        return_trajectory: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Sample using respaced DDPM with 50 steps.
        
        Uses timestep sequence [980, 960, 940, ..., 20, 0] from 1000-step schedule.
        Stride = 20 timesteps per step.
        
        Args:
            unet: Trained diffusion U-Net (should have EMA weights loaded)
            shape: Latent shape (B, C, H, W)
            num_steps: Number of sampling steps (default: 50)
            device: Device to run on
            return_trajectory: If True, return all intermediate z_t
        
        Returns:
            z_0: Denoised latent (B, C, H, W)
            trajectory: Optional list of intermediate latents (if return_trajectory=True)
        """
        # Compute timestep sequence: [980, 960, ..., 20, 0]
        stride = self.num_timesteps // num_steps
        timesteps = list(range(self.num_timesteps - stride, -1, -stride))
        
        # Verify sequence
        assert len(timesteps) == num_steps, \
            f"Expected {num_steps} timesteps, got {len(timesteps)}"
        assert timesteps[0] == self.num_timesteps - stride, \
            f"First timestep should be {self.num_timesteps - stride}, got {timesteps[0]}"
        assert timesteps[-1] == 0, f"Last timestep should be 0, got {timesteps[-1]}"
        
        # Start from random noise
        z_t = torch.randn(shape, device=device)
        trajectory = [z_t.clone()] if return_trajectory else None
        
        batch_size = shape[0]
        
        # Iterative denoising
        for i, t_current in enumerate(timesteps):
            # Determine next timestep
            t_next = timesteps[i + 1] if i < len(timesteps) - 1 else 0
            
            # Prepare timestep tensors
            t_current_tensor = torch.tensor([t_current] * batch_size, device=device, dtype=torch.long)
            t_next_tensor = torch.tensor([t_next] * batch_size, device=device, dtype=torch.long)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = unet(z_t, t_current_tensor)
            
            # Compute posterior mean and variance
            posterior_mean, posterior_var = self.get_respaced_posterior(
                z_t, t_current_tensor, t_next_tensor, noise_pred
            )
            
            # Add noise (except final step)
            if t_next > 0:
                noise = torch.randn_like(z_t)
                # Reshape variance for broadcasting
                posterior_var_reshaped = posterior_var
                while len(posterior_var_reshaped.shape) < len(z_t.shape):
                    posterior_var_reshaped = posterior_var_reshaped.unsqueeze(-1)
                z_t = posterior_mean + torch.sqrt(posterior_var_reshaped) * noise
            else:
                # Final step: no noise
                z_t = posterior_mean
            
            if return_trajectory:
                trajectory.append(z_t.clone())
        
        if return_trajectory:
            return z_t, trajectory
        return z_t
