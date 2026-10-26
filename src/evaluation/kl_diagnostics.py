"""
KL Autoencoder Latent Diagnostics

Analyzes KL-regularized autoencoder latent representations for:
- Posterior statistics (mean, logvar)
- KL divergence per image and channel
- Active dimension analysis
- Sampling variability
- Numerical stability

All diagnostics respect the KL implementation's continuous latent semantics.
"""

import logging
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_kl_latent_statistics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_mean: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute statistics of KL latent representations across validation set.
    
    Args:
        model: KL autoencoder model (must have encode() method returning posterior).
        dataloader: Validation dataloader.
        device: Device to run on.
        use_mean: If True, use posterior mean. If False, sample from posterior.
    
    Returns:
        Dictionary containing:
            - latents: Collected latent values (N, C, H, W)
            - per_channel_mean: Mean per channel (C,)
            - per_channel_std: Std per channel (C,)
            - per_channel_var: Variance per channel (C,)
            - per_sample_mean: Mean per sample (N,)
            - per_sample_std: Std per sample (N,)
    """
    model.eval()
    
    all_latents = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing KL latent statistics"):
            images = batch['image'].to(device)
            
            # Encode to posterior
            posterior = model.encode(images)
            
            # Get latent representation
            if use_mean:
                latents = posterior.mode()  # Deterministic: use mean
            else:
                latents = posterior.sample()  # Stochastic: sample
            
            all_latents.append(latents.cpu().numpy())
    
    # Concatenate all latents
    all_latents = np.concatenate(all_latents, axis=0)  # (N, C, H, W)
    N, C, H, W = all_latents.shape
    
    # Check for non-finite values
    if not np.isfinite(all_latents).all():
        n_nonfinite = (~np.isfinite(all_latents)).sum()
        logger.warning(f"Found {n_nonfinite} non-finite values in latents")
    
    # Per-channel statistics (aggregate over N, H, W)
    per_channel_mean = all_latents.mean(axis=(0, 2, 3))  # (C,)
    per_channel_std = all_latents.std(axis=(0, 2, 3))    # (C,)
    per_channel_var = all_latents.var(axis=(0, 2, 3))    # (C,)
    
    # Per-sample statistics (aggregate over C, H, W)
    per_sample_mean = all_latents.mean(axis=(1, 2, 3))   # (N,)
    per_sample_std = all_latents.std(axis=(1, 2, 3))     # (N,)
    
    logger.info(f"Latent statistics computed over {N} samples")
    logger.info(f"Per-channel mean range: [{per_channel_mean.min():.4f}, {per_channel_mean.max():.4f}]")
    logger.info(f"Per-channel std range: [{per_channel_std.min():.4f}, {per_channel_std.max():.4f}]")
    
    return {
        'latents': all_latents,
        'per_channel_mean': per_channel_mean,
        'per_channel_std': per_channel_std,
        'per_channel_var': per_channel_var,
        'per_sample_mean': per_sample_mean,
        'per_sample_std': per_sample_std
    }


def compute_posterior_statistics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, np.ndarray]:
    """
    Compute statistics of KL posterior distributions (mean and logvar).
    
    Args:
        model: KL autoencoder model.
        dataloader: Validation dataloader.
        device: Device to run on.
    
    Returns:
        Dictionary containing:
            - posterior_means: All posterior means (N, C, H, W)
            - posterior_logvars: All posterior logvars (N, C, H, W)
            - mean_of_means: Mean of posterior means per channel (C,)
            - std_of_means: Std of posterior means per channel (C,)
            - mean_of_logvars: Mean of posterior logvars per channel (C,)
            - std_of_logvars: Std of posterior logvars per channel (C,)
    """
    model.eval()
    
    all_means = []
    all_logvars = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing posterior statistics"):
            images = batch['image'].to(device)
            
            # Encode to posterior
            posterior = model.encode(images)
            
            # Extract mean and logvar
            means = posterior.mean.cpu().numpy()
            logvars = posterior.logvar.cpu().numpy()
            
            all_means.append(means)
            all_logvars.append(logvars)
    
    # Concatenate
    all_means = np.concatenate(all_means, axis=0)       # (N, C, H, W)
    all_logvars = np.concatenate(all_logvars, axis=0)   # (N, C, H, W)
    
    # Check for non-finite values
    if not np.isfinite(all_means).all():
        logger.warning(f"Non-finite values in posterior means")
    if not np.isfinite(all_logvars).all():
        logger.warning(f"Non-finite values in posterior logvars")
    
    # Statistics of posterior parameters
    mean_of_means = all_means.mean(axis=(0, 2, 3))      # (C,)
    std_of_means = all_means.std(axis=(0, 2, 3))        # (C,)
    mean_of_logvars = all_logvars.mean(axis=(0, 2, 3))  # (C,)
    std_of_logvars = all_logvars.std(axis=(0, 2, 3))    # (C,)
    
    logger.info(f"Posterior statistics computed")
    logger.info(f"Mean of posterior means: {mean_of_means.mean():.4f} ± {mean_of_means.std():.4f}")
    logger.info(f"Mean of posterior logvars: {mean_of_logvars.mean():.4f} ± {mean_of_logvars.std():.4f}")
    
    return {
        'posterior_means': all_means,
        'posterior_logvars': all_logvars,
        'mean_of_means': mean_of_means,
        'std_of_means': std_of_means,
        'mean_of_logvars': mean_of_logvars,
        'std_of_logvars': std_of_logvars
    }


def compute_kl_divergence_stats(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, np.ndarray]:
    """
    Compute KL divergence statistics.
    
    KL divergence from standard normal prior: KL(q(z|x) || N(0, I))
    
    Args:
        model: KL autoencoder model.
        dataloader: Validation dataloader.
        device: Device to run on.
    
    Returns:
        Dictionary containing:
            - per_image_kl: KL divergence per image (N,)
            - per_channel_kl: Average KL per channel (C,)
            - total_kl_mean: Mean KL across all images
            - total_kl_std: Std of KL across images
    """
    model.eval()
    
    all_kl_per_image = []
    all_kl_per_channel = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing KL divergence"):
            images = batch['image'].to(device)
            
            # Encode to posterior
            posterior = model.encode(images)
            
            # Compute KL divergence
            # KL is summed over spatial dimensions, returns (B,)
            kl = posterior.kl(other=None)  # None means standard normal prior
            
            all_kl_per_image.append(kl.cpu().numpy())
            
            # For per-channel KL, we need to compute manually
            # KL per pixel: 0.5 * (mean^2 + var - 1 - logvar)
            kl_per_pixel = 0.5 * (
                posterior.mean.pow(2) + posterior.var - 1.0 - posterior.logvar
            )  # (B, C, H, W)
            
            # Sum over spatial, average over batch
            kl_per_channel = kl_per_pixel.sum(dim=(2, 3)).mean(dim=0)  # (C,)
            all_kl_per_channel.append(kl_per_channel.cpu().numpy())
    
    # Concatenate
    per_image_kl = np.concatenate(all_kl_per_image, axis=0)  # (N,)
    per_channel_kl = np.mean(all_kl_per_channel, axis=0)      # (C,)
    
    total_kl_mean = per_image_kl.mean()
    total_kl_std = per_image_kl.std()
    
    logger.info(f"KL divergence: {total_kl_mean:.4f} ± {total_kl_std:.4f}")
    logger.info(f"Per-channel KL range: [{per_channel_kl.min():.4f}, {per_channel_kl.max():.4f}]")
    
    return {
        'per_image_kl': per_image_kl,
        'per_channel_kl': per_channel_kl,
        'total_kl_mean': float(total_kl_mean),
        'total_kl_std': float(total_kl_std)
    }


def compute_active_dimensions(
    latent_stats: Dict[str, np.ndarray],
    threshold: float = 0.01
) -> Dict[str, float]:
    """
    Compute active latent dimensions based on variance threshold.
    
    A dimension is "active" if its variance exceeds the threshold.
    
    Args:
        latent_stats: Output from compute_kl_latent_statistics.
        threshold: Minimum variance for a dimension to be considered active.
    
    Returns:
        Dictionary containing:
            - active_dims: Number of active dimensions
            - total_dims: Total number of dimensions
            - utilization: Fraction of active dimensions
            - inactive_channels: Indices of inactive channels
    """
    per_channel_var = latent_stats['per_channel_var']
    total_dims = len(per_channel_var)
    
    active_mask = per_channel_var > threshold
    active_dims = active_mask.sum()
    inactive_channels = np.where(~active_mask)[0].tolist()
    
    utilization = active_dims / total_dims
    
    logger.info(f"Active dimensions: {active_dims}/{total_dims} ({utilization:.1%})")
    if inactive_channels:
        logger.warning(f"Inactive channels (var < {threshold}): {inactive_channels}")
    
    return {
        'active_dims': int(active_dims),
        'total_dims': int(total_dims),
        'utilization': float(utilization),
        'inactive_channels': inactive_channels,
        'threshold': threshold
    }


def assess_sampling_variability(
    model: torch.nn.Module,
    images: torch.Tensor,
    n_samples: int = 10,
    device: torch.device = torch.device('cpu')
) -> Dict[str, float]:
    """
    Assess reconstruction variability when sampling from posterior vs using mean.
    
    Args:
        model: KL autoencoder model.
        images: Input images (B, C, H, W).
        n_samples: Number of samples to draw per image.
        device: Device to run on.
    
    Returns:
        Dictionary containing:
            - mean_recon_mse: MSE using posterior mean
            - sampled_recon_mse_mean: Mean MSE across samples
            - sampled_recon_mse_std: Std of MSE across samples
            - sampling_variability: Difference between sampled and mean MSE
    """
    model.eval()
    images = images.to(device)
    
    with torch.no_grad():
        # Encode to posterior
        posterior = model.encode(images)
        
        # Reconstruction using mean (deterministic)
        z_mean = posterior.mode()
        recon_mean = model.decode(z_mean)
        mse_mean = F.mse_loss(recon_mean, images).item()
        
        # Reconstructions using samples
        mse_sampled = []
        for _ in range(n_samples):
            z_sample = posterior.sample()
            recon_sample = model.decode(z_sample)
            mse = F.mse_loss(recon_sample, images).item()
            mse_sampled.append(mse)
        
        mse_sampled = np.array(mse_sampled)
        mse_sampled_mean = mse_sampled.mean()
        mse_sampled_std = mse_sampled.std()
        variability = abs(mse_sampled_mean - mse_mean)
    
    logger.info(f"Mean reconstruction MSE: {mse_mean:.6f}")
    logger.info(f"Sampled reconstruction MSE: {mse_sampled_mean:.6f} ± {mse_sampled_std:.6f}")
    logger.info(f"Sampling variability: {variability:.6f}")
    
    return {
        'mean_recon_mse': float(mse_mean),
        'sampled_recon_mse_mean': float(mse_sampled_mean),
        'sampled_recon_mse_std': float(mse_sampled_std),
        'sampling_variability': float(variability)
    }


def compute_all_kl_diagnostics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    active_dim_threshold: float = 0.01,
    assess_sampling: bool = False,
    n_sampling_images: int = 16
) -> Dict:
    """
    Compute all KL-specific diagnostics.
    
    Args:
        model: KL autoencoder model.
        dataloader: Validation dataloader.
        device: Device to run on.
        active_dim_threshold: Variance threshold for active dimensions.
        assess_sampling: Whether to assess sampling variability.
        n_sampling_images: Number of images for sampling assessment.
    
    Returns:
        Comprehensive dictionary of KL diagnostics.
    """
    logger.info("=" * 70)
    logger.info("KL LATENT DIAGNOSTICS")
    logger.info("=" * 70)
    
    results = {}
    
    # 1. Latent statistics (using posterior mean)
    logger.info("\n1. Computing latent statistics (posterior mean)...")
    latent_stats = compute_kl_latent_statistics(model, dataloader, device, use_mean=True)
    results['latent_statistics'] = {
        'per_channel_mean': latent_stats['per_channel_mean'].tolist(),
        'per_channel_std': latent_stats['per_channel_std'].tolist(),
        'per_channel_var': latent_stats['per_channel_var'].tolist(),
        'per_sample_mean_avg': float(latent_stats['per_sample_mean'].mean()),
        'per_sample_mean_std': float(latent_stats['per_sample_mean'].std()),
        'per_sample_std_avg': float(latent_stats['per_sample_std'].mean())
    }
    
    # 2. Posterior statistics
    logger.info("\n2. Computing posterior statistics...")
    posterior_stats = compute_posterior_statistics(model, dataloader, device)
    results['posterior_statistics'] = {
        'mean_of_means': posterior_stats['mean_of_means'].tolist(),
        'std_of_means': posterior_stats['std_of_means'].tolist(),
        'mean_of_logvars': posterior_stats['mean_of_logvars'].tolist(),
        'std_of_logvars': posterior_stats['std_of_logvars'].tolist()
    }
    
    # 3. KL divergence
    logger.info("\n3. Computing KL divergence statistics...")
    kl_stats = compute_kl_divergence_stats(model, dataloader, device)
    results['kl_divergence'] = {
        'per_image_kl_mean': float(kl_stats['per_image_kl'].mean()),
        'per_image_kl_std': float(kl_stats['per_image_kl'].std()),
        'per_image_kl_median': float(np.median(kl_stats['per_image_kl'])),
        'per_channel_kl': kl_stats['per_channel_kl'].tolist(),
        'total_kl_mean': kl_stats['total_kl_mean'],
        'total_kl_std': kl_stats['total_kl_std']
    }
    
    # 4. Active dimensions
    logger.info("\n4. Computing active dimensions...")
    active_dims = compute_active_dimensions(latent_stats, threshold=active_dim_threshold)
    results['active_dimensions'] = active_dims
    
    # 5. Sampling variability (optional)
    if assess_sampling:
        logger.info("\n5. Assessing sampling variability...")
        # Get a batch of images for sampling assessment
        batch = next(iter(dataloader))
        images = batch['image'][:n_sampling_images]
        
        sampling_stats = assess_sampling_variability(model, images, n_samples=10, device=device)
        results['sampling_variability'] = sampling_stats
    
    # 6. Numerical stability check
    logger.info("\n6. Checking numerical stability...")
    latents = latent_stats['latents']
    results['numerical_stability'] = {
        'has_nan': bool(np.isnan(latents).any()),
        'has_inf': bool(np.isinf(latents).any()),
        'min_value': float(latents.min()),
        'max_value': float(latents.max()),
        'extreme_values_count': int((np.abs(latents) > 100).sum())
    }
    
    logger.info("\n" + "=" * 70)
    logger.info("KL DIAGNOSTICS COMPLETE")
    logger.info("=" * 70)
    
    return results



# Wrapper class for batch-wise processing compatibility
class KLDiagnostics:
    """
    Wrapper class for KL diagnostics to provide batch-wise processing API
    compatible with the evaluation pipeline.
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device, latent_channels: int):
        self.model = model
        self.device = device
        self.latent_channels = latent_channels
        self.reset_accumulators()
    
    def reset_accumulators(self):
        """Reset accumulators for batch-wise processing."""
        self.all_means = []
        self.all_logvars = []
        self.kl_divergences = []
    
    def process_batch(self, x: torch.Tensor) -> Dict:
        """
        Process a single batch and extract KL diagnostics.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Dictionary with batch-level diagnostics
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)
            mean = outputs['mean']
            logvar = outputs['logvar']
            
            # Store for later aggregation
            self.all_means.append(mean.cpu())
            self.all_logvars.append(logvar.cpu())
            
            # Compute KL divergence for this batch
            # KL(q||p) = -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
            kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=[1, 2, 3])
            kl_div_mean = kl_div.mean().item()
            self.kl_divergences.append(kl_div_mean)
            
            # Batch statistics
            batch_diagnostics = {
                'posterior_statistics': {
                    'mean_magnitude': mean.abs().mean().item(),
                    'logvar_mean': logvar.mean().item(),
                    'posterior_std': torch.exp(0.5 * logvar).mean().item()
                },
                'kl_divergence': {
                    'mean': kl_div_mean,
                    'per_sample': kl_div.cpu().numpy().tolist()
                }
            }
            
            return batch_diagnostics
    
    def compute_global_statistics(self) -> Dict:
        """
        Compute global statistics across all processed batches.
        
        Returns:
            Dictionary with dataset-level diagnostics
        """
        if not self.all_means:
            return {'error': 'No batches processed'}
        
        # Concatenate all batches
        all_means = torch.cat(self.all_means, dim=0)
        all_logvars = torch.cat(self.all_logvars, dim=0)
        
        # Global posterior statistics
        posterior_stats = {
            'mean': {
                'global_mean': float(all_means.mean()),
                'global_std': float(all_means.std()),
                'per_channel_mean': all_means.mean(dim=[0, 2, 3]).numpy().tolist(),
                'per_channel_std': all_means.std(dim=[0, 2, 3]).numpy().tolist()
            },
            'logvar': {
                'global_mean': float(all_logvars.mean()),
                'global_std': float(all_logvars.std()),
                'per_channel_mean': all_logvars.mean(dim=[0, 2, 3]).numpy().tolist()
            }
        }
        
        # KL divergence statistics
        kl_stats = {
            'mean': float(np.mean(self.kl_divergences)),
            'std': float(np.std(self.kl_divergences)),
            'min': float(np.min(self.kl_divergences)),
            'max': float(np.max(self.kl_divergences))
        }
        
        # Active dimensions (channels with non-negligible variance)
        channel_vars = all_means.var(dim=[0, 2, 3]).numpy()
        active_dims = int(np.sum(channel_vars > 0.01))
        
        return {
            'posterior_statistics': posterior_stats,
            'kl_divergence': kl_stats,
            'active_dimensions': {
                'count': active_dims,
                'ratio': float(active_dims / self.latent_channels),
                'channel_variances': channel_vars.tolist()
            }
        }
    
    def save_diagnostics(self, save_path):
        """Save diagnostic data."""
        from pathlib import Path
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.all_means:
            all_means = torch.cat(self.all_means, dim=0)
            all_logvars = torch.cat(self.all_logvars, dim=0)
            
            np.save(save_path / 'posterior_means.npy', all_means.numpy())
            np.save(save_path / 'posterior_logvars.npy', all_logvars.numpy())
            np.save(save_path / 'kl_divergences.npy', np.array(self.kl_divergences))
        
        logger.info(f"KL diagnostics saved to {save_path}")
