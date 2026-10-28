"""
Latent Distribution Analysis Module

Characterizes the statistical properties of latent representations from
KL and VQ autoencoders to assess LDM-readiness:

- Distribution shape and moments
- Normality assessment
- Spatial correlation structure
- Temporal stability across validation set
- Suitability for diffusion modeling

These analyses inform whether the latent space is well-behaved for
the latent diffusion process in Stage 2.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LatentDistributionAnalyzer:
    """
    Analyze the statistical distribution of latent representations
    for LDM-readiness assessment.
    """
    
    def __init__(self, device: torch.device):
        """
        Args:
            device: Device for computation
        """
        self.device = device
        self.reset_accumulators()
    
    def reset_accumulators(self):
        """Reset all batch-wise accumulators."""
        self.latent_samples = []
        self.latent_means = []
        self.latent_stds = []
        self.spatial_correlations = []
        
    def characterize_distribution(
        self,
        latents: torch.Tensor,
        store_trajectory: bool = True
    ) -> Dict[str, Any]:
        """
        Characterize the statistical distribution of a latent tensor.
        
        Args:
            latents: Latent representation [B, C, H, W]
            store_trajectory: Whether to store in trajectory accumulators
            
        Returns:
            Dictionary with distribution characteristics
        """
        # Flatten spatial dimensions for distribution analysis
        B, C, H, W = latents.shape
        latents_flat = latents.reshape(B, C, -1)  # [B, C, H*W]
        
        # Global statistics
        global_mean = latents.mean().item()
        global_std = latents.std().item()
        global_min = latents.min().item()
        global_max = latents.max().item()
        
        # Per-channel statistics
        channel_means = latents.mean(dim=[0, 2, 3]).cpu().numpy()
        channel_stds = latents.std(dim=[0, 2, 3]).cpu().numpy()
        
        # Kurtosis and skewness (per channel, averaged)
        latents_np = latents.cpu().numpy()
        channel_kurtosis = []
        channel_skewness = []
        
        for c in range(C):
            channel_data = latents_np[:, c, :, :].flatten()
            channel_kurtosis.append(stats.kurtosis(channel_data))
            channel_skewness.append(stats.skew(channel_data))
        
        kurtosis_mean = float(np.mean(channel_kurtosis))
        skewness_mean = float(np.mean(channel_skewness))
        
        # Store for aggregation (only if requested to avoid double-counting)
        if store_trajectory:
            self.latent_samples.append(latents.cpu())
            self.latent_means.append(global_mean)
            self.latent_stds.append(global_std)
        
        return {
            'global_mean': global_mean,
            'global_std': global_std,
            'global_min': global_min,
            'global_max': global_max,
            'channel_means': channel_means.tolist(),
            'channel_stds': channel_stds.tolist(),
            'kurtosis_mean': kurtosis_mean,
            'skewness_mean': skewness_mean
        }
    
    def assess_normality(
        self,
        latents: torch.Tensor,
        sample_size: int = 5000
    ) -> Dict[str, float]:
        """
        Assess how closely the latent distribution approximates normality.
        
        Important for diffusion models which typically assume Gaussian noise.
        
        Args:
            latents: Latent representation [B, C, H, W]
            sample_size: Number of samples for normality test
            
        Returns:
            Dictionary with normality assessment
        """
        # Sample for efficiency
        latents_flat = latents.flatten().cpu().numpy()
        
        if len(latents_flat) > sample_size:
            sample_indices = np.random.choice(
                len(latents_flat),
                size=sample_size,
                replace=False
            )
            sample = latents_flat[sample_indices]
        else:
            sample = latents_flat
        
        # Shapiro-Wilk test (if sample small enough)
        if len(sample) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(sample)
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan
        
        # Kolmogorov-Smirnov test against normal distribution
        # Standardize sample first
        sample_standardized = (sample - sample.mean()) / (sample.std() + 1e-8)
        ks_stat, ks_p = stats.kstest(sample_standardized, 'norm')
        
        # Anderson-Darling test
        anderson_result = stats.anderson(sample_standardized, dist='norm')
        
        return {
            'shapiro_statistic': float(shapiro_stat) if not np.isnan(shapiro_stat) else None,
            'shapiro_p_value': float(shapiro_p) if not np.isnan(shapiro_p) else None,
            'ks_statistic': float(ks_stat),
            'ks_p_value': float(ks_p),
            'anderson_statistic': float(anderson_result.statistic),
            'anderson_critical_values': anderson_result.critical_values.tolist(),
            'normality_assessment': 'approximately_normal' if ks_p > 0.05 else 'non_normal'
        }
    
    def analyze_spatial_correlation(
        self,
        latents: torch.Tensor
    ) -> Dict[str, float]:
        """
        Analyze spatial correlation structure within latent maps.
        
        High spatial correlation may indicate redundancy or smoothness.
        
        Args:
            latents: Latent representation [B, C, H, W]
            
        Returns:
            Dictionary with spatial correlation metrics
        """
        B, C, H, W = latents.shape
        
        # Compute autocorrelation at lag 1 (horizontal and vertical)
        horizontal_corr = []
        vertical_corr = []
        
        for b in range(B):
            for c in range(C):
                latent_map = latents[b, c].cpu().numpy()
                
                # Horizontal autocorrelation
                if W > 1:
                    h_corr = np.corrcoef(
                        latent_map[:, :-1].flatten(),
                        latent_map[:, 1:].flatten()
                    )[0, 1]
                    horizontal_corr.append(h_corr)
                
                # Vertical autocorrelation
                if H > 1:
                    v_corr = np.corrcoef(
                        latent_map[:-1, :].flatten(),
                        latent_map[1:, :].flatten()
                    )[0, 1]
                    vertical_corr.append(v_corr)
        
        mean_h_corr = float(np.mean(horizontal_corr)) if horizontal_corr else 0.0
        mean_v_corr = float(np.mean(vertical_corr)) if vertical_corr else 0.0
        
        # Overall spatial smoothness
        spatial_smoothness = (mean_h_corr + mean_v_corr) / 2.0
        
        self.spatial_correlations.append(spatial_smoothness)
        
        return {
            'mean_horizontal_correlation': mean_h_corr,
            'mean_vertical_correlation': mean_v_corr,
            'spatial_smoothness': spatial_smoothness
        }
    
    def assess_stability(self) -> Dict[str, float]:
        """
        Assess temporal stability of latent statistics across validation set.
        
        Stable statistics suggest consistent encoding behavior.
        
        Returns:
            Dictionary with stability metrics
        """
        if len(self.latent_means) < 2:
            return {
                'mean_stability_cv': 0.0,
                'std_stability_cv': 0.0,
                'note': 'Insufficient samples for stability assessment'
            }
        
        # Coefficient of variation for means and stds
        means_array = np.array(self.latent_means)
        stds_array = np.array(self.latent_stds)
        
        mean_cv = float(np.std(means_array) / (np.mean(means_array) + 1e-8))
        std_cv = float(np.std(stds_array) / (np.mean(stds_array) + 1e-8))
        
        return {
            'mean_stability_cv': mean_cv,
            'std_stability_cv': std_cv,
            'interpretation': 'Lower CV indicates more stable encoding'
        }
    
    def compute_effective_dimensionality(
        self,
        latents: torch.Tensor,
        variance_threshold: float = 0.99
    ) -> Dict[str, Any]:
        """
        Estimate effective dimensionality of latent space via PCA.
        
        Helps assess information capacity and redundancy.
        
        Args:
            latents: Latent representation [B, C, H, W]
            variance_threshold: Cumulative variance threshold
            
        Returns:
            Dictionary with dimensionality metrics
        """
        B, C, H, W = latents.shape
        
        # Reshape to [B*H*W, C] for PCA
        latents_reshaped = latents.permute(0, 2, 3, 1).reshape(-1, C)
        latents_np = latents_reshaped.cpu().numpy()
        
        # Subsample if too large
        if latents_np.shape[0] > 10000:
            indices = np.random.choice(
                latents_np.shape[0],
                size=10000,
                replace=False
            )
            latents_np = latents_np[indices]
        
        # Center the data
        latents_centered = latents_np - latents_np.mean(axis=0)
        
        # Compute covariance and eigenvalues
        cov_matrix = np.cov(latents_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues in descending order
        eigenvalues = eigenvalues[::-1]
        
        # Compute explained variance ratio
        total_variance = eigenvalues.sum()
        explained_variance_ratio = eigenvalues / total_variance
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Effective dimensionality: number of components to reach threshold
        effective_dims = int(np.searchsorted(cumulative_variance, variance_threshold) + 1)
        
        return {
            'total_dimensions': C,
            'effective_dimensions': effective_dims,
            'variance_threshold': variance_threshold,
            'explained_variance_ratio': explained_variance_ratio[:10].tolist(),  # Top 10
            'cumulative_variance': cumulative_variance[:10].tolist(),
            'dimensionality_ratio': float(effective_dims / C)
        }
    
    def assess_ldm_readiness(
        self,
        latents: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Comprehensive assessment of LDM-readiness based on distribution properties.
        
        Args:
            latents: Latent representation [B, C, H, W]
            
        Returns:
            Dictionary with LDM-readiness assessment
        """
        # Run all analyses
        dist_char = self.characterize_distribution(latents)
        normality = self.assess_normality(latents)
        spatial_corr = self.analyze_spatial_correlation(latents)
        eff_dim = self.compute_effective_dimensionality(latents)
        
        # Synthesize readiness score (heuristic)
        readiness_factors = {
            'normality': 1.0 if normality['normality_assessment'] == 'approximately_normal' else 0.5,
            'stability': 1.0 if abs(dist_char['skewness_mean']) < 1.0 else 0.5,
            'effective_capacity': eff_dim['dimensionality_ratio'],
            'spatial_structure': 1.0 if 0.3 < spatial_corr['spatial_smoothness'] < 0.9 else 0.7
        }
        
        overall_readiness = float(np.mean(list(readiness_factors.values())))
        
        return {
            'readiness_score': overall_readiness,
            'readiness_factors': readiness_factors,
            'distribution': dist_char,
            'normality': normality,
            'spatial_correlation': spatial_corr,
            'effective_dimensionality': eff_dim,
            'interpretation': (
                'high_readiness' if overall_readiness > 0.8 else
                'moderate_readiness' if overall_readiness > 0.6 else
                'low_readiness'
            )
        }
    
    def compute_global_statistics(self) -> Dict[str, Any]:
        """
        Compute global statistics across all processed samples.
        
        Returns:
            Dictionary with dataset-level statistics
        """
        if not self.latent_samples:
            return {'error': 'No samples processed'}
        
        # Aggregate all latents
        all_latents = torch.cat(self.latent_samples, dim=0)
        
        # Global distribution (don't store trajectory to avoid double-counting)
        global_dist = self.characterize_distribution(all_latents, store_trajectory=False)
        
        # Stability
        stability = self.assess_stability()
        
        # Effective dimensionality
        eff_dim = self.compute_effective_dimensionality(all_latents)
        
        # Spatial correlation summary
        spatial_corr_summary = {
            'mean': float(np.mean(self.spatial_correlations)) if self.spatial_correlations else 0.0,
            'std': float(np.std(self.spatial_correlations)) if self.spatial_correlations else 0.0
        }
        
        return {
            'global_distribution': global_dist,
            'stability': stability,
            'effective_dimensionality': eff_dim,
            'spatial_correlation_summary': spatial_corr_summary
        }
    
    def save_distribution_data(self, save_path: Path):
        """
        Save distribution data for visualization and further analysis.
        
        Args:
            save_path: Directory to save outputs
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save latent samples (subsampled if large)
        if self.latent_samples:
            all_latents = torch.cat(self.latent_samples, dim=0)
            
            # Save histogram data
            latents_flat = all_latents.flatten().cpu().numpy()
            np.save(
                save_path / 'latent_distribution_samples.npy',
                latents_flat[:100000]  # Subsample for storage
            )
        
        # Save trajectory data
        if self.latent_means:
            np.save(
                save_path / 'latent_means_trajectory.npy',
                np.array(self.latent_means)
            )
            np.save(
                save_path / 'latent_stds_trajectory.npy',
                np.array(self.latent_stds)
            )
        
        if self.spatial_correlations:
            np.save(
                save_path / 'spatial_correlations.npy',
                np.array(self.spatial_correlations)
            )
        
        logger.info(f"Distribution data saved to {save_path}")


def compare_distributions(
    kl_latents: List[torch.Tensor],
    vq_latents: List[torch.Tensor],
    device: torch.device
) -> Dict[str, Any]:
    """
    Compare distributions between KL and VQ latent spaces.
    
    Args:
        kl_latents: List of KL latent tensors
        vq_latents: List of VQ latent tensors
        device: Device for computation
        
    Returns:
        Dictionary with comparative distribution analysis
    """
    kl_all = torch.cat(kl_latents, dim=0)
    vq_all = torch.cat(vq_latents, dim=0)
    
    # Basic statistics comparison
    kl_mean = kl_all.mean().item()
    vq_mean = vq_all.mean().item()
    kl_std = kl_all.std().item()
    vq_std = vq_all.std().item()
    
    # Wasserstein distance (1D approximation on flattened samples)
    kl_sample = kl_all.flatten()[:10000].cpu().numpy()
    vq_sample = vq_all.flatten()[:10000].cpu().numpy()
    wasserstein_dist = float(stats.wasserstein_distance(kl_sample, vq_sample))
    
    # KL divergence approximation (histogram-based)
    kl_hist, bin_edges = np.histogram(kl_sample, bins=100, density=True)
    vq_hist, _ = np.histogram(vq_sample, bins=bin_edges, density=True)
    
    # Add small epsilon to avoid log(0)
    kl_hist = kl_hist + 1e-10
    vq_hist = vq_hist + 1e-10
    
    kl_divergence = float(np.sum(kl_hist * np.log(kl_hist / vq_hist)))
    
    return {
        'kl_autoencoder': {
            'mean': kl_mean,
            'std': kl_std
        },
        'vq_autoencoder': {
            'mean': vq_mean,
            'std': vq_std
        },
        'distribution_distance': {
            'wasserstein': wasserstein_dist,
            'kl_divergence': kl_divergence
        },
        'interpretation': (
            'Distributions are similar' if wasserstein_dist < 0.1 else
            'Distributions differ moderately' if wasserstein_dist < 0.5 else
            'Distributions differ significantly'
        )
    }
