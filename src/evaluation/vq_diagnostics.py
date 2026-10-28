"""
VQ Autoencoder Diagnostic Module

Extracts VQ-specific diagnostic metrics for LDM-readiness evaluation:
- Codebook usage statistics
- Quantization error analysis
- Pre-quantization vs post-quantization latent comparison
- Codebook entry utilization patterns
- Dead code detection
- Commitment loss analysis

These diagnostics assess whether the VQ representation is suitable as
a latent space for diffusion modeling.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VQDiagnostics:
    """
    Compute VQ-specific diagnostic metrics for a trained VQ autoencoder.
    
    Focuses on codebook behavior and quantization quality, which are
    critical for assessing LDM-readiness.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        codebook_size: int,
        latent_channels: int
    ):
        """
        Args:
            model: Trained VQ autoencoder model
            device: Device for computation
            codebook_size: Size of VQ codebook (e.g., 512)
            latent_channels: Number of channels in latent space
        """
        self.model = model
        self.device = device
        self.codebook_size = codebook_size
        self.latent_channels = latent_channels
        
        # Accumulators for batch-wise statistics
        self.reset_accumulators()
    
    def reset_accumulators(self):
        """Reset all batch-wise accumulators."""
        self.codebook_usage_count = np.zeros(self.codebook_size, dtype=np.int64)
        self.quantization_errors = []
        self.commitment_losses = []
        self.perplexity_values = []
        self.pre_quant_latents = []
        self.post_quant_latents = []
        self.indices_per_sample = []
        
    def extract_vq_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract VQ-specific outputs from the model.
        
        The model should return or expose:
        - pre_quant: latent before quantization [B, C, H, W]
        - post_quant: latent after quantization [B, C, H, W]
        - indices: codebook indices [B, H, W]
        - quant_loss: quantization loss (commitment + codebook)
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Dictionary containing VQ-specific outputs
        """
        self.model.eval()
        with torch.no_grad():
            # VQ model encode() returns: (z_quantized, h_pre_quant, vq_losses)
            z_quantized, h_pre_quant, vq_losses = self.model.encode(x)
            
            # Compute indices by finding nearest codebook entry
            # This matches what the quantizer does internally
            # Flatten: (B, C, H, W) -> (B*H*W, C)
            h_flattened = h_pre_quant.permute(0, 2, 3, 1).contiguous().view(-1, h_pre_quant.shape[1])
            
            # Compute distances to all codebook entries
            distances = (
                torch.sum(h_flattened ** 2, dim=1, keepdim=True)
                + torch.sum(self.model.quantizer.embedding.weight ** 2, dim=1)
                - 2 * torch.matmul(h_flattened, self.model.quantizer.embedding.weight.t())
            )
            
            # Find nearest entry
            encoding_indices = torch.argmin(distances, dim=1)
            
            # Reshape to (B, H, W)
            B, C, H, W = h_pre_quant.shape
            indices = encoding_indices.view(B, H, W)
            
            # Build output dict matching expected format
            outputs = {
                'post_quant': z_quantized,
                'pre_quant': h_pre_quant,
                'indices': indices,
                'vq_loss': vq_losses.get('vq_loss', None)
            }
            
        return outputs
    
    def compute_codebook_usage(self, indices: torch.Tensor) -> Dict[str, float]:
        """
        Compute codebook usage statistics from quantization indices.
        
        Args:
            indices: Codebook indices [B, H, W]
            
        Returns:
            Dictionary with usage statistics
        """
        indices_np = indices.cpu().numpy().flatten()
        
        # Update usage counts
        unique, counts = np.unique(indices_np, return_counts=True)
        for idx, count in zip(unique, counts):
            self.codebook_usage_count[idx] += count
        
        # Store indices for this sample
        self.indices_per_sample.append(indices_np)
        
        # Compute perplexity for this batch
        total_codes = len(indices_np)
        probs = counts / total_codes
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        perplexity = np.exp(entropy)
        self.perplexity_values.append(perplexity)
        
        # Active codes in this batch
        active_codes = len(unique)
        active_ratio = active_codes / self.codebook_size
        
        return {
            'batch_perplexity': float(perplexity),
            'batch_active_codes': int(active_codes),
            'batch_active_ratio': float(active_ratio)
        }
    
    def compute_quantization_error(
        self,
        pre_quant: torch.Tensor,
        post_quant: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute quantization error between pre and post quantization.
        
        Args:
            pre_quant: Latent before quantization [B, C, H, W]
            post_quant: Latent after quantization [B, C, H, W]
            
        Returns:
            Dictionary with quantization error metrics
        """
        # MSE between pre and post
        mse = torch.mean((pre_quant - post_quant) ** 2).item()
        
        # Per-channel MSE
        per_channel_mse = torch.mean(
            (pre_quant - post_quant) ** 2,
            dim=[0, 2, 3]
        ).cpu().numpy()
        
        # Relative error
        pre_norm = torch.norm(pre_quant, dim=1).mean().item()
        error_norm = torch.norm(pre_quant - post_quant, dim=1).mean().item()
        relative_error = error_norm / (pre_norm + 1e-8)
        
        self.quantization_errors.append(mse)
        
        return {
            'quant_mse': float(mse),
            'quant_relative_error': float(relative_error),
            'quant_per_channel_mse': per_channel_mse.tolist()
        }
    
    def compute_latent_statistics(
        self,
        pre_quant: torch.Tensor,
        post_quant: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Compute statistical properties of pre and post quantization latents.
        
        Args:
            pre_quant: Latent before quantization [B, C, H, W]
            post_quant: Latent after quantization [B, C, H, W]
            
        Returns:
            Dictionary with latent statistics
        """
        stats = {}
        
        # Pre-quantization statistics
        pre_mean = pre_quant.mean().item()
        pre_std = pre_quant.std().item()
        pre_min = pre_quant.min().item()
        pre_max = pre_quant.max().item()
        
        # Post-quantization statistics
        post_mean = post_quant.mean().item()
        post_std = post_quant.std().item()
        post_min = post_quant.min().item()
        post_max = post_quant.max().item()
        
        # Per-channel statistics
        pre_channel_std = pre_quant.std(dim=[0, 2, 3]).cpu().numpy()
        post_channel_std = post_quant.std(dim=[0, 2, 3]).cpu().numpy()
        
        stats['pre_quant'] = {
            'mean': float(pre_mean),
            'std': float(pre_std),
            'min': float(pre_min),
            'max': float(pre_max),
            'channel_std': pre_channel_std.tolist()
        }
        
        stats['post_quant'] = {
            'mean': float(post_mean),
            'std': float(post_std),
            'min': float(post_min),
            'max': float(post_max),
            'channel_std': post_channel_std.tolist()
        }
        
        # Store for aggregation
        self.pre_quant_latents.append(pre_quant.cpu())
        self.post_quant_latents.append(post_quant.cpu())
        
        return stats
    
    def process_batch(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Process a single batch and extract all VQ diagnostics.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Dictionary with all batch-level diagnostics
        """
        # Extract VQ outputs
        vq_outputs = self.extract_vq_outputs(x)
        
        pre_quant = vq_outputs['pre_quant']
        post_quant = vq_quant = vq_outputs['post_quant']
        indices = vq_outputs['indices']
        
        # Compute diagnostics
        usage_stats = self.compute_codebook_usage(indices)
        quant_error = self.compute_quantization_error(pre_quant, post_quant)
        latent_stats = self.compute_latent_statistics(pre_quant, post_quant)
        
        # Combine results
        batch_diagnostics = {
            'codebook_usage': usage_stats,
            'quantization_error': quant_error,
            'latent_statistics': latent_stats
        }
        
        return batch_diagnostics
    
    def compute_global_statistics(self) -> Dict[str, Any]:
        """
        Compute global statistics across all processed batches.
        
        Returns:
            Dictionary with dataset-level diagnostics
        """
        results = {}
        
        # Codebook usage analysis
        total_usage = self.codebook_usage_count.sum()
        usage_ratios = self.codebook_usage_count / (total_usage + 1e-10)
        
        active_codes = np.sum(self.codebook_usage_count > 0)
        dead_codes = self.codebook_size - active_codes
        
        # Entropy and perplexity
        usage_entropy = -np.sum(
            usage_ratios * np.log(usage_ratios + 1e-10)
        )
        usage_perplexity = np.exp(usage_entropy)
        
        results['codebook_usage'] = {
            'active_codes': int(active_codes),
            'dead_codes': int(dead_codes),
            'active_ratio': float(active_codes / self.codebook_size),
            'usage_entropy': float(usage_entropy),
            'usage_perplexity': float(usage_perplexity),
            'max_code_usage': int(self.codebook_usage_count.max()),
            'min_code_usage': int(self.codebook_usage_count[self.codebook_usage_count > 0].min()) if active_codes > 0 else 0,
            'mean_perplexity': float(np.mean(self.perplexity_values)),
            'std_perplexity': float(np.std(self.perplexity_values))
        }
        
        # Quantization error statistics
        results['quantization_error'] = {
            'mean_quant_mse': float(np.mean(self.quantization_errors)),
            'std_quant_mse': float(np.std(self.quantization_errors)),
            'max_quant_mse': float(np.max(self.quantization_errors)),
            'min_quant_mse': float(np.min(self.quantization_errors))
        }
        
        # Global latent statistics
        if self.pre_quant_latents:
            pre_all = torch.cat(self.pre_quant_latents, dim=0)
            post_all = torch.cat(self.post_quant_latents, dim=0)
            
            results['global_latent_stats'] = {
                'pre_quant_mean': float(pre_all.mean()),
                'pre_quant_std': float(pre_all.std()),
                'post_quant_mean': float(post_all.mean()),
                'post_quant_std': float(post_all.std()),
                'latent_correlation': float(
                    torch.corrcoef(
                        torch.stack([
                            pre_all.flatten()[:10000],  # Sample for efficiency
                            post_all.flatten()[:10000]
                        ])
                    )[0, 1]
                )
            }
        
        return results
    
    def save_diagnostics(self, save_path: Path, expected_samples=None):
        """
        Save codebook usage histogram and other visualizations.
        
        Args:
            save_path: Directory to save diagnostic outputs
            expected_samples: Expected number of samples (for validation)
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Validate sample counts
        if expected_samples is not None:
            actual_perp = len(self.perplexity_values)
            actual_quant = len(self.quantization_errors)
            
            if actual_perp != expected_samples:
                logger.warning(
                    f"Perplexity trajectory count mismatch: "
                    f"expected {expected_samples}, got {actual_perp}"
                )
            if actual_quant != expected_samples:
                logger.warning(
                    f"Quantization error trajectory count mismatch: "
                    f"expected {expected_samples}, got {actual_quant}"
                )
        
        # Save codebook usage histogram data
        logger.info(f"Saving VQ diagnostics: {len(self.perplexity_values)} samples")
        np.save(
            save_path / 'codebook_usage_counts.npy',
            self.codebook_usage_count
        )
        
        # Save perplexity trajectory
        np.save(
            save_path / 'perplexity_trajectory.npy',
            np.array(self.perplexity_values)
        )
        
        # Save quantization error trajectory
        np.save(
            save_path / 'quantization_error_trajectory.npy',
            np.array(self.quantization_errors)
        )
        
        logger.info(f"VQ diagnostics saved to {save_path}")


def analyze_vq_representation_choice(
    pre_quant_latents: List[torch.Tensor],
    post_quant_latents: List[torch.Tensor]
) -> Dict[str, Any]:
    """
    Analyze differences between pre-quant and post-quant representations
    to inform which should be used for LDM Stage 2.
    
    This analysis is deferred per DECISIONS.md but the infrastructure
    is provided for future use.
    
    Args:
        pre_quant_latents: List of pre-quantization latent tensors
        post_quant_latents: List of post-quantization latent tensors
        
    Returns:
        Dictionary with comparative analysis
    """
    pre_all = torch.cat(pre_quant_latents, dim=0)
    post_all = torch.cat(post_quant_latents, dim=0)
    
    analysis = {
        'note': 'Decision on pre-quant vs post-quant deferred to post-evaluation',
        'pre_quant_diversity': float(pre_all.std()),
        'post_quant_diversity': float(post_all.std()),
        'information_loss': float(
            torch.mean((pre_all - post_all) ** 2)
        ),
        'discretization_level': 'Assessed via codebook usage perplexity'
    }
    
    return analysis
