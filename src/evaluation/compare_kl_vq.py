"""
KL vs VQ Comparison Module

Performs direct paired comparisons between KL and VQ autoencoders:
- Side-by-side reconstruction quality
- Latent space characteristics comparison
- Statistical significance testing
- LDM-readiness comparative assessment

All comparisons use identical validation samples (seed=42) for
controlled evaluation.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class KLVQComparator:
    """
    Compare KL and VQ autoencoder performance for LDM Stage 2 selection.
    """
    
    def __init__(self, device: torch.device):
        """
        Args:
            device: Device for computation
        """
        self.device = device
        self.reset_accumulators()
    
    def reset_accumulators(self):
        """Reset all comparison accumulators."""
        self.kl_metrics = []
        self.vq_metrics = []
        self.paired_samples = []
        
    def add_paired_comparison(
        self,
        kl_results: Dict[str, float],
        vq_results: Dict[str, float],
        image_id: Optional[str] = None
    ):
        """
        Add a paired comparison for the same input image.
        
        Args:
            kl_results: Metrics from KL autoencoder
            vq_results: Metrics from VQ autoencoder
            image_id: Optional identifier for the image
        """
        self.kl_metrics.append(kl_results)
        self.vq_metrics.append(vq_results)
        self.paired_samples.append({
            'image_id': image_id,
            'kl': kl_results,
            'vq': vq_results
        })
    
    def compute_metric_comparison(
        self,
        metric_name: str
    ) -> Dict[str, Any]:
        """
        Compare a specific metric between KL and VQ across all samples.
        
        Args:
            metric_name: Name of the metric to compare
            
        Returns:
            Dictionary with comparison statistics
        """
        kl_values = [m[metric_name] for m in self.kl_metrics if metric_name in m]
        vq_values = [m[metric_name] for m in self.vq_metrics if metric_name in m]
        
        if not kl_values or not vq_values:
            return {
                'error': f'Metric {metric_name} not found in results',
                'kl_available': len(kl_values),
                'vq_available': len(vq_values)
            }
        
        kl_array = np.array(kl_values)
        vq_array = np.array(vq_values)
        
        # Basic statistics
        comparison = {
            'metric': metric_name,
            'kl': {
                'mean': float(np.mean(kl_array)),
                'std': float(np.std(kl_array)),
                'median': float(np.median(kl_array)),
                'min': float(np.min(kl_array)),
                'max': float(np.max(kl_array))
            },
            'vq': {
                'mean': float(np.mean(vq_array)),
                'std': float(np.std(vq_array)),
                'median': float(np.median(vq_array)),
                'min': float(np.min(vq_array)),
                'max': float(np.max(vq_array))
            }
        }
        
        # Paired t-test
        if len(kl_array) == len(vq_array):
            t_stat, p_value = stats.ttest_rel(kl_array, vq_array)
            comparison['paired_t_test'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant_at_0.05': bool(p_value < 0.05),
                'interpretation': (
                    'KL significantly better' if p_value < 0.05 and t_stat < 0 else
                    'VQ significantly better' if p_value < 0.05 and t_stat > 0 else
                    'No significant difference'
                )
            }
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        if len(kl_array) == len(vq_array):
            w_stat, w_p = stats.wilcoxon(kl_array, vq_array)
            comparison['wilcoxon_test'] = {
                'statistic': float(w_stat),
                'p_value': float(w_p),
                'significant_at_0.05': bool(w_p < 0.05)
            }
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(kl_array)**2 + np.std(vq_array)**2) / 2)
        cohens_d = (np.mean(kl_array) - np.mean(vq_array)) / (pooled_std + 1e-8)
        comparison['effect_size'] = {
            'cohens_d': float(cohens_d),
            'interpretation': (
                'negligible' if abs(cohens_d) < 0.2 else
                'small' if abs(cohens_d) < 0.5 else
                'medium' if abs(cohens_d) < 0.8 else
                'large'
            )
        }
        
        # Win-lose-tie count
        wins_kl = np.sum(kl_array < vq_array)  # Lower is better for most metrics
        wins_vq = np.sum(vq_array < kl_array)
        ties = np.sum(kl_array == vq_array)
        
        comparison['win_lose_tie'] = {
            'kl_wins': int(wins_kl),
            'vq_wins': int(wins_vq),
            'ties': int(ties),
            'total': len(kl_array)
        }
        
        return comparison
    
    def compare_reconstruction_metrics(self) -> Dict[str, Any]:
        """
        Compare all reconstruction metrics between KL and VQ.
        
        Returns:
            Dictionary with comprehensive reconstruction comparison
        """
        # Standard reconstruction metrics
        metrics_to_compare = [
            'mse', 'ssim', 'psnr',
            'boundary_mse', 'non_boundary_mse',
            'boundary_ratio',
            'gradient_preservation'
        ]
        
        comparisons = {}
        for metric in metrics_to_compare:
            comparisons[metric] = self.compute_metric_comparison(metric)
        
        return comparisons
    
    def compare_latent_properties(
        self,
        kl_latent_stats: Dict[str, Any],
        vq_latent_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare latent space properties between KL and VQ.
        
        Args:
            kl_latent_stats: KL latent distribution statistics
            vq_latent_stats: VQ latent distribution statistics
            
        Returns:
            Dictionary with latent property comparison
        """
        comparison = {
            'kl_properties': kl_latent_stats,
            'vq_properties': vq_latent_stats
        }
        
        # Extract LDM-readiness scores if available
        if 'readiness_score' in kl_latent_stats and 'readiness_score' in vq_latent_stats:
            kl_score = kl_latent_stats['readiness_score']
            vq_score = vq_latent_stats['readiness_score']
            
            comparison['ldm_readiness_comparison'] = {
                'kl_score': kl_score,
                'vq_score': vq_score,
                'difference': float(vq_score - kl_score),
                'recommendation': (
                    'KL preferred' if kl_score > vq_score + 0.1 else
                    'VQ preferred' if vq_score > kl_score + 0.1 else
                    'Comparable'
                )
            }
        
        return comparison
    
    def identify_failure_cases(
        self,
        threshold_mse: float = 0.01,
        threshold_ssim: float = 0.8
    ) -> Dict[str, List[str]]:
        """
        Identify samples where either model performs poorly.
        
        Args:
            threshold_mse: MSE threshold for failure
            threshold_ssim: SSIM threshold for failure
            
        Returns:
            Dictionary with lists of failure case IDs
        """
        kl_failures = []
        vq_failures = []
        both_failures = []
        
        for sample in self.paired_samples:
            image_id = sample.get('image_id', 'unknown')
            kl_mse = sample['kl'].get('mse', 0.0)
            vq_mse = sample['vq'].get('mse', 0.0)
            kl_ssim = sample['kl'].get('ssim', 1.0)
            vq_ssim = sample['vq'].get('ssim', 1.0)
            
            kl_failed = kl_mse > threshold_mse or kl_ssim < threshold_ssim
            vq_failed = vq_mse > threshold_mse or vq_ssim < threshold_ssim
            
            if kl_failed and vq_failed:
                both_failures.append(image_id)
            elif kl_failed:
                kl_failures.append(image_id)
            elif vq_failed:
                vq_failures.append(image_id)
        
        return {
            'kl_only_failures': kl_failures,
            'vq_only_failures': vq_failures,
            'both_failures': both_failures,
            'failure_counts': {
                'kl_only': len(kl_failures),
                'vq_only': len(vq_failures),
                'both': len(both_failures),
                'total_samples': len(self.paired_samples)
            }
        }
    
    def generate_summary_recommendation(
        self,
        reconstruction_comparison: Dict[str, Any],
        latent_comparison: Dict[str, Any],
        kl_diagnostics: Optional[Dict[str, Any]] = None,
        vq_diagnostics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a summary recommendation for LDM Stage 2.
        
        Args:
            reconstruction_comparison: Reconstruction metrics comparison
            latent_comparison: Latent properties comparison
            kl_diagnostics: Optional KL-specific diagnostics
            vq_diagnostics: Optional VQ-specific diagnostics
            
        Returns:
            Dictionary with summary and recommendation
        """
        summary = {
            'evaluation_type': 'LDM Stage 2 First-Stage Selection',
            'comparison_date': None,  # Will be filled by caller
            'total_samples': len(self.paired_samples)
        }
        
        # Aggregate key findings
        key_findings = []
        
        # Reconstruction quality
        if 'mse' in reconstruction_comparison:
            mse_comp = reconstruction_comparison['mse']
            if 'paired_t_test' in mse_comp:
                if mse_comp['paired_t_test']['significant_at_0.05']:
                    winner = 'KL' if mse_comp['paired_t_test']['t_statistic'] < 0 else 'VQ'
                    key_findings.append(
                        f"{winner} shows significantly better reconstruction (MSE, p={mse_comp['paired_t_test']['p_value']:.4f})"
                    )
                else:
                    key_findings.append("No significant difference in reconstruction quality (MSE)")
        
        # SSIM
        if 'ssim' in reconstruction_comparison:
            ssim_comp = reconstruction_comparison['ssim']
            if 'paired_t_test' in ssim_comp:
                if ssim_comp['paired_t_test']['significant_at_0.05']:
                    # For SSIM, higher is better, so reverse interpretation
                    winner = 'KL' if ssim_comp['paired_t_test']['t_statistic'] > 0 else 'VQ'
                    key_findings.append(
                        f"{winner} shows significantly better structural similarity (SSIM, p={ssim_comp['paired_t_test']['p_value']:.4f})"
                    )
        
        # Boundary preservation
        if 'boundary_ratio' in reconstruction_comparison:
            br_comp = reconstruction_comparison['boundary_ratio']
            kl_br = br_comp['kl']['mean']
            vq_br = br_comp['vq']['mean']
            if abs(kl_br - vq_br) > 0.1:
                better = 'KL' if kl_br < vq_br else 'VQ'
                key_findings.append(
                    f"{better} shows better boundary preservation (ratio: KL={kl_br:.3f}, VQ={vq_br:.3f})"
                )
        
        # LDM readiness
        if 'ldm_readiness_comparison' in latent_comparison:
            ldm_comp = latent_comparison['ldm_readiness_comparison']
            key_findings.append(
                f"LDM readiness: {ldm_comp['recommendation']} "
                f"(KL={ldm_comp['kl_score']:.3f}, VQ={ldm_comp['vq_score']:.3f})"
            )
        
        # VQ-specific: codebook usage
        if vq_diagnostics and 'codebook_usage' in vq_diagnostics:
            usage = vq_diagnostics['codebook_usage']
            active_ratio = usage.get('active_ratio', 0.0)
            if active_ratio < 0.5:
                key_findings.append(
                    f"VQ codebook underutilized: {active_ratio*100:.1f}% active codes"
                )
            elif active_ratio > 0.9:
                key_findings.append(
                    f"VQ codebook well-utilized: {active_ratio*100:.1f}% active codes"
                )
        
        # KL-specific: KL divergence
        if kl_diagnostics and 'kl_divergence' in kl_diagnostics:
            kl_div = kl_diagnostics['kl_divergence']
            mean_kl = kl_div.get('mean', 0.0)
            if mean_kl > 10.0:
                key_findings.append(
                    f"KL divergence relatively high: {mean_kl:.3f} (may indicate posterior collapse risk)"
                )
        
        summary['key_findings'] = key_findings
        
        # Generate recommendation
        # This is a heuristic synthesis - user makes final decision
        recommendation = {
            'note': 'This is a technical assessment. Final architectural decision is made by the user.',
            'considerations': []
        }
        
        # Reconstruction quality consideration
        mse_winner = None
        ssim_winner = None
        
        if 'mse' in reconstruction_comparison:
            mse_comp = reconstruction_comparison['mse']
            if 'paired_t_test' in mse_comp and mse_comp['paired_t_test']['significant_at_0.05']:
                mse_winner = 'KL' if mse_comp['kl']['mean'] < mse_comp['vq']['mean'] else 'VQ'
                recommendation['considerations'].append(
                    f"Reconstruction quality (MSE): {mse_winner} performs better"
                )
        
        if 'ssim' in reconstruction_comparison:
            ssim_comp = reconstruction_comparison['ssim']
            if 'paired_t_test' in ssim_comp and ssim_comp['paired_t_test']['significant_at_0.05']:
                ssim_winner = 'KL' if ssim_comp['kl']['mean'] > ssim_comp['vq']['mean'] else 'VQ'
                recommendation['considerations'].append(
                    f"Structural similarity (SSIM): {ssim_winner} performs better"
                )
        
        # LDM readiness consideration
        if 'ldm_readiness_comparison' in latent_comparison:
            ldm_rec = latent_comparison['ldm_readiness_comparison']['recommendation']
            recommendation['considerations'].append(
                f"LDM readiness: {ldm_rec}"
            )
        
        # Overall suggestion (heuristic)
        if mse_winner == ssim_winner and mse_winner is not None:
            recommendation['preliminary_suggestion'] = f"{mse_winner} may be preferred based on reconstruction quality"
        else:
            recommendation['preliminary_suggestion'] = "Trade-offs exist; review detailed metrics for decision"
        
        summary['recommendation'] = recommendation
        
        return summary
    
    def save_comparison_report(
        self,
        save_path: Path,
        reconstruction_comparison: Dict[str, Any],
        latent_comparison: Dict[str, Any],
        summary: Dict[str, Any]
    ):
        """
        Save comprehensive comparison report to JSON.
        
        Args:
            save_path: Path to save the report
            reconstruction_comparison: Reconstruction metrics comparison
            latent_comparison: Latent properties comparison
            summary: Summary and recommendation
        """
        report = {
            'summary': summary,
            'reconstruction_comparison': reconstruction_comparison,
            'latent_comparison': latent_comparison,
            'failure_analysis': self.identify_failure_cases(),
            'sample_count': len(self.paired_samples)
        }
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comparison report saved to {save_path}")


def aggregate_results_for_comparison(
    kl_results: Dict[str, Any],
    vq_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Aggregate results from KL and VQ evaluation for direct comparison.
    
    Args:
        kl_results: Complete KL evaluation results
        vq_results: Complete VQ evaluation results
        
    Returns:
        Dictionary with aggregated comparison-ready data
    """
    aggregated = {
        'kl': {
            'reconstruction_metrics': kl_results.get('reconstruction_metrics', {}),
            'latent_diagnostics': kl_results.get('latent_diagnostics', {}),
            'boundary_metrics': kl_results.get('boundary_metrics', {})
        },
        'vq': {
            'reconstruction_metrics': vq_results.get('reconstruction_metrics', {}),
            'latent_diagnostics': vq_results.get('latent_diagnostics', {}),
            'boundary_metrics': vq_results.get('boundary_metrics', {})
        }
    }
    
    return aggregated
