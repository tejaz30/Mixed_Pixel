"""
Comparison script for Experiment 1: KL vs VQ first-stage autoencoders.

Loads evaluation results from both models and creates comparison tables and visualizations.

Usage:
    python scripts/compare_experiment1.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_evaluation_results(experiment_dir: Path) -> dict:
    """Load evaluation results from JSON file."""
    results_file = experiment_dir / 'evaluation_results.json'
    if not results_file.exists():
        raise FileNotFoundError(f"Results not found: {results_file}")
    
    with open(results_file, 'r') as f:
        return json.load(f)


def create_comparison_table(kl_results: dict, vq_results: dict) -> pd.DataFrame:
    """Create comparison table of metrics."""
    
    metrics = {
        'Metric': [],
        'KL': [],
        'VQ': [],
        'Difference (VQ - KL)': [],
        'Relative Difference (%)': []
    }
    
    # Global metrics
    global_metrics = [
        ('Global MSE', 'global_metrics', 'mse_mean'),
        ('Global SSIM', 'global_metrics', 'ssim_mean'),
        ('Global MAE', 'global_metrics', 'mae_mean'),
    ]
    
    for name, category, key in global_metrics:
        kl_val = kl_results[category][key]
        vq_val = vq_results[category][key]
        diff = vq_val - kl_val
        rel_diff = (diff / kl_val * 100) if kl_val != 0 else 0
        
        metrics['Metric'].append(name)
        metrics['KL'].append(f'{kl_val:.6f}')
        metrics['VQ'].append(f'{vq_val:.6f}')
        metrics['Difference (VQ - KL)'].append(f'{diff:+.6f}')
        metrics['Relative Difference (%)'].append(f'{rel_diff:+.2f}%')
    
    # Boundary metrics
    if 'boundary_metrics' in kl_results:
        boundary_metrics = [
            ('Boundary MSE', 'boundary_metrics', 'boundary_mse_mean'),
            ('Non-boundary MSE', 'boundary_metrics', 'non_boundary_mse_mean'),
            ('Boundary/Non-boundary Ratio', 'boundary_metrics', 'error_ratio_mean'),
            ('Gradient Similarity', 'boundary_metrics', 'gradient_similarity_mean'),
            ('Gradient Correlation', 'boundary_metrics', 'gradient_correlation_mean'),
        ]
        
        for name, category, key in boundary_metrics:
            kl_val = kl_results[category][key]
            vq_val = vq_results[category][key]
            diff = vq_val - kl_val
            rel_diff = (diff / kl_val * 100) if kl_val != 0 else 0
            
            metrics['Metric'].append(name)
            metrics['KL'].append(f'{kl_val:.6f}')
            metrics['VQ'].append(f'{vq_val:.6f}')
            metrics['Difference (VQ - KL)'].append(f'{diff:+.6f}')
            metrics['Relative Difference (%)'].append(f'{rel_diff:+.2f}%')
    
    # Latent diagnostics
    if 'latent_diagnostics' in vq_results:
        vq_diag = vq_results['latent_diagnostics']
        if 'perplexity_mean' in vq_diag:
            metrics['Metric'].append('VQ Perplexity')
            metrics['KL'].append('N/A')
            metrics['VQ'].append(f"{vq_diag['perplexity_mean']:.2f}")
            metrics['Difference (VQ - KL)'].append('N/A')
            metrics['Relative Difference (%)'].append('N/A')
        
        if 'utilization_mean' in vq_diag:
            metrics['Metric'].append('VQ Codebook Utilization')
            metrics['KL'].append('N/A')
            metrics['VQ'].append(f"{vq_diag['utilization_mean']:.4f}")
            metrics['Difference (VQ - KL)'].append('N/A')
            metrics['Relative Difference (%)'].append('N/A')
    
    return pd.DataFrame(metrics)


def create_visualizations(kl_results: dict, vq_results: dict, output_dir: Path):
    """Create comparison visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Global metrics comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_to_plot = [
        ('MSE', 'mse_mean', 'lower is better'),
        ('SSIM', 'ssim_mean', 'higher is better'),
        ('MAE', 'mae_mean', 'lower is better')
    ]
    
    for idx, (name, key, direction) in enumerate(metrics_to_plot):
        kl_val = kl_results['global_metrics'][key]
        vq_val = vq_results['global_metrics'][key]
        
        axes[idx].bar(['KL', 'VQ'], [kl_val, vq_val], color=['#3498db', '#e74c3c'])
        axes[idx].set_ylabel(name)
        axes[idx].set_title(f'{name} ({direction})')
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, val in enumerate([kl_val, vq_val]):
            axes[idx].text(i, val, f'{val:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'global_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Boundary metrics comparison
    if 'boundary_metrics' in kl_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Boundary vs non-boundary MSE
        kl_boundary = kl_results['boundary_metrics']['boundary_mse_mean']
        kl_non_boundary = kl_results['boundary_metrics']['non_boundary_mse_mean']
        vq_boundary = vq_results['boundary_metrics']['boundary_mse_mean']
        vq_non_boundary = vq_results['boundary_metrics']['non_boundary_mse_mean']
        
        x = ['Boundary', 'Non-boundary']
        kl_vals = [kl_boundary, kl_non_boundary]
        vq_vals = [vq_boundary, vq_non_boundary]
        
        x_pos = range(len(x))
        width = 0.35
        
        axes[0].bar([p - width/2 for p in x_pos], kl_vals, width, label='KL', color='#3498db')
        axes[0].bar([p + width/2 for p in x_pos], vq_vals, width, label='VQ', color='#e74c3c')
        axes[0].set_ylabel('MSE')
        axes[0].set_title('MSE: Boundary vs Non-boundary Regions')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(x)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Gradient preservation
        kl_grad_sim = kl_results['boundary_metrics']['gradient_similarity_mean']
        kl_grad_corr = kl_results['boundary_metrics']['gradient_correlation_mean']
        vq_grad_sim = vq_results['boundary_metrics']['gradient_similarity_mean']
        vq_grad_corr = vq_results['boundary_metrics']['gradient_correlation_mean']
        
        x = ['Similarity', 'Correlation']
        kl_vals = [kl_grad_sim, kl_grad_corr]
        vq_vals = [vq_grad_sim, vq_grad_corr]
        
        x_pos = range(len(x))
        axes[1].bar([p - width/2 for p in x_pos], kl_vals, width, label='KL', color='#3498db')
        axes[1].bar([p + width/2 for p in x_pos], vq_vals, width, label='VQ', color='#e74c3c')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Gradient Preservation (higher is better)')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(x)
        axes[1].legend()
        axes[1].set_ylim([0, 1.1])
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'boundary_metrics_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Comparison visualizations saved to {output_dir}")


def generate_report(kl_results: dict, vq_results: dict, output_file: Path):
    """Generate markdown report."""
    
    report = []
    report.append("# Experiment 1: KL vs VQ First-Stage Autoencoder Comparison")
    report.append("")
    report.append("## Research Question")
    report.append("")
    report.append("How do KL-regularized and VQ-regularized first-stage autoencoders differ in their ability to represent and reconstruct thermal imagery, particularly with respect to preserving information at thermal boundaries?")
    report.append("")
    report.append("## Results Summary")
    report.append("")
    
    # Global metrics
    report.append("### 1. Global Reconstruction Quality")
    report.append("")
    gm_kl = kl_results['global_metrics']
    gm_vq = vq_results['global_metrics']
    
    report.append(f"**MSE:**")
    report.append(f"- KL: {gm_kl['mse_mean']:.6f} ± {gm_kl['mse_std']:.6f}")
    report.append(f"- VQ: {gm_vq['mse_mean']:.6f} ± {gm_vq['mse_std']:.6f}")
    mse_diff = ((gm_vq['mse_mean'] - gm_kl['mse_mean']) / gm_kl['mse_mean']) * 100
    report.append(f"- Difference: {mse_diff:+.2f}% {'(VQ better)' if mse_diff < 0 else '(KL better)'}")
    report.append("")
    
    report.append(f"**SSIM:**")
    report.append(f"- KL: {gm_kl['ssim_mean']:.4f} ± {gm_kl['ssim_std']:.4f}")
    report.append(f"- VQ: {gm_vq['ssim_mean']:.4f} ± {gm_vq['ssim_std']:.4f}")
    ssim_diff = ((gm_vq['ssim_mean'] - gm_kl['ssim_mean']) / gm_kl['ssim_mean']) * 100
    report.append(f"- Difference: {ssim_diff:+.2f}% {'(VQ better)' if ssim_diff > 0 else '(KL better)'}")
    report.append("")
    
    # Boundary metrics
    if 'boundary_metrics' in kl_results:
        report.append("### 2. Boundary-Focused Reconstruction")
        report.append("")
        bm_kl = kl_results['boundary_metrics']
        bm_vq = vq_results['boundary_metrics']
        
        report.append(f"**Boundary Region MSE:**")
        report.append(f"- KL: {bm_kl['boundary_mse_mean']:.6f}")
        report.append(f"- VQ: {bm_vq['boundary_mse_mean']:.6f}")
        boundary_diff = ((bm_vq['boundary_mse_mean'] - bm_kl['boundary_mse_mean']) / bm_kl['boundary_mse_mean']) * 100
        report.append(f"- Difference: {boundary_diff:+.2f}% {'(VQ better)' if boundary_diff < 0 else '(KL better)'}")
        report.append("")
        
        report.append(f"**Non-Boundary Region MSE:**")
        report.append(f"- KL: {bm_kl['non_boundary_mse_mean']:.6f}")
        report.append(f"- VQ: {bm_vq['non_boundary_mse_mean']:.6f}")
        report.append("")
        
        report.append(f"**Error Ratio (Boundary/Non-boundary):**")
        report.append(f"- KL: {bm_kl['error_ratio_mean']:.4f}")
        report.append(f"- VQ: {bm_vq['error_ratio_mean']:.4f}")
        report.append("")
        
        report.append(f"**Gradient Preservation:**")
        report.append(f"- KL Similarity: {bm_kl['gradient_similarity_mean']:.4f}")
        report.append(f"- VQ Similarity: {bm_vq['gradient_similarity_mean']:.4f}")
        grad_diff = ((bm_vq['gradient_similarity_mean'] - bm_kl['gradient_similarity_mean']) / bm_kl['gradient_similarity_mean']) * 100
        report.append(f"- Difference: {grad_diff:+.2f}% {'(VQ better)' if grad_diff > 0 else '(KL better)'}")
        report.append("")
    
    # Latent diagnostics
    report.append("### 3. Latent Representation")
    report.append("")
    
    if 'latent_diagnostics' in kl_results:
        ld_kl = kl_results['latent_diagnostics']
        report.append(f"**KL Model:**")
        report.append(f"- Latent mean: {ld_kl.get('latent_mean_mean', 'N/A')}")
        report.append(f"- Latent std: {ld_kl.get('latent_mean_std', 'N/A')}")
        report.append("")
    
    if 'latent_diagnostics' in vq_results:
        ld_vq = vq_results['latent_diagnostics']
        report.append(f"**VQ Model:**")
        report.append(f"- Codebook size: {ld_vq.get('total_codes', 'N/A')}")
        report.append(f"- Active codes: {ld_vq.get('active_codes', 'N/A')}")
        report.append(f"- Codebook utilization: {ld_vq.get('utilization_mean', 'N/A'):.4f}")
        report.append(f"- Perplexity: {ld_vq.get('perplexity_mean', 'N/A'):.2f}")
        report.append("")
    
    # Interpretation
    report.append("## Interpretation")
    report.append("")
    report.append("### Key Observations")
    report.append("")
    
    # Determine winners
    global_winner = "KL" if gm_kl['mse_mean'] < gm_vq['mse_mean'] else "VQ"
    report.append(f"1. **Global Reconstruction:** {global_winner} achieves lower MSE overall")
    
    if 'boundary_metrics' in kl_results:
        boundary_winner = "KL" if bm_kl['boundary_mse_mean'] < bm_vq['boundary_mse_mean'] else "VQ"
        report.append(f"2. **Boundary Preservation:** {boundary_winner} performs better at high-gradient regions")
        
        gradient_winner = "KL" if bm_kl['gradient_similarity_mean'] > bm_vq['gradient_similarity_mean'] else "VQ"
        report.append(f"3. **Gradient Preservation:** {gradient_winner} better preserves gradient structure")
    
    report.append("")
    report.append("### Limitations")
    report.append("")
    report.append("- Boundary regions are identified algorithmically via gradient detection, not ground-truth annotations")
    report.append("- Single dataset/preprocessing configuration tested")
    report.append("- No per-dataset breakdown in this comparison")
    report.append("")
    report.append("## Recommendations for Future Work")
    report.append("")
    report.append("Based on these results, consider investigating:")
    report.append("- Different KL weights and VQ codebook sizes")
    report.append("- Perceptual losses for better boundary preservation")
    report.append("- Attention mechanisms for thermal boundary regions")
    report.append("- Hybrid KL-VQ approaches")
    report.append("")
    
    # Write report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Report saved to: {output_file}")


def main():
    # Paths
    kl_dir = Path("outputs/experiment1/kl")
    vq_dir = Path("outputs/experiment1/vq")
    comparison_dir = Path("outputs/experiment1/comparison")
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("EXPERIMENT 1 COMPARISON: KL vs VQ")
    print("="*70)
    print()
    
    # Load results
    print("Loading evaluation results...")
    kl_results = load_evaluation_results(kl_dir)
    vq_results = load_evaluation_results(vq_dir)
    print("Done.\n")
    
    # Create comparison table
    print("Creating comparison table...")
    comparison_table = create_comparison_table(kl_results, vq_results)
    
    # Save table
    table_file = comparison_dir / 'comparison_table.csv'
    comparison_table.to_csv(table_file, index=False)
    print(f"Saved to: {table_file}")
    
    # Print table
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    print(comparison_table.to_string(index=False))
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(kl_results, vq_results, comparison_dir)
    print()
    
    # Generate report
    print("Generating markdown report...")
    report_file = comparison_dir / 'experiment1_report.md'
    generate_report(kl_results, vq_results, report_file)
    print()
    
    print("="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print(f"\nAll outputs saved to: {comparison_dir}")


if __name__ == '__main__':
    main()
