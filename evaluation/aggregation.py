"""
Dataset-level Metric Aggregation

Computes summary statistics (mean, std, median, min, max) for MSE and SSIM
across each dataset.
"""

import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from .evaluation_types import EvaluationResult


def aggregate_dataset_metrics(
    results: List[EvaluationResult],
    dataset_name: str
) -> Dict[str, float]:
    """
    Aggregate metrics for a single dataset.
    
    Computes:
    - Mean, std, median, min, max for MSE
    - Mean, std, median, min, max for SSIM
    
    Args:
        results: List of per-image evaluation results.
        dataset_name: Name of the dataset.
    
    Returns:
        Dictionary with aggregated statistics.
    """
    if not results:
        raise ValueError(f"No results provided for dataset: {dataset_name}")
    
    # Extract metric values
    mse_values = np.array([r.mse for r in results])
    ssim_values = np.array([r.ssim for r in results])
    
    # Compute statistics
    aggregated = {
        'dataset': dataset_name,
        'num_images': len(results),
        
        # MSE statistics
        'mse_mean': float(np.mean(mse_values)),
        'mse_std': float(np.std(mse_values, ddof=1)),  # Sample std
        'mse_median': float(np.median(mse_values)),
        'mse_min': float(np.min(mse_values)),
        'mse_max': float(np.max(mse_values)),
        
        # SSIM statistics
        'ssim_mean': float(np.mean(ssim_values)),
        'ssim_std': float(np.std(ssim_values, ddof=1)),  # Sample std
        'ssim_median': float(np.median(ssim_values)),
        'ssim_min': float(np.min(ssim_values)),
        'ssim_max': float(np.max(ssim_values))
    }
    
    logging.info(f"Aggregated metrics for {dataset_name}:")
    logging.info(f"  MSE:  {aggregated['mse_mean']:.6f} ± {aggregated['mse_std']:.6f}")
    logging.info(f"  SSIM: {aggregated['ssim_mean']:.4f} ± {aggregated['ssim_std']:.4f}")
    
    return aggregated


def aggregate_all_datasets(
    results_by_dataset: Dict[str, List[EvaluationResult]]
) -> pd.DataFrame:
    """
    Aggregate metrics for all datasets.
    
    Args:
        results_by_dataset: Dictionary mapping dataset names to result lists.
    
    Returns:
        DataFrame with aggregated statistics for all datasets.
    """
    logging.info("Aggregating metrics across all datasets")
    
    aggregated_list = []
    
    for dataset_name, results in results_by_dataset.items():
        if results:
            agg = aggregate_dataset_metrics(results, dataset_name)
            aggregated_list.append(agg)
        else:
            logging.warning(f"No results for dataset: {dataset_name}")
    
    # Convert to DataFrame
    df = pd.DataFrame(aggregated_list)
    
    # Reorder columns for readability
    column_order = [
        'dataset',
        'num_images',
        'mse_mean', 'mse_std', 'mse_median', 'mse_min', 'mse_max',
        'ssim_mean', 'ssim_std', 'ssim_median', 'ssim_min', 'ssim_max'
    ]
    
    df = df[column_order]
    
    logging.info(f"Aggregated {len(df)} datasets")
    
    return df


def save_aggregated_metrics(
    df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Save aggregated metrics to CSV.
    
    Args:
        df: DataFrame with aggregated statistics.
        output_path: Path to save CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False, float_format='%.6f')
    logging.info(f"Aggregated metrics saved to: {output_path}")
    
    # Also create a formatted markdown table for easy viewing
    md_path = output_path.with_suffix('.md')
    
    with open(md_path, 'w') as f:
        f.write("# Dataset-Level Reconstruction Metrics\n\n")
        f.write("## Summary Statistics\n\n")
        
        # Create formatted table
        f.write("| Dataset | N | Mean MSE | Std MSE | Mean SSIM | Std SSIM |\n")
        f.write("|---------|---|----------|---------|-----------|----------|\n")
        
        for _, row in df.iterrows():
            f.write(
                f"| {row['dataset']:<15} | "
                f"{int(row['num_images']):>4} | "
                f"{row['mse_mean']:>8.6f} | "
                f"{row['mse_std']:>7.6f} | "
                f"{row['ssim_mean']:>9.4f} | "
                f"{row['ssim_std']:>8.4f} |\n"
            )
        
        f.write("\n## Detailed Statistics\n\n")
        f.write("### MSE (Mean Squared Error)\n\n")
        f.write("| Dataset | Mean | Std | Median | Min | Max |\n")
        f.write("|---------|------|-----|--------|-----|-----|\n")
        
        for _, row in df.iterrows():
            f.write(
                f"| {row['dataset']:<15} | "
                f"{row['mse_mean']:.6f} | "
                f"{row['mse_std']:.6f} | "
                f"{row['mse_median']:.6f} | "
                f"{row['mse_min']:.6f} | "
                f"{row['mse_max']:.6f} |\n"
            )
        
        f.write("\n### SSIM (Structural Similarity Index Measure)\n\n")
        f.write("| Dataset | Mean | Std | Median | Min | Max |\n")
        f.write("|---------|------|-----|--------|-----|-----|\n")
        
        for _, row in df.iterrows():
            f.write(
                f"| {row['dataset']:<15} | "
                f"{row['ssim_mean']:.4f} | "
                f"{row['ssim_std']:.4f} | "
                f"{row['ssim_median']:.4f} | "
                f"{row['ssim_min']:.4f} | "
                f"{row['ssim_max']:.4f} |\n"
            )
    
    logging.info(f"Formatted table saved to: {md_path}")


def compare_with_paper_results(
    df: pd.DataFrame,
    dataset_name: str = "No_Mixed"
) -> None:
    """
    Compare results with paper's reported values for reference.
    
    The paper reports for Dataset III (before correction):
    - SSIM = 0.4184
    - MSE = 0.1088
    
    Args:
        df: DataFrame with aggregated metrics.
        dataset_name: Name of dataset to compare (default: No_Mixed).
    """
    if dataset_name not in df['dataset'].values:
        logging.warning(f"Dataset {dataset_name} not found in results")
        return
    
    row = df[df['dataset'] == dataset_name].iloc[0]
    
    # Paper's reported values (before correction)
    paper_ssim = 0.4184
    paper_mse = 0.1088
    
    logging.info("\n" + "="*70)
    logging.info("COMPARISON WITH PAPER RESULTS")
    logging.info("="*70)
    logging.info(f"Dataset: {dataset_name}")
    logging.info(f"\nPaper (before correction):")
    logging.info(f"  SSIM: {paper_ssim:.4f}")
    logging.info(f"  MSE:  {paper_mse:.4f}")
    logging.info(f"\nOur results (reconstruction only, no correction):")
    logging.info(f"  SSIM: {row['ssim_mean']:.4f} ± {row['ssim_std']:.4f}")
    logging.info(f"  MSE:  {row['mse_mean']:.6f} ± {row['mse_std']:.6f}")
    logging.info(f"\nNote: Direct comparison may not be meaningful due to:")
    logging.info(f"  - Different preprocessing")
    logging.info(f"  - Different train/test splits")
    logging.info(f"  - Different model initialization")
    logging.info(f"  - This is reconstruction evaluation only (no correction stage)")
    logging.info("="*70 + "\n")
