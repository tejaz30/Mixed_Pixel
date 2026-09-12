"""
Visualization for Reconstruction Quality Evaluation

Creates visual comparisons, error maps, distributions, and boxplots
for reconstruction quality analysis.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
import seaborn as sns

from .evaluation_types import EvaluationResult


# Set style for consistent plotting
plt.style.use('default')
sns.set_palette("husl")


class ReconstructionVisualizer:
    """
    Visualizer for reconstruction quality evaluation.
    
    Creates:
    1. Original vs Reconstructed comparisons
    2. Absolute error maps
    3. MSE and SSIM distributions
    4. Boxplots across datasets
    """
    
    def __init__(
        self,
        output_dir: Path,
        dpi: int = 150,
        figsize: Tuple[int, int] = (15, 5)
    ):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save figures.
            dpi: Resolution for saved figures.
            figsize: Default figure size.
        """
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.figsize = figsize
        
        # Create output directories
        self.figures_dir = self.output_dir / 'figures'
        self.reconstructions_dir = self.output_dir / 'reconstructions'
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reconstructions_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Visualizer initialized. Figures: {self.figures_dir}")
    
    def plot_reconstruction_comparison(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        error_map: np.ndarray,
        title: str,
        mse: float,
        ssim: float,
        save_path: Optional[Path] = None
    ) -> Figure:
        """
        Plot original, reconstructed, and error map side by side.
        
        Args:
            original: Original image (H, W, C) in [0, 1].
            reconstructed: Reconstructed image (H, W, C) in [0, 1].
            error_map: Absolute error map (H, W, C) in [0, 1].
            title: Title for the figure.
            mse: MSE value to display.
            ssim: SSIM value to display.
            save_path: Optional path to save figure.
        
        Returns:
            Matplotlib Figure object.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(np.clip(original, 0, 1))
        axes[0].set_title('Original', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Reconstructed image
        axes[1].imshow(np.clip(reconstructed, 0, 1))
        axes[1].set_title('Reconstructed', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Error map (amplified for visibility)
        # Convert to grayscale for error visualization
        if error_map.ndim == 3:
            error_gray = np.mean(error_map, axis=2)
        else:
            error_gray = error_map
        
        im = axes[2].imshow(error_gray, cmap='hot', vmin=0, vmax=error_gray.max())
        axes[2].set_title('Absolute Error Map', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.set_label('Error Magnitude', fontsize=10)
        
        # Main title with metrics
        fig.suptitle(
            f'{title}\nMSE: {mse:.6f}  |  SSIM: {ssim:.4f}',
            fontsize=14,
            fontweight='bold',
            y=0.98
        )
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logging.debug(f"Saved reconstruction comparison: {save_path.name}")
        
        return fig
    
    def visualize_sample_reconstructions(
        self,
        results: List[EvaluationResult],
        original_images: List[np.ndarray],
        reconstructed_images: List[np.ndarray],
        dataset_name: str,
        num_samples: int = 5
    ) -> None:
        """
        Create visualization for sample reconstructions from a dataset.
        
        Args:
            results: List of evaluation results with error maps.
            original_images: List of original images.
            reconstructed_images: List of reconstructed images.
            dataset_name: Name of the dataset.
            num_samples: Number of samples to visualize.
        """
        num_samples = min(num_samples, len(results))
        
        logging.info(f"Creating {num_samples} sample visualizations for {dataset_name}")
        
        # Select diverse samples (best, worst, median MSE)
        mse_values = [r.mse for r in results]
        sorted_indices = np.argsort(mse_values)
        
        # Pick best, worst, and some median samples
        if num_samples >= 3:
            sample_indices = [
                sorted_indices[0],  # Best (lowest MSE)
                sorted_indices[len(sorted_indices) // 2],  # Median
                sorted_indices[-1]  # Worst (highest MSE)
            ]
            
            # Add more median samples if needed
            remaining = num_samples - 3
            step = len(sorted_indices) // (remaining + 1)
            for i in range(1, remaining + 1):
                idx = min(i * step, len(sorted_indices) - 1)
                sample_indices.append(sorted_indices[idx])
        else:
            sample_indices = sorted_indices[:num_samples].tolist()
        
        # Create visualizations
        for i, idx in enumerate(sample_indices):
            result = results[idx]
            
            if result.error_map is None:
                logging.warning(f"No error map for sample {i}, skipping")
                continue
            
            # Create comparison plot
            save_path = self.reconstructions_dir / f"{dataset_name}_sample_{i+1}.png"
            
            title = f"{dataset_name} - Sample {i+1}"
            
            self.plot_reconstruction_comparison(
                original=original_images[idx],
                reconstructed=reconstructed_images[idx],
                error_map=result.error_map,
                title=title,
                mse=result.mse,
                ssim=result.ssim,
                save_path=save_path
            )
            
            plt.close()
        
        logging.info(f"Sample visualizations saved to: {self.reconstructions_dir}")
    
    def plot_metric_distribution(
        self,
        results: List[EvaluationResult],
        metric: str,
        dataset_name: str,
        bins: int = 50
    ) -> Figure:
        """
        Plot distribution of a metric for a dataset.
        
        Args:
            results: List of evaluation results.
            metric: Metric name ('mse' or 'ssim').
            dataset_name: Name of the dataset.
            bins: Number of histogram bins.
        
        Returns:
            Matplotlib Figure object.
        """
        values = [getattr(r, metric) for r in results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram
        ax.hist(values, bins=bins, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Statistics
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values)
        
        # Add vertical lines for mean and median
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
        
        # Labels
        metric_label = metric.upper()
        ax.set_xlabel(f'{metric_label} Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(
            f'{metric_label} Distribution - {dataset_name}\n'
            f'Mean: {mean_val:.6f}, Std: {std_val:.6f}, N: {len(values)}',
            fontsize=14,
            fontweight='bold'
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def plot_all_distributions(
        self,
        results_by_dataset: Dict[str, List[EvaluationResult]]
    ) -> None:
        """
        Plot MSE and SSIM distributions for all datasets.
        
        Args:
            results_by_dataset: Dictionary mapping dataset names to results.
        """
        logging.info("Creating distribution plots for all datasets")
        
        for dataset_name, results in results_by_dataset.items():
            if not results:
                logging.warning(f"No results for {dataset_name}, skipping distribution plots")
                continue
            
            # MSE distribution
            fig = self.plot_metric_distribution(results, 'mse', dataset_name)
            save_path = self.figures_dir / f'{dataset_name}_mse_distribution.png'
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            
            # SSIM distribution
            fig = self.plot_metric_distribution(results, 'ssim', dataset_name)
            save_path = self.figures_dir / f'{dataset_name}_ssim_distribution.png'
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        
        logging.info(f"Distribution plots saved to: {self.figures_dir}")
    
    def plot_boxplots(
        self,
        results_by_dataset: Dict[str, List[EvaluationResult]]
    ) -> None:
        """
        Plot boxplots comparing metrics across datasets.
        
        Args:
            results_by_dataset: Dictionary mapping dataset names to results.
        """
        logging.info("Creating boxplots for cross-dataset comparison")
        
        # Prepare data
        mse_data = []
        ssim_data = []
        
        for dataset_name, results in results_by_dataset.items():
            for result in results:
                mse_data.append({'Dataset': dataset_name, 'MSE': result.mse})
                ssim_data.append({'Dataset': dataset_name, 'SSIM': result.ssim})
        
        mse_df = pd.DataFrame(mse_data)
        ssim_df = pd.DataFrame(ssim_data)
        
        # MSE boxplot
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=mse_df, x='Dataset', y='MSE', ax=ax)
        ax.set_title('MSE Comparison Across Datasets', fontsize=14, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax.set_ylabel('MSE', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = self.figures_dir / 'mse_boxplot.png'
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        # SSIM boxplot
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=ssim_df, x='Dataset', y='SSIM', ax=ax)
        ax.set_title('SSIM Comparison Across Datasets', fontsize=14, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax.set_ylabel('SSIM', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = self.figures_dir / 'ssim_boxplot.png'
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        logging.info(f"Boxplots saved to: {self.figures_dir}")
    
    def plot_metric_comparison(
        self,
        aggregated_df: pd.DataFrame
    ) -> None:
        """
        Plot bar charts comparing mean metrics across datasets.
        
        Args:
            aggregated_df: DataFrame with aggregated statistics.
        """
        logging.info("Creating metric comparison bar charts")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        datasets = aggregated_df['dataset'].values
        x_pos = np.arange(len(datasets))
        
        # MSE comparison
        mse_means = aggregated_df['mse_mean'].values
        mse_stds = aggregated_df['mse_std'].values
        
        ax1.bar(x_pos, mse_means, yerr=mse_stds, capsize=5, alpha=0.7, edgecolor='black')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(datasets, rotation=45, ha='right')
        ax1.set_ylabel('MSE', fontsize=12, fontweight='bold')
        ax1.set_title('Mean MSE Across Datasets', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # SSIM comparison
        ssim_means = aggregated_df['ssim_mean'].values
        ssim_stds = aggregated_df['ssim_std'].values
        
        ax2.bar(x_pos, ssim_means, yerr=ssim_stds, capsize=5, alpha=0.7, edgecolor='black')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(datasets, rotation=45, ha='right')
        ax2.set_ylabel('SSIM', fontsize=12, fontweight='bold')
        ax2.set_title('Mean SSIM Across Datasets', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        save_path = self.figures_dir / 'metrics_comparison.png'
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        logging.info(f"Comparison chart saved to: {save_path}")
    
    def create_summary_figure(
        self,
        aggregated_df: pd.DataFrame
    ) -> None:
        """
        Create a comprehensive summary figure with all key visualizations.
        
        Args:
            aggregated_df: DataFrame with aggregated statistics.
        """
        logging.info("Creating summary figure")
        
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        datasets = aggregated_df['dataset'].values
        x_pos = np.arange(len(datasets))
        
        # Top left: MSE comparison
        ax1 = fig.add_subplot(gs[0, 0])
        mse_means = aggregated_df['mse_mean'].values
        mse_stds = aggregated_df['mse_std'].values
        ax1.bar(x_pos, mse_means, yerr=mse_stds, capsize=5, alpha=0.7, edgecolor='black')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(datasets, rotation=45, ha='right')
        ax1.set_ylabel('MSE', fontsize=12, fontweight='bold')
        ax1.set_title('Mean MSE Across Datasets', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Top right: SSIM comparison
        ax2 = fig.add_subplot(gs[0, 1])
        ssim_means = aggregated_df['ssim_mean'].values
        ssim_stds = aggregated_df['ssim_std'].values
        ax2.bar(x_pos, ssim_means, yerr=ssim_stds, capsize=5, alpha=0.7, edgecolor='black')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(datasets, rotation=45, ha='right')
        ax2.set_ylabel('SSIM', fontsize=12, fontweight='bold')
        ax2.set_title('Mean SSIM Across Datasets', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Bottom left: Statistics table
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('tight')
        ax3.axis('off')
        
        table_data = []
        for _, row in aggregated_df.iterrows():
            table_data.append([
                row['dataset'],
                f"{int(row['num_images'])}",
                f"{row['mse_mean']:.6f}",
                f"{row['mse_std']:.6f}",
                f"{row['ssim_mean']:.4f}",
                f"{row['ssim_std']:.4f}"
            ])
        
        table = ax3.table(
            cellText=table_data,
            colLabels=['Dataset', 'N', 'Mean MSE', 'Std MSE', 'Mean SSIM', 'Std SSIM'],
            cellLoc='center',
            loc='center',
            colWidths=[0.2, 0.1, 0.15, 0.15, 0.15, 0.15]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(6):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        fig.suptitle(
            'CAE Reconstruction Quality Evaluation - Summary',
            fontsize=16,
            fontweight='bold',
            y=0.98
        )
        
        save_path = self.figures_dir / 'evaluation_summary.png'
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        logging.info(f"Summary figure saved to: {save_path}")
