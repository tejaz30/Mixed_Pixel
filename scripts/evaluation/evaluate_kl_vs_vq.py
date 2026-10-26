"""
Main Evaluation Script: KL vs VQ Autoencoder Comparison

Orchestrates the complete evaluation pipeline comparing trained KL and VQ
autoencoders on identical validation data (seed=42) for LDM Stage 2 selection.

Usage:
    python scripts/evaluation/evaluate_kl_vs_vq.py --sanity  # 20 images
    python scripts/evaluation/evaluate_kl_vs_vq.py           # Full evaluation

Environment:
    Designed for Kaggle GPU execution
    Checkpoints loaded from Kaggle Datasets
    Results saved to /kaggle/working/results/experiment02/
"""

import sys
from pathlib import Path
import argparse
import logging
import json
from datetime import datetime

# Setup logging FIRST before any imports that might fail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path - handle both local and Kaggle environments
project_root = Path(__file__).resolve().parents[2]
src_path = project_root / 'src'

# Add src to path if not already there
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml
import numpy as np

# Project imports - use direct module loading for Kaggle compatibility
import importlib.util

def import_from_path(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

logger.info("Loading project modules...")

# Import each module directly (Kaggle-compatible)
# Use actual file paths from project structure
thermal_dataset_mod = import_from_path('thermal_dataset', src_path / 'data' / 'thermal_dataset.py')
ThermalImageDataset = thermal_dataset_mod.ThermalImageDataset

kl_autoencoder_mod = import_from_path('kl_autoencoder', src_path / 'models' / 'ldm' / 'first_stage' / 'kl' / 'model.py')
KLAutoencoder = kl_autoencoder_mod.KLAutoencoder

vq_autoencoder_mod = import_from_path('vq_autoencoder', src_path / 'models' / 'ldm' / 'first_stage' / 'vq' / 'model.py')
VQAutoencoder = vq_autoencoder_mod.VQAutoencoder

recon_metrics_mod = import_from_path('reconstruction_metrics', src_path / 'evaluation' / 'reconstruction_metrics.py')
ReconstructionMetrics = recon_metrics_mod.ReconstructionMetrics

boundary_metrics_mod = import_from_path('boundary_metrics', src_path / 'evaluation' / 'boundary_metrics.py')
BoundaryMetrics = boundary_metrics_mod.BoundaryMetrics

kl_diag_mod = import_from_path('kl_diagnostics', src_path / 'evaluation' / 'kl_diagnostics.py')
KLDiagnostics = kl_diag_mod.KLDiagnostics

vq_diag_mod = import_from_path('vq_diagnostics', src_path / 'evaluation' / 'vq_diagnostics.py')
VQDiagnostics = vq_diag_mod.VQDiagnostics

latent_dist_mod = import_from_path('latent_distribution', src_path / 'evaluation' / 'latent_distribution.py')
LatentDistributionAnalyzer = latent_dist_mod.LatentDistributionAnalyzer
compare_distributions = latent_dist_mod.compare_distributions

compare_mod = import_from_path('compare_kl_vq', src_path / 'evaluation' / 'compare_kl_vq.py')
KLVQComparator = compare_mod.KLVQComparator
aggregate_results_for_comparison = compare_mod.aggregate_results_for_comparison

# Import transforms (needed for preprocessing images to match training)
transforms_mod = import_from_path('transforms', src_path / 'data' / 'transforms.py')
get_val_transforms = transforms_mod.get_val_transforms

logger.info("✓ All modules loaded successfully")


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device() -> torch.device:
    """Setup computation device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    else:
        device = torch.device('cpu')
        logger.warning("CUDA not available, using CPU")
    return device


def load_kl_model(checkpoint_path: Path, config: dict, device: torch.device) -> KLAutoencoder:
    """
    Load trained KL autoencoder from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Model configuration
        device: Device for model
        
    Returns:
        Loaded KL model in eval mode
    """
    logger.info(f"Loading KL model from {checkpoint_path}")
    
    # Create model with actual signature from model.py
    model = KLAutoencoder(
        in_channels=config['model']['in_channels'],
        latent_channels=config['model']['latent_channels'],
        base_channels=config['model']['base_channels'],
        channel_multipliers=tuple(config['model'].get('channel_multipliers', [1, 2, 4, 4])),
        num_res_blocks=config['model'].get('num_res_blocks', 2)
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"KL model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    return model


def load_vq_model(checkpoint_path: Path, config: dict, device: torch.device) -> VQAutoencoder:
    """
    Load trained VQ autoencoder from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Model configuration
        device: Device for model
        
    Returns:
        Loaded VQ model in eval mode
    """
    logger.info(f"Loading VQ model from {checkpoint_path}")
    
    # Create model with actual signature from model.py
    model = VQAutoencoder(
        in_channels=config['model']['in_channels'],
        latent_channels=config['model']['latent_channels'],
        base_channels=config['model']['base_channels'],
        channel_multipliers=tuple(config['model'].get('channel_multipliers', [1, 2, 4, 4])),
        num_res_blocks=config['model'].get('num_res_blocks', 2),
        num_embeddings=config['model'].get('num_embeddings', 1024),
        commitment_cost=config['model'].get('commitment_cost', 0.25)
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"VQ model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    return model


def create_validation_loader(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    sanity_check: bool = False
) -> DataLoader:
    """
    Create validation data loader with fixed seed=42.
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        sanity_check: If True, use only 20 samples
        
    Returns:
        Validation DataLoader
    """
    logger.info("Creating validation dataset")
    
    # Create transforms matching training (normalize=False means only ToTensor applied)
    # This converts uint8 -> float32 and normalizes to [0, 1]
    val_transform = get_val_transforms(
        image_size=(480, 640),
        normalize=False  # Matches training: only ToTensor, no mean/std normalization
    )
    
    # Create dataset with actual signature: root_dir not data_dir
    dataset = ThermalImageDataset(
        root_dir=data_dir,  # Correct parameter name
        load_mode='rgb',
        return_path=False,
        transform=val_transform  # Apply transforms
    )
    
    # For validation, we need to manually split since the dataset loads all images
    # Use seed=42 for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create train/val split indices
    total_size = len(dataset)
    indices = list(range(total_size))
    np.random.shuffle(indices)
    
    # 80/20 split
    split = int(0.8 * total_size)
    val_indices = indices[split:]
    
    # Sanity check: subsample
    if sanity_check:
        logger.info("SANITY CHECK MODE: Using only 20 validation samples")
        val_indices = val_indices[:min(20, len(val_indices))]
    
    logger.info(f"Validation set size: {len(val_indices)}")
    
    # Create subset
    val_dataset = Subset(dataset, val_indices)
    
    # Create dataloader
    loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Fixed order for reproducibility
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


def evaluate_kl_autoencoder(
    model: KLAutoencoder,
    dataloader: DataLoader,
    device: torch.device,
    result_dir: Path
) -> dict:
    """
    Evaluate KL autoencoder on validation set.
    
    Args:
        model: Trained KL model
        dataloader: Validation data loader
        device: Computation device
        result_dir: Directory to save results
        
    Returns:
        Dictionary with all KL evaluation results
    """
    logger.info("="*60)
    logger.info("Evaluating KL Autoencoder")
    logger.info("="*60)
    
    # Initialize diagnostic modules
    recon_metrics = ReconstructionMetrics(device=device)
    boundary_metrics = BoundaryMetrics(device=device)
    kl_diagnostics = KLDiagnostics(
        model=model,
        device=device,
        latent_channels=model.latent_channels
    )
    latent_analyzer = LatentDistributionAnalyzer(device=device)
    
    # Process validation set
    all_metrics = []
    kl_latents = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            
            # Forward pass - KL model returns tuple: (reconstruction, mean, logvar)
            reconstruction, mean, logvar = model(images, sample=False)  # Use mean for eval
            
            # Reconstruction metrics
            batch_recon = recon_metrics.compute_batch_metrics(images, reconstruction)
            
            # Boundary metrics
            batch_boundary = boundary_metrics.compute_batch_metrics(images, reconstruction)
            
            # KL diagnostics
            batch_kl_diag = kl_diagnostics.process_batch(images)
            
            # Latent distribution (use mean for analysis)
            batch_latent = latent_analyzer.assess_ldm_readiness(mean)
            kl_latents.append(mean)
            
            # Aggregate
            all_metrics.append({
                **batch_recon,
                **batch_boundary,
                'kl_divergence': batch_kl_diag['kl_divergence']['mean']
            })
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # Aggregate results
    logger.info("Aggregating KL results...")
    
    results = {
        'reconstruction_metrics': recon_metrics.aggregate_metrics(),
        'boundary_metrics': boundary_metrics.aggregate_metrics(),
        'kl_diagnostics': kl_diagnostics.compute_global_statistics(),
        'latent_distribution': latent_analyzer.compute_global_statistics(),
        'ldm_readiness': latent_analyzer.assess_ldm_readiness(
            torch.cat(kl_latents, dim=0)
        ),
        'per_sample_metrics': all_metrics
    }
    
    # Save diagnostics
    kl_result_dir = result_dir / 'kl'
    kl_result_dir.mkdir(parents=True, exist_ok=True)
    
    kl_diagnostics.save_diagnostics(kl_result_dir)
    latent_analyzer.save_distribution_data(kl_result_dir)
    
    with open(kl_result_dir / 'kl_evaluation_results.json', 'w') as f:
        # Remove per-sample for summary JSON
        summary = {k: v for k, v in results.items() if k != 'per_sample_metrics'}
        json.dump(summary, f, indent=2)
    
    logger.info(f"KL evaluation complete. Results saved to {kl_result_dir}")
    
    return results


def evaluate_vq_autoencoder(
    model: VQAutoencoder,
    dataloader: DataLoader,
    device: torch.device,
    result_dir: Path,
    codebook_size: int
) -> dict:
    """
    Evaluate VQ autoencoder on validation set.
    
    Args:
        model: Trained VQ model
        dataloader: Validation data loader
        device: Computation device
        result_dir: Directory to save results
        codebook_size: Size of VQ codebook
        
    Returns:
        Dictionary with all VQ evaluation results
    """
    logger.info("="*60)
    logger.info("Evaluating VQ Autoencoder")
    logger.info("="*60)
    
    # Initialize diagnostic modules
    recon_metrics = ReconstructionMetrics(device=device)
    boundary_metrics = BoundaryMetrics(device=device)
    vq_diagnostics = VQDiagnostics(
        model=model,
        device=device,
        codebook_size=codebook_size,
        latent_channels=model.latent_channels
    )
    latent_analyzer = LatentDistributionAnalyzer(device=device)
    
    # Process validation set
    all_metrics = []
    vq_latents = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            
            # Forward pass - VQ model returns tuple: (reconstruction, loss_dict)
            reconstruction, loss_dict = model(images)
            
            # Extract latents from loss_dict
            z_quantized = loss_dict['z_quantized']  # Post-quantization latent
            z_continuous = loss_dict['z_continuous']  # Pre-quantization latent
            
            # Reconstruction metrics
            batch_recon = recon_metrics.compute_batch_metrics(images, reconstruction)
            
            # Boundary metrics
            batch_boundary = boundary_metrics.compute_batch_metrics(images, reconstruction)
            
            # VQ diagnostics
            batch_vq_diag = vq_diagnostics.process_batch(images)
            
            # Latent distribution (use quantized latent for LDM readiness)
            batch_latent = latent_analyzer.assess_ldm_readiness(z_quantized)
            vq_latents.append(z_quantized)
            
            # Aggregate
            all_metrics.append({
                **batch_recon,
                **batch_boundary,
                'quant_mse': batch_vq_diag['quantization_error']['quant_mse']
            })
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # Aggregate results
    logger.info("Aggregating VQ results...")
    
    results = {
        'reconstruction_metrics': recon_metrics.aggregate_metrics(),
        'boundary_metrics': boundary_metrics.aggregate_metrics(),
        'vq_diagnostics': vq_diagnostics.compute_global_statistics(),
        'latent_distribution': latent_analyzer.compute_global_statistics(),
        'ldm_readiness': latent_analyzer.assess_ldm_readiness(
            torch.cat(vq_latents, dim=0)
        ),
        'per_sample_metrics': all_metrics
    }
    
    # Save diagnostics
    vq_result_dir = result_dir / 'vq'
    vq_result_dir.mkdir(parents=True, exist_ok=True)
    
    vq_diagnostics.save_diagnostics(vq_result_dir)
    latent_analyzer.save_distribution_data(vq_result_dir)
    
    with open(vq_result_dir / 'vq_evaluation_results.json', 'w') as f:
        # Remove per-sample for summary JSON
        summary = {k: v for k, v in results.items() if k != 'per_sample_metrics'}
        json.dump(summary, f, indent=2)
    
    logger.info(f"VQ evaluation complete. Results saved to {vq_result_dir}")
    
    return results


def compare_kl_vq(
    kl_results: dict,
    vq_results: dict,
    result_dir: Path
) -> dict:
    """
    Perform comprehensive comparison between KL and VQ results.
    
    Args:
        kl_results: KL evaluation results
        vq_results: VQ evaluation results
        result_dir: Directory to save comparison
        
    Returns:
        Dictionary with comparison results
    """
    logger.info("="*60)
    logger.info("Comparing KL vs VQ Autoencoders")
    logger.info("="*60)
    
    comparator = KLVQComparator(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Add paired comparisons
    kl_per_sample = kl_results['per_sample_metrics']
    vq_per_sample = vq_results['per_sample_metrics']
    
    for idx, (kl_metrics, vq_metrics) in enumerate(zip(kl_per_sample, vq_per_sample)):
        comparator.add_paired_comparison(
            kl_results=kl_metrics,
            vq_results=vq_metrics,
            image_id=f"sample_{idx:04d}"
        )
    
    # Reconstruction comparison
    reconstruction_comparison = comparator.compare_reconstruction_metrics()
    
    # Latent comparison
    latent_comparison = comparator.compare_latent_properties(
        kl_latent_stats=kl_results['ldm_readiness'],
        vq_latent_stats=vq_results['ldm_readiness']
    )
    
    # Summary recommendation
    summary = comparator.generate_summary_recommendation(
        reconstruction_comparison=reconstruction_comparison,
        latent_comparison=latent_comparison,
        kl_diagnostics=kl_results.get('kl_diagnostics'),
        vq_diagnostics=vq_results.get('vq_diagnostics')
    )
    
    summary['comparison_date'] = datetime.now().isoformat()
    
    # Save comparison report
    comparison_dir = result_dir / 'comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    comparator.save_comparison_report(
        save_path=comparison_dir / 'kl_vs_vq_comparison.json',
        reconstruction_comparison=reconstruction_comparison,
        latent_comparison=latent_comparison,
        summary=summary
    )
    
    logger.info("="*60)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*60)
    for finding in summary['key_findings']:
        logger.info(f"  - {finding}")
    
    logger.info("\nRecommendation:")
    for consideration in summary['recommendation']['considerations']:
        logger.info(f"  - {consideration}")
    logger.info(f"\nPreliminary suggestion: {summary['recommendation']['preliminary_suggestion']}")
    logger.info("="*60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate KL vs VQ Autoencoders')
    parser.add_argument(
        '--sanity',
        action='store_true',
        help='Run sanity check on 20 images only'
    )
    parser.add_argument(
        '--kl_checkpoint',
        type=str,
        default='/kaggle/input/kl-autoencoder-checkpoints/best_model.pth',
        help='Path to KL checkpoint'
    )
    parser.add_argument(
        '--vq_checkpoint',
        type=str,
        default='/kaggle/input/vq-autoencoder-checkpoints/best_model.pth',
        help='Path to VQ checkpoint'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/kaggle/input/datasets/tejabulusu/no-mixed/No_mixed',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--result_dir',
        type=str,
        default='/kaggle/working/results/experiment02',
        help='Directory to save results (will be uploaded as Kaggle Dataset)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for evaluation'
    )
    
    args = parser.parse_args()
    
    # Setup
    device = setup_device()
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("KL vs VQ Autoencoder Evaluation")
    logger.info(f"Mode: {'SANITY CHECK (20 images)' if args.sanity else 'FULL EVALUATION'}")
    logger.info("="*60)
    
    # Load configurations
    kl_config_path = project_root / 'configs' / 'experiment1' / 'kl_autoencoder_kaggle_2gpu.yaml'
    vq_config_path = project_root / 'configs' / 'experiment1' / 'vq_autoencoder_kaggle_2gpu.yaml'
    
    kl_config = load_config(kl_config_path)
    vq_config = load_config(vq_config_path)
    
    # Load models
    kl_model = load_kl_model(Path(args.kl_checkpoint), kl_config, device)
    vq_model = load_vq_model(Path(args.vq_checkpoint), vq_config, device)
    
    # Create validation loader
    val_loader = create_validation_loader(
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        num_workers=2,
        sanity_check=args.sanity
    )
    
    # Evaluate KL
    kl_results = evaluate_kl_autoencoder(
        model=kl_model,
        dataloader=val_loader,
        device=device,
        result_dir=result_dir
    )
    
    # Evaluate VQ
    vq_results = evaluate_vq_autoencoder(
        model=vq_model,
        dataloader=val_loader,
        device=device,
        result_dir=result_dir,
        codebook_size=vq_config['model'].get('num_embeddings', 1024)
    )
    
    # Compare
    comparison_summary = compare_kl_vq(
        kl_results=kl_results,
        vq_results=vq_results,
        result_dir=result_dir
    )
    
    logger.info("="*60)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"Results saved to: {result_dir}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
