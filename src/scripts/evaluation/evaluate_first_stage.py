"""
Evaluation script for first-stage autoencoders (KL and VQ).

Computes:
- Global reconstruction metrics (MSE, SSIM)
- Boundary-focused metrics
- Gradient preservation
- Latent diagnostics
- Qualitative visualizations

Usage:
    python scripts/evaluate_first_stage.py --config configs/experiment1/kl_autoencoder.yaml --checkpoint checkpoints/experiment1/kl/best_model.pth
    python scripts/evaluate_first_stage.py --config configs/experiment1/vq_autoencoder.yaml --checkpoint checkpoints/experiment1/vq/best_model.pth
"""

import sys
import argparse
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import cv2

from models.ldm.first_stage.kl import KLAutoencoder
from models.ldm.first_stage.vq import VQAutoencoder
from data import ThermalImageDataset, get_val_transforms
from evaluation.boundary_metrics import (
    compute_gradient_magnitude,
    create_boundary_mask,
    compute_boundary_reconstruction_metrics,
    compute_gradient_preservation
)


def load_config(config_path: Path) -> dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(config: dict, checkpoint_path: Path, device: torch.device):
    """Load model from checkpoint."""
    model_config = config['model']
    model_type = model_config['type']
    
    if model_type == 'kl_autoencoder':
        model = KLAutoencoder(
            in_channels=model_config['in_channels'],
            latent_channels=model_config['latent_channels'],
            base_channels=model_config['base_channels'],
            channel_multipliers=tuple(model_config['channel_multipliers']),
            num_res_blocks=model_config['num_res_blocks']
        )
    elif model_type == 'vq_autoencoder':
        model = VQAutoencoder(
            in_channels=model_config['in_channels'],
            latent_channels=model_config['latent_channels'],
            base_channels=model_config['base_channels'],
            channel_multipliers=tuple(model_config['channel_multipliers']),
            num_res_blocks=model_config['num_res_blocks'],
            num_embeddings=model_config['num_embeddings'],
            commitment_cost=model_config['commitment_cost']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def compute_ssim_batch(images1: torch.Tensor, images2: torch.Tensor) -> float:
    """Compute SSIM for a batch."""
    images1_np = images1.cpu().numpy()
    images2_np = images2.cpu().numpy()
    
    ssim_scores = []
    for i in range(images1_np.shape[0]):
        # Transpose to (H, W, C)
        img1 = np.transpose(images1_np[i], (1, 2, 0))
        img2 = np.transpose(images2_np[i], (1, 2, 0))
        
        # Compute SSIM
        score = ssim(img1, img2, multichannel=True, data_range=1.0, channel_axis=2)
        ssim_scores.append(score)
    
    return np.mean(ssim_scores)


def evaluate_global_metrics(
    model,
    dataloader: DataLoader,
    device: torch.device
) -> dict:
    """Evaluate global reconstruction metrics."""
    model.eval()
    
    all_mse = []
    all_mae = []
    all_ssim = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing global metrics"):
            images = batch['image'].to(device)
            
            # Reconstruct
            if isinstance(model, KLAutoencoder):
                reconstruction, _ = model(images, sample=False)
            elif isinstance(model, VQAutoencoder):
                reconstruction, _ = model(images)
            else:
                raise ValueError(f"Unknown model type")
            
            # MSE
            mse = F.mse_loss(reconstruction, images, reduction='none')
            mse = mse.view(mse.size(0), -1).mean(dim=1)
            all_mse.extend(mse.cpu().numpy().tolist())
            
            # MAE
            mae = F.l1_loss(reconstruction, images, reduction='none')
            mae = mae.view(mae.size(0), -1).mean(dim=1)
            all_mae.extend(mae.cpu().numpy().tolist())
            
            # SSIM
            ssim_batch = compute_ssim_batch(reconstruction, images)
            all_ssim.append(ssim_batch)
    
    return {
        'mse_mean': float(np.mean(all_mse)),
        'mse_std': float(np.std(all_mse)),
        'mse_median': float(np.median(all_mse)),
        'mae_mean': float(np.mean(all_mae)),
        'mae_std': float(np.std(all_mae)),
        'ssim_mean': float(np.mean(all_ssim)),
        'ssim_std': float(np.std(all_ssim))
    }


def evaluate_boundary_metrics(
    model,
    dataloader: DataLoader,
    device: torch.device,
    threshold_percentile: float = 75.0
) -> dict:
    """Evaluate boundary-focused metrics."""
    model.eval()
    
    boundary_metrics_accum = {
        'boundary_mse': [],
        'non_boundary_mse': [],
        'boundary_mae': [],
        'non_boundary_mae': [],
        'error_ratio': [],
        'boundary_fraction': [],
        'gradient_mse': [],
        'gradient_correlation': [],
        'gradient_similarity': []
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing boundary metrics"):
            images = batch['image'].to(device)
            
            # Reconstruct
            if isinstance(model, KLAutoencoder):
                reconstruction, _ = model(images, sample=False)
            elif isinstance(model, VQAutoencoder):
                reconstruction, _ = model(images)
            
            # Convert to numpy
            images_np = images.cpu().numpy()
            reconstruction_np = reconstruction.cpu().numpy()
            
            # Process each image in batch
            for i in range(images_np.shape[0]):
                orig_img = np.transpose(images_np[i], (1, 2, 0))
                recon_img = np.transpose(reconstruction_np[i], (1, 2, 0))
                
                # Compute gradient and boundary mask
                gradient_mag = compute_gradient_magnitude(orig_img)
                boundary_mask = create_boundary_mask(gradient_mag, threshold_percentile)
                
                # Boundary reconstruction metrics
                boundary_metrics = compute_boundary_reconstruction_metrics(
                    orig_img, recon_img, boundary_mask
                )
                
                # Gradient preservation
                gradient_metrics = compute_gradient_preservation(orig_img, recon_img)
                
                # Accumulate
                for key in boundary_metrics:
                    boundary_metrics_accum[key].append(boundary_metrics[key])
                for key in gradient_metrics:
                    boundary_metrics_accum[key].append(gradient_metrics[key])
    
    # Compute statistics
    results = {}
    for key, values in boundary_metrics_accum.items():
        results[f'{key}_mean'] = float(np.mean(values))
        results[f'{key}_std'] = float(np.std(values))
        results[f'{key}_median'] = float(np.median(values))
    
    return results


def visualize_reconstructions(
    model,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    num_samples: int = 16
):
    """Create visualization of reconstructions with error maps and boundary masks."""
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect samples
    samples_collected = 0
    all_originals = []
    all_reconstructions = []
    
    with torch.no_grad():
        for batch in dataloader:
            if samples_collected >= num_samples:
                break
            
            images = batch['image'].to(device)
            
            # Reconstruct
            if isinstance(model, KLAutoencoder):
                reconstruction, _ = model(images, sample=False)
            elif isinstance(model, VQAutoencoder):
                reconstruction, _ = model(images)
            
            batch_size = min(images.size(0), num_samples - samples_collected)
            all_originals.append(images[:batch_size].cpu())
            all_reconstructions.append(reconstruction[:batch_size].cpu())
            samples_collected += batch_size
    
    all_originals = torch.cat(all_originals, dim=0)
    all_reconstructions = torch.cat(all_reconstructions, dim=0)
    
    # Create visualizations for each sample
    for idx in range(num_samples):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        orig_np = all_originals[idx].numpy().transpose(1, 2, 0)
        recon_np = all_reconstructions[idx].numpy().transpose(1, 2, 0)
        
        # Original
        axes[0, 0].imshow(orig_np)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Reconstruction
        axes[0, 1].imshow(recon_np)
        axes[0, 1].set_title('Reconstruction')
        axes[0, 1].axis('off')
        
        # Error map
        error_map = np.abs(orig_np - recon_np)
        error_map_vis = np.mean(error_map, axis=2)
        im = axes[0, 2].imshow(error_map_vis, cmap='hot')
        axes[0, 2].set_title('Error Map (Abs Diff)')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2])
        
        # Gradient magnitude (original)
        grad_orig = compute_gradient_magnitude(orig_np)
        axes[1, 0].imshow(grad_orig, cmap='gray')
        axes[1, 0].set_title('Gradient (Original)')
        axes[1, 0].axis('off')
        
        # Boundary mask
        boundary_mask = create_boundary_mask(grad_orig, threshold_percentile=75.0)
        axes[1, 1].imshow(boundary_mask, cmap='gray')
        axes[1, 1].set_title('Boundary Mask (75th percentile)')
        axes[1, 1].axis('off')
        
        # Overlay boundary on error
        axes[1, 2].imshow(error_map_vis, cmap='hot', alpha=0.7)
        axes[1, 2].contour(boundary_mask, colors='cyan', linewidths=1)
        axes[1, 2].set_title('Error + Boundary Overlay')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'reconstruction_{idx:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {num_samples} visualization samples to {output_dir}")


def evaluate_latent_diagnostics(
    model,
    dataloader: DataLoader,
    device: torch.device
) -> dict:
    """Evaluate latent representation diagnostics."""
    model.eval()
    
    diagnostics = {}
    
    if isinstance(model, KLAutoencoder):
        # KL-specific diagnostics
        all_means = []
        all_logvars = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing latent diagnostics"):
                images = batch['image'].to(device)
                posterior = model.encode(images)
                
                all_means.append(posterior.mean.cpu())
                all_logvars.append(posterior.logvar.cpu())
        
        all_means = torch.cat(all_means, dim=0)
        all_logvars = torch.cat(all_logvars, dim=0)
        
        diagnostics['latent_mean_mean'] = float(all_means.mean())
        diagnostics['latent_mean_std'] = float(all_means.std())
        diagnostics['latent_logvar_mean'] = float(all_logvars.mean())
        diagnostics['latent_logvar_std'] = float(all_logvars.std())
        
    elif isinstance(model, VQAutoencoder):
        # VQ-specific diagnostics
        codebook_usage = model.get_codebook_usage()
        diagnostics.update(codebook_usage)
        
        # Collect perplexity during inference
        all_perplexities = []
        all_utilizations = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing latent diagnostics"):
                images = batch['image'].to(device)
                _, vq_outputs = model(images)
                
                if 'perplexity' in vq_outputs:
                    all_perplexities.append(vq_outputs['perplexity'].item())
                if 'codebook_utilization' in vq_outputs:
                    all_utilizations.append(vq_outputs['codebook_utilization'].item())
        
        if all_perplexities:
            diagnostics['perplexity_mean'] = float(np.mean(all_perplexities))
            diagnostics['perplexity_std'] = float(np.std(all_perplexities))
        
        if all_utilizations:
            diagnostics['utilization_mean'] = float(np.mean(all_utilizations))
    
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description="Evaluate first-stage autoencoder")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    # Load config
    config = load_config(Path(args.config))
    print("="*70)
    print(f"EVALUATION: {config['experiment']['name']}")
    print("="*70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load model
    print("Loading model...")
    model = load_model(config, Path(args.checkpoint), device)
    print(model)
    print()
    
    # Create dataloader
    print("Creating dataloader...")
    val_transform = get_val_transforms(normalize=False)
    dataset = ThermalImageDataset(
        root_dir=config['data']['dataset_root'],
        transform=val_transform,
        load_mode=config['data']['load_mode']
    )
    
    # Use validation split
    val_size = int(config['data']['val_ratio'] * len(dataset))
    train_size = len(dataset) - val_size
    _, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['data']['seed'])
    )
    
    dataloader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    print(f"Evaluation dataset size: {len(val_dataset)}\n")
    
    # Evaluate
    results = {}
    
    # Global metrics
    print("1. Computing global reconstruction metrics...")
    global_metrics = evaluate_global_metrics(model, dataloader, device)
    results['global_metrics'] = global_metrics
    print("Done.\n")
    
    # Boundary metrics
    if config['evaluation']['boundary_metrics']['enabled']:
        print("2. Computing boundary-focused metrics...")
        boundary_metrics = evaluate_boundary_metrics(
            model, dataloader, device,
            config['evaluation']['boundary_metrics']['threshold_percentile']
        )
        results['boundary_metrics'] = boundary_metrics
        print("Done.\n")
    
    # Latent diagnostics
    print("3. Computing latent diagnostics...")
    latent_diagnostics = evaluate_latent_diagnostics(model, dataloader, device)
    results['latent_diagnostics'] = latent_diagnostics
    print("Done.\n")
    
    # Visualizations
    if config['evaluation']['save_reconstructions']:
        print("4. Creating visualizations...")
        output_dir = Path(config['paths']['output_dir']) / 'visualizations'
        visualize_reconstructions(
            model, dataloader, device, output_dir,
            num_samples=config['evaluation']['num_vis_samples']
        )
        print("Done.\n")
    
    # Save results
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print("\nGlobal Metrics:")
    print(f"  MSE: {global_metrics['mse_mean']:.6f} ± {global_metrics['mse_std']:.6f}")
    print(f"  MAE: {global_metrics['mae_mean']:.6f} ± {global_metrics['mae_std']:.6f}")
    print(f"  SSIM: {global_metrics['ssim_mean']:.4f} ± {global_metrics['ssim_std']:.4f}")
    
    if 'boundary_metrics' in results:
        print("\nBoundary Metrics:")
        bm = boundary_metrics
        print(f"  Boundary MSE: {bm['boundary_mse_mean']:.6f}")
        print(f"  Non-boundary MSE: {bm['non_boundary_mse_mean']:.6f}")
        print(f"  Error Ratio: {bm['error_ratio_mean']:.4f}")
        print(f"  Gradient Preservation: {bm['gradient_similarity_mean']:.4f}")
    
    if 'latent_diagnostics' in results:
        print("\nLatent Diagnostics:")
        for key, value in latent_diagnostics.items():
            print(f"  {key}: {value}")
    
    print("="*70)


if __name__ == '__main__':
    main()
