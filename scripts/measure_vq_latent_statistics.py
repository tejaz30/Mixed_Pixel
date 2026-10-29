"""
Measure VQ latent statistics from training data.

Computes mean, std, min, max, and per-channel statistics
for proper latent scaling in LDM Stage 2.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
from tqdm import tqdm
import yaml
from data.thermal_dataset import ThermalImageDataset, train_val_test_split
from data.transforms import get_val_transforms
from models.ldm.first_stage.vq.model import VQAutoencoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_vq_model(checkpoint_path, config_path, device):
    """Load trained VQ model."""
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create model
    model_config = config['model']
    model = VQAutoencoder(
        in_channels=model_config['in_channels'],
        latent_channels=model_config['latent_channels'],
        base_channels=model_config['base_channels'],
        channel_multipliers=model_config['channel_multipliers'],
        num_res_blocks=model_config['num_res_blocks'],
        num_embeddings=model_config['num_embeddings'],
        commitment_cost=model_config['commitment_cost']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded VQ model from {checkpoint_path}")
    return model


def compute_latent_statistics(
    model: VQAutoencoder,
    dataset: ThermalImageDataset,
    device: torch.device,
    batch_size: int = 8,
    num_samples: int = None
):
    """
    Compute latent statistics from training data.
    
    Args:
        model: Trained VQ model
        dataset: Training dataset
        device: Computation device
        batch_size: Batch size for processing
        num_samples: Number of samples to process (None = all)
    
    Returns:
        Dictionary with statistics
    """
    from torch.utils.data import DataLoader, Subset
    
    # Limit samples if requested
    if num_samples is not None and num_samples < len(dataset):
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataset = Subset(dataset, indices)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    
    # Collect latents
    all_latents_quantized = []
    all_latents_continuous = []
    
    logger.info(f"Processing {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding"):
            if isinstance(batch, dict):
                images = batch['image'].to(device)
            else:
                images, _ = batch
                images = images.to(device)
            
            # Encode
            z_quantized, z_continuous, _ = model.encode(images)
            
            all_latents_quantized.append(z_quantized.cpu())
            all_latents_continuous.append(z_continuous.cpu())
    
    # Concatenate
    z_quant = torch.cat(all_latents_quantized, dim=0)
    z_cont = torch.cat(all_latents_continuous, dim=0)
    
    logger.info(f"Collected latents: quantized={z_quant.shape}, continuous={z_cont.shape}")
    
    # Compute statistics
    stats = {
        'quantized': {
            'shape': list(z_quant.shape),
            'dtype': str(z_quant.dtype),
            'global_mean': float(z_quant.mean()),
            'global_std': float(z_quant.std()),
            'global_min': float(z_quant.min()),
            'global_max': float(z_quant.max()),
            'per_channel_mean': z_quant.mean(dim=(0, 2, 3)).tolist(),
            'per_channel_std': z_quant.std(dim=(0, 2, 3)).tolist(),
            'per_channel_min': z_quant.min(dim=0)[0].flatten().min(dim=0)[0].tolist(),
            'per_channel_max': z_quant.max(dim=0)[0].flatten().max(dim=0)[0].tolist(),
        },
        'continuous': {
            'shape': list(z_cont.shape),
            'dtype': str(z_cont.dtype),
            'global_mean': float(z_cont.mean()),
            'global_std': float(z_cont.std()),
            'global_min': float(z_cont.min()),
            'global_max': float(z_cont.max()),
            'per_channel_mean': z_cont.mean(dim=(0, 2, 3)).tolist(),
            'per_channel_std': z_cont.std(dim=(0, 2, 3)).tolist(),
            'per_channel_min': z_cont.min(dim=0)[0].flatten().min(dim=0)[0].tolist(),
            'per_channel_max': z_cont.max(dim=0)[0].flatten().max(dim=0)[0].tolist(),
        },
        'num_samples': len(dataset)
    }
    
    return stats, z_quant, z_cont


def main():
    # Paths
    checkpoint_path = Path('outputs/experiment1/vq_nomixed/best_model (1).pth')  # VQ checkpoint
    config_path = Path('configs/experiment1/vq_autoencoder_kaggle_2gpu.yaml')
    dataset_root = Path('src/datasets/processed/No_Mixed/No_mixed')
    output_path = Path('outputs/vq_latent_statistics.yaml')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_vq_model(checkpoint_path, config_path, device)
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_root}")
    
    # Use same transforms as training (no normalization, only ToTensor)
    transform = get_val_transforms(normalize=False)
    
    dataset = ThermalImageDataset(
        root_dir=dataset_root,
        load_mode='rgb',
        return_path=False,
        transform=transform
    )
    
    # Split to get training set only
    logger.info("Splitting dataset (80/20)...")
    train_ds, val_ds, _ = train_val_test_split(
        dataset,
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0,
        random_seed=42
    )
    
    logger.info(f"Training set size: {len(train_ds)}")
    logger.info(f"Validation set size: {len(val_ds)}")
    
    # Compute statistics on training set
    stats, z_quant, z_cont = compute_latent_statistics(
        model,
        train_ds,
        device,
        batch_size=8,
        num_samples=None  # Use all training samples
    )
    
    # Print statistics
    print("\n" + "="*60)
    print("VQ LATENT STATISTICS (Training Set)")
    print("="*60)
    
    print("\nQuantized Latents:")
    print(f"  Shape: {stats['quantized']['shape']}")
    print(f"  Global mean: {stats['quantized']['global_mean']:.6f}")
    print(f"  Global std:  {stats['quantized']['global_std']:.6f}")
    print(f"  Global range: [{stats['quantized']['global_min']:.6f}, {stats['quantized']['global_max']:.6f}]")
    print(f"  Per-channel mean: {[f'{x:.4f}' for x in stats['quantized']['per_channel_mean']]}")
    print(f"  Per-channel std:  {[f'{x:.4f}' for x in stats['quantized']['per_channel_std']]}")
    
    print("\nContinuous (Pre-Quant) Latents:")
    print(f"  Shape: {stats['continuous']['shape']}")
    print(f"  Global mean: {stats['continuous']['global_mean']:.6f}")
    print(f"  Global std:  {stats['continuous']['global_std']:.6f}")
    print(f"  Global range: [{stats['continuous']['global_min']:.6f}, {stats['continuous']['global_max']:.6f}]")
    print(f"  Per-channel mean: {[f'{x:.4f}' for x in stats['continuous']['per_channel_mean']]}")
    print(f"  Per-channel std:  {[f'{x:.4f}' for x in stats['continuous']['per_channel_std']]}")
    
    print("\nRecommended Scaling:")
    quant_std = stats['quantized']['global_std']
    print(f"  For quantized latents: z_scaled = z / {quant_std:.4f}")
    print(f"  Before decoding: z = z_scaled * {quant_std:.4f}")
    
    # Save statistics
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
    
    logger.info(f"\nStatistics saved to {output_path}")
    
    # Also save sample latents for inspection
    sample_save_path = output_path.parent / 'vq_latent_samples.npz'
    np.savez(
        sample_save_path,
        quantized=z_quant[:100].numpy(),  # Save first 100 samples
        continuous=z_cont[:100].numpy()
    )
    logger.info(f"Sample latents saved to {sample_save_path}")


if __name__ == '__main__':
    main()
