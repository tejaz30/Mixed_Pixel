"""
Visual test script for Step 1: DRAEM dataset and augmentation.

Generates sample visualizations to verify:
- Original clean images
- Corrupted images with DRAEM augmentation
- Anomaly masks
- Overlay visualizations

Run this before proceeding to Step 2.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import matplotlib.pyplot as plt
import numpy as np

from src.datasets.diffad import DiffADDataset, DRAEMAugmenter


def visualize_draem_samples(dataset, num_samples=5, save_path=None):
    """
    Visualize DRAEM augmentation samples.
    
    Args:
        dataset: DiffADDataset instance
        num_samples: Number of samples to visualize
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes[np.newaxis, :]
    
    for i in range(num_samples):
        # Get sample
        sample = dataset[i]
        
        clean = sample['image'].cpu().numpy()
        corrupted = sample['augmented_image'].cpu().numpy()
        mask = sample['mask'].squeeze().cpu().numpy()
        
        # Handle both grayscale and RGB
        if clean.shape[0] == 1:
            # Grayscale
            clean = clean.squeeze()
            corrupted = corrupted.squeeze()
            cmap = 'gray'
        else:
            # RGB - transpose to HWC for matplotlib
            clean = clean.transpose(1, 2, 0)
            corrupted = corrupted.transpose(1, 2, 0)
            cmap = None
        
        # Original clean image
        axes[i, 0].imshow(clean, cmap=cmap)
        axes[i, 0].set_title(f'Sample {i}: Clean (Target)')
        axes[i, 0].axis('off')
        
        # Corrupted image
        axes[i, 1].imshow(corrupted, cmap=cmap)
        axes[i, 1].set_title('Corrupted (Condition)')
        axes[i, 1].axis('off')
        
        # Anomaly mask
        axes[i, 2].imshow(mask, cmap='Reds', vmin=0, vmax=1)
        axes[i, 2].set_title('Anomaly Mask')
        axes[i, 2].axis('off')
        
        # Overlay: clean + mask outline
        if clean.ndim == 2:
            # Convert grayscale to RGB for overlay
            overlay = np.stack([clean, clean, clean], axis=-1)
        else:
            overlay = clean.copy()
        
        # Add red outline where mask is 1
        mask_boundary = mask > 0.5
        overlay[mask_boundary, 0] = 1.0  # Red channel
        overlay[mask_boundary, 1] = 0.0  # Green channel
        overlay[mask_boundary, 2] = 0.0  # Blue channel
        
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay (Clean + Mask)')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def test_augmenter_directly(image_size=256, num_samples=3, save_path=None):
    """
    Test DRAEM augmenter directly without loading dataset.
    
    Args:
        image_size: Size of test images
        num_samples: Number of test samples
        save_path: Optional path to save the figure
    """
    augmenter = DRAEMAugmenter(
        texture_source_dir=None,  # Use random noise textures
        resize_shape=(image_size, image_size),
    )
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes[np.newaxis, :]
    
    for i in range(num_samples):
        # Create a test pattern (gradient)
        y, x = np.meshgrid(np.linspace(0, 1, image_size), 
                          np.linspace(0, 1, image_size))
        test_image = ((x + y) / 2).astype(np.float32)
        test_image = torch.from_numpy(test_image).unsqueeze(0)  # (1, H, W)
        
        # Apply augmentation
        clean, corrupted, mask = augmenter(test_image, return_mask=True)
        
        clean = clean.squeeze().cpu().numpy()
        corrupted = corrupted.squeeze().cpu().numpy()
        mask = mask.squeeze().cpu().numpy()
        
        # Visualize
        axes[i, 0].imshow(clean, cmap='gray')
        axes[i, 0].set_title(f'Test {i}: Clean')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(corrupted, cmap='gray')
        axes[i, 1].set_title('Corrupted')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(mask, cmap='Reds', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Mask (Area: {mask.sum() / mask.size:.2%})')
        axes[i, 2].axis('off')
        
        # Difference image
        diff = np.abs(corrupted - clean)
        axes[i, 3].imshow(diff, cmap='hot')
        axes[i, 3].set_title('Absolute Difference')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved augmenter test to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_dataset_stats(dataset):
    """Print statistics about the dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total samples: {len(dataset)}")
    
    # Sample a few to get stats
    num_check = min(20, len(dataset))
    masks_sizes = []
    
    for i in range(num_check):
        sample = dataset[i]
        mask = sample['mask']
        mask_area = (mask.sum() / mask.numel()).item()
        masks_sizes.append(mask_area)
    
    masks_sizes = np.array(masks_sizes)
    print(f"\nAnomalyarea statistics (from {num_check} samples):")
    print(f"  Mean: {masks_sizes.mean():.2%}")
    print(f"  Std:  {masks_sizes.std():.2%}")
    print(f"  Min:  {masks_sizes.min():.2%}")
    print(f"  Max:  {masks_sizes.max():.2%}")
    
    # Check image statistics
    sample = dataset[0]
    print(f"\nImage shapes:")
    print(f"  Clean: {sample['image'].shape}")
    print(f"  Corrupted: {sample['augmented_image'].shape}")
    print(f"  Mask: {sample['mask'].shape}")
    
    print(f"\nValue ranges:")
    print(f"  Clean: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"  Corrupted: [{sample['augmented_image'].min():.3f}, {sample['augmented_image'].max():.3f}]")
    print(f"  Mask: [{sample['mask'].min():.3f}, {sample['mask'].max():.3f}]")
    print(f"{'='*60}\n")


def main():
    """Main test function."""
    print("Step 1: DRAEM Dataset Visual Test")
    print("=" * 60)
    
    # Test 1: Direct augmenter test
    print("\n[Test 1] Testing DRAEM augmenter with synthetic patterns...")
    test_augmenter_directly(
        image_size=256,
        num_samples=3,
        save_path="results/diffad/step1_augmenter_test.png"
    )
    
    # Test 2: Load actual dataset
    print("\n[Test 2] Loading DiffAD dataset from No_Mixed...")
    dataset_path = Path("src/datasets/processed/No_Mixed/No_mixed")
    
    if not dataset_path.exists():
        print(f"ERROR: Dataset path not found: {dataset_path}")
        print("Please verify the path to your thermal images.")
        return
    
    # Create dataset
    dataset = DiffADDataset(
        image_dir=dataset_path,
        image_size=256,
        texture_source_dir=None,  # Use random noise textures
        clean_sample_ratio=0.0,  # All samples are corrupted for training
        return_path=False,
    )
    
    # Print dataset stats
    print_dataset_stats(dataset)
    
    # Visualize samples
    print("\n[Test 3] Visualizing DRAEM augmentation on real thermal images...")
    visualize_draem_samples(
        dataset=dataset,
        num_samples=5,
        save_path="results/diffad/step1_dataset_samples.png"
    )
    
    print("\n" + "=" * 60)
    print("Step 1 Visual Test Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - results/diffad/step1_augmenter_test.png")
    print("  - results/diffad/step1_dataset_samples.png")
    print("\nReview these visualizations to verify:")
    print("  1. Masks are plausible blob-like regions")
    print("  2. Corrupted images show texture/noise in masked regions")
    print("  3. Non-masked regions remain unchanged")
    print("  4. Image values are in valid ranges [0, 1]")
    print("\nIf visualizations look good, Step 1 is complete!")


if __name__ == "__main__":
    # Create output directory
    Path("results/diffad").mkdir(parents=True, exist_ok=True)
    main()
