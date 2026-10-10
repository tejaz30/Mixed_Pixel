"""
Create minimal synthetic dataset for smoke testing.
"""

import torch
from pathlib import Path
from PIL import Image
import numpy as np


def create_smoke_dataset(output_dir: Path, num_samples: int = 20):
    """
    Create a minimal synthetic thermal dataset for testing.
    
    Args:
        output_dir: Directory to create dataset in
        num_samples: Number of synthetic samples to create
    """
    # Create directory structure
    class_dir = output_dir / 'thermal' / 'test_class'
    class_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_samples} synthetic thermal images...")
    print(f"Output: {class_dir}")
    
    # Create synthetic thermal images
    for i in range(num_samples):
        # Create random thermal-like image (480x640)
        img_array = np.random.rand(480, 640, 3) * 255
        img_array = img_array.astype(np.uint8)
        
        # Convert to PIL Image
        img = Image.fromarray(img_array, mode='RGB')
        
        # Save
        img_path = class_dir / f'synthetic_{i:03d}.jpg'
        img.save(img_path, quality=95)
    
    print(f"Created {num_samples} images in {class_dir}")
    print(f"\nDataset structure:")
    print(f"{output_dir}/")
    print(f"  thermal/")
    print(f"    test_class/")
    print(f"      synthetic_000.jpg")
    print(f"      ...")
    print(f"      synthetic_{num_samples-1:03d}.jpg")
    
    return output_dir


if __name__ == '__main__':
    smoke_dataset = Path('temp_smoke_dataset')
    create_smoke_dataset(smoke_dataset, num_samples=20)
