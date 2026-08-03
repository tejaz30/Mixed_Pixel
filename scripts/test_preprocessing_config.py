"""
Test script to verify configurable preprocessing pipeline.

This script tests different preprocessing configurations without processing
the entire dataset.
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing import (
    ImageLoader,
    PreprocessingPipeline,
    ConfigurableImageProcessor,
    load_config
)


def test_configuration(config_dict, test_image_path):
    """
    Test a preprocessing configuration on a single image.
    
    Args:
        config_dict: Preprocessing configuration dictionary.
        test_image_path: Path to test image.
    """
    print("\n" + "="*70)
    print("Testing Preprocessing Configuration")
    print("="*70)
    
    # Initialize components
    loader = ImageLoader(
        valid_extensions=['.jpg', '.jpeg', '.png', '.tif', '.tiff'],
        load_mode='unchanged',
        skip_corrupted=True
    )
    pipeline = PreprocessingPipeline(config_dict)
    
    # Show enabled operations
    enabled_ops = pipeline.get_enabled_operations()
    print(f"\nEnabled operations: {', '.join(enabled_ops) if enabled_ops else 'None'}")
    
    # Load test image
    print(f"\nLoading test image: {test_image_path}")
    result = loader.load_image(test_image_path)
    
    if result is None:
        print("ERROR: Failed to load test image")
        return False
    
    image, load_metadata = result
    print(f"Original size: {load_metadata['width']}x{load_metadata['height']}")
    print(f"Original dtype: {image.dtype}")
    print(f"Original shape: {image.shape}")
    
    # Process through pipeline
    try:
        processed_image, metadata = pipeline.process(image, load_metadata.copy())
        
        print(f"\nProcessed size: {processed_image.shape[1]}x{processed_image.shape[0]}")
        print(f"Processed dtype: {processed_image.dtype}")
        print(f"Processed shape: {processed_image.shape}")
        print(f"Value range: [{processed_image.min():.2f}, {processed_image.max():.2f}]")
        
        # Show metadata
        print("\nMetadata:")
        for key, value in metadata.items():
            if key.startswith('operations'):
                print(f"  {key}: {value}")
        
        print("\n✓ Configuration test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Configuration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run configuration tests."""
    
    # Find a test image
    test_images = [
        Path("datasets/deduplicated/Okra/Okra_Dataset/adequate_matured_Okra"),
        Path("datasets/deduplicated/Chilli_Leaves/Bacterial Spot"),
        Path("datasets/deduplicated/No_Mixed/No_mixed"),
    ]
    
    test_image_path = None
    for img_dir in test_images:
        if img_dir.exists():
            images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg"))
            if images:
                test_image_path = images[0]
                break
    
    if test_image_path is None:
        print("ERROR: No test images found")
        print("Expected directories:")
        for d in test_images:
            print(f"  - {d}")
        return 1
    
    print("="*70)
    print("PREPROCESSING CONFIGURATION TESTER")
    print("="*70)
    print(f"Test image: {test_image_path}")
    
    # Test configurations
    configs_to_test = [
        {
            'name': 'Paper Default (Resize only)',
            'config': {
                'resize': {
                    'enabled': True,
                    'width': 640,
                    'height': 480,
                    'preserve_aspect_ratio': False,
                    'interpolation': {'downsample': 'area', 'upsample': 'linear'}
                },
                'grayscale': {'enabled': False},
                'normalization': {'enabled': False},
                'histogram_equalization': {'enabled': False},
                'clahe': {'enabled': False},
                'padding': {'enabled': False}
            }
        },
        {
            'name': 'Resize + Grayscale',
            'config': {
                'resize': {
                    'enabled': True,
                    'width': 640,
                    'height': 480,
                    'preserve_aspect_ratio': False,
                    'interpolation': {'downsample': 'area', 'upsample': 'linear'}
                },
                'grayscale': {'enabled': True},
                'normalization': {'enabled': False},
                'histogram_equalization': {'enabled': False},
                'clahe': {'enabled': False},
                'padding': {'enabled': False}
            }
        },
        {
            'name': 'Resize + CLAHE + MinMax Normalization',
            'config': {
                'resize': {
                    'enabled': True,
                    'width': 640,
                    'height': 480,
                    'preserve_aspect_ratio': False,
                    'interpolation': {'downsample': 'area', 'upsample': 'linear'}
                },
                'grayscale': {'enabled': False},
                'normalization': {
                    'enabled': True,
                    'method': 'minmax',
                    'min_value': 0.0,
                    'max_value': 1.0,
                    'per_image': True
                },
                'histogram_equalization': {'enabled': False},
                'clahe': {
                    'enabled': True,
                    'clip_limit': 2.0,
                    'tile_grid_size': [8, 8]
                },
                'padding': {'enabled': False}
            }
        },
        {
            'name': 'Resize + Z-score Normalization',
            'config': {
                'resize': {
                    'enabled': True,
                    'width': 640,
                    'height': 480,
                    'preserve_aspect_ratio': False,
                    'interpolation': {'downsample': 'area', 'upsample': 'linear'}
                },
                'grayscale': {'enabled': False},
                'normalization': {
                    'enabled': True,
                    'method': 'zscore',
                    'per_image': True
                },
                'histogram_equalization': {'enabled': False},
                'clahe': {'enabled': False},
                'padding': {'enabled': False}
            }
        }
    ]
    
    # Run tests
    results = []
    for i, test_case in enumerate(configs_to_test, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(configs_to_test)}: {test_case['name']}")
        print(f"{'='*70}")
        
        success = test_configuration(test_case['config'], test_image_path)
        results.append((test_case['name'], success))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(success for _, success in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
