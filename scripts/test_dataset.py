"""
Test script for thermal image dataset and datamodule.

Tests:
1. Dataset loading and discovery
2. Train/val/test splitting
3. DataLoader creation
4. Transforms
5. Batch iteration
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from data import (
    ThermalImageDataset,
    train_val_test_split,
    ThermalImageDataModule,
    create_simple_dataloaders,
    get_train_transforms,
    get_val_transforms
)


def test_dataset_loading():
    """Test basic dataset loading."""
    print("="*70)
    print("TEST 1: Dataset Loading")
    print("="*70)
    
    try:
        dataset = ThermalImageDataset(
            root_dir="datasets/processed",
            load_mode="rgb"
        )
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Number of classes: {len(dataset.classes)}")
        print(f"  Classes: {dataset.classes}")
        
        # Test class distribution
        class_dist = dataset.get_class_distribution()
        print(f"\n  Class distribution:")
        for class_name, count in sorted(class_dist.items()):
            print(f"    {class_name}: {count}")
        
        # Test dataset distribution
        dataset_dist = dataset.get_dataset_distribution()
        print(f"\n  Dataset distribution:")
        for ds_name, count in sorted(dataset_dist.items()):
            print(f"    {ds_name}: {count}")
        
        return dataset
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_sample_access(dataset):
    """Test accessing individual samples."""
    print("\n" + "="*70)
    print("TEST 2: Sample Access")
    print("="*70)
    
    if dataset is None:
        print("✗ Skipping (dataset not loaded)")
        return False
    
    try:
        # Get first sample
        sample = dataset[0]
        
        print(f"✓ Sample accessed successfully")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Image dtype: {sample['image'].dtype}")
        print(f"  Label: {sample['label']} ({dataset.idx_to_class[sample['label']]})")
        print(f"  Metadata keys: {list(sample['metadata'].keys())}")
        print(f"  Dataset: {sample['metadata']['dataset']}")
        print(f"  Class name: {sample['metadata']['class_name']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Sample access failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_splitting(dataset):
    """Test train/val/test splitting."""
    print("\n" + "="*70)
    print("TEST 3: Train/Val/Test Splitting")
    print("="*70)
    
    if dataset is None:
        print("✗ Skipping (dataset not loaded)")
        return None, None, None
    
    try:
        train_ds, val_ds, test_ds = train_val_test_split(
            dataset,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            stratify=True,
            random_seed=42
        )
        
        print(f"✓ Dataset split successfully")
        print(f"  Train: {len(train_ds)} samples")
        print(f"  Val: {len(val_ds)} samples")
        print(f"  Test: {len(test_ds)} samples")
        
        # Check class distributions
        print(f"\n  Train class distribution:")
        train_dist = train_ds.get_class_distribution()
        for class_name, count in sorted(train_dist.items()):
            print(f"    {class_name}: {count}")
        
        return train_ds, val_ds, test_ds
        
    except Exception as e:
        print(f"✗ Splitting failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_transforms():
    """Test transform pipelines."""
    print("\n" + "="*70)
    print("TEST 4: Transforms")
    print("="*70)
    
    try:
        # Create dataset with transforms
        train_transform = get_train_transforms(augment=True, normalize=True)
        val_transform = get_val_transforms(normalize=True)
        
        dataset = ThermalImageDataset(
            root_dir="datasets/processed",
            load_mode="rgb",
            transform=train_transform
        )
        
        sample = dataset[0]
        
        print(f"✓ Transforms applied successfully")
        print(f"  Image type: {type(sample['image'])}")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Image dtype: {sample['image'].dtype}")
        
        if isinstance(sample['image'], torch.Tensor):
            print(f"  Value range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Transform test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader():
    """Test DataLoader creation and iteration."""
    print("\n" + "="*70)
    print("TEST 5: DataLoader")
    print("="*70)
    
    try:
        # Create simple dataloaders
        train_transform = get_train_transforms(augment=False, normalize=True)
        
        train_loader, val_loader, test_loader = create_simple_dataloaders(
            data_dir="datasets/processed",
            batch_size=8,
            num_workers=0,  # Use 0 for testing
            transform=train_transform
        )
        
        print(f"✓ DataLoaders created successfully")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Test iteration
        batch = next(iter(train_loader))
        
        print(f"\n  Sample batch:")
        print(f"    Images shape: {batch['image'].shape}")
        print(f"    Labels shape: {batch['label'].shape}")
        print(f"    Metadata: {len(batch['metadata'])} items")
        
        return True
        
    except Exception as e:
        print(f"✗ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_datamodule():
    """Test PyTorch Lightning DataModule."""
    print("\n" + "="*70)
    print("TEST 6: DataModule (PyTorch Lightning)")
    print("="*70)
    
    try:
        train_transform = get_train_transforms(augment=True, normalize=True)
        val_transform = get_val_transforms(normalize=True)
        
        datamodule = ThermalImageDataModule(
            data_dir="datasets/processed",
            batch_size=8,
            num_workers=0,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=val_transform
        )
        
        # Setup
        datamodule.setup('fit')
        
        print(f"✓ DataModule setup successfully")
        
        # Get dataset info
        info = datamodule.get_dataset_info()
        print(f"\n  Dataset Info:")
        print(f"    Total samples: {info['total_samples']}")
        print(f"    Classes: {info['num_classes']}")
        print(f"    Train samples: {info['train_samples']}")
        print(f"    Val samples: {info['val_samples']}")
        print(f"    Test samples: {info['test_samples']}")
        
        # Test dataloaders
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        
        print(f"\n  Train batch:")
        print(f"    Images: {batch['image'].shape}")
        print(f"    Labels: {batch['label'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ DataModule test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("THERMAL IMAGE DATASET TESTS")
    print("="*70)
    print()
    
    results = []
    
    # Test 1: Dataset loading
    dataset = test_dataset_loading()
    results.append(('Dataset Loading', dataset is not None))
    
    # Test 2: Sample access
    success = test_sample_access(dataset)
    results.append(('Sample Access', success))
    
    # Test 3: Splitting
    train_ds, val_ds, test_ds = test_splitting(dataset)
    results.append(('Train/Val/Test Split', all([train_ds, val_ds, test_ds])))
    
    # Test 4: Transforms
    success = test_transforms()
    results.append(('Transforms', success))
    
    # Test 5: DataLoader
    success = test_dataloader()
    results.append(('DataLoader', success))
    
    # Test 6: DataModule
    success = test_datamodule()
    results.append(('DataModule', success))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
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
