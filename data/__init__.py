"""
Data loading and processing for thermal image mixed-pixel detection.

This package provides:
- ThermalImageDataset: PyTorch Dataset for thermal images
- ThermalImageDataModule: PyTorch Lightning DataModule
- Transform pipelines for training and inference
- Train/val/test splitting utilities
"""

from .thermal_dataset import (
    ThermalImageDataset,
    SubsetDataset,
    train_val_test_split,
    collate_fn
)

from .datamodule import (
    ThermalImageDataModule,
    create_simple_dataloaders
)

from .transforms import (
    ToTensor,
    Normalize,
    Resize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    Compose,
    get_default_transforms,
    get_train_transforms,
    get_val_transforms,
    get_thermal_transforms
)

__all__ = [
    # Dataset
    'ThermalImageDataset',
    'SubsetDataset',
    'train_val_test_split',
    'collate_fn',
    
    # DataModule
    'ThermalImageDataModule',
    'create_simple_dataloaders',
    
    # Transforms
    'ToTensor',
    'Normalize',
    'Resize',
    'RandomHorizontalFlip',
    'RandomVerticalFlip',
    'RandomRotation',
    'Compose',
    'get_default_transforms',
    'get_train_transforms',
    'get_val_transforms',
    'get_thermal_transforms',
]

__version__ = '1.0.0'
