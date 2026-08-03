"""
PyTorch Lightning DataModule for thermal image datasets.

Provides a high-level interface for data loading with automatic
train/val/test splits, transforms, and DataLoader creation.
"""

import logging
from pathlib import Path
from typing import Optional, Callable, Dict, List, Any, Union, Tuple
import torch
from torch.utils.data import DataLoader
try:
    import pytorch_lightning as pl
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    # Create a dummy base class if Lightning not available
    class LightningDataModule:
        pass
    pl = type('pl', (), {'LightningDataModule': LightningDataModule})

from .thermal_dataset import (
    ThermalImageDataset,
    SubsetDataset,
    train_val_test_split,
    collate_fn
)

logger = logging.getLogger("data.datamodule")


class ThermalImageDataModule(pl.LightningDataModule if LIGHTNING_AVAILABLE else object):
    """
    PyTorch Lightning DataModule for thermal image datasets.
    
    Handles:
    - Dataset loading and splitting
    - Transform pipeline
    - DataLoader creation
    - Multi-GPU support
    
    Compatible with all PyTorch Lightning trainers and future architectures.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        datasets: Optional[List[str]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify: bool = True,
        random_seed: int = 42,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_extensions: Optional[List[str]] = None,
        load_mode: str = "rgb",
        pin_memory: bool = True,
        drop_last: bool = False,
        persistent_workers: bool = False
    ):
        """
        Initialize DataModule.
        
        Args:
            data_dir: Root directory containing processed datasets.
            datasets: List of dataset names to include.
            batch_size: Batch size for DataLoaders.
            num_workers: Number of worker processes for data loading.
            train_ratio: Proportion of data for training.
            val_ratio: Proportion of data for validation.
            test_ratio: Proportion of data for testing.
            stratify: If True, maintain class distribution across splits.
            random_seed: Random seed for reproducibility.
            train_transform: Transform for training data.
            val_transform: Transform for validation data.
            test_transform: Transform for test data.
            target_transform: Transform for labels.
            image_extensions: Valid image file extensions.
            load_mode: How to load images ('rgb', 'grayscale', 'unchanged').
            pin_memory: Pin memory for faster GPU transfer.
            drop_last: Drop last incomplete batch.
            persistent_workers: Keep workers alive between epochs.
        """
        if LIGHTNING_AVAILABLE:
            super().__init__()
        
        self.data_dir = Path(data_dir)
        self.datasets = datasets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.stratify = stratify
        self.random_seed = random_seed
        self.image_extensions = image_extensions
        self.load_mode = load_mode
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers
        
        # Transforms
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.target_transform = target_transform
        
        # Datasets (will be initialized in setup())
        self.dataset_full = None
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        
        # Save hyperparameters if using Lightning
        if LIGHTNING_AVAILABLE:
            self.save_hyperparameters(ignore=['train_transform', 'val_transform', 
                                              'test_transform', 'target_transform'])
    
    def prepare_data(self):
        """
        Download or prepare data (called only once on single GPU).
        
        For thermal images, data is already preprocessed, so this is a no-op.
        """
        # Verify data directory exists
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        
        logger.info(f"Data directory verified: {self.data_dir}")
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for each stage (fit, validate, test, predict).
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict', or None).
        """
        # Create full dataset
        if self.dataset_full is None:
            logger.info("Loading full dataset...")
            self.dataset_full = ThermalImageDataset(
                root_dir=self.data_dir,
                datasets=self.datasets,
                transform=None,  # Transforms applied per split
                target_transform=self.target_transform,
                image_extensions=self.image_extensions,
                load_mode=self.load_mode,
                return_path=False
            )
            
            logger.info(f"Full dataset loaded: {len(self.dataset_full)} samples")
            logger.info(f"Classes: {self.dataset_full.classes}")
            
            # Log class distribution
            class_dist = self.dataset_full.get_class_distribution()
            logger.info("Class distribution:")
            for class_name, count in sorted(class_dist.items()):
                logger.info(f"  {class_name}: {count}")
        
        # Split dataset
        if stage == 'fit' or stage is None:
            if self.dataset_train is None or self.dataset_val is None:
                logger.info("Splitting dataset...")
                train_ds, val_ds, test_ds = train_val_test_split(
                    self.dataset_full,
                    train_ratio=self.train_ratio,
                    val_ratio=self.val_ratio,
                    test_ratio=self.test_ratio,
                    stratify=self.stratify,
                    random_seed=self.random_seed
                )
                
                # Apply transforms
                train_ds.dataset.transform = self.train_transform
                val_ds.dataset.transform = self.val_transform
                test_ds.dataset.transform = self.test_transform
                
                self.dataset_train = train_ds
                self.dataset_val = val_ds
                self.dataset_test = test_ds
                
                logger.info(f"Train set: {len(self.dataset_train)} samples")
                logger.info(f"Val set: {len(self.dataset_val)} samples")
                logger.info(f"Test set: {len(self.dataset_test)} samples")
        
        if stage == 'test' or stage is None:
            if self.dataset_test is None:
                # Split if not already done
                if self.dataset_train is None:
                    self.setup('fit')
        
        if stage == 'predict':
            # For prediction, use full dataset or create a predict dataset
            pass
    
    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle training data
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers and self.num_workers > 0
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,  # Don't shuffle validation data
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers and self.num_workers > 0
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,  # Don't shuffle test data
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers and self.num_workers > 0
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Create prediction DataLoader."""
        # Use full dataset for prediction
        predict_dataset = self.dataset_full
        predict_dataset.transform = self.test_transform
        
        return DataLoader(
            predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers and self.num_workers > 0
        )
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        if self.dataset_full is None:
            self.setup()
        return self.dataset_full.classes
    
    def get_class_to_idx(self) -> Dict[str, int]:
        """Get mapping from class names to indices."""
        if self.dataset_full is None:
            self.setup()
        return self.dataset_full.class_to_idx
    
    def get_idx_to_class(self) -> Dict[int, str]:
        """Get mapping from indices to class names."""
        if self.dataset_full is None:
            self.setup()
        return self.dataset_full.idx_to_class
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get comprehensive dataset information.
        
        Returns:
            Dictionary with dataset statistics and metadata.
        """
        if self.dataset_full is None:
            self.setup()
        
        info = {
            'total_samples': len(self.dataset_full),
            'num_classes': len(self.dataset_full.classes),
            'classes': self.dataset_full.classes,
            'class_to_idx': self.dataset_full.class_to_idx,
            'class_distribution': self.dataset_full.get_class_distribution(),
            'dataset_distribution': self.dataset_full.get_dataset_distribution(),
        }
        
        if self.dataset_train is not None:
            info['train_samples'] = len(self.dataset_train)
            info['train_class_distribution'] = self.dataset_train.get_class_distribution()
        
        if self.dataset_val is not None:
            info['val_samples'] = len(self.dataset_val)
            info['val_class_distribution'] = self.dataset_val.get_class_distribution()
        
        if self.dataset_test is not None:
            info['test_samples'] = len(self.dataset_test)
            info['test_class_distribution'] = self.dataset_test.get_class_distribution()
        
        return info
    
    def __repr__(self) -> str:
        """String representation of DataModule."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  data_dir={self.data_dir},\n"
            f"  batch_size={self.batch_size},\n"
            f"  num_workers={self.num_workers},\n"
            f"  total_samples={len(self.dataset_full) if self.dataset_full else 'Not loaded'},\n"
            f"  num_classes={len(self.dataset_full.classes) if self.dataset_full else 'Not loaded'}\n"
            f")"
        )


def create_simple_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    transform: Optional[Callable] = None,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders without Lightning dependency.
    
    Simple interface for users who don't use PyTorch Lightning.
    
    Args:
        data_dir: Root directory containing processed datasets.
        batch_size: Batch size for all DataLoaders.
        num_workers: Number of worker processes.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        test_ratio: Proportion of data for testing.
        transform: Transform to apply to all splits.
        **kwargs: Additional arguments passed to ThermalImageDataset.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Create full dataset
    dataset = ThermalImageDataset(
        root_dir=data_dir,
        transform=transform,
        **kwargs
    )
    
    # Split dataset
    train_ds, val_ds, test_ds = train_val_test_split(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    logger.info(f"Created DataLoaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    
    return train_loader, val_loader, test_loader
