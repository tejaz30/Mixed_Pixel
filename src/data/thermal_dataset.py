"""
PyTorch dataset for thermal image mixed-pixel detection.

This module provides a flexible dataset interface that supports:
- Multiple thermal image datasets
- Automatic label discovery from directory structure
- Configurable transforms and augmentations
- Train/validation/test splits
- Future detector architectures
"""

import logging
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple, Any, Union
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from collections import defaultdict

logger = logging.getLogger("data.thermal_dataset")


class ThermalImageDataset(Dataset):
    """
    PyTorch dataset for thermal images.
    
    Features:
    - Recursive dataset discovery
    - Multiple dataset support
    - Automatic label extraction from folder names
    - Configurable transforms
    - Metadata preservation
    - Support for supervised and unsupervised learning
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        datasets: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_extensions: List[str] = None,
        load_mode: str = "rgb",
        return_path: bool = False,
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        """
        Initialize thermal image dataset.
        
        Args:
            root_dir: Root directory containing dataset folders.
            datasets: List of dataset names to include. If None, include all.
            transform: Optional transform to apply to images.
            target_transform: Optional transform to apply to labels.
            image_extensions: List of valid image extensions.
            load_mode: How to load images ('rgb', 'grayscale', 'unchanged').
            return_path: If True, include image path in returned dict.
            class_to_idx: Manual mapping from class names to indices.
                         If None, will be auto-generated.
        """
        self.root_dir = Path(root_dir)
        self.datasets = datasets
        self.transform = transform
        self.target_transform = target_transform
        self.load_mode = load_mode.lower()
        self.return_path = return_path
        
        if image_extensions is None:
            self.image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        else:
            self.image_extensions = image_extensions
        
        # Validate root directory
        if not self.root_dir.exists():
            raise ValueError(f"Root directory does not exist: {self.root_dir}")
        
        # Discover dataset structure
        logger.info(f"Discovering datasets in {self.root_dir}")
        self.samples = []
        self.classes = []
        self.class_to_idx = class_to_idx or {}
        
        self._discover_datasets()
        
        # Auto-generate class mapping if not provided
        if not self.class_to_idx:
            self._generate_class_mapping()
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        logger.info(f"Dataset initialized:")
        logger.info(f"  Total samples: {len(self.samples)}")
        logger.info(f"  Number of classes: {len(self.classes)}")
        logger.info(f"  Classes: {self.classes}")
    
    def _discover_datasets(self):
        """
        Recursively discover all images and their labels.
        
        Expected structure:
        root_dir/
            Dataset1/
                ClassA/
                    image1.jpg
                    image2.jpg
                ClassB/
                    image3.jpg
            Dataset2/
                ClassC/
                    image4.jpg
        
        Labels are extracted from the immediate parent directory.
        """
        # Get dataset directories
        if self.datasets is None:
            # Use all subdirectories
            dataset_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        else:
            # Use specified datasets
            dataset_dirs = [self.root_dir / ds for ds in self.datasets]
        
        # Track statistics per dataset
        dataset_stats = defaultdict(lambda: defaultdict(int))
        
        for dataset_dir in dataset_dirs:
            if not dataset_dir.exists():
                logger.warning(f"Dataset directory not found: {dataset_dir}")
                continue
            
            dataset_name = dataset_dir.name
            logger.info(f"Scanning dataset: {dataset_name}")
            
            # Find all images recursively
            for ext in self.image_extensions:
                for img_path in dataset_dir.rglob(f"*{ext}"):
                    # Extract class name from parent directory
                    class_name = img_path.parent.name
                    
                    # Store sample
                    self.samples.append({
                        'path': img_path,
                        'class': class_name,
                        'dataset': dataset_name
                    })
                    
                    # Track unique classes
                    if class_name not in self.classes:
                        self.classes.append(class_name)
                    
                    # Update statistics
                    dataset_stats[dataset_name][class_name] += 1
        
        # Sort classes for consistency
        self.classes.sort()
        
        # Log statistics
        logger.info(f"Dataset discovery complete:")
        for dataset_name, class_counts in dataset_stats.items():
            logger.info(f"  {dataset_name}:")
            for class_name, count in sorted(class_counts.items()):
                logger.info(f"    {class_name}: {count} images")
    
    def _generate_class_mapping(self):
        """Generate class name to index mapping."""
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        logger.info(f"Generated class mapping: {self.class_to_idx}")
    
    def _load_image(self, path: Path) -> np.ndarray:
        """
        Load image from disk.
        
        Args:
            path: Path to image file.
            
        Returns:
            Image as numpy array.
        """
        if self.load_mode == 'grayscale':
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        elif self.load_mode == 'unchanged':
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        else:  # rgb (default)
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            # Convert BGR to RGB
            if image is not None and len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image is None:
            raise IOError(f"Failed to load image: {path}")
        
        return image
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample.
            
        Returns:
            Dictionary containing:
                - image: Image tensor (after transforms)
                - label: Class index (int) or None if unlabeled
                - metadata: Dict with additional information
        """
        sample_info = self.samples[idx]
        
        # Load image
        image = self._load_image(sample_info['path'])
        
        # Get label
        class_name = sample_info['class']
        label = self.class_to_idx.get(class_name, None)
        
        # Create metadata
        metadata = {
            'dataset': sample_info['dataset'],
            'class_name': class_name,
            'original_shape': image.shape,
        }
        
        if self.return_path:
            metadata['path'] = str(sample_info['path'])
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None and label is not None:
            label = self.target_transform(label)
        
        return {
            'image': image,
            'label': label,
            'metadata': metadata
        }
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get distribution of samples per class.
        
        Returns:
            Dictionary mapping class names to sample counts.
        """
        distribution = defaultdict(int)
        for sample in self.samples:
            distribution[sample['class']] += 1
        return dict(distribution)
    
    def get_dataset_distribution(self) -> Dict[str, int]:
        """
        Get distribution of samples per dataset.
        
        Returns:
            Dictionary mapping dataset names to sample counts.
        """
        distribution = defaultdict(int)
        for sample in self.samples:
            distribution[sample['dataset']] += 1
        return dict(distribution)
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Get information about a sample without loading the image.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary with sample information.
        """
        return self.samples[idx].copy()
    
    def filter_by_dataset(self, dataset_names: List[str]) -> 'ThermalImageDataset':
        """
        Create a new dataset containing only specified datasets.
        
        Args:
            dataset_names: List of dataset names to include.
            
        Returns:
            New ThermalImageDataset instance.
        """
        new_dataset = ThermalImageDataset(
            root_dir=self.root_dir,
            datasets=dataset_names,
            transform=self.transform,
            target_transform=self.target_transform,
            image_extensions=self.image_extensions,
            load_mode=self.load_mode,
            return_path=self.return_path,
            class_to_idx=self.class_to_idx
        )
        return new_dataset
    
    def filter_by_class(self, class_names: List[str]) -> 'ThermalImageDataset':
        """
        Create a new dataset containing only specified classes.
        
        Args:
            class_names: List of class names to include.
            
        Returns:
            New ThermalImageDataset instance.
        """
        # Create filtered dataset manually
        filtered = ThermalImageDataset.__new__(ThermalImageDataset)
        filtered.root_dir = self.root_dir
        filtered.datasets = self.datasets
        filtered.transform = self.transform
        filtered.target_transform = self.target_transform
        filtered.load_mode = self.load_mode
        filtered.return_path = self.return_path
        filtered.image_extensions = self.image_extensions
        
        # Filter samples
        filtered.samples = [
            s for s in self.samples 
            if s['class'] in class_names
        ]
        
        # Update classes
        filtered.classes = [c for c in self.classes if c in class_names]
        filtered.class_to_idx = {
            k: v for k, v in self.class_to_idx.items() 
            if k in class_names
        }
        filtered.idx_to_class = {v: k for k, v in filtered.class_to_idx.items()}
        
        logger.info(f"Filtered dataset: {len(filtered.samples)} samples, {len(filtered.classes)} classes")
        
        return filtered


class SubsetDataset(Dataset):
    """
    Subset of a ThermalImageDataset using indices.
    
    Useful for train/val/test splits.
    """
    
    def __init__(self, dataset: ThermalImageDataset, indices: List[int]):
        """
        Initialize subset dataset.
        
        Args:
            dataset: Parent ThermalImageDataset.
            indices: List of indices to include in subset.
        """
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.dataset[self.indices[idx]]
    
    @property
    def classes(self) -> List[str]:
        return self.dataset.classes
    
    @property
    def class_to_idx(self) -> Dict[str, int]:
        return self.dataset.class_to_idx
    
    @property
    def idx_to_class(self) -> Dict[int, str]:
        return self.dataset.idx_to_class
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution for this subset."""
        distribution = defaultdict(int)
        for idx in self.indices:
            sample = self.dataset.samples[idx]
            distribution[sample['class']] += 1
        return dict(distribution)


def train_val_test_split(
    dataset: ThermalImageDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    random_seed: int = 42
) -> Tuple[SubsetDataset, SubsetDataset, SubsetDataset]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: ThermalImageDataset to split.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        test_ratio: Proportion of data for testing.
        stratify: If True, maintain class distribution across splits.
        random_seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    np.random.seed(random_seed)
    
    if stratify:
        # Group indices by class
        class_indices = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            class_indices[sample['class']].append(idx)
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        # Split each class separately
        for class_name, indices in class_indices.items():
            # Shuffle indices
            indices = np.array(indices)
            np.random.shuffle(indices)
            
            n = len(indices)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            train_indices.extend(indices[:n_train].tolist())
            val_indices.extend(indices[n_train:n_train + n_val].tolist())
            test_indices.extend(indices[n_train + n_val:].tolist())
        
        logger.info(f"Stratified split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    
    else:
        # Random split without stratification
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_indices = indices[:n_train].tolist()
        val_indices = indices[n_train:n_train + n_val].tolist()
        test_indices = indices[n_train + n_val:].tolist()
        
        logger.info(f"Random split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    
    # Create subset datasets
    train_dataset = SubsetDataset(dataset, train_indices)
    val_dataset = SubsetDataset(dataset, val_indices)
    test_dataset = SubsetDataset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for thermal image batches.
    
    Args:
        batch: List of samples from dataset.
        
    Returns:
        Batched dictionary with stacked tensors and lists of metadata.
    """
    images = []
    labels = []
    metadata = []
    
    for sample in batch:
        images.append(sample['image'])
        labels.append(sample['label'] if sample['label'] is not None else -1)
        metadata.append(sample['metadata'])
    
    # Stack images
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images)
    else:
        # Convert to tensor if not already
        images = torch.stack([torch.from_numpy(img) for img in images])
    
    # Stack labels
    labels = torch.tensor(labels, dtype=torch.long)
    
    return {
        'image': images,
        'label': labels,
        'metadata': metadata
    }
