"""
Image normalization operations.
"""

import logging
from typing import Tuple, Dict, Any
import numpy as np

logger = logging.getLogger("preprocessing.normalization")


class ImageNormalizer:
    """
    Normalize pixel intensities using various methods.
    """
    
    def __init__(
        self,
        method: str = "minmax",
        min_value: float = 0.0,
        max_value: float = 1.0,
        per_image: bool = True
    ):
        """
        Initialize normalizer.
        
        Args:
            method: Normalization method ('minmax', 'zscore', 'none').
            min_value: Minimum value for minmax normalization.
            max_value: Maximum value for minmax normalization.
            per_image: If True, use per-image statistics. If False, use dataset stats.
        """
        self.method = method.lower()
        self.min_value = min_value
        self.max_value = max_value
        self.per_image = per_image
        
        # Dataset statistics (to be set externally if per_image=False)
        self.dataset_mean = None
        self.dataset_std = None
        self.dataset_min = None
        self.dataset_max = None
        
        valid_methods = ['minmax', 'zscore', 'divide255', 'none']
        if self.method not in valid_methods:
            raise ValueError(f"Invalid normalization method: {method}. Choose from {valid_methods}")
    
    def normalize(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Normalize an image.
        
        Args:
            image: Input image (uint8 or uint16).
            
        Returns:
            Tuple of (normalized_image, metadata).
        """
        if self.method == 'none':
            return image, {'normalization': 'none'}
        
        # Convert to float for normalization
        image_float = image.astype(np.float32)
        
        metadata = {
            'normalization_method': self.method,
            'original_dtype': str(image.dtype)
        }
        
        if self.method == 'minmax':
            normalized = self._minmax_normalize(image_float, metadata)
        elif self.method == 'zscore':
            normalized = self._zscore_normalize(image_float, metadata)
        elif self.method == 'divide255':
            normalized = self._divide255_normalize(image_float, metadata)
        else:
            normalized = image_float
        
        return normalized, metadata
    
    def _minmax_normalize(
        self, 
        image: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """
        Min-max normalization to [min_value, max_value] range.
        
        Args:
            image: Input image as float32.
            metadata: Dictionary to update with normalization info.
            
        Returns:
            Normalized image.
        """
        if self.per_image:
            img_min = image.min()
            img_max = image.max()
        else:
            if self.dataset_min is None or self.dataset_max is None:
                logger.warning("Dataset statistics not set, using per-image statistics")
                img_min = image.min()
                img_max = image.max()
            else:
                img_min = self.dataset_min
                img_max = self.dataset_max
        
        metadata['data_min'] = float(img_min)
        metadata['data_max'] = float(img_max)
        metadata['target_min'] = self.min_value
        metadata['target_max'] = self.max_value
        
        # Avoid division by zero
        if img_max - img_min < 1e-8:
            logger.warning("Image has constant intensity, returning zeros")
            return np.zeros_like(image)
        
        # Normalize to [0, 1]
        normalized = (image - img_min) / (img_max - img_min)
        
        # Scale to [min_value, max_value]
        normalized = normalized * (self.max_value - self.min_value) + self.min_value
        
        return normalized
    
    def _divide255_normalize(
        self,
        image: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """
        Fixed division by 255.0 as specified in paper Section 4.1.3.
        
        "Each image was converted to 'float32' format and normalized 
        by dividing the pixel values by 255.0"
        
        Args:
            image: Input image as float32.
            metadata: Dictionary to update with normalization info.
            
        Returns:
            Normalized image.
        """
        metadata['divide_factor'] = 255.0
        
        # Simple division by 255.0
        normalized = image / 255.0
        
        # Clip to [0, 1] to handle any potential overflow
        normalized = np.clip(normalized, 0.0, 1.0)
        
        return normalized
    
    def _zscore_normalize(
        self,
        image: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """
        Z-score normalization (standardization) to mean=0, std=1.
        
        Args:
            image: Input image as float32.
            metadata: Dictionary to update with normalization info.
            
        Returns:
            Normalized image.
        """
        if self.per_image:
            img_mean = image.mean()
            img_std = image.std()
        else:
            if self.dataset_mean is None or self.dataset_std is None:
                logger.warning("Dataset statistics not set, using per-image statistics")
                img_mean = image.mean()
                img_std = image.std()
            else:
                img_mean = self.dataset_mean
                img_std = self.dataset_std
        
        metadata['data_mean'] = float(img_mean)
        metadata['data_std'] = float(img_std)
        
        # Avoid division by zero
        if img_std < 1e-8:
            logger.warning("Image has zero std, returning centered values")
            return image - img_mean
        
        # Standardize
        normalized = (image - img_mean) / img_std
        
        return normalized
    
    def set_dataset_statistics(
        self,
        mean: float = None,
        std: float = None,
        min_val: float = None,
        max_val: float = None
    ):
        """
        Set dataset-level statistics for normalization.
        
        Args:
            mean: Dataset mean.
            std: Dataset standard deviation.
            min_val: Dataset minimum value.
            max_val: Dataset maximum value.
        """
        self.dataset_mean = mean
        self.dataset_std = std
        self.dataset_min = min_val
        self.dataset_max = max_val
        
        logger.info(
            f"Dataset statistics set: mean={mean}, std={std}, "
            f"min={min_val}, max={max_val}"
        )
