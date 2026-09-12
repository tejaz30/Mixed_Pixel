"""
Reconstruction Quality Metrics

Implements MSE and SSIM as specified in the paper:
"Unsupervised autoencoder detection with a local context-based correction
for mixed pixels in thermal images"

The paper evaluates reconstruction quality using:
1. Mean Squared Error (MSE)
2. Structural Similarity Index Measure (SSIM)

All parameters are explicit and configurable to ensure reproducibility.
"""

import logging
from typing import Tuple, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch

# Lazy import of skimage to avoid potential hanging issues on Windows
# Import happens inside compute_ssim function instead
_ssim_skimage = None

def _get_ssim_function():
    """Lazy load SSIM function from skimage."""
    global _ssim_skimage
    if _ssim_skimage is None:
        from skimage.metrics import structural_similarity
        _ssim_skimage = structural_similarity
    return _ssim_skimage


@dataclass
class SSIMConfig:
    """
    Configuration for SSIM computation.
    
    All parameters are explicit to ensure reproducibility.
    
    Attributes:
        data_range: The data range of the input images.
                   For normalized images in [0, 1], use 1.0.
                   For images in [0, 255], use 255.0.
        win_size: Window size for SSIM computation. Must be odd.
                 Default is 7 (as commonly used in literature).
        channel_axis: Axis corresponding to color channels.
                     For (H, W, C) format, use -1 or 2.
        gaussian_weights: Whether to use Gaussian weighting.
        sigma: Standard deviation for Gaussian kernel.
        use_sample_covariance: Whether to use sample covariance or biased estimator.
        k1: Algorithm parameter for SSIM (default: 0.01).
        k2: Algorithm parameter for SSIM (default: 0.03).
    """
    data_range: float = 1.0
    win_size: int = 7
    channel_axis: int = -1
    gaussian_weights: bool = True
    sigma: float = 1.5
    use_sample_covariance: bool = True
    k1: float = 0.01
    k2: float = 0.03
    
    def __post_init__(self):
        """Validate configuration."""
        if self.win_size % 2 == 0:
            raise ValueError(f"win_size must be odd, got {self.win_size}")
        if self.data_range <= 0:
            raise ValueError(f"data_range must be positive, got {self.data_range}")
        
        logging.debug(f"SSIM Config: data_range={self.data_range}, "
                     f"win_size={self.win_size}, channel_axis={self.channel_axis}, "
                     f"gaussian_weights={self.gaussian_weights}, sigma={self.sigma}")


def compute_mse(
    original: np.ndarray,
    reconstructed: np.ndarray,
    data_range: float = 1.0
) -> float:
    """
    Compute Mean Squared Error between original and reconstructed images.
    
    MSE = mean((original - reconstructed)^2)
    
    As specified in the paper, MSE quantifies pixel-level reconstruction quality.
    Lower MSE indicates better reconstruction.
    
    Args:
        original: Original image with shape (H, W, C) or (H, W).
                 Values should be in [0, data_range].
        reconstructed: Reconstructed image with same shape as original.
                      Values should be in [0, data_range].
        data_range: The data range of the images. Used for validation only.
                   For normalized images in [0, 1], use 1.0.
    
    Returns:
        MSE value as float.
    
    Notes:
        - Both images must be in the same value range
        - No normalization is applied inside this function
        - The images must have been preprocessed consistently
    
    Example:
        >>> original = np.random.rand(480, 640, 3)  # [0, 1]
        >>> reconstructed = np.random.rand(480, 640, 3)  # [0, 1]
        >>> mse = compute_mse(original, reconstructed, data_range=1.0)
    """
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} vs reconstructed {reconstructed.shape}"
        )
    
    # Validate value range (with tolerance for floating point errors)
    if np.any(original < -0.01) or np.any(original > data_range + 0.01):
        logging.warning(
            f"Original image values outside expected range [0, {data_range}]: "
            f"min={original.min():.4f}, max={original.max():.4f}"
        )
    
    if np.any(reconstructed < -0.01) or np.any(reconstructed > data_range + 0.01):
        logging.warning(
            f"Reconstructed image values outside expected range [0, {data_range}]: "
            f"min={reconstructed.min():.4f}, max={reconstructed.max():.4f}"
        )
    
    # Compute MSE
    mse = np.mean((original - reconstructed) ** 2)
    
    return float(mse)


def compute_ssim(
    original: np.ndarray,
    reconstructed: np.ndarray,
    config: Optional[SSIMConfig] = None
) -> float:
    """
    Compute Structural Similarity Index Measure between original and reconstructed images.
    
    As specified in the paper, SSIM quantifies structural consistency.
    Higher SSIM indicates better structural preservation.
    SSIM ranges from -1 to 1, where 1 indicates perfect similarity.
    
    Args:
        original: Original image with shape (H, W, C) or (H, W).
                 Values should be in [0, data_range].
        reconstructed: Reconstructed image with same shape as original.
                      Values should be in [0, data_range].
        config: SSIM configuration. If None, uses default config.
    
    Returns:
        SSIM value as float.
    
    Notes:
        - Uses scikit-image implementation for reproducibility
        - All parameters are explicit in SSIMConfig
        - Window size, Gaussian weighting, and data range are configurable
        - Channel axis must be specified for color images
    
    Example:
        >>> config = SSIMConfig(data_range=1.0, win_size=7)
        >>> original = np.random.rand(480, 640, 3)  # [0, 1]
        >>> reconstructed = np.random.rand(480, 640, 3)  # [0, 1]
        >>> ssim_val = compute_ssim(original, reconstructed, config)
    """
    if config is None:
        config = SSIMConfig()
    
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} vs reconstructed {reconstructed.shape}"
        )
    
    # Validate value range
    if np.any(original < -0.01) or np.any(original > config.data_range + 0.01):
        logging.warning(
            f"Original image values outside expected range [0, {config.data_range}]: "
            f"min={original.min():.4f}, max={original.max():.4f}"
        )
    
    if np.any(reconstructed < -0.01) or np.any(reconstructed > config.data_range + 0.01):
        logging.warning(
            f"Reconstructed image values outside expected range [0, {config.data_range}]: "
            f"min={reconstructed.min():.4f}, max={reconstructed.max():.4f}"
        )
    
    # Compute SSIM using scikit-image (lazy-loaded)
    ssim_function = _get_ssim_function()
    ssim_value = ssim_function(
        original,
        reconstructed,
        data_range=config.data_range,
        win_size=config.win_size,
        channel_axis=config.channel_axis if original.ndim == 3 else None,
        gaussian_weights=config.gaussian_weights,
        sigma=config.sigma,
        use_sample_covariance=config.use_sample_covariance,
        K1=config.k1,
        K2=config.k2
    )
    
    return float(ssim_value)


def compute_error_map(
    original: np.ndarray,
    reconstructed: np.ndarray
) -> np.ndarray:
    """
    Compute absolute error map between original and reconstructed images.
    
    Error map = |original - reconstructed|
    
    This visualizes pixel-wise reconstruction errors.
    
    Args:
        original: Original image with shape (H, W, C) or (H, W).
        reconstructed: Reconstructed image with same shape.
    
    Returns:
        Absolute error map with same shape as input.
    
    Notes:
        - No normalization is applied (use same range as input)
        - For visualization, you may want to normalize separately
        - The error map uses the same value range as the input images
    """
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} vs reconstructed {reconstructed.shape}"
        )
    
    error_map = np.abs(original - reconstructed)
    
    return error_map


class ReconstructionMetrics:
    """
    Container for computing reconstruction metrics.
    
    This class provides a unified interface for computing MSE, SSIM,
    and error maps with consistent configuration.
    """
    
    def __init__(
        self,
        data_range: float = 1.0,
        ssim_config: Optional[SSIMConfig] = None
    ):
        """
        Initialize metrics computer.
        
        Args:
            data_range: Data range for images (1.0 for [0,1], 255.0 for [0,255]).
            ssim_config: SSIM configuration. If None, creates default with data_range.
        """
        self.data_range = data_range
        
        if ssim_config is None:
            self.ssim_config = SSIMConfig(data_range=data_range)
        else:
            self.ssim_config = ssim_config
            # Ensure data_range matches
            if self.ssim_config.data_range != data_range:
                logging.warning(
                    f"SSIM config data_range ({self.ssim_config.data_range}) "
                    f"differs from provided data_range ({data_range})"
                )
        
        logging.info(f"ReconstructionMetrics initialized with data_range={self.data_range}")
        logging.info(f"SSIM parameters: win_size={self.ssim_config.win_size}, "
                    f"gaussian_weights={self.ssim_config.gaussian_weights}, "
                    f"sigma={self.ssim_config.sigma}")
    
    def compute_all(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        compute_error: bool = True
    ) -> dict:
        """
        Compute all reconstruction metrics.
        
        Args:
            original: Original image.
            reconstructed: Reconstructed image.
            compute_error: Whether to compute error map.
        
        Returns:
            Dictionary with keys: 'mse', 'ssim', and optionally 'error_map'.
        """
        results = {
            'mse': compute_mse(original, reconstructed, self.data_range),
            'ssim': compute_ssim(original, reconstructed, self.ssim_config)
        }
        
        if compute_error:
            results['error_map'] = compute_error_map(original, reconstructed)
        
        return results
    
    def get_config(self) -> dict:
        """
        Get metric configuration as dictionary.
        
        Returns:
            Dictionary with all metric parameters.
        """
        return {
            'data_range': self.data_range,
            'ssim': {
                'win_size': self.ssim_config.win_size,
                'channel_axis': self.ssim_config.channel_axis,
                'gaussian_weights': self.ssim_config.gaussian_weights,
                'sigma': self.ssim_config.sigma,
                'use_sample_covariance': self.ssim_config.use_sample_covariance,
                'k1': self.ssim_config.k1,
                'k2': self.ssim_config.k2
            }
        }
