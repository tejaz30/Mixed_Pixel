"""
Perlin noise generation for DRAEM synthetic anomaly masks.

Simple implementation using random noise and Gaussian smoothing.
"""

import numpy as np
from scipy.ndimage import zoom, gaussian_filter


def rand_perlin_2d_np(shape, res, fade=None):
    """
    Generate smooth 2D noise similar to Perlin noise.
    
    Uses random noise at low resolution with smooth upsampling to create
    blob-like patterns suitable for anomaly masks.
    
    Args:
        shape: Tuple (height, width) for output shape
        res: Tuple (res_y, res_x) for gradient resolution (controls blob size)
        fade: Unused (for API compatibility)
    
    Returns:
        noise: 2D numpy array of shape (height, width) with values in [0, 1]
    """
    # Generate random noise at low resolution
    low_res_noise = np.random.rand(res[0], res[1])
    
    # Apply Gaussian smoothing at low resolution
    low_res_noise = gaussian_filter(low_res_noise, sigma=1.0)
    
    # Calculate zoom factors to reach target shape
    zoom_factors = (shape[0] / res[0], shape[1] / res[1])
    
    # Upsample to target shape with smooth interpolation
    noise = zoom(low_res_noise, zoom_factors, order=3)
    
    # Ensure exact output shape (zoom might be off by 1 pixel)
    if noise.shape != shape:
        # Crop or pad to exact shape
        noise_fixed = np.zeros(shape, dtype=noise.dtype)
        h_min = min(noise.shape[0], shape[0])
        w_min = min(noise.shape[1], shape[1])
        noise_fixed[:h_min, :w_min] = noise[:h_min, :w_min]
        noise = noise_fixed
    
    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    
    return noise.astype(np.float32)


