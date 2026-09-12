"""
Boundary-focused evaluation metrics for thermal imagery.

Since we do not have ground-truth mixed-pixel or boundary annotations,
we construct gradient-defined boundary proxies from the original thermal images.

These high-gradient regions are treated as CANDIDATE boundary regions,
not verified mixed-pixel annotations.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple
import cv2


def compute_gradient_magnitude(image: np.ndarray, method: str = 'sobel') -> np.ndarray:
    """
    Compute gradient magnitude from image.
    
    Args:
        image: Image array of shape (H, W) or (H, W, C).
        method: Gradient computation method ('sobel' or 'scharr').
    
    Returns:
        Gradient magnitude of shape (H, W).
    """
    # Convert to grayscale if color
    if len(image.shape) == 3:
        # Use luminance weighting for thermal RGB representation
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    else:
        gray = image
    
    # Ensure float32
    gray = gray.astype(np.float32)
    
    # Compute gradients
    if method == 'sobel':
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    elif method == 'scharr':
        grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    else:
        raise ValueError(f"Unknown gradient method: {method}")
    
    # Magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    return magnitude


def create_boundary_mask(
    gradient_magnitude: np.ndarray,
    threshold_percentile: float = 75.0,
    min_threshold: float = 0.01
) -> np.ndarray:
    """
    Create binary boundary mask from gradient magnitude.
    
    Args:
        gradient_magnitude: Gradient magnitude array (H, W).
        threshold_percentile: Percentile threshold for high-gradient regions.
        min_threshold: Minimum absolute threshold to avoid noise.
    
    Returns:
        Binary mask (H, W) where 1 indicates high-gradient/boundary regions.
    """
    # Compute percentile threshold
    threshold = np.percentile(gradient_magnitude, threshold_percentile)
    
    # Apply minimum threshold
    threshold = max(threshold, min_threshold)
    
    # Create binary mask
    mask = (gradient_magnitude >= threshold).astype(np.uint8)
    
    return mask


def compute_boundary_reconstruction_metrics(
    original: np.ndarray,
    reconstruction: np.ndarray,
    boundary_mask: np.ndarray
) -> Dict[str, float]:
    """
    Compute reconstruction metrics separately for boundary and non-boundary regions.
    
    Args:
        original: Original image (H, W, C) or (H, W), values in [0, 1].
        reconstruction: Reconstructed image (H, W, C) or (H, W), values in [0, 1].
        boundary_mask: Binary mask (H, W) where 1 = boundary, 0 = non-boundary.
    
    Returns:
        Dictionary containing:
            - boundary_mse: MSE in boundary regions
            - non_boundary_mse: MSE in non-boundary regions
            - boundary_mae: MAE in boundary regions
            - non_boundary_mae: MAE in non-boundary regions
            - error_ratio: boundary_mse / non_boundary_mse
            - boundary_fraction: Fraction of pixels in boundary region
    """
    # Ensure same shape
    assert original.shape == reconstruction.shape
    
    # Flatten if needed for easier indexing
    if len(original.shape) == 3:
        H, W, C = original.shape
        original_flat = original.reshape(-1, C)
        reconstruction_flat = reconstruction.reshape(-1, C)
        mask_flat = boundary_mask.reshape(-1)
    else:
        original_flat = original.reshape(-1, 1)
        reconstruction_flat = reconstruction.reshape(-1, 1)
        mask_flat = boundary_mask.reshape(-1)
    
    # Separate boundary and non-boundary pixels
    boundary_indices = mask_flat == 1
    non_boundary_indices = mask_flat == 0
    
    # Compute metrics for boundary region
    if boundary_indices.sum() > 0:
        boundary_orig = original_flat[boundary_indices]
        boundary_recon = reconstruction_flat[boundary_indices]
        
        boundary_mse = np.mean((boundary_orig - boundary_recon) ** 2)
        boundary_mae = np.mean(np.abs(boundary_orig - boundary_recon))
    else:
        boundary_mse = 0.0
        boundary_mae = 0.0
    
    # Compute metrics for non-boundary region
    if non_boundary_indices.sum() > 0:
        non_boundary_orig = original_flat[non_boundary_indices]
        non_boundary_recon = reconstruction_flat[non_boundary_indices]
        
        non_boundary_mse = np.mean((non_boundary_orig - non_boundary_recon) ** 2)
        non_boundary_mae = np.mean(np.abs(non_boundary_orig - non_boundary_recon))
    else:
        non_boundary_mse = 0.0
        non_boundary_mae = 0.0
    
    # Compute ratio
    if non_boundary_mse > 0:
        error_ratio = boundary_mse / non_boundary_mse
    else:
        error_ratio = float('inf') if boundary_mse > 0 else 1.0
    
    # Boundary fraction
    boundary_fraction = boundary_indices.sum() / len(mask_flat)
    
    return {
        'boundary_mse': float(boundary_mse),
        'non_boundary_mse': float(non_boundary_mse),
        'boundary_mae': float(boundary_mae),
        'non_boundary_mae': float(non_boundary_mae),
        'error_ratio': float(error_ratio),
        'boundary_fraction': float(boundary_fraction)
    }


def compute_gradient_preservation(
    original: np.ndarray,
    reconstruction: np.ndarray,
    method: str = 'sobel'
) -> Dict[str, float]:
    """
    Compute how well reconstruction preserves gradient structure.
    
    Args:
        original: Original image (H, W, C) or (H, W).
        reconstruction: Reconstructed image (H, W, C) or (H, W).
        method: Gradient computation method.
    
    Returns:
        Dictionary containing:
            - gradient_mse: MSE between gradient magnitudes
            - gradient_correlation: Correlation between gradients
            - gradient_similarity: Normalized similarity [0, 1]
    """
    # Compute gradients
    grad_orig = compute_gradient_magnitude(original, method=method)
    grad_recon = compute_gradient_magnitude(reconstruction, method=method)
    
    # Gradient MSE
    gradient_mse = np.mean((grad_orig - grad_recon) ** 2)
    
    # Gradient correlation
    grad_orig_flat = grad_orig.flatten()
    grad_recon_flat = grad_recon.flatten()
    
    if grad_orig_flat.std() > 0 and grad_recon_flat.std() > 0:
        gradient_correlation = np.corrcoef(grad_orig_flat, grad_recon_flat)[0, 1]
    else:
        gradient_correlation = 0.0
    
    # Gradient similarity (normalized)
    # 1 - normalized_distance
    grad_norm = np.linalg.norm(grad_orig - grad_recon)
    orig_norm = np.linalg.norm(grad_orig)
    
    if orig_norm > 0:
        gradient_similarity = 1.0 - (grad_norm / orig_norm)
    else:
        gradient_similarity = 1.0
    
    return {
        'gradient_mse': float(gradient_mse),
        'gradient_correlation': float(gradient_correlation),
        'gradient_similarity': float(gradient_similarity)
    }


def evaluate_boundary_reconstruction(
    original_batch: torch.Tensor,
    reconstruction_batch: torch.Tensor,
    threshold_percentile: float = 75.0
) -> Dict[str, float]:
    """
    Evaluate boundary reconstruction for a batch of images.
    
    Args:
        original_batch: Original images (B, C, H, W) in [0, 1].
        reconstruction_batch: Reconstructed images (B, C, H, W) in [0, 1].
        threshold_percentile: Percentile for boundary detection.
    
    Returns:
        Dictionary of averaged boundary metrics across the batch.
    """
    # Convert to numpy
    original_np = original_batch.cpu().numpy()
    reconstruction_np = reconstruction_batch.cpu().numpy()
    
    batch_size = original_np.shape[0]
    
    # Accumulate metrics
    metrics_accum = {
        'boundary_mse': 0.0,
        'non_boundary_mse': 0.0,
        'boundary_mae': 0.0,
        'non_boundary_mae': 0.0,
        'error_ratio': 0.0,
        'boundary_fraction': 0.0,
        'gradient_mse': 0.0,
        'gradient_correlation': 0.0,
        'gradient_similarity': 0.0
    }
    
    for i in range(batch_size):
        # Get single image (C, H, W) -> (H, W, C)
        orig_img = np.transpose(original_np[i], (1, 2, 0))
        recon_img = np.transpose(reconstruction_np[i], (1, 2, 0))
        
        # Compute gradient and boundary mask
        gradient_mag = compute_gradient_magnitude(orig_img)
        boundary_mask = create_boundary_mask(gradient_mag, threshold_percentile)
        
        # Boundary reconstruction metrics
        boundary_metrics = compute_boundary_reconstruction_metrics(
            orig_img, recon_img, boundary_mask
        )
        
        # Gradient preservation metrics
        gradient_metrics = compute_gradient_preservation(orig_img, recon_img)
        
        # Accumulate
        for key in boundary_metrics:
            metrics_accum[key] += boundary_metrics[key]
        
        for key in gradient_metrics:
            metrics_accum[key] += gradient_metrics[key]
    
    # Average
    metrics_avg = {k: v / batch_size for k, v in metrics_accum.items()}
    
    return metrics_avg
