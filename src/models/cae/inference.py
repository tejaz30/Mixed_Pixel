"""
Inference module for Convolutional Autoencoder.

Provides functions for:
- Running inference on images
- Computing reconstruction error maps
- Batch prediction
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from .model import ConvolutionalAutoencoder

logger = logging.getLogger("models.cae.inference")


class CAEInference:
    """
    Inference handler for Convolutional Autoencoder.
    
    Provides methods for:
    - predict(): Run inference on image batches
    - reconstruct_image(): Reconstruct a single image
    - compute_reconstruction_error(): Generate reconstruction error maps
    """
    
    def __init__(
        self,
        model: ConvolutionalAutoencoder,
        device: torch.device,
        checkpoint_path: Optional[Path] = None
    ):
        """
        Initialize inference handler.
        
        Args:
            model: CAE model for inference.
            device: Device to run inference on (CPU/GPU).
            checkpoint_path: Optional path to load model weights.
        """
        self.model = model.to(device)
        self.device = device
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        # Set to evaluation mode
        self.model.eval()
        
        logger.info(f"Inference handler initialized on device: {device}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model weights from: {checkpoint_path}")
    
    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference on image batch.
        
        Args:
            images: Input tensor of shape (B, C, H, W).
                   Expected: (B, 3, 480, 640) normalized to [0, 1].
        
        Returns:
            Dictionary containing:
                - 'reconstructed': Reconstructed images (B, C, H, W)
                - 'error_map': Reconstruction error maps (B, C, H, W)
                - 'latent': Latent representations (B, 8, 60, 80)
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Move to device
        images = images.to(self.device)
        
        # Encode
        latent = self.model.encode(images)
        
        # Decode
        reconstructed = self.model.decode(latent)
        
        # Compute reconstruction error (per-pixel absolute difference)
        error_map = torch.abs(images - reconstructed)
        
        return {
            'reconstructed': reconstructed,
            'error_map': error_map,
            'latent': latent
        }
    
    @torch.no_grad()
    def reconstruct_image(
        self,
        image: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstruct a single image.
        
        Args:
            image: Input tensor of shape (C, H, W) or (1, C, H, W).
                  Expected: (3, 480, 640) normalized to [0, 1].
        
        Returns:
            Dictionary containing:
                - 'reconstructed': Reconstructed image (C, H, W)
                - 'error_map': Reconstruction error map (C, H, W)
                - 'latent': Latent representation
        """
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Run prediction
        results = self.predict(image)
        
        # Remove batch dimension
        return {
            'reconstructed': results['reconstructed'].squeeze(0),
            'error_map': results['error_map'].squeeze(0),
            'latent': results['latent'].squeeze(0)
        }
    
    @torch.no_grad()
    def compute_reconstruction_error(
        self,
        images: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute reconstruction error with optional reduction.
        
        Args:
            images: Input tensor of shape (B, C, H, W).
            reduction: Reduction method:
                - 'none': Return per-pixel errors (B, C, H, W)
                - 'channel': Average over channels (B, H, W)
                - 'spatial': Average over spatial dims (B, C)
                - 'mean': Return scalar mean error per image (B,)
                - 'sum': Return scalar sum error per image (B,)
        
        Returns:
            Reconstruction error tensor with shape depending on reduction.
        """
        # Run prediction
        results = self.predict(images)
        error_map = results['error_map']
        
        # Apply reduction
        if reduction == 'none':
            return error_map
        elif reduction == 'channel':
            return error_map.mean(dim=1)  # (B, H, W)
        elif reduction == 'spatial':
            return error_map.mean(dim=[2, 3])  # (B, C)
        elif reduction == 'mean':
            return error_map.mean(dim=[1, 2, 3])  # (B,)
        elif reduction == 'sum':
            return error_map.sum(dim=[1, 2, 3])  # (B,)
        else:
            raise ValueError(
                f"Invalid reduction: {reduction}. "
                f"Choose from: none, channel, spatial, mean, sum"
            )
    
    @torch.no_grad()
    def encode_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent representations.
        
        Args:
            images: Input tensor of shape (B, C, H, W).
        
        Returns:
            Latent representations of shape (B, 8, 60, 80).
        """
        self.model.eval()
        images = images.to(self.device)
        return self.model.encode(images)
    
    @torch.no_grad()
    def decode_batch(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representations to images.
        
        Args:
            latent: Latent tensor of shape (B, 8, 60, 80).
        
        Returns:
            Reconstructed images of shape (B, C, H, W).
        """
        self.model.eval()
        latent = latent.to(self.device)
        return self.model.decode(latent)


def load_model_for_inference(
    checkpoint_path: Path,
    device: Optional[torch.device] = None
) -> CAEInference:
    """
    Load trained CAE model for inference.
    
    Args:
        checkpoint_path: Path to model checkpoint.
        device: Device to load model on. If None, auto-select.
    
    Returns:
        CAEInference handler ready for inference.
    """
    # Auto-select device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint to get model config
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get('model_config', {})
    
    # Create model
    model = ConvolutionalAutoencoder(
        input_channels=model_config.get('input_channels', 3),
        input_height=model_config.get('input_height', 480),
        input_width=model_config.get('input_width', 640),
        encoder_filters=tuple(model_config.get('encoder_filters', [1, 6, 8]))
    )
    
    # Create inference handler (loads weights automatically)
    inference = CAEInference(
        model=model,
        device=device,
        checkpoint_path=checkpoint_path
    )
    
    logger.info(f"Model loaded for inference from: {checkpoint_path}")
    return inference


def visualize_reconstruction(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    error_map: torch.Tensor
) -> Dict[str, np.ndarray]:
    """
    Convert tensors to numpy arrays for visualization.
    
    Args:
        original: Original image tensor (C, H, W) in [0, 1].
        reconstructed: Reconstructed image tensor (C, H, W) in [0, 1].
        error_map: Error map tensor (C, H, W).
    
    Returns:
        Dictionary with numpy arrays (H, W, C) in [0, 255] uint8.
    """
    # Move to CPU and convert to numpy
    original_np = (original.cpu().numpy() * 255).astype(np.uint8)
    reconstructed_np = (reconstructed.cpu().numpy() * 255).astype(np.uint8)
    error_np = (error_map.cpu().numpy() * 255).astype(np.uint8)
    
    # Transpose from (C, H, W) to (H, W, C)
    original_np = np.transpose(original_np, (1, 2, 0))
    reconstructed_np = np.transpose(reconstructed_np, (1, 2, 0))
    error_np = np.transpose(error_np, (1, 2, 0))
    
    return {
        'original': original_np,
        'reconstructed': reconstructed_np,
        'error': error_np
    }
