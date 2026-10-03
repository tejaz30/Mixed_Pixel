"""
Utility functions for reconstruction evaluation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import torch
import numpy as np
from PIL import Image


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to log file.
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def save_evaluation_config(
    config: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Save evaluation configuration with timestamp.
    
    Args:
        config: Configuration dictionary.
        output_path: Path to save config JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    config_with_meta = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'config': config
    }
    
    with open(output_path, 'w') as f:
        json.dump(config_with_meta, f, indent=2)
    
    logging.info(f"Evaluation config saved to: {output_path}")


def load_image(image_path: Path) -> np.ndarray:
    """
    Load image as numpy array.
    
    Args:
        image_path: Path to image file.
    
    Returns:
        Image as numpy array with shape (H, W, C) and values in [0, 1].
    """
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image, dtype=np.float32) / 255.0
    return image_array


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array.
    
    Args:
        tensor: PyTorch tensor with shape (C, H, W) or (B, C, H, W).
    
    Returns:
        Numpy array with shape (H, W, C) or (B, H, W, C).
    """
    if tensor.dim() == 4:  # Batch
        # (B, C, H, W) -> (B, H, W, C)
        array = tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
    elif tensor.dim() == 3:  # Single image
        # (C, H, W) -> (H, W, C)
        array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
    
    return array


def numpy_to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.
    
    Args:
        array: Numpy array with shape (H, W, C) or (B, H, W, C).
        device: Device to place tensor on.
    
    Returns:
        PyTorch tensor with shape (C, H, W) or (B, C, H, W).
    """
    if array.ndim == 4:  # Batch
        # (B, H, W, C) -> (B, C, H, W)
        tensor = torch.from_numpy(array).permute(0, 3, 1, 2)
    elif array.ndim == 3:  # Single image
        # (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(array).permute(2, 0, 1)
    else:
        raise ValueError(f"Unexpected array shape: {array.shape}")
    
    return tensor.to(device)


def create_directories(paths: List[Path]) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        paths: List of directory paths to create.
    """
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        logging.debug(f"Directory ensured: {path}")


def get_checkpoint_info(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Extract information from checkpoint path.
    
    Args:
        checkpoint_path: Path to model checkpoint.
    
    Returns:
        Dictionary with checkpoint metadata.
    """
    return {
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_name': checkpoint_path.name,
        'checkpoint_exists': checkpoint_path.exists(),
        'checkpoint_size_mb': checkpoint_path.stat().st_size / (1024 * 1024) if checkpoint_path.exists() else 0
    }


def clamp_to_valid_range(array: np.ndarray, data_range: float = 1.0) -> np.ndarray:
    """
    Clamp array values to valid range [0, data_range].
    
    Args:
        array: Input array.
        data_range: Maximum value of valid range.
    
    Returns:
        Clamped array.
    """
    return np.clip(array, 0, data_range)
