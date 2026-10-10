"""
Training utility functions.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_name: str = 'auto') -> torch.device:
    """
    Get torch device.
    
    Args:
        device_name: Device specification ('auto', 'cuda', 'cpu', 'cuda:0', etc.)
    
    Returns:
        torch.device instance
    """
    if device_name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_name)


def create_directories(paths: dict) -> None:
    """
    Create necessary directories for experiment outputs.
    
    Args:
        paths: Dictionary of path keys to directory paths
    """
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_device(model: torch.nn.Module) -> torch.device:
    """
    Get the device a model is on.
    
    Args:
        model: PyTorch model
    
    Returns:
        Device the model is on
    """
    return next(model.parameters()).device
