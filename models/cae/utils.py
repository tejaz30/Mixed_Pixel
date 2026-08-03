"""
Utility functions for CAE module.

Provides helper functions for:
- Device selection
- Model initialization
- Checkpoint management
- Configuration handling
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import yaml

logger = logging.getLogger("models.cae.utils")


def get_device(device_name: Optional[str] = None) -> torch.device:
    """
    Get PyTorch device.
    
    Args:
        device_name: Device name ('cuda', 'cpu', 'cuda:0', etc.).
                    If None, auto-select best available device.
    
    Returns:
        torch.device object.
    """
    if device_name is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Auto-selected device: {device}")
    else:
        device = torch.device(device_name)
        logger.info(f"Using specified device: {device}")
    
    # Log GPU info if using CUDA
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(device)
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    return device


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model.
    
    Returns:
        Dictionary with parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file.
    
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from: {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: Path):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary.
        config_path: Path to save YAML file.
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved configuration to: {config_path}")


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Optional file path to save logs.
    
    Returns:
        Root logger.
    """
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    root_logger = logging.getLogger()
    
    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    return root_logger


def get_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate model size in memory.
    
    Args:
        model: PyTorch model.
    
    Returns:
        Dictionary with size in different units.
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    return {
        'total_bytes': total_size,
        'total_kb': total_size / 1024,
        'total_mb': total_size / (1024 ** 2),
        'param_mb': param_size / (1024 ** 2),
        'buffer_mb': buffer_size / (1024 ** 2)
    }


def create_directories(dirs: list):
    """
    Create directories if they don't exist.
    
    Args:
        dirs: List of directory paths (Path or str).
    """
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {dir_path}")


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """
    Find the latest checkpoint in directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints.
    
    Returns:
        Path to latest checkpoint, or None if no checkpoints found.
    """
    if not checkpoint_dir.exists():
        return None
    
    # Look for best_model.pth first
    best_model = checkpoint_dir / "best_model.pth"
    if best_model.exists():
        return best_model
    
    # Otherwise find latest epoch checkpoint
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if not checkpoints:
        return None
    
    # Sort by epoch number
    checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
    return checkpoints[-1]


def validate_input_shape(
    tensor: torch.Tensor,
    expected_shape: Tuple[int, ...]
) -> bool:
    """
    Validate tensor shape matches expected shape.
    
    Args:
        tensor: Input tensor.
        expected_shape: Expected shape tuple (can contain -1 for any size).
    
    Returns:
        True if shape matches, False otherwise.
    """
    if len(tensor.shape) != len(expected_shape):
        return False
    
    for actual, expected in zip(tensor.shape, expected_shape):
        if expected != -1 and actual != expected:
            return False
    
    return True


def get_checkpoint_info(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Extract information from checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint.
    
    Returns:
        Dictionary with checkpoint information.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    info = {
        'has_model': 'model_state_dict' in checkpoint,
        'has_optimizer': 'optimizer_state_dict' in checkpoint,
        'epoch': checkpoint.get('epoch', None),
        'best_val_loss': checkpoint.get('best_val_loss', None),
        'model_config': checkpoint.get('model_config', None)
    }
    
    return info


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.
    
    Args:
        seconds: Time in seconds.
    
    Returns:
        Formatted string (e.g., "1h 23m 45s").
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def seed_everything(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to: {seed}")
