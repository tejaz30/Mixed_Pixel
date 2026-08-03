"""
Utility functions for preprocessing pipeline.
"""

import yaml
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil


def setup_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    logger_name: str = "preprocessing"
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file. If None, logs only to console.
        level: Logging level.
        logger_name: Name of the logger.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        Configuration dictionary.
        
    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If config file is malformed.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def compute_file_hash(
    file_path: Path,
    algorithm: str = "sha256",
    chunk_size: int = 4096
) -> str:
    """
    Compute cryptographic hash of a file.
    
    Args:
        file_path: Path to file.
        algorithm: Hash algorithm (sha256, md5, sha1, sha512).
        chunk_size: Size of chunks to read (bytes).
        
    Returns:
        Hexadecimal hash string.
        
    Raises:
        ValueError: If algorithm is not supported.
        FileNotFoundError: If file does not exist.
    """
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{algorithm}' not supported")
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(chunk_size), b""):
            hash_obj.update(byte_block)
    
    return hash_obj.hexdigest()


def is_image_file(file_path: Path, valid_extensions: List[str]) -> bool:
    """
    Check if a file is a valid image based on extension.
    
    Args:
        file_path: Path to file.
        valid_extensions: List of valid extensions (e.g., ['.jpg', '.png']).
        
    Returns:
        True if file has valid image extension.
    """
    return file_path.suffix.lower() in [ext.lower() for ext in valid_extensions]


def find_all_images(
    root_path: Path,
    valid_extensions: List[str],
    skip_hidden: bool = True
) -> List[Path]:
    """
    Recursively find all image files in a directory.
    
    Args:
        root_path: Root directory to search.
        valid_extensions: List of valid image extensions.
        skip_hidden: Skip hidden files and directories.
        
    Returns:
        List of paths to image files.
    """
    image_files = []
    
    for file_path in root_path.rglob('*'):
        if file_path.is_file():
            # Skip hidden files if requested
            if skip_hidden and any(part.startswith('.') for part in file_path.parts):
                continue
            
            if is_image_file(file_path, valid_extensions):
                image_files.append(file_path)
    
    return image_files


def get_relative_path(file_path: Path, root_path: Path) -> Path:
    """
    Get relative path from root.
    
    Args:
        file_path: Full path to file.
        root_path: Root directory.
        
    Returns:
        Relative path.
    """
    try:
        return file_path.relative_to(root_path)
    except ValueError:
        return file_path


def get_relative_path(file_path: Path, root_path: Path) -> Path:
    """
    Get relative path from root.
    
    Args:
        file_path: Full path to file.
        root_path: Root directory.
        
    Returns:
        Relative path.
    """
    try:
        return file_path.relative_to(root_path)
    except ValueError:
        # If not relative, return just the filename
        return Path(file_path.name)


def format_bytes(size_bytes: int) -> str:
    """
    Format bytes into human-readable string.
    
    Args:
        size_bytes: Size in bytes.
        
    Returns:
        Formatted string (e.g., "1.5 MB").
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def safe_copy_file(
    src: Path,
    dst: Path,
    verify: bool = True,
    hash_algorithm: str = "sha256"
) -> bool:
    """
    Safely copy a file with optional verification.
    
    Args:
        src: Source file path.
        dst: Destination file path.
        verify: Verify copy by comparing hashes.
        hash_algorithm: Hash algorithm for verification.
        
    Returns:
        True if copy succeeded (and verified if requested).
        
    Raises:
        FileNotFoundError: If source file doesn't exist.
        IOError: If copy fails.
    """
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    
    # Create destination directory
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy file
    shutil.copy2(src, dst)
    
    # Verify if requested
    if verify:
        src_hash = compute_file_hash(src, algorithm=hash_algorithm)
        dst_hash = compute_file_hash(dst, algorithm=hash_algorithm)
        
        if src_hash != dst_hash:
            dst.unlink()  # Remove corrupted copy
            raise IOError(f"Copy verification failed: {src} -> {dst}")
    
    return True


def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path.
        
    Returns:
        The directory path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_directory_size(path: Path) -> int:
    """
    Calculate total size of all files in a directory.
    
    Args:
        path: Directory path.
        
    Returns:
        Total size in bytes.
    """
    total_size = 0
    for file_path in path.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    return total_size


def count_files_by_extension(root_path: Path) -> Dict[str, int]:
    """
    Count files by extension in a directory tree.
    
    Args:
        root_path: Root directory to search.
        
    Returns:
        Dictionary mapping extensions to counts.
    """
    extension_counts = {}
    
    for file_path in root_path.rglob('*'):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            extension_counts[ext] = extension_counts.get(ext, 0) + 1
    
    return extension_counts
