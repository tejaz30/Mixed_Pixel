"""
Image loading utilities for preprocessing pipeline.
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger("preprocessing.loader")


class ImageLoader:
    """
    Handles loading images from disk with multiple fallback methods.
    """
    
    def __init__(
        self,
        valid_extensions: List[str],
        load_mode: str = "unchanged",
        skip_corrupted: bool = True
    ):
        """
        Initialize image loader.
        
        Args:
            valid_extensions: List of valid image file extensions.
            load_mode: Loading mode - 'unchanged', 'rgb', or 'grayscale'.
            skip_corrupted: If True, skip corrupted images instead of raising error.
        """
        self.valid_extensions = [ext.lower() for ext in valid_extensions]
        self.load_mode = load_mode
        self.skip_corrupted = skip_corrupted
    
    def is_valid_image(self, file_path: Path) -> bool:
        """
        Check if file is a valid image based on extension.
        
        Args:
            file_path: Path to file.
            
        Returns:
            True if file has valid image extension.
        """
        return file_path.suffix.lower() in self.valid_extensions
    
    def load_image(self, file_path: Path) -> Optional[Tuple[np.ndarray, dict]]:
        """
        Load an image from disk.
        
        Tries OpenCV first (preserves bit depth), then PIL as fallback.
        
        Args:
            file_path: Path to image file.
            
        Returns:
            Tuple of (image_array, metadata) or None if loading fails.
            metadata contains: width, height, channels, dtype, bit_depth
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        # Try OpenCV first (better for preserving 16-bit thermal data)
        image, metadata = self._load_with_opencv(file_path)
        
        if image is None:
            # Fallback to PIL
            image, metadata = self._load_with_pil(file_path)
        
        if image is None:
            if self.skip_corrupted:
                logger.warning(f"Could not load image: {file_path}")
                return None
            else:
                raise IOError(f"Failed to load image: {file_path}")
        
        return image, metadata
    
    def _load_with_opencv(self, file_path: Path) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """
        Load image using OpenCV.
        
        Args:
            file_path: Path to image file.
            
        Returns:
            Tuple of (image_array, metadata) or (None, None) if failed.
        """
        try:
            # Load based on mode
            if self.load_mode == "grayscale":
                image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            elif self.load_mode == "rgb":
                image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:  # unchanged
                image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            
            if image is None:
                return None, None
            
            # Extract metadata
            metadata = self._extract_metadata(image)
            
            return image, metadata
            
        except Exception as e:
            logger.debug(f"OpenCV failed to load {file_path}: {e}")
            return None, None
    
    def _load_with_pil(self, file_path: Path) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """
        Load image using PIL (fallback method).
        
        Args:
            file_path: Path to image file.
            
        Returns:
            Tuple of (image_array, metadata) or (None, None) if failed.
        """
        try:
            pil_image = Image.open(file_path)
            
            # Convert based on mode
            if self.load_mode == "grayscale":
                pil_image = pil_image.convert('L')
            elif self.load_mode == "rgb":
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            image = np.array(pil_image)
            
            # Extract metadata
            metadata = self._extract_metadata(image)
            
            return image, metadata
            
        except Exception as e:
            logger.debug(f"PIL failed to load {file_path}: {e}")
            return None, None
    
    def _extract_metadata(self, image: np.ndarray) -> dict:
        """
        Extract metadata from loaded image array.
        
        Args:
            image: Image array.
            
        Returns:
            Dictionary with image metadata.
        """
        metadata = {
            'dtype': str(image.dtype),
            'shape': image.shape,
        }
        
        # Determine dimensions
        if len(image.shape) == 2:
            metadata['height'], metadata['width'] = image.shape
            metadata['channels'] = 1
        elif len(image.shape) == 3:
            metadata['height'], metadata['width'], metadata['channels'] = image.shape
        else:
            metadata['height'] = image.shape[0] if len(image.shape) > 0 else 0
            metadata['width'] = image.shape[1] if len(image.shape) > 1 else 0
            metadata['channels'] = 1
        
        # Determine bit depth
        if image.dtype == np.uint8:
            metadata['bit_depth'] = 8
        elif image.dtype == np.uint16:
            metadata['bit_depth'] = 16
        elif image.dtype == np.float32:
            metadata['bit_depth'] = 32
        elif image.dtype == np.float64:
            metadata['bit_depth'] = 64
        else:
            metadata['bit_depth'] = image.itemsize * 8
        
        return metadata
    
    def find_images(self, root_path: Path, recursive: bool = True) -> List[Path]:
        """
        Find all valid image files in a directory.
        
        Args:
            root_path: Root directory to search.
            recursive: If True, search recursively.
            
        Returns:
            List of image file paths.
        """
        image_files = []
        
        if recursive:
            pattern = '**/*'
        else:
            pattern = '*'
        
        for file_path in root_path.glob(pattern):
            if file_path.is_file() and self.is_valid_image(file_path):
                image_files.append(file_path)
        
        return sorted(image_files)
