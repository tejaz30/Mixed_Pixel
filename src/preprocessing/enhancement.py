"""
Image enhancement operations (histogram equalization, CLAHE).
"""

import logging
from typing import Tuple, Dict, Any
import numpy as np
import cv2

logger = logging.getLogger("preprocessing.enhancement")


class HistogramEqualizer:
    """
    Apply histogram equalization for contrast enhancement.
    """
    
    def __init__(self):
        """Initialize histogram equalizer."""
        pass
    
    def equalize(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply histogram equalization.
        
        For grayscale images: Apply directly
        For color images: Convert to HSV, equalize V channel, convert back
        
        Args:
            image: Input image.
            
        Returns:
            Tuple of (equalized_image, metadata).
        """
        metadata = {
            'operation': 'histogram_equalization',
            'channels': image.shape[2] if len(image.shape) == 3 else 1
        }
        
        # Grayscale image
        if len(image.shape) == 2 or image.shape[2] == 1:
            equalized = cv2.equalizeHist(image)
            metadata['method'] = 'grayscale'
        
        # Color image
        else:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Equalize V channel
            hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
            
            # Convert back to BGR
            equalized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            metadata['method'] = 'hsv_value_channel'
        
        return equalized, metadata


class CLAHEEnhancer:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    """
    
    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8)
    ):
        """
        Initialize CLAHE enhancer.
        
        Args:
            clip_limit: Threshold for contrast limiting.
            tile_grid_size: Size of grid for histogram equalization (width, height).
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tuple(tile_grid_size)
        
        # Create CLAHE object
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )
    
    def enhance(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply CLAHE enhancement.
        
        For grayscale images: Apply directly
        For color images: Convert to HSV, apply to V channel, convert back
        
        Args:
            image: Input image.
            
        Returns:
            Tuple of (enhanced_image, metadata).
        """
        metadata = {
            'operation': 'clahe',
            'clip_limit': self.clip_limit,
            'tile_grid_size': self.tile_grid_size,
            'channels': image.shape[2] if len(image.shape) == 3 else 1
        }
        
        # Grayscale image
        if len(image.shape) == 2 or image.shape[2] == 1:
            enhanced = self.clahe.apply(image)
            metadata['method'] = 'grayscale'
        
        # Color image
        else:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Apply CLAHE to V channel
            hsv[:, :, 2] = self.clahe.apply(hsv[:, :, 2])
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            metadata['method'] = 'hsv_value_channel'
        
        return enhanced, metadata


class GrayscaleConverter:
    """
    Convert color images to grayscale.
    """
    
    def __init__(self):
        """Initialize grayscale converter."""
        pass
    
    def convert(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Convert image to grayscale.
        
        Args:
            image: Input image.
            
        Returns:
            Tuple of (grayscale_image, metadata).
        """
        metadata = {
            'operation': 'grayscale_conversion',
            'original_channels': image.shape[2] if len(image.shape) == 3 else 1
        }
        
        # Already grayscale
        if len(image.shape) == 2 or image.shape[2] == 1:
            metadata['converted'] = False
            return image, metadata
        
        # Convert to grayscale
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        metadata['converted'] = True
        metadata['method'] = 'opencv_BGR2GRAY'
        
        return grayscale, metadata


class ImagePadder:
    """
    Add padding to images.
    """
    
    def __init__(
        self,
        mode: str = "constant",
        fill_value: int = 0
    ):
        """
        Initialize image padder.
        
        Args:
            mode: Padding mode ('constant', 'edge', 'reflect', 'symmetric').
            fill_value: Fill value for constant padding (0-255).
        """
        self.mode = mode.lower()
        self.fill_value = fill_value
        
        # Map to OpenCV border types
        self.border_type_map = {
            'constant': cv2.BORDER_CONSTANT,
            'edge': cv2.BORDER_REPLICATE,
            'reflect': cv2.BORDER_REFLECT,
            'symmetric': cv2.BORDER_REFLECT_101
        }
        
        if self.mode not in self.border_type_map:
            raise ValueError(
                f"Invalid padding mode: {mode}. "
                f"Choose from {list(self.border_type_map.keys())}"
            )
    
    def pad(
        self,
        image: np.ndarray,
        target_width: int,
        target_height: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Pad image to target dimensions.
        
        Args:
            image: Input image.
            target_width: Target width after padding.
            target_height: Target height after padding.
            
        Returns:
            Tuple of (padded_image, metadata).
        """
        h, w = image.shape[:2]
        
        metadata = {
            'operation': 'padding',
            'original_size': (w, h),
            'target_size': (target_width, target_height),
            'padding_mode': self.mode
        }
        
        # Calculate padding amounts
        pad_width = max(0, target_width - w)
        pad_height = max(0, target_height - h)
        
        # Distribute padding evenly
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        
        metadata['padding'] = {
            'left': pad_left,
            'right': pad_right,
            'top': pad_top,
            'bottom': pad_bottom
        }
        
        # No padding needed
        if pad_width == 0 and pad_height == 0:
            metadata['padded'] = False
            return image, metadata
        
        # Apply padding
        border_type = self.border_type_map[self.mode]
        
        if self.mode == 'constant':
            padded = cv2.copyMakeBorder(
                image,
                pad_top, pad_bottom, pad_left, pad_right,
                border_type,
                value=self.fill_value
            )
        else:
            padded = cv2.copyMakeBorder(
                image,
                pad_top, pad_bottom, pad_left, pad_right,
                border_type
            )
        
        metadata['padded'] = True
        return padded, metadata
