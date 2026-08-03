"""
Image resizing utilities for preprocessing pipeline.
"""

import logging
from typing import Tuple, Optional
import numpy as np
import cv2

logger = logging.getLogger("preprocessing.resize")


class ImageResizer:
    """
    Handles image resizing with configurable interpolation methods.
    """
    
    # Interpolation method mapping
    INTERPOLATION_METHODS = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA,
        'lanczos': cv2.INTER_LANCZOS4,
    }
    
    def __init__(
        self,
        target_width: int,
        target_height: int,
        interpolation_downsample: str = "area",
        interpolation_upsample: str = "linear",
        preserve_aspect_ratio: bool = False
    ):
        """
        Initialize image resizer.
        
        Args:
            target_width: Target width in pixels.
            target_height: Target height in pixels.
            interpolation_downsample: Interpolation method for downsampling.
            interpolation_upsample: Interpolation method for upsampling.
            preserve_aspect_ratio: If True, preserve aspect ratio with padding.
        """
        self.target_width = target_width
        self.target_height = target_height
        self.preserve_aspect_ratio = preserve_aspect_ratio
        
        # Get interpolation methods
        self.interp_downsample = self._get_interpolation_method(interpolation_downsample)
        self.interp_upsample = self._get_interpolation_method(interpolation_upsample)
    
    def _get_interpolation_method(self, method_name: str) -> int:
        """
        Get OpenCV interpolation constant from method name.
        
        Args:
            method_name: Name of interpolation method.
            
        Returns:
            OpenCV interpolation constant.
            
        Raises:
            ValueError: If method name is invalid.
        """
        method_name = method_name.lower()
        if method_name not in self.INTERPOLATION_METHODS:
            raise ValueError(
                f"Invalid interpolation method: {method_name}. "
                f"Valid options: {list(self.INTERPOLATION_METHODS.keys())}"
            )
        return self.INTERPOLATION_METHODS[method_name]
    
    def resize(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Resize image to target resolution.
        
        Args:
            image: Input image array.
            
        Returns:
            Tuple of (resized_image, metadata).
            metadata contains: original_size, new_size, was_downsampled, was_upsampled
        """
        original_height, original_width = image.shape[:2]
        
        # Determine if downsampling or upsampling
        total_pixels_original = original_width * original_height
        total_pixels_target = self.target_width * self.target_height
        is_downsampling = total_pixels_original > total_pixels_target
        
        # Select appropriate interpolation method
        if is_downsampling:
            interpolation = self.interp_downsample
        else:
            interpolation = self.interp_upsample
        
        # Resize
        if self.preserve_aspect_ratio:
            resized, metadata = self._resize_with_aspect_ratio(
                image, interpolation
            )
        else:
            resized, metadata = self._resize_direct(
                image, interpolation
            )
        
        # Add metadata
        metadata.update({
            'original_width': original_width,
            'original_height': original_height,
            'original_resolution': f"{original_width}x{original_height}",
            'target_width': self.target_width,
            'target_height': self.target_height,
            'target_resolution': f"{self.target_width}x{self.target_height}",
            'was_downsampled': is_downsampling,
            'was_upsampled': not is_downsampling,
        })
        
        return resized, metadata
    
    def _resize_direct(
        self,
        image: np.ndarray,
        interpolation: int
    ) -> Tuple[np.ndarray, dict]:
        """
        Resize image directly to target resolution.
        
        Args:
            image: Input image.
            interpolation: OpenCV interpolation method.
            
        Returns:
            Tuple of (resized_image, metadata).
        """
        resized = cv2.resize(
            image,
            (self.target_width, self.target_height),
            interpolation=interpolation
        )
        
        metadata = {
            'resize_method': 'direct',
            'interpolation': self._get_interpolation_name(interpolation),
        }
        
        return resized, metadata
    
    def _resize_with_aspect_ratio(
        self,
        image: np.ndarray,
        interpolation: int
    ) -> Tuple[np.ndarray, dict]:
        """
        Resize image while preserving aspect ratio, then pad to target size.
        
        Args:
            image: Input image.
            interpolation: OpenCV interpolation method.
            
        Returns:
            Tuple of (resized_image, metadata).
        """
        original_height, original_width = image.shape[:2]
        
        # Calculate scaling factor to fit within target resolution
        scale_w = self.target_width / original_width
        scale_h = self.target_height / original_height
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize
        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=interpolation
        )
        
        # Pad to target size
        if len(image.shape) == 3:
            padded = np.zeros(
                (self.target_height, self.target_width, image.shape[2]),
                dtype=image.dtype
            )
        else:
            padded = np.zeros(
                (self.target_height, self.target_width),
                dtype=image.dtype
            )
        
        # Center the resized image
        y_offset = (self.target_height - new_height) // 2
        x_offset = (self.target_width - new_width) // 2
        
        if len(image.shape) == 3:
            padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width, :] = resized
        else:
            padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        metadata = {
            'resize_method': 'aspect_ratio_preserved',
            'interpolation': self._get_interpolation_name(interpolation),
            'scaled_width': new_width,
            'scaled_height': new_height,
            'padding_x': x_offset,
            'padding_y': y_offset,
        }
        
        return padded, metadata
    
    def _get_interpolation_name(self, interpolation: int) -> str:
        """
        Get name of interpolation method from OpenCV constant.
        
        Args:
            interpolation: OpenCV interpolation constant.
            
        Returns:
            Name of interpolation method.
        """
        for name, value in self.INTERPOLATION_METHODS.items():
            if value == interpolation:
                return name
        return "unknown"
    
    def verify_resolution(self, image: np.ndarray) -> bool:
        """
        Verify that image has target resolution.
        
        Args:
            image: Image array to verify.
            
        Returns:
            True if image matches target resolution.
        """
        height, width = image.shape[:2]
        return width == self.target_width and height == self.target_height
