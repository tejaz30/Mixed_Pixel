"""
Main image processing pipeline.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2

from .image_loader import ImageLoader
from .resize import ImageResizer

logger = logging.getLogger("preprocessing.processor")


class ImageProcessor:
    """
    Orchestrates the image preprocessing pipeline.
    """
    
    def __init__(
        self,
        image_loader: ImageLoader,
        image_resizer: ImageResizer,
        verify: bool = True
    ):
        """
        Initialize image processor.
        
        Args:
            image_loader: ImageLoader instance.
            image_resizer: ImageResizer instance.
            verify: If True, verify processed images.
        """
        self.loader = image_loader
        self.resizer = image_resizer
        self.verify = verify
    
    def process_image(
        self,
        input_path: Path,
        output_path: Path,
        overwrite: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single image through the pipeline.
        
        Args:
            input_path: Path to input image.
            output_path: Path to save processed image.
            overwrite: If True, overwrite existing output.
            
        Returns:
            Dictionary with processing metadata or None if failed.
        """
        start_time = time.time()
        
        # Check if output already exists
        if output_path.exists() and not overwrite:
            logger.debug(f"Skipping existing file: {output_path}")
            return None
        
        # Load image
        result = self.loader.load_image(input_path)
        if result is None:
            logger.error(f"Failed to load image: {input_path}")
            return None
        
        image, load_metadata = result
        
        # Resize image
        resized_image, resize_metadata = self.resizer.resize(image)
        
        # Verify resolution if requested
        if self.verify:
            if not self.resizer.verify_resolution(resized_image):
                logger.error(
                    f"Resolution verification failed for {input_path}: "
                    f"expected {self.resizer.target_width}x{self.resizer.target_height}, "
                    f"got {resized_image.shape[1]}x{resized_image.shape[0]}"
                )
                return None
        
        # Save processed image
        success = self._save_image(resized_image, output_path)
        if not success:
            logger.error(f"Failed to save image: {output_path}")
            return None
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Combine metadata
        metadata = {
            'input_path': str(input_path),
            'output_path': str(output_path),
            'processing_time': processing_time,
            **load_metadata,
            **resize_metadata,
        }
        
        # Add file sizes
        try:
            metadata['file_size_original'] = input_path.stat().st_size
            metadata['file_size_processed'] = output_path.stat().st_size
        except Exception as e:
            logger.warning(f"Could not get file sizes: {e}")
        
        return metadata
    
    def _save_image(self, image: np.ndarray, output_path: Path) -> bool:
        """
        Save processed image to disk.
        
        Args:
            image: Image array to save.
            output_path: Path to save image.
            
        Returns:
            True if save succeeded.
        """
        try:
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with OpenCV (preserves bit depth)
            success = cv2.imwrite(str(output_path), image)
            
            if not success:
                logger.error(f"OpenCV failed to write image: {output_path}")
                return False
            
            # Verify file was created
            if not output_path.exists():
                logger.error(f"Output file was not created: {output_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving image {output_path}: {e}")
            return False
    
    def verify_processed_image(self, output_path: Path) -> bool:
        """
        Verify that a processed image is readable and has correct resolution.
        
        Args:
            output_path: Path to processed image.
            
        Returns:
            True if image is valid.
        """
        try:
            # Try to load the image
            result = self.loader.load_image(output_path)
            if result is None:
                logger.error(f"Processed image is not readable: {output_path}")
                return False
            
            image, metadata = result
            
            # Verify resolution
            if not self.resizer.verify_resolution(image):
                logger.error(
                    f"Processed image has wrong resolution: {output_path} "
                    f"({metadata['width']}x{metadata['height']})"
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying processed image {output_path}: {e}")
            return False
