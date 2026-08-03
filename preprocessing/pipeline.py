"""
Configurable preprocessing pipeline.

This module orchestrates all preprocessing operations based on configuration.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import cv2

from .image_loader import ImageLoader
from .resize import ImageResizer
from .normalization import ImageNormalizer
from .enhancement import (
    HistogramEqualizer,
    CLAHEEnhancer,
    GrayscaleConverter,
    ImagePadder
)

logger = logging.getLogger("preprocessing.pipeline")


class PreprocessingPipeline:
    """
    Configurable preprocessing pipeline that applies operations based on config.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessing pipeline from configuration.
        
        Args:
            config: Preprocessing configuration dictionary.
        """
        self.config = config
        self.preserve_hierarchy = config.get('preserve_hierarchy', True)
        
        # Initialize components based on configuration
        self._init_components()
    
    def _init_components(self):
        """Initialize preprocessing components from config."""
        
        # Resize operation
        resize_config = self.config.get('resize', {})
        self.resize_enabled = resize_config.get('enabled', True)
        
        if self.resize_enabled:
            interp_config = resize_config.get('interpolation', {})
            self.resizer = ImageResizer(
                target_width=resize_config.get('width', 640),
                target_height=resize_config.get('height', 480),
                interpolation_downsample=interp_config.get('downsample', 'area'),
                interpolation_upsample=interp_config.get('upsample', 'linear'),
                preserve_aspect_ratio=resize_config.get('preserve_aspect_ratio', False)
            )
            logger.info(
                f"Resize enabled: {resize_config.get('width')}x{resize_config.get('height')}"
            )
        else:
            self.resizer = None
            logger.info("Resize disabled")
        
        # Grayscale conversion
        grayscale_config = self.config.get('grayscale', {})
        self.grayscale_enabled = grayscale_config.get('enabled', False)
        
        if self.grayscale_enabled:
            self.grayscale_converter = GrayscaleConverter()
            logger.info("Grayscale conversion enabled")
        else:
            self.grayscale_converter = None
        
        # Normalization
        norm_config = self.config.get('normalization', {})
        self.normalization_enabled = norm_config.get('enabled', False)
        
        if self.normalization_enabled:
            self.normalizer = ImageNormalizer(
                method=norm_config.get('method', 'minmax'),
                min_value=norm_config.get('min_value', 0.0),
                max_value=norm_config.get('max_value', 1.0),
                per_image=norm_config.get('per_image', True)
            )
            logger.info(
                f"Normalization enabled: method={norm_config.get('method')}"
            )
        else:
            self.normalizer = None
        
        # Histogram equalization
        histeq_config = self.config.get('histogram_equalization', {})
        self.histeq_enabled = histeq_config.get('enabled', False)
        
        if self.histeq_enabled:
            self.histogram_equalizer = HistogramEqualizer()
            logger.info("Histogram equalization enabled")
        else:
            self.histogram_equalizer = None
        
        # CLAHE
        clahe_config = self.config.get('clahe', {})
        self.clahe_enabled = clahe_config.get('enabled', False)
        
        if self.clahe_enabled:
            self.clahe_enhancer = CLAHEEnhancer(
                clip_limit=clahe_config.get('clip_limit', 2.0),
                tile_grid_size=tuple(clahe_config.get('tile_grid_size', [8, 8]))
            )
            logger.info(
                f"CLAHE enabled: clip_limit={clahe_config.get('clip_limit')}, "
                f"tile_grid_size={clahe_config.get('tile_grid_size')}"
            )
        else:
            self.clahe_enhancer = None
        
        # Padding
        padding_config = self.config.get('padding', {})
        self.padding_enabled = padding_config.get('enabled', False)
        
        if self.padding_enabled:
            self.padder = ImagePadder(
                mode=padding_config.get('mode', 'constant'),
                fill_value=padding_config.get('fill_value', 0)
            )
            logger.info(
                f"Padding enabled: mode={padding_config.get('mode')}"
            )
        else:
            self.padder = None
    
    def process(
        self,
        image: np.ndarray,
        metadata: Dict[str, Any]
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Process an image through all enabled operations.
        
        Operations are applied in this order:
        1. Resize
        2. Grayscale conversion
        3. Histogram equalization
        4. CLAHE
        5. Normalization
        6. Padding
        
        Args:
            image: Input image array.
            metadata: Initial metadata dict (will be updated).
            
        Returns:
            Tuple of (processed_image, updated_metadata).
        """
        processed = image
        
        # Track which operations were applied
        operations_applied = []
        
        # 1. Resize
        if self.resize_enabled and self.resizer is not None:
            processed, resize_meta = self.resizer.resize(processed)
            metadata.update(resize_meta)
            operations_applied.append('resize')
        
        # 2. Grayscale conversion
        if self.grayscale_enabled and self.grayscale_converter is not None:
            processed, gray_meta = self.grayscale_converter.convert(processed)
            metadata.update(gray_meta)
            operations_applied.append('grayscale')
        
        # 3. Histogram equalization (should be before normalization)
        if self.histeq_enabled and self.histogram_equalizer is not None:
            processed, histeq_meta = self.histogram_equalizer.equalize(processed)
            metadata.update(histeq_meta)
            operations_applied.append('histogram_equalization')
        
        # 4. CLAHE (should be before normalization)
        if self.clahe_enabled and self.clahe_enhancer is not None:
            processed, clahe_meta = self.clahe_enhancer.enhance(processed)
            metadata.update(clahe_meta)
            operations_applied.append('clahe')
        
        # 5. Normalization (should be after contrast enhancement)
        if self.normalization_enabled and self.normalizer is not None:
            processed, norm_meta = self.normalizer.normalize(processed)
            metadata.update(norm_meta)
            operations_applied.append('normalization')
        
        # 6. Padding (should be last)
        if self.padding_enabled and self.padder is not None:
            if self.resize_enabled and self.resizer is not None:
                target_w = self.resizer.target_width
                target_h = self.resizer.target_height
            else:
                # If no resize, padding makes no sense without target dimensions
                logger.warning("Padding enabled but no target dimensions available")
                target_w = processed.shape[1]
                target_h = processed.shape[0]
            
            processed, pad_meta = self.padder.pad(processed, target_w, target_h)
            metadata.update(pad_meta)
            operations_applied.append('padding')
        
        # Add summary of operations
        metadata['operations_applied'] = operations_applied
        metadata['num_operations'] = len(operations_applied)
        
        return processed, metadata
    
    def get_enabled_operations(self) -> list[str]:
        """
        Get list of enabled preprocessing operations.
        
        Returns:
            List of operation names.
        """
        operations = []
        
        if self.resize_enabled:
            operations.append('resize')
        if self.grayscale_enabled:
            operations.append('grayscale')
        if self.histeq_enabled:
            operations.append('histogram_equalization')
        if self.clahe_enabled:
            operations.append('clahe')
        if self.normalization_enabled:
            operations.append('normalization')
        if self.padding_enabled:
            operations.append('padding')
        
        return operations


class ConfigurableImageProcessor:
    """
    Image processor that uses configurable preprocessing pipeline.
    """
    
    def __init__(
        self,
        image_loader: ImageLoader,
        pipeline: PreprocessingPipeline,
        verify: bool = True
    ):
        """
        Initialize configurable image processor.
        
        Args:
            image_loader: ImageLoader instance.
            pipeline: PreprocessingPipeline instance.
            verify: If True, verify processed images.
        """
        self.loader = image_loader
        self.pipeline = pipeline
        self.verify = verify
    
    def process_image(
        self,
        input_path: Path,
        output_path: Path,
        overwrite: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single image through the configurable pipeline.
        
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
        
        # Process through pipeline
        try:
            processed_image, processing_metadata = self.pipeline.process(
                image,
                load_metadata.copy()
            )
        except Exception as e:
            logger.error(f"Pipeline processing failed for {input_path}: {e}")
            return None
        
        # Verify if requested
        if self.verify and self.pipeline.resize_enabled:
            if not self.pipeline.resizer.verify_resolution(processed_image):
                logger.error(
                    f"Resolution verification failed for {input_path}: "
                    f"expected {self.pipeline.resizer.target_width}x"
                    f"{self.pipeline.resizer.target_height}, "
                    f"got {processed_image.shape[1]}x{processed_image.shape[0]}"
                )
                return None
        
        # Save processed image
        success = self._save_image(processed_image, output_path)
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
            **processing_metadata,
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
        
        Handles both uint8/uint16 images and float images (from normalization).
        
        Args:
            image: Image array to save.
            output_path: Path to save image.
            
        Returns:
            True if save succeeded.
        """
        try:
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # If image is float, need to convert based on original dtype
            if image.dtype in [np.float32, np.float64]:
                # Check value range to determine output format
                if image.min() >= 0 and image.max() <= 1.0:
                    # Normalized to [0, 1], scale to uint8
                    image_to_save = (image * 255).astype(np.uint8)
                else:
                    # Other float range, clip and scale
                    image_to_save = np.clip(image, 0, 255).astype(np.uint8)
            else:
                image_to_save = image
            
            # Save with OpenCV
            success = cv2.imwrite(str(output_path), image_to_save)
            
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
