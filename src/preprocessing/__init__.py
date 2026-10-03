"""
Preprocessing pipeline for thermal image mixed-pixel detection.

This package contains modules for:
- Dataset auditing
- Deduplication
- Image preprocessing (resizing, etc.)
- Normalization (future)
- Augmentation (future)
- Patch generation (future)
- Dataset splitting (future)
"""

# Deduplication modules
from .deduplicator import DatasetDeduplicator, DeduplicationResult
from .hash_computer import HashComputer
from .deduplication_report import DeduplicationReportGenerator

# Preprocessing modules
from .image_loader import ImageLoader
from .resize import ImageResizer
from .image_processor import ImageProcessor
from .metadata import MetadataManager
from .normalization import ImageNormalizer
from .enhancement import (
    HistogramEqualizer,
    CLAHEEnhancer,
    GrayscaleConverter,
    ImagePadder
)
from .pipeline import PreprocessingPipeline, ConfigurableImageProcessor
from .validate import DatasetValidator, ValidationResult, ValidationReportGenerator

# Utilities
from .utils import (
    setup_logging,
    load_config,
    compute_file_hash,
    find_all_images,
    get_relative_path,
    format_bytes
)

__all__ = [
    # Deduplication
    'DatasetDeduplicator',
    'DeduplicationResult',
    'HashComputer',
    'DeduplicationReportGenerator',
    
    # Preprocessing
    'ImageLoader',
    'ImageResizer',
    'ImageProcessor',
    'MetadataManager',
    'ImageNormalizer',
    'HistogramEqualizer',
    'CLAHEEnhancer',
    'GrayscaleConverter',
    'ImagePadder',
    'PreprocessingPipeline',
    'ConfigurableImageProcessor',
    'DatasetValidator',
    'ValidationResult',
    'ValidationReportGenerator',
    
    # Utilities
    'setup_logging',
    'load_config',
    'compute_file_hash',
    'find_all_images',
    'get_relative_path',
    'format_bytes',
]

__version__ = '2.0.0'
