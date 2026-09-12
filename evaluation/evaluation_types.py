"""
Evaluation data types.

Shared type definitions to avoid circular dependencies between evaluation modules.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EvaluationResult:
    """
    Container for per-image evaluation results.
    
    Attributes:
        dataset: Dataset name (e.g., 'Chilli_Leaves', 'Okra').
        image_path: Relative path to the image.
        mse: Mean Squared Error.
        ssim: Structural Similarity Index Measure.
        error_map: Optional absolute error map (numpy array).
    """
    dataset: str
    image_path: str
    mse: float
    ssim: float
    error_map: Optional[np.ndarray] = None
    
    def to_dict(self, include_error_map: bool = False) -> dict:
        """Convert to dictionary for saving."""
        result = {
            'dataset': self.dataset,
            'image_path': self.image_path,
            'mse': self.mse,
            'ssim': self.ssim
        }
        if include_error_map and self.error_map is not None:
            result['error_map'] = self.error_map
        return result
