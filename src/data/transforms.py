"""
Image transforms for thermal image datasets.

Provides common transforms compatible with both NumPy arrays and PyTorch tensors.
"""

import logging
from typing import Union, Tuple, Optional, List
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

logger = logging.getLogger("data.transforms")


class ToTensor:
    """
    Convert numpy array or PIL Image to PyTorch tensor.
    
    Handles different input formats:
    - HWC numpy array -> CHW tensor
    - Grayscale (H, W) -> (1, H, W) tensor
    - Normalizes to [0, 1] if input is uint8
    """
    
    def __call__(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            return image
        
        # Handle different formats
        if isinstance(image, np.ndarray):
            # Grayscale: (H, W) -> (1, H, W)
            if image.ndim == 2:
                image = image[np.newaxis, :, :]
            # Color: (H, W, C) -> (C, H, W)
            elif image.ndim == 3:
                image = np.transpose(image, (2, 0, 1))
            
            # Convert to tensor
            tensor = torch.from_numpy(image.copy())
            
            # Normalize to [0, 1] if uint8
            if tensor.dtype == torch.uint8:
                tensor = tensor.float() / 255.0
            elif tensor.dtype in [torch.uint16, torch.int16]:
                tensor = tensor.float() / 65535.0
            
            return tensor
        
        raise TypeError(f"Unsupported type: {type(image)}")


class Normalize:
    """
    Normalize tensor with mean and std.
    
    Args:
        mean: Mean for each channel.
        std: Standard deviation for each channel.
    """
    
    def __init__(
        self,
        mean: Union[float, List[float]],
        std: Union[float, List[float]]
    ):
        if isinstance(mean, (int, float)):
            mean = [mean]
        if isinstance(std, (int, float)):
            std = [std]
        
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std


class Resize:
    """
    Resize image to target size.
    
    Args:
        size: Target size as (height, width) or int for square.
        interpolation: Interpolation mode.
    """
    
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        interpolation: str = 'bilinear'
    ):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.interpolation = interpolation
    
    def __call__(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(image, np.ndarray):
            import cv2
            interp_map = {
                'nearest': cv2.INTER_NEAREST,
                'bilinear': cv2.INTER_LINEAR,
                'bicubic': cv2.INTER_CUBIC,
                'area': cv2.INTER_AREA
            }
            # OpenCV uses (width, height)
            return cv2.resize(
                image,
                (self.size[1], self.size[0]),
                interpolation=interp_map.get(self.interpolation, cv2.INTER_LINEAR)
            )
        elif isinstance(image, torch.Tensor):
            return TF.resize(image, self.size)
        else:
            raise TypeError(f"Unsupported type: {type(image)}")


class RandomHorizontalFlip:
    """
    Randomly flip image horizontally.
    
    Args:
        p: Probability of flipping.
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if np.random.rand() < self.p:
            if isinstance(image, np.ndarray):
                return np.fliplr(image).copy()
            elif isinstance(image, torch.Tensor):
                return TF.hflip(image)
        return image


class RandomVerticalFlip:
    """
    Randomly flip image vertically.
    
    Args:
        p: Probability of flipping.
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if np.random.rand() < self.p:
            if isinstance(image, np.ndarray):
                return np.flipud(image).copy()
            elif isinstance(image, torch.Tensor):
                return TF.vflip(image)
        return image


class RandomRotation:
    """
    Randomly rotate image.
    
    Args:
        degrees: Range of degrees as (min, max) or single value for (-degrees, degrees).
    """
    
    def __init__(self, degrees: Union[float, Tuple[float, float]]):
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
    
    def __call__(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        angle = np.random.uniform(self.degrees[0], self.degrees[1])
        
        if isinstance(image, np.ndarray):
            import cv2
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, matrix, (w, h))
        elif isinstance(image, torch.Tensor):
            return TF.rotate(image, angle)
        else:
            raise TypeError(f"Unsupported type: {type(image)}")


class Compose:
    """
    Compose multiple transforms together.
    
    Args:
        transforms: List of transform functions.
    """
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


# Predefined transform pipelines

def get_default_transforms(
    image_size: Tuple[int, int] = (480, 640),
    normalize: bool = True,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> Compose:
    """
    Get default transform pipeline for inference.
    
    Args:
        image_size: Target image size (height, width).
        normalize: Whether to normalize.
        mean: Mean for normalization (per channel).
        std: Std for normalization (per channel).
        
    Returns:
        Composed transforms.
    """
    transforms = [
        ToTensor()
    ]
    
    if normalize:
        if mean is None or std is None:
            # ImageNet defaults (common for pretrained models)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        transforms.append(Normalize(mean=mean, std=std))
    
    return Compose(transforms)


def get_train_transforms(
    image_size: Tuple[int, int] = (480, 640),
    augment: bool = True,
    normalize: bool = True,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> Compose:
    """
    Get transform pipeline for training with optional augmentation.
    
    Args:
        image_size: Target image size (height, width).
        augment: Whether to apply augmentations.
        normalize: Whether to normalize.
        mean: Mean for normalization (per channel).
        std: Std for normalization (per channel).
        
    Returns:
        Composed transforms.
    """
    transforms = []
    
    if augment:
        transforms.extend([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(degrees=15)
        ])
    
    transforms.append(ToTensor())
    
    if normalize:
        if mean is None or std is None:
            # ImageNet defaults
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        transforms.append(Normalize(mean=mean, std=std))
    
    return Compose(transforms)


def get_val_transforms(
    image_size: Tuple[int, int] = (480, 640),
    normalize: bool = True,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> Compose:
    """
    Get transform pipeline for validation/testing (no augmentation).
    
    Args:
        image_size: Target image size (height, width).
        normalize: Whether to normalize.
        mean: Mean for normalization (per channel).
        std: Std for normalization (per channel).
        
    Returns:
        Composed transforms.
    """
    return get_default_transforms(
        image_size=image_size,
        normalize=normalize,
        mean=mean,
        std=std
    )


def get_thermal_transforms(
    image_size: Tuple[int, int] = (480, 640),
    normalize_method: str = 'minmax',
    augment: bool = False
) -> Compose:
    """
    Get transform pipeline optimized for thermal images.
    
    Args:
        image_size: Target image size (height, width).
        normalize_method: 'minmax' for [0,1], 'zscore' for mean=0 std=1, or None.
        augment: Whether to apply augmentations.
        
    Returns:
        Composed transforms.
    """
    transforms = []
    
    if augment:
        transforms.extend([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
        ])
    
    transforms.append(ToTensor())
    
    if normalize_method == 'minmax':
        # Assume images are already [0, 1] after ToTensor
        pass
    elif normalize_method == 'zscore':
        # Use per-channel mean and std for thermal images
        transforms.append(Normalize(mean=[0.5], std=[0.5]))
    
    return Compose(transforms)
