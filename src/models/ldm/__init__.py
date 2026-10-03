"""
Latent Diffusion Model (LDM) components.

This module contains first-stage autoencoders for compression before diffusion.
"""

from .first_stage.kl.model import KLAutoencoder
from .first_stage.vq.model import VQAutoencoder

__all__ = [
    'KLAutoencoder',
    'VQAutoencoder',
]
