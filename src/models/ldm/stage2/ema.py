"""
Exponential Moving Average (EMA) for model parameters.

Maintains shadow copies of model parameters that are updated using
exponential moving average. EMA weights are used for sampling and evaluation.
"""

import torch
import torch.nn as nn
from typing import Dict
from copy import deepcopy


class EMA:
    """
    Exponential Moving Average of model parameters.
    
    Maintains shadow parameters:
        θ_shadow ← decay * θ_shadow + (1 - decay) * θ_model
    
    Use EMA weights for sampling/evaluation for more stable results.
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: torch.device = None
    ):
        """
        Initialize EMA.
        
        Args:
            model: Model to track
            decay: EMA decay rate (default: 0.9999)
            device: Device for shadow parameters
        """
        self.decay = decay
        self.device = device if device is not None else next(model.parameters()).device
        
        # Create shadow copy of model parameters
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)
        
        # Track number of updates
        self.num_updates = 0
    
    def update(self, model: nn.Module):
        """
        Update EMA shadow parameters.
        
        Args:
            model: Current model with updated parameters
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name] = (
                        self.decay * self.shadow[name] +
                        (1 - self.decay) * param.data.to(self.device)
                    )
        
        self.num_updates += 1
    
    def apply_shadow(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Replace model parameters with EMA shadow parameters.
        
        Returns original parameters for restoration.
        
        Args:
            model: Model to apply EMA to
        
        Returns:
            backup: Original parameter values
        """
        backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    backup[name] = param.data.clone()
                    param.data.copy_(self.shadow[name].to(param.device))
        
        return backup
    
    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]):
        """
        Restore original model parameters from backup.
        
        Args:
            model: Model to restore
            backup: Original parameter values from apply_shadow
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in backup:
                    param.data.copy_(backup[name])
    
    def state_dict(self) -> Dict:
        """
        Get EMA state for checkpointing.
        
        Returns:
            State dict with shadow parameters and num_updates
        """
        return {
            'shadow': deepcopy(self.shadow),
            'decay': self.decay,
            'num_updates': self.num_updates
        }
    
    def load_state_dict(self, state_dict: Dict):
        """
        Load EMA state from checkpoint.
        
        Args:
            state_dict: EMA state dict
        """
        self.shadow = state_dict['shadow']
        self.decay = state_dict.get('decay', self.decay)
        self.num_updates = state_dict.get('num_updates', 0)
    
    def __repr__(self) -> str:
        return f'EMA(decay={self.decay}, num_updates={self.num_updates})'
