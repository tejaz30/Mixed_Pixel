"""
Vector Quantizer for VQ-regularized autoencoder.

Implements the vector quantization mechanism with exponential moving average (EMA)
updates as described in the VQ-VAE papers and used in Rombach et al. LDM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with EMA updates.
    
    Maps continuous encoder outputs to discrete codebook entries.
    Uses exponential moving average for codebook updates.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5
    ):
        """
        Initialize vector quantizer.
        
        Args:
            num_embeddings: Size of the codebook (number of discrete codes).
            embedding_dim: Dimension of each codebook entry.
            commitment_cost: Weight for commitment loss (beta in VQ-VAE).
            decay: Decay rate for EMA updates.
            epsilon: Small constant for numerical stability.
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
        # EMA parameters
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_embed_avg', self.embedding.weight.data.clone())
        
        self.decay = decay
        self.epsilon = epsilon
    
    def forward(
        self,
        z: torch.Tensor,
        compute_loss: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Quantize continuous latent representation.
        
        Args:
            z: Continuous latent of shape (B, C, H, W).
            compute_loss: Whether to compute and return VQ losses.
        
        Returns:
            Tuple of (quantized_latent, loss_dict)
        """
        # Flatten spatial dimensions: (B, C, H, W) -> (B*H*W, C)
        z_flattened = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z_flattened.view(-1, self.embedding_dim)
        
        # Compute distances to codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2<z, e>
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )
        
        # Find nearest codebook entry for each latent
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(z_flattened.dtype)
        
        # Quantize using nearest codebook entry
        z_quantized_flattened = torch.matmul(encodings, self.embedding.weight)
        
        # Reshape back to spatial dimensions
        z_quantized = z_quantized_flattened.view(z.permute(0, 2, 3, 1).shape)
        z_quantized = z_quantized.permute(0, 3, 1, 2).contiguous()
        
        # Compute losses if requested
        loss_dict = {}
        if compute_loss:
            # Commitment loss: ||sg[z] - e||^2
            # Encourages encoder to commit to codebook entries
            commitment_loss = F.mse_loss(z_quantized.detach(), z)
            
            # Codebook loss: ||z - sg[e]||^2
            # Not used with EMA updates, but compute for monitoring
            codebook_loss = F.mse_loss(z_quantized, z.detach())
            
            # Total VQ loss
            vq_loss = codebook_loss + self.commitment_cost * commitment_loss
            
            loss_dict = {
                'vq_loss': vq_loss,
                'codebook_loss': codebook_loss,
                'commitment_loss': commitment_loss
            }
        
        # Straight-through estimator
        # Forward: use quantized values
        # Backward: copy gradients from decoder to encoder
        z_quantized = z + (z_quantized - z).detach()
        
        # Update EMA statistics during training
        if self.training and compute_loss:
            with torch.no_grad():
                # Update cluster sizes
                encodings_sum = encodings.sum(0)
                self.ema_cluster_size.mul_(self.decay).add_(
                    encodings_sum, alpha=1 - self.decay
                )
                
                # Update embeddings
                embed_sum = torch.matmul(encodings.t(), z_flattened)
                self.ema_embed_avg.mul_(self.decay).add_(
                    embed_sum, alpha=1 - self.decay
                )
                
                # Laplace smoothing
                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon) * n
                )
                
                # Update codebook
                normalized_embed = self.ema_embed_avg / cluster_size.unsqueeze(1)
                self.embedding.weight.data.copy_(normalized_embed)
        
        # Compute perplexity for monitoring
        if compute_loss:
            avg_probs = encodings.mean(0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            
            # Codebook utilization
            active_codes = (encodings.sum(0) > 0).sum().float()
            utilization = active_codes / self.num_embeddings
            
            loss_dict['perplexity'] = perplexity
            loss_dict['codebook_utilization'] = utilization
            loss_dict['active_codes'] = active_codes
        
        return z_quantized, loss_dict
    
    def get_codebook_entry(self, indices: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Get codebook entries for given indices.
        
        Args:
            indices: Codebook indices of shape (B, H, W).
            shape: Desired output shape (B, C, H, W).
        
        Returns:
            Quantized tensor of specified shape.
        """
        # Get embeddings
        z_quantized = self.embedding(indices)
        
        # Reshape
        z_quantized = z_quantized.view(shape)
        z_quantized = z_quantized.permute(0, 3, 1, 2).contiguous()
        
        return z_quantized
