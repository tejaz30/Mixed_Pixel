# LDM First-Stage Autoencoders - Code Review Documentation

## Overview

This document details the implementation of KL-regularized and VQ-regularized first-stage autoencoders based on Rombach et al.'s "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022).

**Purpose**: Controlled comparison of latent regularization strategies for thermal image compression BEFORE introducing diffusion models.

---

## 1. Conceptual Background

### 1.1 Latent Diffusion Models (LDM) Architecture

```
Full LDM Pipeline (not implemented yet):

Thermal Image (640×480)
    ↓
[FIRST STAGE: Autoencoder] ← THIS IMPLEMENTATION
    Encoder → Latent (60×80) → Decoder
    ↓
[SECOND STAGE: Diffusion Model] ← FUTURE WORK
    Latent diffusion for generation/manipulation
    ↓
Reconstructed/Generated Image
```

### 1.2 Why First-Stage Matters

**Motivation from Rombach et al.**:
1. **Perceptual compression**: Remove high-frequency imperceptible details
2. **Computational efficiency**: Diffusion on low-dimensional latents
3. **Better sampling**: Continuous/discrete latent representations

**Our Research Question**:
> "How do KL-regularized and VQ-regularized latent representations differ for thermal imagery, particularly at thermal boundaries?"

### 1.3 Experiment 1 Scope

**What We're Comparing**:
- KL-regularized autoencoder (continuous latent)
- VQ-regularized autoencoder (discrete latent)

**What We're NOT Doing**:
- ❌ Diffusion models
- ❌ Conditioning mechanisms
- ❌ Attention layers
- ❌ Perceptual losses
- ❌ Architectural variations

**Controlled Variables**:
- Same encoder/decoder architecture
- Same spatial compression (8×)
- Same optimizer (Adam, lr=1e-3)
- Same dataset and splits
- Same training protocol

---

## 2. Project Structure

```
models/ldm/
├── __init__.py
└── first_stage/
    ├── __init__.py
    ├── encoder.py           # Shared encoder (both KL and VQ)
    ├── decoder.py           # Shared decoder (both KL and VQ)
    ├── kl/
    │   ├── __init__.py
    │   ├── model.py         # KL autoencoder
    │   └── loss.py          # KL loss (MSE + KL divergence)
    └── vq/
        ├── __init__.py
        ├── model.py         # VQ autoencoder
        ├── vector_quantizer.py  # VQ mechanism with EMA
        └── loss.py          # VQ loss (MSE + codebook + commitment)

configs/experiment1/
├── kl_autoencoder.yaml      # KL configuration
└── vq_autoencoder.yaml      # VQ configuration

scripts/
├── train_first_stage.py     # Unified training script
├── evaluate_first_stage.py  # Evaluation with boundary metrics
├── compare_experiment1.py   # KL vs VQ comparison
├── run_experiment1.py       # Complete pipeline runner
└── test_experiment1_models.py  # Model validation

evaluation/
└── boundary_metrics.py      # Gradient-based boundary analysis
```

---

## 3. Shared Architecture Components

### 3.1 Encoder (`models/ldm/first_stage/encoder.py`)

**Philosophy**: ResNet-inspired blocks with progressive downsampling

```python
class Encoder(nn.Module):
    """
    Encoder for first-stage autoencoders
    
    Architecture:
        Input (3, 480, 640)
          ↓
        Initial Conv: 3 → 128 channels
          ↓
        Level 1: 128 channels, ResBlocks × 2
        ↓ Downsample (2×)
        Level 2: 256 channels, ResBlocks × 2
        ↓ Downsample (2×)
        Level 3: 512 channels, ResBlocks × 2
        ↓ Downsample (2×)
        Level 4: 512 channels, ResBlocks × 2
          ↓
        Middle: ResBlocks × 2
          ↓
        Output Conv: 512 → latent_channels
          ↓
        Latent (latent_channels, 60, 80)
    
    For KL: latent_channels = 2×4 = 8 (mean + logvar)
    For VQ: latent_channels = 4 (continuous pre-quantization)
    """
```

**Key Components**:

```python
class ResBlock(nn.Module):
    """
    Residual block with two convolutions
    
    Architecture:
        x → GroupNorm → SiLU → Conv → GroupNorm → SiLU → Conv → + → out
        ↓_______________________________________________|
                      (skip connection)
    """
    def __init__(self, in_channels, out_channels):
        self.norm1 = GroupNorm(32, in_channels)
        self.conv1 = Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = GroupNorm(32, out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = Conv2d(in_channels, out_channels, 1) if in_channels != out_channels


class DownsampleBlock(nn.Module):
    """Strided convolution for downsampling"""
    def __init__(self, channels):
        self.conv = Conv2d(channels, channels, 3, stride=2, padding=0)
    
    def forward(self, x):
        # Pad to ensure proper dimensions
        x = F.pad(x, (0, 1, 0, 1), mode='constant', value=0)
        return self.conv(x)
```

**Design Rationale**:
- **GroupNorm instead of BatchNorm**: More stable with small batches
- **SiLU (Swish) activation**: Smoother gradients than ReLU
- **Progressive channel expansion**: 128 → 256 → 512 → 512
- **ResBlocks for gradient flow**: Enables deeper networks
- **Strided convolutions for downsampling**: Learnable vs fixed pooling

### 3.2 Decoder (`models/ldm/first_stage/decoder.py`)

**Philosophy**: Mirror encoder with upsampling

```python
class Decoder(nn.Module):
    """
    Decoder for first-stage autoencoders
    
    Architecture:
        Latent (latent_channels, 60, 80)
          ↓
        Initial Conv: latent_channels → 512
          ↓
        Middle: ResBlocks × 2
          ↓
        Level 4: 512 channels, ResBlocks × 2
        ↑ Upsample (2×)
        Level 3: 512 channels, ResBlocks × 2
        ↑ Upsample (2×)
        Level 2: 256 channels, ResBlocks × 2
        ↑ Upsample (2×)
        Level 1: 128 channels, ResBlocks × 2
          ↓
        Output Conv: 128 → 3
          ↓
        Output (3, 480, 640)
    """
```

**Key Components**:

```python
class UpsampleBlock(nn.Module):
    """Nearest-neighbor interpolation + convolution"""
    def __init__(self, channels):
        self.conv = Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)
```

**Design Rationale**:
- **Nearest-neighbor + conv**: Avoids checkerboard artifacts of transposed conv
- **Progressive channel reduction**: 512 → 512 → 256 → 128
- **No Sigmoid output**: Raw output (loss handles range)
- **Symmetric with encoder**: Facilitates information flow

---

## 4. KL-Regularized Autoencoder

### 4.1 Model Architecture (`models/ldm/first_stage/kl/model.py`)

```python
class KLAutoencoder(nn.Module):
    """
    KL-regularized continuous latent autoencoder
    
    Key Idea: Learn a Gaussian posterior q(z|x) that's regularized
              toward a standard normal prior p(z) = N(0, I)
    
    Forward Pass:
        x → Encoder → [μ, log σ²] → Sample z ~ N(μ, σ²) → Decoder → x̂
    
    Loss:
        L = ||x - x̂||² + β * KL(q(z|x) || p(z))
        
        where β = kl_weight (typically 1e-6)
    """
```

**Latent Representation**:

```python
class DiagonalGaussianDistribution:
    """
    Parameterizes q(z|x) as diagonal Gaussian
    
    Encoder outputs: (B, 2*latent_channels, 60, 80)
                   = (B, 8, 60, 80) for latent_channels=4
    
    Split into:
        mean:   (B, 4, 60, 80)
        logvar: (B, 4, 60, 80)
    
    Sample via reparameterization:
        z = μ + σ * ε,  where ε ~ N(0, I)
    """
    
    def __init__(self, parameters):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)  # Stability
        self.std = torch.exp(0.5 * self.logvar)
    
    def sample(self):
        """Reparameterization trick for backprop through sampling"""
        epsilon = torch.randn_like(self.mean)
        return self.mean + self.std * epsilon
    
    def mode(self):
        """Deterministic encoding (inference)"""
        return self.mean
    
    def kl(self):
        """
        KL divergence to standard normal:
        KL(q||p) = 0.5 * Σ(μ² + σ² - 1 - log σ²)
        """
        return 0.5 * torch.sum(
            self.mean.pow(2) + self.var - 1.0 - self.logvar,
            dim=[1, 2, 3]
        )
```

**Model Methods**:

```python
def encode(self, x):
    """x → posterior distribution q(z|x)"""
    h = self.encoder(x)
    return DiagonalGaussianDistribution(h)

def decode(self, z):
    """z → reconstruction x̂"""
    return self.decoder(z)

def forward(self, x, sample=True):
    """
    Complete forward pass
    
    Args:
        x: Input (B, 3, 480, 640)
        sample: If True, sample z. If False, use mean (deterministic)
    
    Returns:
        reconstruction: (B, 3, 480, 640)
        posterior: DiagonalGaussianDistribution
    """
    posterior = self.encode(x)
    z = posterior.sample() if sample else posterior.mode()
    reconstruction = self.decode(z)
    return reconstruction, posterior
```

### 4.2 KL Loss (`models/ldm/first_stage/kl/loss.py`)

```python
class KLAutoencoderLoss(nn.Module):
    """
    Combined reconstruction + KL regularization loss
    
    L_total = L_recon + kl_weight * L_KL
    
    where:
        L_recon = MSE(x, x̂)
        L_KL = KL(q(z|x) || N(0, I))
        kl_weight = 1e-6 (from Rombach et al.)
    """
    
    def __init__(self, kl_weight=1e-6):
        super().__init__()
        self.kl_weight = kl_weight
    
    def forward(self, reconstruction, target, posterior):
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, target, reduction='mean')
        
        # KL divergence loss
        kl_loss = posterior.kl().mean()
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'kl': kl_loss,
            'weighted_kl': self.kl_weight * kl_loss
        }
```

**KL Weight Rationale**:
- **Large weight** (e.g., 1.0): Strong regularization → posterior collapse
- **Small weight** (e.g., 1e-6): Weak regularization → good reconstruction
- **Trade-off**: Balance between compression and reconstruction quality

### 4.3 KL Latent Space Properties

**Characteristics**:
- **Continuous**: z ∈ ℝ^(4×60×80)
- **Gaussian**: Each dimension ~ N(μ, σ²)
- **Smooth interpolation**: Interpolating in latent space is meaningful
- **Regularized**: Pushed toward N(0, I), prevents overfitting
- **Stochastic during training**: Sampling provides regularization
- **Deterministic during inference**: Use mean for consistency

---

## 5. VQ-Regularized Autoencoder

### 5.1 Model Architecture (`models/ldm/first_stage/vq/model.py`)

```python
class VQAutoencoder(nn.Module):
    """
    VQ-regularized discrete latent autoencoder
    
    Key Idea: Quantize continuous encoder outputs to discrete
              codebook entries, learning a discrete representation
    
    Forward Pass:
        x → Encoder → z_e → Quantizer → z_q → Decoder → x̂
                               ↑
                            Codebook
    
    Loss:
        L = ||x - x̂||² + ||z_e - sg[z_q]||² + β||sg[z_e] - z_q||²
        
        where:
            - First term: Reconstruction loss
            - Second term: Commitment loss (β = 0.25)
            - Third term: Codebook loss (handled by EMA updates)
            sg[·] = stop_gradient
    """
```

**Model Components**:

```python
def __init__(self, ...):
    # Encoder
    self.encoder = Encoder(...)
    
    # Pre-quantization conv (align representation)
    self.quant_conv = Conv2d(latent_channels, latent_channels, 1)
    
    # Vector Quantizer
    self.quantizer = VectorQuantizer(
        num_embeddings=1024,      # Codebook size
        embedding_dim=4,          # Latent channels
        commitment_cost=0.25      # β in commitment loss
    )
    
    # Post-quantization conv
    self.post_quant_conv = Conv2d(latent_channels, latent_channels, 1)
    
    # Decoder
    self.decoder = Decoder(...)

def encode(self, x):
    """x → quantized latent"""
    h = self.encoder(x)
    h = self.quant_conv(h)
    z_quantized, vq_losses = self.quantizer(h, compute_loss=True)
    return z_quantized, h, vq_losses

def decode(self, z):
    """z → reconstruction"""
    z = self.post_quant_conv(z)
    return self.decoder(z)

def forward(self, x):
    """Complete forward pass"""
    z_quantized, z_continuous, vq_losses = self.encode(x)
    reconstruction = self.decode(z_quantized)
    return reconstruction, vq_losses
```

### 5.2 Vector Quantizer (`models/ldm/first_stage/vq/vector_quantizer.py`)

**Core Algorithm**:

```python
class VectorQuantizer(nn.Module):
    """
    Vector Quantization with EMA updates
    
    Codebook: Learnable embedding matrix (num_embeddings × embedding_dim)
              e.g., (1024 × 4)
    
    Quantization Process:
        1. Flatten spatial dims: z_e (B,C,H,W) → (B*H*W, C)
        2. Find nearest codebook entry for each vector
        3. Replace with codebook vector: z_q
        4. Reshape back: (B*H*W, C) → (B,C,H,W)
        5. Use straight-through estimator for gradients
    """
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        
        # Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0/num_embeddings, 1.0/num_embeddings
        )
        
        # EMA parameters (for codebook updates)
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_embed_avg', self.embedding.weight.data.clone())
        
        self.commitment_cost = commitment_cost
        self.decay = 0.99  # EMA decay rate
    
    def forward(self, z, compute_loss=True):
        """
        Quantize continuous latent z
        
        Args:
            z: (B, C, H, W) continuous latent
        
        Returns:
            z_q: (B, C, H, W) quantized latent
            losses: Dict of VQ losses
        """
        # Flatten spatial dimensions
        z_flat = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z_flat.view(-1, self.embedding_dim)
        
        # Compute distances to codebook entries
        # ||z - e||² = ||z||² + ||e||² - 2⟨z, e⟩
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )
        
        # Find nearest codebook entry
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Quantize
        z_q_flat = torch.matmul(encodings, self.embedding.weight)
        
        # Reshape back to spatial dimensions
        z_q = z_q_flat.view(z.permute(0, 2, 3, 1).shape)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        # Compute losses
        if compute_loss:
            # Commitment loss: encoder commits to codebook
            commitment_loss = F.mse_loss(z_q.detach(), z)
            
            # Codebook loss: codebook learns encoder outputs
            codebook_loss = F.mse_loss(z_q, z.detach())
            
            # Total VQ loss
            vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator
        # Forward: use quantized z_q
        # Backward: copy gradients from decoder to encoder
        z_q = z + (z_q - z).detach()
        
        # Update codebook with EMA (during training)
        if self.training and compute_loss:
            self._ema_update(encodings, z_flat)
        
        # Compute perplexity (codebook utilization metric)
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        losses = {
            'vq_loss': vq_loss,
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss,
            'perplexity': perplexity,
            'codebook_utilization': (encodings.sum(0) > 0).sum() / self.num_embeddings
        }
        
        return z_q, losses
```

**EMA Codebook Updates**:

```python
def _ema_update(self, encodings, z_flat):
    """
    Update codebook using exponential moving average
    
    Instead of gradient descent on codebook, use EMA:
        - More stable than gradient-based updates
        - Prevents codebook collapse
        - Laplace smoothing for unused codes
    """
    with torch.no_grad():
        # Update cluster sizes
        encodings_sum = encodings.sum(0)
        self.ema_cluster_size.mul_(self.decay).add_(
            encodings_sum, alpha=1 - self.decay
        )
        
        # Update embeddings
        embed_sum = torch.matmul(encodings.t(), z_flat)
        self.ema_embed_avg.mul_(self.decay).add_(
            embed_sum, alpha=1 - self.decay
        )
        
        # Laplace smoothing
        n = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + self.epsilon)
            / (n + self.num_embeddings * self.epsilon) * n
        )
        
        # Normalize and update codebook
        normalized_embed = self.ema_embed_avg / cluster_size.unsqueeze(1)
        self.embedding.weight.data.copy_(normalized_embed)
```

### 5.3 VQ Loss (`models/ldm/first_stage/vq/loss.py`)

```python
class VQAutoencoderLoss(nn.Module):
    """
    Combined reconstruction + VQ regularization loss
    
    L_total = L_recon + L_VQ
    
    where:
        L_recon = MSE(x, x̂)
        L_VQ = ||z_e - sg[z_q]||² + β||sg[z_e] - z_q||²
               (already computed by VectorQuantizer)
    """
    
    def forward(self, reconstruction, target, vq_losses):
        recon_loss = F.mse_loss(reconstruction, target)
        vq_loss = vq_losses['vq_loss']
        total_loss = recon_loss + vq_loss
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'vq_loss': vq_loss,
            'codebook_loss': vq_losses['codebook_loss'],
            'commitment_loss': vq_losses['commitment_loss'],
            'perplexity': vq_losses['perplexity'],
            'codebook_utilization': vq_losses['codebook_utilization']
        }
```

### 5.4 VQ Latent Space Properties

**Characteristics**:
- **Discrete**: z_q drawn from finite codebook (1024 entries)
- **Deterministic**: No sampling, always nearest neighbor
- **Structured**: Codebook learns common patterns in data
- **Perplexity**: Measures effective codebook usage (ideal ≈ 1024)
- **Utilization**: Fraction of codebook actually used
- **No posterior collapse**: Discrete space prevents over-compression

---

## 6. Training Infrastructure (`scripts/train_first_stage.py`)

### 6.1 Unified Training Script

**Key Feature**: Single script handles both KL and VQ models

```python
def create_model(config):
    """Factory pattern for model creation"""
    model_type = config['model']['type']
    
    if model_type == 'kl_autoencoder':
        return KLAutoencoder(...)
    elif model_type == 'vq_autoencoder':
        return VQAutoencoder(...)

def create_criterion(config):
    """Factory pattern for loss creation"""
    model_type = config['model']['type']
    
    if model_type == 'kl_autoencoder':
        return KLAutoencoderLoss(kl_weight=config['model']['kl_weight'])
    elif model_type == 'vq_autoencoder':
        return VQAutoencoderLoss()
```

### 6.2 Training Loop

```python
def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    epoch_losses = {}
    
    for batch in train_loader:
        images = batch['image'].to(device)
        
        # Forward pass (model-specific)
        if isinstance(model, KLAutoencoder):
            reconstruction, posterior = model(images, sample=True)
            losses = criterion(reconstruction, images, posterior)
        elif isinstance(model, VQAutoencoder):
            reconstruction, vq_outputs = model(images)
            losses = criterion(reconstruction, images, vq_outputs)
        
        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
        
        # Accumulate losses
        for key, value in losses.items():
            epoch_losses[key] = epoch_losses.get(key, 0) + value.item()
    
    return epoch_losses

def validate(model, val_loader, criterion, device):
    """Validation loop"""
    model.eval()
    val_losses = {}
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            
            # Forward pass (use deterministic mode)
            if isinstance(model, KLAutoencoder):
                reconstruction, posterior = model(images, sample=False)
                losses = criterion(reconstruction, images, posterior)
            elif isinstance(model, VQAutoencoder):
                reconstruction, vq_outputs = model(images)
                losses = criterion(reconstruction, images, vq_outputs)
            
            # Accumulate
            for key, value in losses.items():
                val_losses[key] = val_losses.get(key, 0) + value.item()
    
    return val_losses
```

### 6.3 Configuration

**Shared Configuration** (`configs/experiment1/`):

```yaml
# Common to both KL and VQ
model:
  in_channels: 3
  latent_channels: 4
  base_channels: 64          # Reduced from 128 for 6GB GPU
  channel_multipliers: [1, 2, 4, 4]
  num_res_blocks: 2

data:
  batch_size: 2              # Reduced for memory
  train_ratio: 0.8
  val_ratio: 0.2
  seed: 42

training:
  max_epochs: 100
  learning_rate: 1.0e-3
  optimizer: "adam"
  early_stopping:
    patience: 5

reproducibility:
  deterministic: true
  seed: 42
```

**KL-Specific** (`kl_autoencoder.yaml`):
```yaml
model:
  type: "kl_autoencoder"
  kl_weight: 1.0e-6          # From Rombach et al.
```

**VQ-Specific** (`vq_autoencoder.yaml`):
```yaml
model:
  type: "vq_autoencoder"
  num_embeddings: 1024       # Codebook size
  commitment_cost: 0.25      # β in commitment loss
```

---

## 7. Evaluation Framework (`scripts/evaluate_first_stage.py`)

### 7.1 Global Metrics

```python
def evaluate_global_metrics(model, dataloader, device):
    """
    Standard reconstruction metrics:
    
    - MSE: Mean Squared Error
    - MAE: Mean Absolute Error
    - SSIM: Structural Similarity Index
    
    Returns statistics: mean, std, median, min, max
    """
```

### 7.2 Boundary-Focused Metrics (`evaluation/boundary_metrics.py`)

**Critical Component**: Since we lack ground-truth boundary annotations

```python
def compute_gradient_magnitude(image, method='sobel'):
    """
    Compute gradient magnitude using Sobel operator
    
    Steps:
        1. Convert RGB to grayscale (luminance weighted)
        2. Apply Sobel filter (x and y directions)
        3. Compute magnitude: sqrt(g_x² + g_y²)
    
    Returns: (H, W) gradient magnitude map
    """

def create_boundary_mask(gradient_magnitude, threshold_percentile=75.0):
    """
    Create binary boundary mask from gradients
    
    High-gradient regions (75th percentile and above) are
    treated as CANDIDATE boundary regions, NOT verified
    mixed-pixel locations.
    
    Returns: (H, W) binary mask (1 = boundary, 0 = non-boundary)
    """

def compute_boundary_reconstruction_metrics(original, reconstruction, boundary_mask):
    """
    Separate MSE/MAE for boundary vs non-boundary regions
    
    Returns:
        - boundary_mse: Error in high-gradient regions
        - non_boundary_mse: Error in smooth regions
        - error_ratio: boundary_mse / non_boundary_mse
        - boundary_fraction: % of pixels in boundary regions
    """

def compute_gradient_preservation(original, reconstruction):
    """
    Measure how well reconstruction preserves gradient structure
    
    Returns:
        - gradient_mse: MSE between gradient magnitudes
        - gradient_correlation: Correlation of gradients
        - gradient_similarity: Normalized similarity [0, 1]
    """
```

**Important Limitation**:
> Boundary masks are algorithmically derived from gradients, NOT ground-truth annotations. Interpret as "high-gradient regions" rather than "true mixed pixels."

### 7.3 Latent Diagnostics

**KL Model**:
```python
def evaluate_latent_diagnostics_kl(model, dataloader, device):
    """
    Analyze learned Gaussian posterior
    
    Returns:
        - latent_mean_mean: Average of means
        - latent_mean_std: Spread of means
        - latent_logvar_mean: Average log-variance
        - latent_logvar_std: Spread of log-variances
    """
```

**VQ Model**:
```python
def evaluate_latent_diagnostics_vq(model, dataloader, device):
    """
    Analyze codebook usage
    
    Returns:
        - active_codes: Number of used codebook entries
        - total_codes: Codebook size (1024)
        - utilization: Fraction of codebook used
        - perplexity_mean: Effective codebook size
    """
```

### 7.4 Visualization

```python
def visualize_reconstructions(model, dataloader, device, output_dir, num_samples=16):
    """
    Create comprehensive visualization for each sample:
    
    Row 1:
        [Original] [Reconstruction] [Error Map]
    
    Row 2:
        [Gradient (Original)] [Boundary Mask] [Error + Boundary Overlay]
    
    Saved as: reconstruction_{idx:03d}.png
    """
```

---

## 8. Comparison Pipeline (`scripts/compare_experiment1.py`)

### 8.1 Loading Results

```python
def load_evaluation_results(experiment_dir):
    """
    Load evaluation_results.json from both KL and VQ
    
    Returns: dict with all metrics
    """
```

### 8.2 Comparison Table

```python
def create_comparison_table(kl_results, vq_results):
    """
    Generate side-by-side comparison
    
    Columns:
        - Metric name
        - KL value
        - VQ value
        - Difference (VQ - KL)
        - Relative difference (%)
    
    Rows:
        - Global metrics (MSE, SSIM, MAE)
        - Boundary metrics
        - Gradient preservation
        - Latent diagnostics
    """
```

### 8.3 Comparison Visualizations

```python
def create_visualizations(kl_results, vq_results, output_dir):
    """
    Generate comparison plots:
    
    1. global_metrics_comparison.png
       - Bar charts: MSE, SSIM, MAE
       - KL vs VQ side-by-side
    
    2. boundary_metrics_comparison.png
       - Boundary vs non-boundary MSE
       - Gradient preservation scores
    """
```

### 8.4 Markdown Report

```python
def generate_report(kl_results, vq_results, output_file):
    """
    Automated interpretation report:
    
    Sections:
        1. Research question
        2. Results summary
        3. Key observations
        4. Limitations
        5. Recommendations for future work
    
    Clearly distinguishes:
        - Observed facts
        - Interpretations
        - Hypotheses
    """
```

---

## 9. Complete Experiment Runner (`scripts/run_experiment1.py`)

### 9.1 Full Pipeline

```bash
python scripts/run_experiment1.py --all

# Or step-by-step:
python scripts/run_experiment1.py --train --evaluate --compare
```

**Pipeline Steps**:
1. Train KL autoencoder
2. Train VQ autoencoder
3. Evaluate KL model
4. Evaluate VQ model
5. Generate comparison report

### 9.2 Partial Execution

```bash
# Skip training, only evaluate
python scripts/run_experiment1.py --evaluate-only

# Skip KL model entirely
python scripts/run_experiment1.py --all --skip-kl

# Skip VQ model entirely
python scripts/run_experiment1.py --all --skip-vq
```

---

## 10. Key Design Decisions

### 10.1 Why These Architectures?

**Encoder/Decoder**:
- **ResNet blocks**: Proven effective for image reconstruction
- **GroupNorm**: More stable than BatchNorm with small batches
- **Progressive channels**: Gradual information compression
- **SiLU activation**: Smoother gradients than ReLU

**KL Regularization**:
- **Small weight (1e-6)**: Prevents posterior collapse
- **Diagonal Gaussian**: Simplest factorization, computationally efficient
- **Reparameterization**: Enables backprop through sampling

**VQ Regularization**:
- **EMA updates**: More stable than gradient-based codebook learning
- **Straight-through estimator**: Allows gradients through discrete operation
- **Codebook size (1024)**: Balances expressiveness vs memory
- **Commitment cost (0.25)**: From original VQ-VAE paper

### 10.2 Controlled Comparison

**What's Identical**:
- ✅ Encoder/decoder architecture
- ✅ Spatial compression (8×)
- ✅ Latent channels (4)
- ✅ Optimizer (Adam, lr=1e-3)
- ✅ Data split (80/20, seed=42)
- ✅ Training budget (100 epochs, early stopping)
- ✅ Evaluation protocol

**What Differs (By Design)**:
- KL: Continuous latent, Gaussian posterior, KL loss
- VQ: Discrete latent, codebook quantization, VQ loss

**What's Excluded (For Now)**:
- ❌ Diffusion models
- ❌ Attention mechanisms
- ❌ Perceptual/adversarial losses
- ❌ Thermal-specific preprocessing
- ❌ Architectural variations

### 10.3 Memory Optimization

**Original (81M params, OOM on 6GB GPU)**:
```yaml
base_channels: 128
batch_size: 8
```

**Reduced (20M params, fits 6GB GPU)**:
```yaml
base_channels: 64
batch_size: 2
```

**Impact**:
- ✅ Still a valid controlled comparison
- ⚠️ Slower training (smaller batches)
- ⚠️ Reduced model capacity

---

## 11. Interpretation Guidelines

### 11.1 What Results Can Tell Us

**Valid Conclusions**:
- Which regularization reconstructs better globally
- Which preserves high-gradient regions better
- Computational trade-offs (params, time, memory)
- Qualitative reconstruction differences
- Latent space properties (continuous vs discrete)

**Invalid Conclusions**:
- ❌ "This is the optimal architecture for thermal imagery"
- ❌ "Results generalize to all thermal datasets"
- ❌ "Performance at true mixed pixels" (no ground truth)
- ❌ "Impact on downstream detection" (not tested)

### 11.2 Boundary Metrics Caveat

**What We Measure**:
- High-gradient regions (75th percentile threshold)
- Sobel-based edge detection
- Proxy for thermal boundaries

**What We DON'T Have**:
- Ground-truth mixed-pixel annotations
- Verified boundary locations
- Temperature calibration

**Interpretation**:
> "VQ preserves high-gradient structure better"
> NOT "VQ is better for mixed-pixel detection"

---

## 12. Future Extensions

### 12.1 Architectural Improvements

- [ ] Add attention to encoder/decoder
- [ ] Test different channel multipliers
- [ ] Vary compression factors (4×, 16×)
- [ ] Hybrid KL-VQ approaches
- [ ] Thermal-specific normalization

### 12.2 Loss Improvements

- [ ] Perceptual loss (VGG features)
- [ ] Adversarial loss (discriminator)
- [ ] Edge-aware loss for boundaries
- [ ] Thermal-gradient preservation loss

### 12.3 Experiment Extensions

- [ ] Per-dataset analysis
- [ ] Different KL weights / codebook sizes
- [ ] Cross-dataset generalization
- [ ] Integration with diffusion models
- [ ] Downstream mixed-pixel detection

---

## 13. References

### 13.1 Primary Reference

**Rombach et al. (2022)**:
> "High-Resolution Image Synthesis with Latent Diffusion Models"
> CVPR 2022
> 
> Key contributions:
> - First-stage compression (KL/VQ regularized)
> - Latent diffusion (not implemented yet)
> - Conditioning mechanisms

### 13.2 Foundational Works

**VAE** (Kingma & Welling, 2014):
> "Auto-Encoding Variational Bayes"
> ICLR 2014
> - KL divergence regularization
> - Reparameterization trick

**VQ-VAE** (van den Oord et al., 2017):
> "Neural Discrete Representation Learning"
> NeurIPS 2017
> - Vector quantization
> - EMA codebook updates
> - Straight-through estimator

---

## 14. Git Repository

**Tracked Files**:
```
models/ldm/                        ✓ All implementations
configs/experiment1/               ✓ Configurations
scripts/train_first_stage.py       ✓ Training
scripts/evaluate_first_stage.py    ✓ Evaluation
scripts/compare_experiment1.py     ✓ Comparison
evaluation/boundary_metrics.py     ✓ Boundary analysis
```

**Ignored Files**:
```
checkpoints/experiment1/           ✗ Model weights
logs/experiment1/                  ✗ Training logs
outputs/experiment1/               ✗ Results
runs/experiment1/                  ✗ TensorBoard
```

---

## 15. Code Quality

### 15.1 Documentation

- **Docstrings**: Complete for all public methods
- **Type hints**: Full annotations
- **Comments**: Algorithm explanations, not obvious code
- **README**: This document + EXPERIMENT1_README.md

### 15.2 Testing

```python
# scripts/test_experiment1_models.py

def test_kl_model():
    """Instantiate and forward pass"""

def test_vq_model():
    """Instantiate and forward pass"""

def test_latent_dimensions():
    """Verify 8× compression"""

def test_loss_computation():
    """Verify loss calculations"""
```

---

## 16. Usage Summary

### 16.1 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Test models
python scripts/test_experiment1_models.py

# Run complete experiment
python scripts/run_experiment1.py --all

# Monitor training
tensorboard --logdir runs/experiment1
```

### 16.2 Output Locations

```
checkpoints/experiment1/
├── kl/best_model.pth
└── vq/best_model.pth

outputs/experiment1/
├── kl/
│   ├── evaluation_results.json
│   └── visualizations/
├── vq/
│   ├── evaluation_results.json
│   └── visualizations/
└── comparison/
    ├── comparison_table.csv
    ├── experiment1_report.md
    └── *.png (comparison plots)
```

---

**Document Version**: 1.0  
**Last Updated**: September 2026  
**Authors**: Research team  
**Status**: Implementation complete, ready for training  
**Next Step**: Execute experiment to obtain KL vs VQ results
