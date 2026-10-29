# Latent Diffusion Model (Stage 2) — Design

## Overview

This document specifies the design of the LDM Stage 2 baseline: a diffusion model operating in the frozen VQ first-stage autoencoder's latent space.

The architecture follows the two-stage paradigm from Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models":

```
Stage 1 (Frozen):  Image → VQ Encoder → Quantized Latent → VQ Decoder → Reconstruction
Stage 2 (Trained): Noise → Diffusion U-Net → Denoised Latent → VQ Decoder → Generated Image
```

The baseline implements unconditional generation without conditioning mechanisms, architecture modifications, or custom losses.

## System Architecture

### High-Level Pipeline

```
Training:
    Input image (B, 3, 480, 640)
         ↓
    [Frozen VQ Encoder]
         ↓
    z_quantized (B, 4, 60, 80)
         ↓
    Standardize: z_scaled = (z - μ) / σ
         ↓
    Add noise: z_t = √(ᾱ_t) z_0 + √(1 - ᾱ_t) ε
         ↓
    [Diffusion U-Net]
         ↓
    Predict noise: ε_pred = U-Net(z_t, t)
         ↓
    Loss: ||ε - ε_pred||²

Sampling:
    z_T ~ N(0, I)  (B, 4, 60, 80)
         ↓
    [Diffusion U-Net] (iterative denoising)
         ↓
    z_0_scaled (B, 4, 60, 80)
         ↓
    Unscale: z_0 = z_0_scaled * σ + μ
         ↓
    [Frozen VQ Decoder]
         ↓
    Generated image (B, 3, 480, 640)
```

### Component Responsibilities

| Component | Responsibility | Trainable | Source |
|-----------|---------------|-----------|--------|
| VQ Encoder | Extract quantized latents | ❌ Frozen | Experiment 1 checkpoint |
| VQ Decoder | Decode latents to images | ❌ Frozen | Experiment 1 checkpoint |
| VQ Quantizer | Codebook lookup | ❌ Frozen | Experiment 1 checkpoint |
| Latent Scaler | Standardize/unstandardize | ❌ Fixed transform | Measured statistics |
| Noise Scheduler | Forward/reverse diffusion | ❌ Fixed schedule | Cosine schedule |
| Diffusion U-Net | Denoise latents | ✅ Trained | New implementation |
| EMA Module | Maintain inference weights | ✅ Updated | New implementation |

## First-Stage Interface (Frozen)

### VQ Autoencoder Specification

**Checkpoint:** `outputs/experiment1/vq_nomixed/best_model (1).pth`

**Architecture Configuration:**
```python
VQAutoencoder(
    in_channels=3,
    latent_channels=4,
    base_channels=32,
    channel_multipliers=[1, 2, 4, 4],
    num_res_blocks=2,
    num_embeddings=1024,
    commitment_cost=0.25
)
```

**Total parameters:** 5,119,311
- Encoder: 2,077,156
- Decoder: 3,038,019
- Quantizer: 4,096

### Encode Method

```python
def encode(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Encode input image to quantized latent.
    
    Args:
        x: Input image tensor (B, 3, 480, 640) float32, range [0, 1]
    
    Returns:
        z_quantized: Post-quantization embeddings (B, 4, 60, 80) float32
        z_continuous: Pre-quantization encoder output (B, 4, 60, 80) float32
        vq_losses: Dict with VQ diagnostics (unused in LDM training)
    """
```

**Internal flow:**
```
x → encoder → quant_conv → z_continuous → quantizer → z_quantized
```

**For LDM baseline:** Use `z_quantized` only. Discard `z_continuous` and `vq_losses`.

### Decode Method

```python
def decode(z: torch.Tensor) -> torch.Tensor:
    """
    Decode quantized latent to image.
    
    Args:
        z: Quantized latent tensor (B, 4, 60, 80) float32
           Must be in the original unscaled range from VQ training.
    
    Returns:
        reconstruction: Image tensor (B, 3, 480, 640) float32, range ≈ [0, 1]
    """
```

**Internal flow:**
```
z → post_quant_conv → decoder → reconstruction
```

**Critical:** Decoder expects unscaled latents in the original VQ training range. Must apply inverse standardization before decoding.

### Loading and Freezing

```python
# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
vq_model = VQAutoencoder(...)
vq_model.load_state_dict(checkpoint['model_state_dict'])

# Freeze all parameters
vq_model.eval()
for param in vq_model.parameters():
    param.requires_grad = False

# Verify frozen
assert all(not p.requires_grad for p in vq_model.parameters())

# Verify codebook loaded
assert hasattr(vq_model.quantizer, 'ema_cluster_size')
assert vq_model.quantizer.ema_cluster_size.sum() > 0
```

### Known Limitations

**Codebook underutilization:**
- Active codes: 178 / 1024 (17.4%)
- Dead codes: 846 (82.6%)

**Implication:** Generated latents may lie off the 178-active-code manifold. Decoder robustness to such latents is unknown and requires testing.

**Potential mitigation (deferred):** Project sampled latents to nearest active codebook entry before decoding.

## Latent Scaling

### Rationale

VQ latents have non-zero mean and non-unit variance. Diffusion training is more stable when operating in approximately zero-mean, unit-variance space.

**Measured statistics (from training set, 1654 samples):**

From `outputs/vq_latent_statistics.yaml`:
```yaml
quantized:
  global_mean: 0.01289583370089531
  global_std: 0.19909434020519257
  global_min: -1.8445848226547241
  global_max: 0.5819209814071655
  per_channel_mean: [-0.10774, 0.06486, -0.01323, 0.10769]
  per_channel_std: [0.06423, 0.19236, 0.04955, 0.29673]
```

### Transformation

**Standardization (forward):**
```python
μ = 0.01289583370089531  # Global mean
σ = 0.19909434020519257  # Global std

z_scaled = (z_quantized - μ) / σ
```

**Unstandardization (inverse):**
```python
z_unscaled = z_scaled * σ + μ
```

**Properties:**
- Linear, invertible transformation.
- After standardization: global mean ≈ 0, global std ≈ 1.
- Decoder receives latents in original VQ training range.

### Implementation

```python
class LatentScaler:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
        assert std > 1e-6, "Standard deviation too small"
    
    def scale(self, z: torch.Tensor) -> torch.Tensor:
        """Standardize latent."""
        return (z - self.mean) / self.std
    
    def unscale(self, z_scaled: torch.Tensor) -> torch.Tensor:
        """Inverse standardization."""
        return z_scaled * self.std + self.mean
    
    def test_invertibility(self, z: torch.Tensor, tol: float = 1e-5) -> bool:
        """Verify unscale(scale(z)) ≈ z."""
        z_reconstructed = self.unscale(self.scale(z))
        error = (z - z_reconstructed).abs().max().item()
        return error < tol
```

**Test before training:** Verify invertibility on sample batch.

### Per-Channel vs Global

**Baseline uses global standardization** (single μ, σ for all channels).

**Rationale:**
- Simpler implementation.
- Preserves relative channel magnitudes.
- Sufficient for baseline.

**Deferred:** Per-channel standardization as ablation experiment.

## Diffusion Model Architecture

**⚠️ PROVISIONAL: Requires memory validation and user approval.**

### U-Net Overview

**Type:** Denoising U-Net with timestep embeddings  
**Input:** Noised latent (B, 4, 60, 80) + timestep t  
**Output:** Predicted noise (B, 4, 60, 80)

**Architecture:** Symmetric encoder-decoder with skip connections.

### Provisional Configuration

```python
UNetConfig(
    in_channels=4,          # VQ latent channels
    out_channels=4,         # Same as input
    base_channels=128,      # Base channel count
    channel_multipliers=[1, 2, 4, 4],  # 4 resolution levels
    num_res_blocks=2,       # Residual blocks per level
    attention_levels=[2, 3],  # Apply attention at levels 2 and 3
    dropout=0.0,            # No dropout in baseline
    time_emb_dim=256,       # Timestep embedding dimension
    num_groups=32,          # GroupNorm groups
    activation='silu'       # SiLU (Swish) activation
)
```

### Resolution Levels

| Level | Resolution | Channels | Attention | Residual Blocks |
|-------|-----------|----------|-----------|-----------------|
| 0 (input) | 60×80 | 128×1=128 | ❌ | 2 |
| 1 | 30×40 | 128×2=256 | ❌ | 2 |
| 2 | 15×20 | 128×4=512 | ✅ | 2 |
| 3 (bottleneck) | 8×10 | 128×4=512 | ✅ | 2 |

**Downsampling:** 2×2 strided convolution  
**Upsampling:** Nearest-neighbor interpolation + convolution

**Skip connections:** Concatenate encoder features to decoder at each level.

### Residual Block

```
Input
  ↓
GroupNorm → SiLU → Conv3x3
  ↓
Add time embedding (projected + SiLU)
  ↓
GroupNorm → SiLU → Conv3x3
  ↓
Residual connection (+ input projection if channels differ)
  ↓
Output
```

### Attention Block

**Type:** Self-attention with single head (multi-head deferred)

```
Input (B, C, H, W)
  ↓
GroupNorm
  ↓
Q, K, V = Linear projections
  ↓
Attention(Q, K, V) = softmax(QK^T / √d) V
  ↓
Output projection
  ↓
Residual connection (+ input)
  ↓
Output (B, C, H, W)
```

**Applied at:** Levels 2 (15×20) and 3 (8×10)  
**Rationale:** Attention at low resolutions captures global structure without excessive memory.

### Timestep Embedding

**Type:** Sinusoidal positional encoding

```python
def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Args:
        timesteps: (B,) integer timesteps
        dim: Embedding dimension (must be even)
    
    Returns:
        (B, dim) sinusoidal embeddings
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb
```

**Projection to time_emb_dim:**
```python
t_emb = sinusoidal_embedding(timesteps, dim=128)  # (B, 128)
t_emb = silu(linear1(t_emb))                      # (B, 256)
t_emb = linear2(t_emb)                            # (B, 256)
```

**Injection:** Add to residual block features after first convolution.

### Parameter and Memory Estimates

**Parameter count (estimated):**
- Encoder: ~4 levels × 2 blocks × (Conv + GroupNorm + TimeProj) ≈ 10M
- Bottleneck attention: ≈ 2M
- Decoder: ~4 levels × 2 blocks ≈ 10M
- Skip projections: ≈ 1M
- **Total:** ~25-30M parameters (~100-120MB)

**Memory estimate for batch size 8 on T4 GPU (16GB):**
- Model parameters: ~120 MB
- Gradients: ~120 MB
- Optimizer state (Adam): ~240 MB (2× params)
- Activations (forward): ~2-3 GB (estimated for 128 base channels, 4 levels, attention at 2-3)
- **Total training memory:** ~3-4 GB per batch
- **Safety margin:** ~12 GB free for multiple batches, caching

**Verdict:** Configuration should fit comfortably within T4 16GB limit. If memory issues arise during implementation, reduce base_channels from 128 to 96 or 64.

## Forward Diffusion Process

### Mathematical Definition

**Forward process:** Gradually add Gaussian noise to latent.

```
q(z_t | z_0) = N(z_t; √(ᾱ_t) z_0, (1 - ᾱ_t) I)
```

Where:
- `z_0`: Scaled latent from VQ encoder
- `z_t`: Noised latent at timestep t
- `α_t = 1 - β_t`: Noise schedule parameter
- `ᾱ_t = ∏_{s=1}^t α_s`: Cumulative product

**Sampling:**
```python
ε ~ N(0, I)
z_t = √(ᾱ_t) z_0 + √(1 - ᾱ_t) ε
```

**Properties:**
- Markov process: `q(z_t | z_{t-1}, ..., z_0) = q(z_t | z_{t-1})`
- Single-step sampling: Can sample `z_t` directly from `z_0` using `ᾱ_t`.

### Noise Schedule

**Type:** Cosine schedule (Nichol & Dhariwal 2021)

**Mathematical definition:**

**Definition:**
```python
def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule from Nichol & Dhariwal 2021.
    
    Args:
        timesteps: Total number of diffusion timesteps (T=1000)
        s: Offset to prevent β_t from being too small at t=0
    
    Returns:
        beta: (T,) tensor of β_t values
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.0001, 0.9999)
    return betas
```

**Explicit values:**
- T = 1000 total timesteps
- s = 0.008 (offset parameter)
- β_t ∈ [0.0001, 0.9999] (clipped)
- α_t = 1 - β_t
- ᾱ_t = ∏_{i=1}^t α_i

**Properties:**
- Smooth transition from low to high noise
- Avoids extreme β values near 0 or 1
- Better preservation of signal at early timesteps
- β_1 ≈ 0.0001, β_1000 ≈ 0.02 (approximate, computed from cosine formula)

**Precomputed values:**
```python
betas = cosine_beta_schedule(1000)       # (1000,)
alphas = 1 - betas                        # (1000,)
alphas_cumprod = torch.cumprod(alphas, dim=0)  # (1000,) ᾱ_t values
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
```

**Register as buffers (non-trainable, moved with model to device).**

### NoiseScheduler Implementation

```python
class NoiseScheduler:
    def __init__(self, num_timesteps: int = 1000, schedule: str = 'cosine'):
        self.num_timesteps = num_timesteps
        
        if schedule == 'cosine':
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Precompute values for forward diffusion
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        
        # Precompute values for reverse diffusion
        self.alphas = alphas
        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
    
    def add_noise(self, z_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Add noise to latent at timestep t.
        
        Args:
            z_0: Original latent (B, C, H, W)
            t: Timesteps (B,) integers in [0, num_timesteps)
            noise: Noise tensor (B, C, H, W) ~ N(0, I)
        
        Returns:
            z_t: Noised latent (B, C, H, W)
        """
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        z_t = sqrt_alpha_cumprod * z_0 + sqrt_one_minus_alpha_cumprod * noise
        return z_t
```

## Training Procedure

### Training Objective

**Type:** Simplified noise prediction (ε-prediction)

```
L_simple = E_{t, z_0, ε} [||ε - ε_θ(z_t, t)||²]
```

**Procedure:**
1. Sample batch of images from training set.
2. Encode to z_quantized using frozen VQ encoder.
3. Standardize: `z_0 = (z_quantized - μ) / σ`.
4. Sample timestep: `t ~ Uniform(0, T-1)`.
5. Sample noise: `ε ~ N(0, I)`.
6. Add noise: `z_t = √(ᾱ_t) z_0 + √(1 - ᾱ_t) ε`.
7. Predict noise: `ε_pred = UNet(z_t, t)`.
8. Compute loss: `L = ||ε - ε_pred||²`.
9. Backpropagate and update U-Net parameters.
10. Update EMA weights.

### Training Loop

```python
for epoch in range(max_epochs):
    for batch in train_loader:
        images, _ = batch
        
        # Encode (frozen)
        with torch.no_grad():
            z_quantized, _, _ = vq_model.encode(images)
            z_0 = scaler.scale(z_quantized)
        
        # Sample timesteps and noise
        t = torch.randint(0, num_timesteps, (batch_size,), device=device)
        noise = torch.randn_like(z_0)
        
        # Forward diffusion
        z_t = scheduler.add_noise(z_0, t, noise)
        
        # Predict noise
        noise_pred = unet(z_t, t)
        
        # Compute loss
        loss = F.mse_loss(noise_pred, noise)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update EMA
        ema.update(unet)
    
    # Validation
    val_loss = validate(unet, val_loader, vq_model, scaler, scheduler)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(unet, ema, optimizer, epoch)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```

### Optimizer Configuration

```python
optimizer = torch.optim.Adam(
    unet.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0
)
```

**Learning rate schedule:** Constant (no warmup or decay in baseline).

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
```

Prevents exploding gradients, especially early in training.

### Exponential Moving Average (EMA)

Maintain EMA of model parameters for inference:

```python
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {name: param.clone().detach() 
                       for name, param in model.named_parameters()}
    
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_shadow(self, model: nn.Module):
        """Apply EMA weights to model (for inference)."""
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])
    
    def restore(self, model: nn.Module, backup: dict):
        """Restore original weights after inference."""
        for name, param in model.named_parameters():
            param.data.copy_(backup[name])
```

**Usage:**
- Update EMA every training step.
- Use EMA weights for sampling and evaluation.
- Save EMA shadow dict separately in checkpoint.

### Early Stopping

**Monitor:** Validation loss  
**Patience:** 10 epochs  
**Criterion:** Save checkpoint when `val_loss < best_val_loss`

**Rationale:** Prevents overfitting, especially with limited validation set (413 images).

## Reverse Sampling Process

### DDPM Reverse Process

**Sampling configuration:** 50-step DDPM with uniform stride

**Reverse process:**
```
p_θ(z_{t-1} | z_t) = N(z_{t-1}; μ_θ(z_t, t), σ_t² I)
```

**Mean prediction:**
```
μ_θ(z_t, t) = (1 / √α_t) (z_t - (β_t / √(1 - ᾱ_t)) ε_θ(z_t, t))
```

**Variance:**
```
σ_t² = β_t  (or other variance schedules)
```

### Sampling Algorithm

```python
def sample(unet: nn.Module, 
           scheduler: NoiseScheduler,
           scaler: LatentScaler,
           vq_decoder: nn.Module,
           batch_size: int,
           device: torch.device) -> torch.Tensor:
    """
    Generate images using DDPM sampling.
    
    Args:
        unet: Trained diffusion model (with EMA weights)
        scheduler: Noise scheduler
        scaler: Latent scaler
        vq_decoder: Frozen VQ decoder
        batch_size: Number of images to generate
        device: torch device
    
    Returns:
        Generated images (batch_size, 3, 480, 640)
    """
    # Start from pure noise
    z_t = torch.randn(batch_size, 4, 60, 80, device=device)
    
    # Iterative denoising (50 steps from 1000)
    # Uniform stride: every 20th timestep
    timesteps = list(range(0, 1000, 20))[::-1]  # [980, 960, 940, ..., 40, 20, 0]
    # Exact sequence: [980, 960, 940, 920, 900, 880, 860, 840, 820, 800,
    #                  780, 760, 740, 720, 700, 680, 660, 640, 620, 600,
    #                  580, 560, 540, 520, 500, 480, 460, 440, 420, 400,
    #                  380, 360, 340, 320, 300, 280, 260, 240, 220, 200,
    #                  180, 160, 140, 120, 100,  80,  60,  40,  20,   0]
    
    for i, t in enumerate(timesteps):
        t_tensor = torch.tensor([t] * batch_size, device=device)
        
        # Predict noise
        with torch.no_grad():
            noise_pred = unet(z_t, t_tensor)
        
        # Compute mean
        alpha_t = scheduler.alphas[t]
        alpha_cumprod_t = scheduler.alphas_cumprod[t]
        beta_t = scheduler.betas[t]
        
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            z_t - (beta_t / torch.sqrt(1.0 - alpha_cumprod_t)) * noise_pred
        )
        
        # Add noise (except last step)
        if i < len(timesteps) - 1:
            noise = torch.randn_like(z_t)
            variance = beta_t
            z_t = mean + torch.sqrt(variance) * noise
        else:
            z_t = mean
    
    # z_t is now z_0 (denoised latent)
    z_0_scaled = z_t
    
    # Unscale
    z_0 = scaler.unscale(z_0_scaled)
    
    # Decode
    with torch.no_grad():
        images = vq_decoder(z_0)
    
    return images
```

### Timestep Selection (PROVISIONAL)

**Options:**

1. **Uniform stride (proposed baseline):**
   - Sample every 20th timestep: [1000, 980, 960, ..., 20, 0]
   - Total: 50 steps
   - Simple, deterministic

2. **Quadratic spacing:**
   - More steps at low noise, fewer at high noise
   - May improve quality
   - More complex

3. **DDIM (deferred):**
   - Deterministic, non-Markovian
   - Fewer steps possible
   - Requires separate implementation

**Baseline uses uniform stride pending user approval.**

### Variance Schedule

**Baseline:** Fixed variance `σ_t² = β_t`

**Alternatives (deferred):**
- Learned variance
- `σ_t² = β̃_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) × β_t` (DDPM paper)

## Checkpoint Management

### Checkpoint Structure

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': unet.state_dict(),
    'ema_state_dict': ema.shadow,
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_loss': best_val_loss,
    'train_losses': train_losses,
    'val_losses': val_losses,
    'config': config_dict,
    'latent_stats': {'mean': μ, 'std': σ},
    'vq_checkpoint_path': vq_checkpoint_path
}
```

**Save location:** `outputs/experiment_ldm_baseline/`

**Naming:**
- `best_model.pth`: Best validation loss
- `ema_weights.pth`: EMA shadow parameters (optional separate save)
- `checkpoint_epoch_{epoch}.pth`: Periodic checkpoints

### Loading for Inference

```python
# Load checkpoint
checkpoint = torch.load(checkpoint_path)
unet.load_state_dict(checkpoint['model_state_dict'])

# Apply EMA weights
ema_weights = checkpoint['ema_state_dict']
for name, param in unet.named_parameters():
    param.data.copy_(ema_weights[name])

# Load scaler
scaler = LatentScaler(
    mean=checkpoint['latent_stats']['mean'],
    std=checkpoint['latent_stats']['std']
)

# Load frozen VQ model
vq_checkpoint = torch.load(checkpoint['vq_checkpoint_path'])
vq_model = VQAutoencoder(...)
vq_model.load_state_dict(vq_checkpoint['model_state_dict'])
vq_model.eval()
for param in vq_model.parameters():
    param.requires_grad = False
```

## Testing Strategy

### Pre-Training Tests

**Test suite:** `tests/ldm/test_diffusion_baseline.py`

**Required tests:**

1. **test_load_frozen_vq:**
   - Load checkpoint successfully.
   - All parameters frozen.
   - Quantizer buffers present.
   - Reconstruction matches baseline (within tolerance).

2. **test_latent_scaling:**
   - Invertibility: `unscale(scale(z)) ≈ z` within 1e-5.
   - Scaled latents: mean ≈ 0, std ≈ 1 (within 0.1).

3. **test_unet_forward:**
   - Forward pass with random input and timestep.
   - Output shape matches input shape.
   - No NaN or Inf in output.

4. **test_noise_scheduler:**
   - `add_noise` produces correct shape.
   - Noise increases with timestep (verify `ᾱ_t` decreases).
   - At t=T-1, noised latent ≈ pure noise.

5. **test_training_step:**
   - Single training step completes.
   - Loss is finite.
   - Gradients are finite.
   - EMA updates correctly.

6. **test_sampling:**
   - Sample from random noise.
   - Output shape: (B, 4, 60, 80).
   - Unscaled latent can be decoded.
   - Generated image shape: (B, 3, 480, 640).
   - No NaN or Inf.

### Smoke Training

**Script:** `scripts/smoke_test_ldm.py`

**Procedure:**
1. Load frozen VQ model and verify reconstruction.
2. Initialize U-Net, scheduler, scaler, optimizer, EMA.
3. Train for 1-5 epochs on small subset (e.g., 100 images).
4. Log loss every 10 batches.
5. Validate after each epoch.
6. Generate 4 samples and save.
7. Check:
   - Loss decreases or behaves reasonably.
   - No NaN or Inf.
   - Samples are not pure noise.

**Success criteria:**
- Training completes without errors.
- Loss trajectory shows learning signal.
- Generated samples are decodable.

**NOT required:**
- Converged loss.
- High-quality samples.

### Integration Test

**After smoke test approval:**

Full training run on Kaggle:
1. Train for up to 100 epochs with early stopping.
2. Monitor training and validation loss.
3. Generate sample grids every 10 epochs.
4. Save best checkpoint.
5. Evaluate against VQ baseline.

## Evaluation Procedure

### Reconstruction Metrics

**Compare LDM generation vs VQ baseline reconstruction:**

1. Encode validation images with frozen VQ encoder.
2. Standardize latents.
3. Sample from diffusion model conditioned on those latents (if conditional sampling added later) OR compare unconditional generations to validation set statistics.

**Baseline is unconditional:** Direct input-output reconstruction comparison not possible.

**Alternative:** Compare generated sample statistics (mean, std, gradient distribution) to validation set.

### Generation Quality

**Qualitative:**
- Visual inspection of sample grids.
- Presence of realistic thermal structure.
- Absence of artifacts (checkerboard, noise, mode collapse).

**Quantitative (optional, requires reference set):**
- Fréchet Inception Distance (FID) (if pre-trained thermal feature extractor available).
- Boundary proxy metrics on generated samples.

### Boundary Analysis

**On generated samples:**
1. Compute gradient-based boundary proxy.
2. Measure boundary sharpness.
3. Compare to validation set boundary statistics.

**Limitation:** Proxy is NOT ground truth. Cannot claim improved mixed-pixel detection.

### Sampling Efficiency

**Measure:**
- Time per sample (seconds).
- Time per batch.
- Memory usage during sampling.

**Compare:**
- 50-step DDPM sampling vs VQ encode/decode (baseline).

## Known Limitations

1. **Codebook underutilization:** Diffusion may generate off-manifold latents. Decoder robustness unknown.
2. **No ground truth:** Cannot evaluate true mixed-pixel detection accuracy.
3. **Unconditional:** Cannot reconstruct specific input images.
4. **Small validation set:** 413 images limits statistical power.
5. **Provisional architecture:** Memory and compute not yet validated on T4.

## Future Extensions (Deferred)

1. **Conditioning:** Class labels, text, spatial maps.
2. **Accelerated sampling:** DDIM, DPM-Solver, consistency models.
3. **Architecture:** Multi-head attention, cross-attention, adaptive group norm.
4. **Latent variants:** Pre-quantization, discrete indices.
5. **Losses:** Perceptual loss, edge-aware loss.
6. **Codebook projection:** Nearest-neighbor post-processing.
7. **Hybrid training:** Fine-tune first-stage with diffusion feedback.

All require separate specification and approval.

## Implementation Checklist

- [ ] Implement U-Net with timestep embeddings
- [ ] Implement noise scheduler (cosine schedule)
- [ ] Implement latent scaler with invertibility test
- [ ] Implement EMA module
- [ ] Implement training loop with gradient clipping and early stopping
- [ ] Implement sampling algorithm (DDPM 50-step)
- [ ] Write pre-training test suite
- [ ] Write smoke test script
- [ ] Verify memory usage estimate
- [ ] Run pre-training tests
- [ ] Run smoke test
- [ ] Get approval for full training
- [ ] Document all provisional decisions requiring approval
