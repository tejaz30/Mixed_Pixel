# DiffAD Detection - Design

**Experiment:** Mixed-Pixel Detection via Conditional Latent Diffusion  
**Status:** Specification Phase  
**Date:** 2026-10-03

---

## 1. System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    DiffAD System Overview                       │
└────────────────────────────────────────────────────────────────┘

Phase 1: Conditional Reconstruction Network Training
┌────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Input: No_Mixed Dataset (Clean Thermal Images)                │
│         ↓                                                       │
│    DRAEM Augmentation                                          │
│         ├──────────────┬─────────────┐                        │
│         │              │             │                         │
│    x_normal      x_synth        mask                          │
│    (clean)     (corrupted)    (binary)                        │
│         │              │             │                         │
│    [Target]      [Condition]   [Unused in LDM]                │
│         │              │                                        │
│         └──→ VQ Encode ←──┘                                    │
│              (Frozen)                                          │
│         ├────────┬────────┐                                   │
│         │        │        │                                    │
│    z_normal  c_synth     │                                    │
│         │        │        │                                    │
│      Scale    Scale       │                                    │
│         │        │        │                                    │
│    Sample t  Sample t     │                                    │
│    (target)  (condition)  │                                    │
│         │        │        │                                    │
│    Add noise Add noise    │                                    │
│         │        │        │                                    │
│       z_t    c_noisy      │                                    │
│         │        │        │                                    │
│         └───┬────┘        │                                    │
│             │             │                                     │
│        Concatenate        │                                     │
│        (8 channels)       │                                     │
│             ↓             │                                     │
│    Conditional U-Net      │                                     │
│    (23M params, 8→4)      │                                     │
│             ↓             │                                     │
│    Predict eps_target     │                                     │
│             ↓             │                                     │
│    Loss: MSE(eps_pred,    │                                     │
│              eps_target)  │                                     │
│                           │                                     │
└───────────────────────────┼─────────────────────────────────────┘
                            │
Phase 2: Discriminative Network Training
┌───────────────────────────┼─────────────────────────────────────┐
│                           │ (Frozen LDM + VQ)                    │
│  x_synth ────────┬────────┼────────┐                           │
│      │           │        │        │                            │
│   Encode    Reconstruct   │    mask (Ground Truth)             │
│      │       via LDM      │        │                            │
│      ↓           ↓        │        │                            │
│  c_synth      z_r ────────┼────────┼───────┐                  │
│      │           │        │        │       │                    │
│      └─→ Interpolate ←────┘        │       │                    │
│           (λ=0.5)                  │       │                    │
│              ↓                     │       │                    │
│       VQ Decode (Frozen)           │       │                    │
│              ↓                     │       │                    │
│          x_inter                   │       │                    │
│              │                     │       │                    │
│   ┌──────────┼────────────┐        │       │                    │
│   │          │            │        │       │                    │
│ x_synth   x_inter      x_r        │       │                    │
│   │          │            │        │       │                    │
│   └──────────┴────────────┘        │       │                    │
│              │                     │       │                    │
│        Concatenate                 │       │                    │
│        (9 channels)                │       │                    │
│              ↓                     │       │                    │
│    Discriminator U-Net             │       │                    │
│              ↓                     │       │                    │
│        prob_map (1×480×640)        │       │                    │
│              ↓                     │       │                    │
│    Loss: Focal(prob_map, mask)────┘       │                    │
│                                            │                    │
└────────────────────────────────────────────┼────────────────────┘
                                             │
Phase 3: Inference (Mixed-Pixel Detection)
┌────────────────────────────────────────────┼────────────────────┐
│                                            │                     │
│  Test Image x_test (Potentially Mixed-Pixel)                    │
│         ↓                                  │                     │
│    VQ Encode (Frozen) ─────────────────────┘                    │
│         ↓                                                        │
│    c_test (4×60×80)                                            │
│         ↓                                                        │
│      Scale                                                       │
│         ↓                                                        │
│  Sample t_inf ∈ [400,500]                                       │
│         ↓                                                        │
│    Add noise                                                     │
│         ↓                                                        │
│    c_noisy (or c for ablation)                                  │
│         │                                                        │
│    Initialize z_T ~ N(0,I)                                      │
│         │                                                        │
│    DDIM Reverse Sampling (50 steps):                           │
│         for t in [T, ..., 1]:                                   │
│             concat([z_t, c_noisy]) → U-Net → eps_pred           │
│             z_t ← DDIM_step(z_t, eps_pred, t)                  │
│         ↓                                                        │
│      z_0 (reconstructed latent)                                │
│         ↓                                                        │
│     Unscale                                                      │
│         ↓                                                        │
│  VQ Decode (Frozen)                                            │
│         ↓                                                        │
│  x_r (Reconstructed Normal Image)                              │
│         │                                                        │
│   ┌─────┴─────┬──────────┐                                     │
│   │           │          │                                      │
│ x_test    Interpolate  x_r                                     │
│   │      (c_test, z_0) │                                       │
│   │           │          │                                      │
│   │       x_inter        │                                      │
│   │           │          │                                      │
│   └───────────┴──────────┘                                      │
│               │                                                  │
│         Concatenate (9 ch)                                      │
│               ↓                                                  │
│    Discriminator (Frozen)                                       │
│               ↓                                                  │
│    prob_map (Mixed-Pixel Probability)                          │
│                                                                  │
│  Alternative: Residual Analysis                                 │
│    residual = |x_test - x_r|                                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1 VQ Autoencoder (Frozen, Experiment 1)

**Purpose:** First-stage compression to latent space

**Architecture:**
```
Encoder:
  Conv 3→32, stride 2, downsample ×2
  ResBlock ×2 (32 channels)
  Conv 32→64, stride 2, downsample ×2
  ResBlock ×2 (64 channels)
  Conv 64→128, stride 2, downsample ×2
  ResBlock ×2 (128 channels)
  Conv 128→256, stride 2, downsample ×2
  ResBlock ×2 (256 channels)
  Conv 256→4 (latent channels)
  
Quantizer:
  Codebook: 1024 embeddings × 4 dimensions
  EMA update (not used during inference)
  
Decoder:
  Symmetric upsampling (4 levels)
  ResBlocks ×2 per level
  Final Conv 32→3
```

**Interface:**
```python
class VQAutoencoder(nn.Module):
    def encode(self, x):
        """
        Args: x (B, 3, 480, 640)
        Returns: 
            z_quantized (B, 4, 60, 80)
            z_continuous (B, 4, 60, 80)  # Unused
            vq_losses dict
        """
    
    def decode(self, z):
        """
        Args: z (B, 4, 60, 80) - quantized or interpolated
        Returns: x_recon (B, 3, 480, 640)
        """
```

**Checkpoint:** `checkpoints/experiment1/vq_nomixed/best_model (1).pth`

**Frozen:** All parameters `requires_grad=False`

**Latent Statistics (Quantized):**
```yaml
mean: 0.012896
std: 0.199094
min: -1.844585
max: 0.581921
source: 1654 training images
```

---

### 2.2 Conditional Diffusion U-Net

**Purpose:** Denoise target latent conditioned on noisy corrupted latent

**Architecture:**

```python
class ConditionalDiffusionUNet(nn.Module):
    """
    Conditional U-Net for latent diffusion.
    Receives concatenated target and condition latents.
    """
    def __init__(self):
        # Input/output
        self.in_channels = 8      # 4 (target z_t) + 4 (condition c_noisy)
        self.out_channels = 4     # Predicted noise for target
        
        # Network structure
        self.base_channels = 64
        self.channel_multipliers = [1, 2, 4, 4]  # [64, 128, 256, 256]
        self.num_res_blocks = 2
        self.attention_levels = [2, 3]  # Attend at 15×20 and 8×10
        
        # Timestep embedding
        self.time_emb_dim = 256
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(128),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )
        
        # Encoder (4 levels, spatial: 60×80 → 8×10)
        self.encoder_blocks = nn.ModuleList([
            # Level 0: 60×80, ch=64
            EncoderBlock(8, 64, time_emb_dim=256, num_res_blocks=2),
            # Level 1: 30×40, ch=128
            EncoderBlock(64, 128, time_emb_dim=256, num_res_blocks=2),
            # Level 2: 15×20, ch=256, attention
            EncoderBlock(128, 256, time_emb_dim=256, num_res_blocks=2, 
                         attention=True, num_heads=8),
            # Level 3: 8×10, ch=256, attention
            EncoderBlock(256, 256, time_emb_dim=256, num_res_blocks=2,
                         attention=True, num_heads=8)
        ])
        
        # Middle block
        self.middle_block = MiddleBlock(256, time_emb_dim=256, 
                                        attention=True, num_heads=8)
        
        # Decoder (symmetric)
        self.decoder_blocks = nn.ModuleList([...])
        
        # Output
        self.out_norm = nn.GroupNorm(32, 64)
        self.out_conv = nn.Conv2d(64, 4, kernel_size=3, padding=1)
    
    def forward(self, x_concat, t):
        """
        Args:
            x_concat: (B, 8, 60, 80) - [z_t, c_noisy] concatenated
            t: (B,) - timesteps
        Returns:
            eps_pred: (B, 4, 60, 80) - predicted noise for z_t
        """
        # Timestep embedding
        t_emb = self.time_mlp(t)  # (B, 256)
        
        # Encoder with skip connections
        skips = []
        h = x_concat
        for encoder in self.encoder_blocks:
            h, skip = encoder(h, t_emb)
            skips.append(skip)
        
        # Middle
        h = self.middle_block(h, t_emb)
        
        # Decoder with skip connections
        for decoder, skip in zip(self.decoder_blocks, reversed(skips)):
            h = decoder(h, skip, t_emb)
        
        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        eps_pred = self.out_conv(h)
        
        return eps_pred
```

**Key Properties:**
- Input is ALWAYS 8 channels (target + condition)
- No separate conditioning path (concatenation only)
- Timestep embedding injected at each residual block
- Self-attention at lower resolutions for global structure
- No cross-attention (conditioning via concatenation)

**Parameters:**
- Total: ~23M (target, same scale as unconditional LDM)
- Trainable: All (new model, not fine-tuned)

**Initialization:** Xavier/Kaiming normal (standard)

---

### 2.3 Noise Scheduler

**Purpose:** Manage forward and reverse diffusion processes

**Configuration:**
```python
from diffusers import DDPMScheduler

scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="squaredcos_cap_v2",  # Cosine
    prediction_type="epsilon",           # Predict noise
    clip_sample=False,
    thresholding=False,
    variance_type="fixed_small",
    timestep_spacing="leading"
)
```

**Forward Process (Training):**
```python
def add_noise(z_0, t, eps):
    """
    q(z_t | z_0) = N(√ᾱ_t·z_0, (1-ᾱ_t)·I)
    
    Args:
        z_0: Clean latent (B, 4, 60, 80)
        t: Timestep (B,) ∈ [0, 999]
        eps: Noise (B, 4, 60, 80) ~ N(0,I)
    Returns:
        z_t: Noisy latent
    """
    sqrt_alpha_bar = scheduler.alphas_cumprod[t] ** 0.5
    sqrt_one_minus_alpha_bar = (1 - scheduler.alphas_cumprod[t]) ** 0.5
    
    return sqrt_alpha_bar * z_0 + sqrt_one_minus_alpha_bar * eps
```

**Reverse Process (Inference):**
```python
scheduler.set_timesteps(50)  # DDIM with 50 steps

for t in scheduler.timesteps:
    # Predict noise
    eps_pred = unet(concat([z_t, c_noisy]), t)
    
    # DDIM step (deterministic, eta=0)
    result = scheduler.step(eps_pred, t, z_t, eta=0.0)
    z_t = result.prev_sample
```

**Timestep Sampling:**
- Training: `t ~ Uniform(0, 999)` per sample
- Inference condition noise: `t_inf ~ Uniform(400, 500)`

---

### 2.4 Latent Scaler

**Purpose:** Normalize latents to ~N(0,1) for stable diffusion training

**Implementation:**
```python
class LatentScaler:
    def __init__(self, mean=0.012896, std=0.199094):
        self.mean = mean
        self.std = std
    
    def scale(self, z):
        """z_scaled = (z - μ) / σ"""
        return (z - self.mean) / self.std
    
    def unscale(self, z_scaled):
        """z = z_scaled * σ + μ"""
        return z_scaled * self.std + self.mean
```

**Usage:**
- Scale BOTH target and condition latents before diffusion
- Use SAME statistics for both (measured on quantized latents)
- Unscale before VQ decode

---

### 2.5 DRAEM Augmentation

**Purpose:** Generate synthetic anomalies for training

**Implementation:**
```python
class DRAEMAugmenter:
    def __init__(self, 
                 texture_source_paths,  # DTD dataset
                 min_perlin_scale=0,
                 max_perlin_scale=6,
                 beta_range=(0.0, 0.8)):
        
        self.texture_paths = texture_source_paths
        self.min_perlin = min_perlin_scale
        self.max_perlin = max_perlin_scale
        self.beta_range = beta_range
        
        # Image augmentations
        self.augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2), add=(-30,30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50,50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32,128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]
        
        self.rotation = iaa.Affine(rotate=(-90, 90))
    
    def augment(self, image):
        """
        Generate synthetic anomaly from clean image.
        
        Args:
            image: (H, W, 3) float32 in [0,1]
        Returns:
            augmented: (H, W, 3) float32 in [0,1]
            mask: (H, W, 1) float32 binary
            has_anomaly: bool
        """
        H, W = image.shape[:2]
        
        # 50% chance: return clean image
        if random.random() > 0.5:
            return image, np.zeros((H, W, 1)), False
        
        # Load random texture
        texture_path = random.choice(self.texture_paths)
        texture = cv2.imread(texture_path)
        texture = cv2.resize(texture, (W, H))
        
        # Augment texture
        aug_ops = random.sample(self.augmenters, 3)
        aug_seq = iaa.Sequential(aug_ops)
        texture_aug = aug_seq(image=texture)
        
        # Generate Perlin mask
        scale_x = 2 ** random.randint(self.min_perlin, self.max_perlin)
        scale_y = 2 ** random.randint(self.min_perlin, self.max_perlin)
        perlin = rand_perlin_2d_np((H, W), (scale_x, scale_y))
        perlin = self.rotation(image=perlin)
        
        # Threshold to binary
        mask = (perlin > 0.5).astype(float)
        mask = np.expand_dims(mask, axis=2)  # (H, W, 1)
        
        # Blend
        beta = random.uniform(*self.beta_range)
        texture_masked = texture_aug.astype(float) / 255.0 * mask
        
        augmented = (image * (1 - mask) + 
                    (1 - beta) * texture_masked + 
                    beta * image * mask)
        
        # Ensure mask has actual content
        has_anomaly = (mask.sum() > 0)
        
        return augmented, mask, has_anomaly
```

**Parameters (from official DiffAD):**
- Perlin scale range: 2^0 to 2^6 (1 to 64 pixels)
- Beta range: 0.0 to 0.8 (texture opacity)
- Threshold: 0.5 (binary mask)
- Texture source: DTD dataset (5640 images, 47 categories)

**Output Shape:**
- `augmented`: (480, 640, 3) float32 [0, 1]
- `mask`: (480, 640, 1) float32 {0, 1}
- `has_anomaly`: bool

---

### 2.6 Discriminative U-Net

**Purpose:** Pixel-level anomaly segmentation from three-image input

**Architecture:**
```python
class DiscriminativeUNet(nn.Module):
    """
    Segmentation U-Net for anomaly localization.
    Takes 3 RGB images (9 channels) as input.
    """
    def __init__(self, base_channels=64):
        self.in_channels = 9   # 3 images × 3 channels
        self.out_channels = 1  # Probability map
        
        # Encoder (4 levels: 480→240→120→60→30)
        self.encoder1 = EncoderBlock(9, 64)      # 480×640
        self.encoder2 = EncoderBlock(64, 128)    # 240×320
        self.encoder3 = EncoderBlock(128, 256)   # 120×160
        self.encoder4 = EncoderBlock(256, 512)   # 60×80
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (symmetric with skip connections)
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 64)
        
        # Output
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x_concat):
        """
        Args:
            x_concat: (B, 9, 480, 640) - [x_a, x_inter, x_r]
        Returns:
            prob_map: (B, 1, 480, 640) - P(anomaly)
        """
        # Encoder with skip connections
        e1, skip1 = self.encoder1(x_concat)
        e2, skip2 = self.encoder2(e1)
        e3, skip3 = self.encoder3(e2)
        e4, skip4 = self.encoder4(e3)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decoder with skip connections
        d4 = self.decoder4(b, skip4)
        d3 = self.decoder3(d4, skip3)
        d2 = self.decoder2(d3, skip2)
        d1 = self.decoder1(d2, skip1)
        
        # Output
        logits = self.out_conv(d1)
        prob_map = self.sigmoid(logits)
        
        return prob_map
```

**Blocks:**
```python
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        h = F.relu(self.norm1(self.conv1(x)))
        h = F.relu(self.norm2(self.conv2(h)))
        skip = h
        h = self.pool(h)
        return h, skip

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)  # in_ch for concat
        self.norm1 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
    
    def forward(self, x, skip):
        h = self.upsample(x)
        h = torch.cat([h, skip], dim=1)
        h = F.relu(self.norm1(self.conv1(h)))
        h = F.relu(self.norm2(self.conv2(h)))
        return h
```

**Parameters:** ~15M (rough estimate)

---

## 3. Training Procedures

### 3.1 Conditional LDM Training

**Pseudocode:**
```python
def train_conditional_ldm():
    # Setup
    vq_model = load_vq_checkpoint()  # Frozen
    vq_model.eval()
    for param in vq_model.parameters():
        param.requires_grad = False
    
    unet = ConditionalDiffusionUNet()  # Trainable, from scratch
    optimizer = Adam(unet.parameters(), lr=1e-4)
    scheduler = DDPMScheduler(...)
    scaler = LatentScaler(mean=0.012896, std=0.199094)
    
    dataset = DiffADDataset(
        image_dir='No_Mixed/train/',
        augmenter=DRAEMAugmenter(texture_paths=DTD_PATHS)
    )
    
    # Training loop
    for epoch in range(100):
        for batch in dataloader:
            # Get paired samples
            x_normal = batch['image']         # (B, 3, 480, 640)
            x_synth = batch['augmented_image']  # (B, 3, 480, 640)
            
            # Encode (frozen VQ)
            with torch.no_grad():
                z_normal, _, _ = vq_model.encode(x_normal)
                c_synth, _, _ = vq_model.encode(x_synth)
                
                # Scale
                z_normal_scaled = scaler.scale(z_normal)
                c_synth_scaled = scaler.scale(c_synth)
            
            # Sample INDEPENDENT timesteps
            B = z_normal_scaled.shape[0]
            t_target = torch.randint(0, 1000, (B,), device=device)
            t_cond = torch.randint(0, 1000, (B,), device=device)
            
            # Add noise independently
            eps_target = torch.randn_like(z_normal_scaled)
            z_t = scheduler.add_noise(z_normal_scaled, eps_target, t_target)
            
            eps_cond = torch.randn_like(c_synth_scaled)
            c_noisy = scheduler.add_noise(c_synth_scaled, eps_cond, t_cond)
            
            # Concatenate (8 channels)
            model_input = torch.cat([z_t, c_noisy], dim=1)
            
            # Predict noise
            eps_pred = unet(model_input, t_target)
            
            # Loss
            loss = F.mse_loss(eps_pred, eps_target)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
        
        # Validation
        val_loss = validate(...)
        
        # Checkpoint
        if val_loss < best_val_loss:
            save_checkpoint(unet, optimizer, epoch, val_loss)
```

**Key Points:**
- VQ frozen, U-Net trained from scratch
- Independent timesteps sampled per batch
- Loss computed on TARGET noise only
- Gradient clipping for stability
- Validation every epoch

### 3.2 Discriminator Training

**Pseudocode:**
```python
def train_discriminator():
    # Setup (frozen)
    vq_model = load_vq_checkpoint()
    unet = load_conditional_ldm_checkpoint()
    vq_model.eval()
    unet.eval()
    for model in [vq_model, unet]:
        for param in model.parameters():
            param.requires_grad = False
    
    # Trainable
    discriminator = DiscriminativeUNet()
    optimizer = Adam(discriminator.parameters(), lr=1e-3)
    
    # Training loop
    for epoch in range(700):
        for batch in dataloader:
            x_synth = batch['augmented_image']  # Synthetic anomaly
            mask = batch['mask']                # Binary ground truth
            
            # Generate reconstruction (frozen LDM)
            with torch.no_grad():
                c_synth, _, _ = vq_model.encode(x_synth)
                c_synth_scaled = scaler.scale(c_synth)
                
                # Add inference noise
                t_inf = random.randint(400, 500)
                eps = torch.randn_like(c_synth_scaled)
                c_noisy = scheduler.add_noise(c_synth_scaled, eps, 
                                              torch.tensor([t_inf]))
                
                # DDIM sampling
                z_T = torch.randn_like(c_synth_scaled)
                z_t = z_T
                scheduler.set_timesteps(50)
                
                for t in scheduler.timesteps:
                    model_input = torch.cat([z_t, c_noisy], dim=1)
                    eps_pred = unet(model_input, t)
                    z_t = scheduler.step(eps_pred, t, z_t).prev_sample
                
                z_r_scaled = z_t
                z_r = scaler.unscale(z_r_scaled)
                x_r = vq_model.decode(z_r)
                
                # Interpolate
                lambda_interp = 0.5
                c_synth_unscaled = scaler.unscale(c_synth_scaled)
                z_inter = lambda_interp * c_synth_unscaled + \
                          (1 - lambda_interp) * z_r
                x_inter = vq_model.decode(z_inter)
            
            # Concatenate (9 channels)
            disc_input = torch.cat([x_synth, x_inter, x_r], dim=1)
            
            # Predict
            prob_map = discriminator(disc_input)
            
            # Loss
            loss = focal_loss(prob_map, mask)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        val_auroc = validate(...)
```

**Key Points:**
- VQ and LDM completely frozen
- Reconstructions generated on-the-fly (expensive!)
- Interpolation in unscaled latent space
- Focal loss for imbalanced segmentation
- AUROC primary validation metric

---

## 4. Inference Pipelines

### 4.1 Reconstruction

```python
def reconstruct_image(x_test, 
                      vq_model, 
                      unet, 
                      scheduler, 
                      scaler,
                      t_inf_range=(400, 500),
                      use_noisy_cond=True,
                      num_steps=50):
    """
    Reconstruct potentially anomalous test image as normal.
    
    Args:
        x_test: (1, 3, 480, 640) test image
        ...
        use_noisy_cond: If True, use c_noisy; else use c
    Returns:
        x_r: (1, 3, 480, 640) reconstructed normal image
    """
    device = x_test.device
    
    # Encode test image
    with torch.no_grad():
        c_test, _, _ = vq_model.encode(x_test)
        c_test_scaled = scaler.scale(c_test)
        
        # Prepare condition
        if use_noisy_cond:
            t_inf = random.randint(*t_inf_range)
            eps_cond = torch.randn_like(c_test_scaled)
            c_cond = scheduler.add_noise(c_test_scaled, eps_cond,
                                         torch.tensor([t_inf], device=device))
        else:
            c_cond = c_test_scaled  # Clean condition (ablation)
        
        # Initialize reconstruction from noise
        z_T = torch.randn_like(c_test_scaled)
        
        # DDIM reverse sampling
        scheduler.set_timesteps(num_steps)
        z_t = z_T
        
        for t in scheduler.timesteps:
            # Concatenate with condition
            model_input = torch.cat([z_t, c_cond], dim=1)
            
            # Predict noise
            eps_pred = unet(model_input, t)
            
            # DDIM step
            z_t = scheduler.step(eps_pred, t, z_t, eta=0.0).prev_sample
        
        # Unscale and decode
        z_0 = z_t
        z_0_unscaled = scaler.unscale(z_0)
        x_r = vq_model.decode(z_0_unscaled)
    
    return x_r
```

### 4.2 Detection

```python
def detect_anomalies(x_test,
                    vq_model,
                    unet,
                    discriminator,
                    scheduler,
                    scaler,
                    lambda_interp=0.5):
    """
    Full DiffAD detection pipeline.
    
    Returns:
        x_r: Reconstructed image
        x_inter: Interpolated image
        prob_map: Anomaly probability map
        residual_map: Simple reconstruction residual
    """
    # Reconstruct
    x_r = reconstruct_image(x_test, vq_model, unet, scheduler, scaler)
    
    with torch.no_grad():
        # Encode for interpolation
        c_test, _, _ = vq_model.encode(x_test)
        c_test_scaled = scaler.scale(c_test)
        
        # Get reconstruction latent (recompute or pass from reconstruct)
        z_r, _, _ = vq_model.encode(x_r)  # Re-encode for simplicity
        z_r_scaled = scaler.scale(z_r)
        
        # Interpolate (in scaled space, then unscale)
        z_inter_scaled = lambda_interp * c_test_scaled + \
                         (1 - lambda_interp) * z_r_scaled
        z_inter = scaler.unscale(z_inter_scaled)
        x_inter = vq_model.decode(z_inter)
        
        # Concatenate for discriminator
        disc_input = torch.cat([x_test, x_inter, x_r], dim=1)
        
        # Predict anomaly probability
        prob_map = discriminator(disc_input)
        
        # Compute residual (baseline)
        residual_map = torch.abs(x_test - x_r).mean(dim=1, keepdim=True)
    
    return {
        'reconstructed': x_r,
        'interpolated': x_inter,
        'probability_map': prob_map,
        'residual_map': residual_map
    }
```

---

## 5. Loss Functions

### 5.1 Diffusion Loss (LDM Training)

**MSE on predicted noise:**
```python
def diffusion_loss(eps_pred, eps_target):
    return F.mse_loss(eps_pred, eps_target, reduction='mean')
```

**Simple, standard DDPM objective.**

### 5.2 Focal Loss (Discriminator Training)

**Purpose:** Handle class imbalance (small anomaly regions)

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) probabilities [0,1]
            target: (B, 1, H, W) binary {0,1}
        """
        # BCE component
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        # Focal modulation
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha balancing
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        loss = alpha_t * focal_weight * bce
        return loss.mean()
```

**Alternative:** Standard BCE if focal loss causes issues

---

## 6. Evaluation Metrics

### 6.1 Reconstruction Quality

**Compute on clean validation images:**

```python
def compute_reconstruction_metrics(x_true, x_recon):
    mse = F.mse_loss(x_recon, x_true)
    mae = F.l1_loss(x_recon, x_true)
    psnr = -10 * torch.log10(mse)
    ssim = structural_similarity(x_true, x_recon)
    
    return {'MSE': mse, 'MAE': mae, 'PSNR': psnr, 'SSIM': ssim}
```

**Boundary-specific metrics:**
```python
def compute_boundary_metrics(x_true, x_recon, boundary_mask):
    """
    boundary_mask: Sobel gradient > 75th percentile (proxy)
    """
    # MSE on boundary pixels
    boundary_mse = ((x_true - x_recon) ** 2 * boundary_mask).sum() / \
                   boundary_mask.sum()
    
    # MSE on non-boundary pixels
    non_boundary_mse = ((x_true - x_recon) ** 2 * (1 - boundary_mask)).sum() / \
                       (1 - boundary_mask).sum()
    
    # Ratio (should be close to 1.0)
    ratio = boundary_mse / (non_boundary_mse + 1e-8)
    
    # Gradient preservation
    grad_true = sobel_gradient(x_true)
    grad_recon = sobel_gradient(x_recon)
    grad_preservation = F.cosine_similarity(grad_true, grad_recon).mean()
    
    return {
        'boundary_MSE': boundary_mse,
        'non_boundary_MSE': non_boundary_mse,
        'boundary_ratio': ratio,
        'gradient_preservation': grad_preservation
    }
```

### 6.2 Detection Performance (Synthetic Anomalies)

**Pixel-level metrics:**
```python
from sklearn.metrics import roc_auc_score, average_precision_score

def compute_detection_metrics(prob_map, mask):
    """
    Args:
        prob_map: (H, W) anomaly probabilities [0,1]
        mask: (H, W) binary ground truth {0,1}
    """
    # Flatten
    prob_flat = prob_map.flatten()
    mask_flat = mask.flatten()
    
    # AUROC
    auroc = roc_auc_score(mask_flat, prob_flat)
    
    # Average Precision
    ap = average_precision_score(mask_flat, prob_flat)
    
    # F1 at optimal threshold
    thresholds = np.linspace(0, 1, 100)
    f1_scores = []
    for thresh in thresholds:
        pred_binary = (prob_flat > thresh).astype(int)
        f1 = f1_score(mask_flat, pred_binary)
        f1_scores.append(f1)
    
    best_f1 = max(f1_scores)
    best_thresh = thresholds[np.argmax(f1_scores)]
    
    # IoU and Dice at best threshold
    pred_binary = (prob_flat > best_thresh).astype(int)
    intersection = (pred_binary * mask_flat).sum()
    union = pred_binary.sum() + mask_flat.sum() - intersection
    iou = intersection / (union + 1e-8)
    dice = 2 * intersection / (pred_binary.sum() + mask_flat.sum() + 1e-8)
    
    return {
        'AUROC': auroc,
        'AP': ap,
        'F1': best_f1,
        'threshold': best_thresh,
        'IoU': iou,
        'Dice': dice
    }
```

---

## 7. Configuration Files

### 7.1 LDM Training Config

**`configs/diffad/ldm_train.yaml`:**
```yaml
model:
  in_channels: 8
  out_channels: 4
  base_channels: 64
  channel_multipliers: [1, 2, 4, 4]
  num_res_blocks: 2
  attention_levels: [2, 3]
  time_emb_dim: 256

first_stage:
  checkpoint: checkpoints/experiment1/vq_nomixed/best_model (1).pth
  latent_stats:
    mean: 0.012896
    std: 0.199094

noise_schedule:
  type: squaredcos_cap_v2
  num_timesteps: 1000
  s_param: 0.008

training:
  optimizer: Adam
  learning_rate: 1.0e-4
  batch_size: 8
  epochs: 100
  gradient_clip: 1.0
  ema_decay: 0.9999

data:
  train_dir: src/datasets/processed/No_Mixed/No_mixed
  train_split: 0.8
  val_split: 0.2
  random_seed: 42
  
draem:
  texture_dir: datasets/dtd/images
  perlin_min: 0
  perlin_max: 6
  beta_range: [0.0, 0.8]
  anomaly_ratio: 0.5

checkpointing:
  save_every: 10
  save_best: true
  output_dir: checkpoints/diffad/ldm

logging:
  log_every: 100
  sample_every: 1000
```

### 7.2 Discriminator Training Config

**`configs/diffad/discriminator_train.yaml`:**
```yaml
model:
  in_channels: 9
  out_channels: 1
  base_channels: 64

ldm:
  checkpoint: checkpoints/diffad/ldm/best_model.pth
  inference_steps: 50
  t_inf_range: [400, 500]
  use_noisy_cond: true

interpolation:
  lambda: 0.5

training:
  optimizer: Adam
  learning_rate: 1.0e-3
  batch_size: 32
  epochs: 700

loss:
  type: focal
  alpha: 0.25
  gamma: 2.0

checkpointing:
  save_every: 50
  save_best: true
  output_dir: checkpoints/diffad/discriminator
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

1. **DRAEM augmentation:**
   - Mask aligns with augmented region
   - Values in [0, 1]
   - Clean samples when has_anomaly=False

2. **Dataset:**
   - Returns correct shapes
   - Clean and augmented are different
   - Clean and augmented from same source

3. **Conditional U-Net:**
   - Forward pass with 8-channel input
   - Output is 4-channel
   - Timestep embedding works

4. **Latent scaling:**
   - scale/unscale are inverses
   - Statistics match expected values

5. **Discriminator:**
   - Forward pass with 9-channel input
   - Output is 1-channel probability

6. **Interpolation:**
   - Decoder handles interpolated latents
   - No NaN/Inf values
   - Visual quality reasonable

### 8.2 Integration Tests

1. **Full training step:**
   - No crashes
   - Loss is finite
   - Gradients computed

2. **Full inference:**
   - Test image → reconstruction
   - All intermediate shapes correct
   - Output is valid image

3. **Full detection:**
   - Test image → probability map
   - All steps execute
   - Metrics computable

### 8.3 Smoke Test

**Run before full training:**
- 5 epochs conditional LDM training
- Validation loss computed
- Sample reconstructions generated
- Visual inspection of outputs

**Acceptance criteria:**
- Training completes without error
- Loss decreases (even slightly)
- Samples are valid images (not noise/black)

---

## 9. Known Limitations and Risks

### Limitations

1. **Synthetic ≠ Real:**
   - DRAEM textures don't model thermal mixed pixels
   - Perlin masks don't match real boundary shapes
   - No physics-based temperature mixing

2. **Decoder Off-Manifold:**
   - VQ decoder trained on discrete codebook only
   - Interpolated latents are off-manifold
   - May produce artifacts (needs testing)

3. **Boundary Proxy:**
   - Sobel gradient mask is NOT ground truth
   - Cannot claim real boundary preservation
   - Only for relative comparison

4. **Computational Cost:**
   - Discriminator training requires full LDM inference
   - 50 DDIM steps per sample
   - Slow training (consider pre-generating reconstructions)

### Risks

1. **Conditional training diverges:**
   - Mitigation: Lower LR, gradient clipping, monitor closely

2. **Interpolation artifacts:**
   - Mitigation: Test decoder first, adjust lambda if needed

3. **Poor synthetic detection:**
   - Mitigation: Verify DRAEM implementation, check official code

4. **No improvement over baselines:**
   - Mitigation: Implement ablations, check methodology

---

## 10. Success Criteria

**Phase 1 (Conditional LDM):**
- ✅ Training converges (loss decreases)
- ✅ Reconstructions visually sensible
- ✅ Conditioning demonstrably affects output

**Phase 2 (Discriminator):**
- ✅ Training converges
- ✅ AUROC > 0.7 on synthetic validation
- ✅ Probability maps visually align with masks

**Phase 3 (Comparison):**
- ✅ DiffAD ≥ Residual baseline (AUROC)
- ✅ DiffAD reconstructions ≈ VQ quality
- ✅ Clear limitation documentation

**Stop and Review Before:**
- Full 100-epoch training
- Expensive Kaggle runs
- Publishing any results

