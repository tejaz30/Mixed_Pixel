# Latent Diffusion Model (Stage 2) — Requirements

## 1. Objective

Implement and evaluate the second stage of the Latent Diffusion Model (LDM) architecture: a diffusion model operating in the learned latent space of the trained VQ first-stage autoencoder.

This experiment investigates whether latent diffusion can generate high-quality thermal images by learning to denoise in the compressed VQ latent space, rather than directly in pixel space.

The purpose is to establish a working LDM baseline that can be compared against the VQ first-stage autoencoder alone and against the CAE baseline in terms of reconstruction quality, generation quality, and boundary preservation.

**This experiment implements unconditional latent diffusion only. No conditioning mechanisms are included in the baseline.**

## 2. Experimental Question

Can a diffusion model operating in the VQ latent space generate realistic agricultural thermal images with quality comparable to or exceeding the VQ reconstruction baseline?

## 3. Experimental Scope

The experiment shall implement and train a diffusion model operating over the frozen VQ first-stage autoencoder's latent representation.

The VQ first-stage model shall remain completely frozen throughout diffusion training. No first-stage parameters, codebook entries, or preprocessing shall be modified.

The diffusion model shall learn to reverse a forward noising process applied to VQ latent representations extracted from training images.

The experiment shall evaluate:
- Generated sample quality.
- Reconstruction quality when conditioning on input images (if implemented).
- Comparison with VQ autoencoder baseline.
- Training stability and convergence.
- Sampling efficiency.

The baseline shall not include:
- Conditioning mechanisms (class, text, or image).
- Architecture modifications beyond standard U-Net with attention.
- Edge-aware or boundary-specific losses.
- Thermal-specific architectural modules.
- Custom sampling algorithms beyond standard DDPM.

These are explicitly deferred to future ablation experiments.

## 4. Controlled Experimental Conditions

The experiment shall use:
- The same dataset as Experiment 1 (No_Mixed thermal images).
- The same preprocessing pipeline (resize to 640×480, ToTensor).
- The same train/validation split (80/20, stratified, seed=42).
- The frozen VQ checkpoint from Experiment 1: `outputs/experiment1/vq_nomixed/best_model (1).pth`.
- Measured latent statistics from the VQ training set: `outputs/vq_latent_statistics.yaml`.

Changes from Experiment 1:
- New model: Diffusion U-Net operating in latent space.
- New training objective: Denoising diffusion probabilistic model (DDPM).
- New evaluation: Generation quality, not just reconstruction.

Record all training configurations, random seeds, and hyperparameters for reproducibility.

If essential settings cannot be verified or measured statistics are unavailable, report the gap rather than assuming values.

## 5. Independent Variable

The primary independent variable is the introduction of latent diffusion.

Comparison groups:
1. **VQ baseline:** First-stage autoencoder alone (encode → decode).
2. **LDM baseline:** Frozen first-stage with learned diffusion (encode → diffuse → denoise → decode).

All other conditions shall remain controlled wherever technically possible.

## 6. First-Stage Interface (Frozen)

### 6.1 VQ Autoencoder

**Checkpoint:** `outputs/experiment1/vq_nomixed/best_model (1).pth`  
**Architecture:** VQAutoencoder with 1024-entry codebook  
**Configuration:**
- `in_channels`: 3
- `latent_channels`: 4
- `base_channels`: 32
- `channel_multipliers`: [1, 2, 4, 4]
- `num_res_blocks`: 2
- `num_embeddings`: 1024
- `commitment_cost`: 0.25

### 6.2 Encode/Decode Interface

**Verified signatures from source code:**

```python
def encode(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Args:
        x: Input image (B, 3, 480, 640) float32
    
    Returns:
        z_quantized: (B, 4, 60, 80) float32 - post-quantization embeddings
        z_continuous: (B, 4, 60, 80) float32 - pre-quantization encoder output
        vq_losses: dict with VQ-specific diagnostics
    """

def decode(z: torch.Tensor) -> torch.Tensor:
    """
    Args:
        z: Quantized latent (B, 4, 60, 80) float32
    
    Returns:
        reconstruction: (B, 3, 480, 640) float32
    """
```

### 6.3 Latent Representation

**Shape:** (B, 4, 60, 80)  
**Data type:** float32  
**Representation:** Post-quantization continuous embeddings (`z_quantized`)

**Measured statistics from training set** (from `outputs/vq_latent_statistics.yaml`):
- Global mean: 0.01289583370089531
- Global std: 0.19909434020519257
- Global min: -1.8445848226547241
- Global max: 0.5819209814071655

**Per-channel statistics:**
- Channel 0: mean=-0.10774, std=0.06423
- Channel 1: mean=0.06486, std=0.19236
- Channel 2: mean=-0.01323, std=0.04955
- Channel 3: mean=0.10769, std=0.29673

**Scaling transformation:**

The diffusion model shall operate on standardized latents:

```python
# Standardize (use measured statistics)
μ = 0.01289583370089531
σ = 0.19909434020519257
z_scaled = (z_quantized - μ) / σ

# Inverse (before decoding)
z_unscaled = z_scaled * σ + μ
reconstruction = vq_model.decode(z_unscaled)
```

This ensures diffusion operates in an approximately zero-mean, unit-variance space while maintaining decoder compatibility.

### 6.4 Frozen First-Stage Requirements

- Load checkpoint in evaluation mode (`model.eval()`).
- Set all parameters to non-trainable (`requires_grad=False`).
- Verify no gradient computation during diffusion training.
- Verify codebook and EMA buffers are preserved from checkpoint.
- Verify reconstruction quality matches Experiment 1 baseline before diffusion training begins.

## 7. Diffusion Model Architecture (PROVISIONAL)

**⚠️ WARNING: The following architecture specifications are PROVISIONAL and require user approval and memory validation on T4 GPU before implementation.**

### 7.1 Model Type

**Architecture:** U-Net with timestep embeddings  
**Conditioning:** None (unconditional)  
**Input/Output:** (B, 4, 60, 80) scaled latent tensors

### 7.2 Provisional U-Net Configuration

- **Base channels:** 128
- **Channel multipliers:** [1, 2, 4, 4] (4 resolution levels)
- **Attention levels:** [2, 3] (applied at resolutions 30×40 and 15×20)
- **Residual blocks per level:** 2
- **Time embedding dimension:** 256
- **Time embedding type:** Sinusoidal positional encoding
- **Normalization:** GroupNorm
- **Activation:** SiLU (Swish)

**Resolution progression:**
- Level 0: 60×80, channels=128×1=128
- Level 1: 30×40, channels=128×2=256
- Level 2: 15×20, channels=128×4=512, **attention**
- Level 3: 8×10, channels=128×4=512, **attention**

**⚠️ Memory validation required:** Estimated parameter count and peak memory must fit within T4 GPU constraints (16 GB) with batch size 8.

### 7.3 Timestep Embedding

Sinusoidal positional encoding mapped to 256-dimensional embedding:

```python
t_emb = sinusoidal_embedding(timesteps, dim=128)
t_emb = linear_project(t_emb, dim=256)
t_emb = silu(t_emb)
t_emb = linear_project(t_emb, dim=256)
```

Injected into each residual block via addition or adaptive group normalization (AdaGN).

## 8. Forward Diffusion Process (PROVISIONAL)

**⚠️ WARNING: Noise schedule requires precise mathematical definition and user approval.**

### 8.1 Forward Process

The forward process gradually adds Gaussian noise to scaled latents:

```
q(z_t | z_0) = N(z_t; √(ᾱ_t) z_0, (1 - ᾱ_t) I)
```

Where:
- `z_0` is the scaled latent from VQ encoder
- `z_t` is the noised latent at timestep t
- `ᾱ_t = ∏_{s=1}^t α_s` is the cumulative product
- `α_t = 1 - β_t` defines the noise schedule

### 8.2 Noise Schedule (PROVISIONAL)

**Type:** Cosine schedule (Nichol & Dhariwal 2021)

**⚠️ REQUIRES PRECISE DEFINITION:** The exact β_t formulation, timestep distribution, and cumulative alpha bar schedule must be mathematically specified before implementation.

**Provisional specification:**
- Training timesteps T: 1000
- Schedule: Cosine with offset
- β range: [β_min, β_max] (values TBD)

### 8.3 Sampling from q(z_t | z_0)

```python
ε ~ N(0, I)
z_t = √(ᾱ_t) z_0 + √(1 - ᾱ_t) ε
```

## 9. Training Objective

### 9.1 Denoising Objective

The model learns to predict the noise ε added to latents:

```
L_simple = E_{t, z_0, ε} [||ε - ε_θ(z_t, t)||²]
```

Where:
- `ε_θ` is the U-Net denoising model
- `z_t` is the noised latent at timestep t
- `t ~ Uniform(1, T)` is sampled uniformly
- `ε ~ N(0, I)` is the ground-truth noise

### 9.2 Loss Weighting

Use simplified objective (unweighted MSE) initially:

```python
loss = mse_loss(predicted_noise, target_noise)
```

Velocity parameterization, v-prediction, and other variants are deferred to ablation studies.

## 10. Training Configuration (PROVISIONAL)

**⚠️ WARNING: Training hyperparameters are provisional and require user approval.**

### 10.1 Optimizer

- **Type:** Adam
- **Learning rate:** 1e-4
- **Betas:** (0.9, 0.999)
- **Weight decay:** 0.0
- **Epsilon:** 1e-8

### 10.2 Training Schedule

- **Batch size:** 8 (match first-stage training)
- **Max epochs:** 100
- **Early stopping:** Yes (patience=10 epochs, monitor validation loss)
- **Gradient clipping:** 1.0 (global norm)

### 10.3 Exponential Moving Average (EMA)

Maintain EMA of model parameters for inference:

- **Decay:** 0.9999
- **Update frequency:** Every training step
- **Usage:** EMA weights used for sampling and evaluation

### 10.4 Random Seed

- **Seed:** 42 (consistent with Experiment 1)
- **Apply to:** PyTorch, NumPy, Python random module

### 10.5 Hardware

- **Target:** Kaggle T4 GPU
- **Precision:** float32 (mixed precision deferred)
- **Gradient accumulation:** None initially

## 11. Reverse Sampling Process (PROVISIONAL)

**⚠️ WARNING: Sampling method requires precise specification and user approval.**

### 11.1 DDPM Sampling (Initial Approach)

Sample from the learned model using the DDPM reverse process:

```
p_θ(z_{t-1} | z_t) = N(z_{t-1}; μ_θ(z_t, t), σ_t² I)
```

Where `μ_θ` is computed from the predicted noise `ε_θ`.

### 11.2 Sampling Steps

**Training timesteps:** 1000  
**Sampling timesteps:** 50

**⚠️ REQUIRES SPECIFICATION:** How are the 50 sampling steps selected from the 1000 training steps?

Options:
1. Uniform stride: Sample every 20th timestep
2. Quadratic schedule: Non-uniform spacing
3. DDIM: Deterministic sampling (requires separate implementation)

**User decision required before implementation.**

### 11.3 Sampling Procedure

```python
# Start from pure noise
z_T ~ N(0, I)

# Iterative denoising
for t in [T, T-stride, ..., 1]:
    ε_pred = model(z_t, t)
    z_{t-1} = denoise_step(z_t, ε_pred, t)

# Unscale and decode
z_0_unscaled = z_0 * σ + μ
reconstruction = vq_model.decode(z_0_unscaled)
```

## 12. Evaluation Metrics

### 12.1 Training Metrics

- Training loss (per epoch, per batch)
- Validation loss (per epoch)
- Gradient norms
- EMA update tracking
- Training time per epoch

### 12.2 Reconstruction Evaluation

Compare LDM against VQ baseline on held-out validation set:

- **Mean Squared Error (MSE)**
- **Structural Similarity Index (SSIM)**
- **Boundary-region MSE** (using gradient-based proxy)
- **Non-boundary-region MSE**
- **Boundary/non-boundary error ratio**

**Critical:** Boundary proxy is NOT ground-truth mixed-pixel annotation.

### 12.3 Generation Evaluation

Evaluate unconditional samples from LDM:

- **Sample quality:** Visual inspection of generated thermal images
- **Sample diversity:** Variation across multiple samples
- **Artifacts:** Presence of noise, blurriness, or unrealistic patterns
- **Sampling time:** Time per image and per batch

### 12.4 What NOT to Claim

Do not claim:
- Ground-truth mixed-pixel detection accuracy (no annotations available).
- Superiority in downstream classification (not evaluated).
- Input-specific reconstruction quality (unconditional model cannot condition on specific inputs for reconstruction).
- That lower boundary error proves better mixed-pixel detection.

## 13. Research Constraints

Do not introduce any of the following into the baseline:

- Conditioning mechanisms (class labels, text, spatial conditioning).
- Attention architecture modifications beyond standard self-attention.
- Edge-aware, boundary-aware, or thermal-specific losses.
- Thermal-specific architectural modules or custom layers.
- Compressed or expanded latent dimensions.
- Changes to frozen first-stage architecture, weights, or preprocessing.
- Advanced sampling methods (DDIM, DPM-Solver, etc.).
- Velocity prediction, v-parameterization, or alternative objectives.

All such modifications must be proposed as separate ablation experiments with explicit approval.

## 14. Interpretation Framework

Separate observations, interpretations, and hypotheses:

### Observed Results

Measured metrics, loss curves, generated samples, and reconstruction quality.

### Interpretation

Explain what results indicate about:
- LDM's ability to model VQ latent distribution.
- Generation quality relative to VQ reconstruction baseline.
- Training stability and convergence.
- Computational efficiency.

### Hypotheses

Identify questions requiring future experiments:
- Would conditioning improve reconstruction quality?
- Would accelerated sampling reduce inference time?
- Does diffusion over pre-quantization latents differ?
- Does generation quality translate to better mixed-pixel detection?

Do not present hypotheses as established findings.

## 15. Known Limitations

### 15.1 Codebook Underutilization

From Experiment 1:
- Active codes: 178 / 1024 (17.4%)
- Dead codes: 846 (82.6%)

**Implication:** Diffusion operates over a 178-mode discrete manifold embedded in continuous 4×60×80 space. Generated latents may interpolate between active codes, producing off-manifold representations.

**Unknown:** Decoder robustness to off-manifold latents. Requires empirical testing.

**Mitigation:** Accept as known limitation. Document in evaluation. Consider nearest-codebook projection as optional post-processing (defer to ablation).

### 15.2 Ground Truth Limitations

**No verified mixed-pixel masks:** Cannot evaluate mixed-pixel detection accuracy.

**Boundary proxy:** Sobel gradient filter with 75th percentile threshold. This is a structural proxy, NOT ground-truth annotation.

### 15.3 Unconditional Generation

Baseline is unconditional and cannot:
- Reconstruct specific input images.
- Generate images with specified attributes.
- Be directly compared to CAE reconstruction on matched inputs.

Conditional variants deferred to future experiments.

### 15.4 Evaluation Set Size

Validation set: 413 images (from 80/20 split of 2068 total).

Small evaluation set limits statistical power. Avoid repeated hyperparameter tuning on this set.

## 16. Testing Requirements

### 16.1 Pre-Training Tests (REQUIRED)

All tests must pass before full training:

1. **Checkpoint Loading**
   - Load frozen VQ checkpoint successfully.
   - Verify all encoder, decoder, quantizer, and codebook parameters present.
   - Verify EMA buffers loaded correctly.

2. **Frozen Parameters**
   - Verify all VQ parameters have `requires_grad=False`.
   - Verify no gradients computed for first-stage during diffusion training.

3. **Latent Scaling**
   - Verify scaling/unscaling invertibility: `unscale(scale(z)) ≈ z`.
   - Verify scaled latents have mean ≈ 0, std ≈ 1.
   - Verify unscaled latents reproduce VQ baseline reconstruction quality.

4. **Diffusion Model**
   - Forward pass produces correct shapes at all timesteps.
   - Forward diffusion q(z_t | z_0) produces correct shapes.
   - Loss computation returns finite values.
   - Backward pass produces finite gradients.
   - Optimizer step succeeds without errors.

5. **Sampling**
   - Sample z_T from N(0, I) with correct shape.
   - Reverse process produces z_0 with correct shape.
   - Unscaled z_0 can be decoded to (B, 3, 480, 640).
   - No NaN or Inf in sampled outputs.

### 16.2 Smoke Training (After Pre-Training Tests)

**Duration:** 1-5 epochs  
**Purpose:** Verify training pipeline works end-to-end

**Success criteria:**
- No NaN or Inf in loss or gradients.
- Loss trajectory shows learning (decreasing, not flat or diverging).
- Checkpoint save/load works correctly.
- EMA weights update correctly.
- Sampling produces finite outputs with correct shapes.
- Decoded samples are not pure noise (visual inspection).

**NOT required in smoke test:**
- Converged loss.
- High-quality generated samples.
- Improved quality over VQ baseline.

### 16.3 Full Training Evaluation

**After smoke test approval:**

- Training and validation loss curves.
- Generated sample grids (qualitative).
- Reconstruction metrics (MSE, SSIM, boundary analysis).
- Sampling time measurement.
- Peak memory usage.
- Parameter count.

## 17. Deliverables

The completed experiment shall produce:

1. **Implementation**
   - Diffusion U-Net model under `src/models/ldm/diffusion/`.
   - Noise scheduler under `src/models/ldm/diffusion/`.
   - Training script under `scripts/` or training module under `src/training/`.
   - Sampling utilities under `src/models/ldm/diffusion/`.

2. **Tests**
   - Pre-training test suite under `tests/ldm/`.
   - Checkpoint loading tests.
   - Scaling/unscaling tests.
   - Forward pass and shape tests.
   - Smoke training test.

3. **Configuration**
   - Training configuration YAML under `configs/experiment_ldm/`.
   - Separate from first-stage config.

4. **Outputs**
   - Best model checkpoint.
   - EMA weights checkpoint.
   - Training/validation loss history.
   - Generated sample grids.
   - Reconstruction comparison (LDM vs VQ baseline).
   - Evaluation metrics report.

5. **Documentation**
   - Experiment report (Markdown).
   - Methodology, results, interpretation, limitations.
   - Updated `docs/CURRENT_STATE.md`.
   - Updated `docs/DECISIONS.md`.

Use existing repository conventions for directory structure, naming, and output formats.

## 18. Acceptance Criteria

The experiment is complete when:

1. Frozen VQ checkpoint loads correctly and reconstruction matches baseline.
2. Latent scaling/unscaling is verified as invertible.
3. Diffusion model implements documented architecture.
4. Noise schedule is precisely defined and implemented.
5. Training objective (ε-prediction) is correctly implemented.
6. Pre-training tests pass.
7. Smoke training demonstrates learning signal without divergence.
8. Full training completes with saved checkpoints.
9. Generated samples are qualitatively inspected.
10. Reconstruction metrics are computed and compared with VQ baseline.
11. Sampling procedure works and produces finite outputs.
12. Evaluation report distinguishes observations, interpretations, and hypotheses.
13. Known limitations are documented (codebook underutilization, no ground truth, unconditional only).
14. No claims of ground-truth mixed-pixel detection or downstream classification improvement.

## 19. Decisions Requiring User Approval

Before implementation:

1. **Approve single baseline approach:** Post-quantization embeddings only (defer pre-quant and discrete diffusion).
2. **Approve provisional U-Net config:** 128 base channels, 4 levels, attention at levels 2-3.
3. **Approve or specify noise schedule:** Precise mathematical definition of cosine schedule required.
4. **Approve or specify sampling method:** 50-step DDPM with timestep selection method.
5. **Approve training hyperparameters:** Learning rate 1e-4, batch size 8, 100 epochs max.
6. **Approve smoke test criteria:** 1-5 epochs, finite outputs, learning signal.

**Memory and compute validation required before full implementation.**

## 20. Reproducibility

Record:
- Code version / commit hash.
- Experiment configuration files.
- Frozen VQ checkpoint path and verification hash.
- Latent statistics file path and verification hash.
- Dataset version and split.
- Random seed.
- Hardware environment (GPU type, CUDA version, PyTorch version).
- Training duration and convergence epoch.
- Best checkpoint selection criterion.

The experiment must be reproducible from documented configuration and recorded environment.
