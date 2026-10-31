# DiffAD Detection - Requirements

**Experiment:** Mixed-Pixel Detection via Conditional Latent Diffusion (DiffAD)  
**Status:** Specification Phase  
**Dependencies:** Experiment 1 VQ Autoencoder (frozen)

---

## 1. Objective

Implement DiffAD (Zhang et al., ICCV 2023) for mixed-pixel detection in agricultural thermal imagery.

DiffAD uses a conditional latent diffusion model to reconstruct potentially anomalous inputs as normal, enabling anomaly detection through reconstruction analysis.

---

## 2. Scope

### In Scope

✅ **Conditional reconstruction network:**
- Conditional latent diffusion model
- Training on paired clean-target and synthetic-condition images
- DRAEM-style synthetic anomaly generation
- Reconstruction inference with noisy condition embedding

✅ **Discriminative network:**
- Pixel-level anomaly segmentation
- Trained on synthetic anomalies with reconstruction network frozen
- Three-channel input: anomalous, interpolated, reconstructed

✅ **Evaluation:**
- Reconstruction quality on clean validation images
- Detection performance on synthetic anomalies
- Comparison with unconditional LDM and VQ/CAE baselines
- Boundary preservation analysis

### Out of Scope

❌ Modifications to VQ autoencoder weights or architecture  
❌ Retraining unconditional LDM  
❌ Real mixed-pixel annotations (use synthetic evaluation only)  
❌ Thermal-specific corruption models (DRAEM baseline first)  
❌ Architecture search or hyperparameter tuning beyond standard DiffAD settings

---

## 3. Method Overview

### 3.1 Conditional Reconstruction Network

**Architecture:** Latent diffusion model with channel-concatenation conditioning

**Training Data:**
- Source: No_Mixed thermal images (clean normal samples)
- Augmentation: DRAEM synthetic anomalies (texture + Perlin mask)
- Pairing: Each clean image generates one synthetic corruption
- Split: Use existing Experiment 1 train/validation split

**Training Objective:**
```
For each sample:
  1. Clean image x_normal (diffusion target)
  2. Synthetic corruption x_synth (condition)
  3. Encode both: z_normal = E(x_normal), c_synth = E(x_synth)
  4. Sample independent timesteps: t_target, t_cond
  5. Add noise independently: z_t, c_noisy
  6. Concatenate: [z_t, c_noisy] → 8 channels
  7. Predict target noise: eps_pred = U-Net([z_t, c_noisy], t_target)
  8. Loss: MSE(eps_pred, eps_target)
```

**Key Properties:**
- Clean normal is diffusion target (learns normal distribution)
- Synthetic corruption is condition (guides reconstruction)
- Both use same frozen VQ encoder
- Independent noise and timesteps for target and condition
- U-Net has 8-channel input (4 target + 4 condition)

**Inference:**
```
For test image x_test:
  1. Encode: c_test = E(x_test)
  2. Add moderate noise: c_noisy = q_sample(c_test, t ∈ [400,500])
  3. Initialize reconstruction from noise: z_T ~ N(0,I)
  4. DDIM sampling conditioned on c_noisy (50 steps)
  5. Decode: x_reconstructed = D(z_0)
```

### 3.2 Discriminative Network

**Architecture:** U-Net segmentation network

**Input:** 9-channel RGB concatenation:
- Anomalous input image (3 channels)
- Interpolated image (3 channels)  
- Reconstructed image (3 channels)

**Interpolation:**
```
z_inter = λ·c + (1-λ)·z_r
x_inter = D(z_inter)
where λ = 0.5 (default, configurable)
```

**Training:**
- Data: Synthetic DRAEM anomalies
- Target: Binary anomaly masks
- Loss: Focal loss (or BCE, verify from official code)
- Frozen: VQ encoder/decoder and reconstruction network

**Output:** Pixel-level anomaly probability map (1 channel, 480×640)

---

## 4. First-Stage Model (Frozen)

### 4.1 VQ Autoencoder

**Checkpoint:** `checkpoints/experiment1/vq_nomixed/best_model (1).pth`

**Architecture:**
- Encoder: 3 → 32 → 64 → 128 → 256 (4 downsampling levels)
- Latent: 4×60×80 (8× spatial compression)
- Codebook: 1024 embeddings, ~178 active
- Decoder: Symmetric upsampling

**Latent Representation:** Quantized (discrete codebook entries)

**Interface:**
```python
z_quantized, z_continuous, vq_losses = vq_model.encode(x)
x_reconstructed = vq_model.decode(z_quantized)
```

**Statistics (Quantized Latents):**
- Mean: 0.012896
- Std: 0.199094
- Measured on: 1654 training images (No_Mixed)

**Scaling:**
```python
z_scaled = (z_quantized - 0.012896) / 0.199094
z_unscaled = z_scaled * 0.199094 + 0.012896
```

**Frozen:** All weights frozen during DiffAD training

### 4.2 Latent Conventions

**Use quantized latents for:**
- Clean target encoding
- Synthetic condition encoding
- Inference condition encoding
- Reconstruction decoding
- Latent interpolation

**Scaling applied to:**
- Both target and condition latents (same statistics)
- Before diffusion (training and inference)
- Unscaled before final decode

**Do NOT:**
- Mix quantized and continuous latents
- Use different statistics for target vs condition
- Apply scaling twice
- Decode without unscaling

---

## 5. Conditional Diffusion Model

### 5.1 Architecture

**U-Net Configuration:**
- Input channels: 8 (4 latent + 4 condition)
- Output channels: 4 (predicted noise)
- Base channels: 64 (adjust for target ~25M params)
- Channel multipliers: [1, 2, 4, 4]
- Attention levels: [2, 3]
- Resolution: 60×80 latent spatial dimensions

**Conditioning Mechanism:** Channel concatenation

**Timestep Embedding:** Standard sinusoidal + MLP (256-dim)

### 5.2 Noise Schedule

**Type:** Cosine schedule  
**Timesteps:** T = 1000  
**Parameter:** s = 0.008 (same as unconditional LDM)

**Forward Diffusion:**
```python
q_sample(z, t, eps):
  return sqrt(alpha_bar_t) * z + sqrt(1 - alpha_bar_t) * eps
```

### 5.3 Training Configuration

**Optimizer:** Adam
- Learning rate: 1e-4
- Betas: (0.9, 0.999)
- Weight decay: 0.0

**Batch size:** 8  
**Epochs:** 100 (initial target)  
**Device:** GPU (Kaggle P100/T4)

**Data:**
- Training: ~1654 No_Mixed images with DRAEM augmentation
- Validation: ~414 No_Mixed images (held-out)
- 50% training samples are clean (no synthetic anomaly)

**Checkpointing:**
- Save every 10 epochs
- Save best model (validation loss)
- Track EMA weights (decay=0.9999)

**Random Seeds:**
- Fixed seeds for reproducibility
- Different seed per epoch for DRAEM variation

### 5.4 Sampling

**Training:** Random timesteps `t ~ Uniform(0, 1000)`

**Inference:**
- Timesteps: 50 DDIM steps
- Condition noise: `t_inf ~ Uniform(400, 500)`
- Condition mode: `c_noisy` (default), `c` (ablation)
- DDIM eta: 0.0 (deterministic)

**Configurable Parameters:**
- Inference timestep range [t_min, t_max]
- Number of DDIM steps
- Condition noise mode (c vs c_noisy)

---

## 6. DRAEM Synthetic Anomaly Generation

### 6.1 Purpose

Generate synthetic corruptions for:
1. Conditional diffusion training (teach reconstruction)
2. Discriminative network training (teach segmentation)

**Not equivalent to:** Real thermal mixed pixels

### 6.2 Method

**Based on:** DRAEM (Zavrtanik et al., 2021) as used in official DiffAD

**Texture Source:** DTD dataset (describable textures)

**Augmentation Pipeline:**
1. Load random texture image
2. Apply 3 random augmentations:
   - Gamma contrast (0.5-2.0)
   - Brightness (±30)
   - Sharpness enhancement
   - Hue/saturation shift (±50)
   - Solarize
   - Posterize
   - Invert
   - Autocontrast
   - Equalize
   - Rotation (±45°)
3. Generate Perlin noise mask:
   - Scales: 2^0 to 2^6
   - Threshold: 0.5 (binary mask)
4. Blend texture with image:
   ```python
   beta = random(0, 0.8)
   x_synth = x_normal * (1 - mask) + \
             (1 - beta) * texture * mask + \
             beta * x_normal * mask
   ```

**Properties:**
- Smooth boundaries (Perlin noise)
- Texture diversity (DTD + augmentation)
- Variable opacity (beta parameter)
- Random location and scale

**Training Ratio:**
- 50% synthetic anomalies
- 50% clean (no augmentation)

### 6.3 Output

```python
{
    'image': x_normal,          # Clean (target)
    'augmented_image': x_synth,  # Corrupted (condition)
    'mask': binary_mask,        # Ground truth for discriminator
    'has_anomaly': bool         # Sample indicator
}
```

### 6.4 Limitations

**Known limitations:**
- Synthetic textures ≠ real thermal mixed pixels
- No physics-based temperature mixing
- No camera PSF or MTF modeling
- Perlin masks may not match real boundary shapes

**Future work:** Thermal-aware corruption (separate experiment)

**Current scope:** DRAEM baseline replication only

---

## 7. Discriminative Network

### 7.1 Architecture

**Base:** U-Net encoder-decoder

**Configuration:**
- Input channels: 9 (3 RGB images concatenated)
- Output channels: 1 (anomaly probability)
- Base channels: 64 (adjust for memory)
- Encoder: 4 downsampling levels
- Decoder: Symmetric upsampling
- Skip connections: Concatenation
- Output activation: Sigmoid

### 7.2 Input Construction

**Three-image concatenation:**
1. Anomalous input `x_a` (or synthetic `x_synth`)
2. Interpolated image `x_inter`
3. Reconstructed image `x_r`

**Interpolation in latent space:**
```python
# Option 1: Interpolate scaled latents
z_inter_scaled = 0.5 * c_scaled + 0.5 * z_r_scaled
z_inter_unscaled = scaler.unscale(z_inter_scaled)
x_inter = vq_decoder(z_inter_unscaled)

# Option 2: Interpolate unscaled latents (verify which is correct)
z_inter = 0.5 * c_unscaled + 0.5 * z_r_unscaled
x_inter = vq_decoder(z_inter)
```

**Lambda parameter:** 0.5 (default), configurable

### 7.3 Training

**Data:**
- Synthetic DRAEM anomalies (generated on-the-fly or pre-generated)
- Binary masks as ground truth
- No_Mixed as source images

**Frozen components:**
- VQ encoder/decoder (from Experiment 1)
- Conditional LDM (from DiffAD Phase 1)

**Trainable:** Discriminative U-Net only

**Loss:** Focal loss (or BCE, verify from `seg_network/train.py`)
- Handles class imbalance (small anomaly regions)
- Focuses on hard examples

**Optimizer:** Adam
- Learning rate: 1e-3
- Batch size: 32
- Epochs: 700 (from official DiffAD)

**Augmentation:** None (images already processed)

### 7.4 Output

**Pixel-level probability map:**
- Shape: (1, 480, 640)
- Range: [0, 1] (after sigmoid)
- Interpretation: P(anomaly | pixel)

**Post-processing:** None (raw probabilities)

---

## 8. Evaluation

### 8.1 Reconstruction Quality (Clean Validation)

**Dataset:** No_Mixed validation set (~414 images)

**Metrics:**
- MSE (mean squared error)
- MAE (mean absolute error)
- PSNR (peak signal-to-noise ratio)
- SSIM (structural similarity)

**Comparisons:**
- Conditional DiffAD reconstruction
- Unconditional LDM generation (baseline)
- VQ autoencoder reconstruction (baseline)
- CAE reconstruction (Experiment 0 baseline)

**Boundary Analysis:**
- Boundary MSE (Sobel 75th percentile mask)
- Non-boundary MSE
- Boundary/non-boundary ratio
- Gradient magnitude preservation

### 8.2 Detection Performance (Synthetic Anomalies)

**Dataset:** Held-out synthetic DRAEM validation set

**Metrics:**
- Pixel-level AUROC
- Pixel-level average precision (AP)
- Pixel-level F1-score (optimal threshold)
- IoU (intersection over union)
- Dice coefficient

**Detection methods:**
- DiffAD discriminator (primary)
- Reconstruction residual: `|x_test - x_r|` (baseline)
- Reconstruction MSE per pixel (baseline)

**Ablations:**
- Condition mode: c_noisy vs c (clean)
- Interpolation lambda: 0.3, 0.5, 0.7
- Inference timestep range: [300,400], [400,500], [500,600]

### 8.3 Qualitative Analysis

**Visualizations:**
- Input → Reconstruction pairs
- Input → Interpolation → Reconstruction triplets
- Anomaly probability heatmaps
- Reconstruction residuals
- Boundary overlays

**Inspection:**
- Reconstruction quality on clean images
- Anomaly removal effectiveness
- Boundary preservation
- False positive patterns
- Off-manifold interpolation artifacts

### 8.4 Limitations

**No real mixed-pixel evaluation:**
- No verified ground-truth mixed-pixel annotations
- Synthetic anomaly performance ≠ real detection performance
- Sobel boundary mask is a proxy, not ground truth

**Claims forbidden without evidence:**
- Real mixed-pixel detection accuracy
- Real boundary preservation
- Generalization to real thermal anomalies

**Permitted claims:**
- Synthetic anomaly detection performance
- Reconstruction quality on clean validation
- Relative comparison with baselines (same limitations)

---

## 9. Implementation Requirements

### 9.1 Code Organization

**New modules:**
```
src/models/diffad/
  ├── __init__.py
  ├── conditional_ldm.py        # Conditional diffusion model
  ├── discriminator.py          # Discriminative U-Net
  └── draem.py                  # DRAEM augmentation

src/datasets/
  └── diffad_dataset.py         # Paired clean/corrupted dataset

src/training/
  └── diffad_trainer.py         # Training loop

scripts/
  ├── train_diffad_ldm.py       # Conditional LDM training
  ├── train_diffad_discriminator.py  # Discriminator training
  └── evaluate_diffad.py        # Evaluation pipeline

configs/diffad/
  ├── ldm_train.yaml            # LDM training config
  ├── discriminator_train.yaml  # Discriminator config
  └── eval.yaml                 # Evaluation config

tests/
  ├── test_diffad_dataset.py
  ├── test_conditional_ldm.py
  ├── test_discriminator.py
  └── test_draem.py
```

**Preserve existing:**
- VQ autoencoder code and checkpoint
- Unconditional LDM code and checkpoint
- Experiment 1 evaluation code

### 9.2 Configuration Files

**YAML configs for:**
- Model architecture (U-Net channels, attention, etc.)
- Training hyperparameters
- Data paths and splits
- DRAEM augmentation parameters
- Evaluation settings
- Inference modes

**No hard-coded:**
- Dataset paths
- Checkpoint paths
- Model dimensions
- Hyperparameters

### 9.3 Tests

**Unit tests:**
- DRAEM augmentation (mask alignment, value ranges)
- Dataset __getitem__ (shape, paired samples)
- Conditional U-Net forward (8-channel input, 4-channel output)
- Latent scaling/unscaling (invertibility)
- Discriminator forward (9-channel input, 1-channel output)
- Latent interpolation (off-manifold handling)

**Integration tests:**
- Full training step (no crashes)
- Full inference pipeline (input → reconstruction)
- Discriminator pipeline (input → probability map)

**Smoke test:**
- 5-10 epoch training run
- Validation metrics logged
- Checkpoint saving/loading
- Sample generation working

### 9.4 Reproducibility

**Fixed seeds:**
- Model initialization
- Data shuffling
- Noise sampling
- DRAEM augmentation (per-epoch seed)

**Logged:**
- All hyperparameters
- Exact dataset splits
- Random seeds used
- Software versions (PyTorch, etc.)

**Checkpoints include:**
- Model state dict
- Optimizer state
- EMA state (if used)
- Training epoch
- Configuration dict
- Loss history

---

## 10. Success Criteria

### 10.1 Conditional LDM

✅ **Training converges:**
- Validation loss decreases
- No NaN/Inf values
- No mode collapse

✅ **Reconstructions are valid:**
- Visual quality comparable to unconditional LDM
- Boundary preservation comparable to VQ
- No obvious artifacts

✅ **Conditioning works:**
- Reconstructions respect input structure
- Different inputs produce different reconstructions
- Clean inputs reconstruct cleanly
- Ablation: c_noisy ≠ c produces different results

### 10.2 Discriminative Network

✅ **Training converges:**
- Validation segmentation loss decreases
- AUROC > 0.7 on synthetic validation

✅ **Segmentation is sensible:**
- High probability on anomalous regions
- Low probability on normal regions
- Smooth probability maps (not noisy)

✅ **Interpolation works:**
- Decoder handles interpolated latents
- x_inter visually between x_a and x_r
- No severe artifacts

### 10.3 Comparison with Baselines

✅ **DiffAD ≥ Reconstruction residual baseline:**
- AUROC improvement on synthetic anomalies
- Visual inspection shows better localization

✅ **DiffAD reconstructions ≥ Unconditional LDM:**
- Boundary preservation equal or better
- Reconstruction quality comparable

✅ **Clear documentation of limitations:**
- Synthetic vs real distinction clear
- No unsupported real detection claims
- Proxy vs ground-truth distinction maintained

### 10.4 Ready for Review

✅ **All tests passing**  
✅ **Smoke test successful**  
✅ **Qualitative samples generated**  
✅ **Quantitative metrics computed**  
✅ **Implementation report written**  
✅ **Code documented and clean**

**Then:** STOP and wait for user approval before full training

---

## 11. Out of Scope (Future Work)

### Not in this implementation:

❌ **Thermal-aware corruption:**
- Physics-based temperature mixing
- Emissivity modeling
- Camera PSF/MTF modeling
→ Separate research proposal required

❌ **Real mixed-pixel annotation:**
- Expert labeling
- Multi-camera validation
- Ground-truth establishment
→ Data collection effort required

❌ **Architecture search:**
- Different U-Net configurations
- Alternative conditioning mechanisms
- Hyperparameter tuning
→ Ablation studies after baseline works

❌ **Multi-dataset evaluation:**
- Okra, Chilli_Leaves, Paddy_Leaves
- Cross-dataset generalization
→ Investigate dataset roles first

❌ **Deployment optimization:**
- Model quantization
- Inference speedup
- Memory optimization
→ Research phase focuses on correctness

---

## 12. Risk Mitigation

### Risk: Decoder fails on interpolated latents

**Mitigation:**
- Test decoder on interpolated latents before discriminator training
- Visualize interpolations qualitatively
- Measure reconstruction quality (MSE, SSIM)
- If severe artifacts: adjust lambda or use alternative interpolation

### Risk: Conditional training diverges

**Mitigation:**
- Start with small learning rate (1e-4)
- Monitor validation loss closely
- Use gradient clipping if needed
- Compare with unconditional LDM training curves

### Risk: DRAEM doesn't match real anomalies

**Mitigation:**
- Document as known limitation
- Use for internal comparison only
- Don't claim real detection without validation
- Prepare thermal-aware proposal for future work

### Risk: Discriminator overfits to DRAEM

**Mitigation:**
- Use 50% clean samples in training
- Strong augmentation in DRAEM
- Validation set from different source images
- Monitor validation AUROC

### Risk: Implementation deviates from official DiffAD

**Mitigation:**
- Cross-reference with official code at every step
- Document all assumptions and deviations
- Implement ablations for ambiguous choices
- Preserve traceability to paper and code

---

## 13. Approval Required

Before proceeding to implementation:

☐ User approves overall methodology  
☐ User approves using quantized latents (not continuous)  
☐ User approves conditional training (not fine-tuning unconditional)  
☐ User approves DRAEM baseline (not thermal-specific yet)  
☐ User approves independent timesteps for target and condition  
☐ User approves smoke test before full training

**Status:** Awaiting user approval

