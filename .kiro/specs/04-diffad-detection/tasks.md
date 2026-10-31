# DiffAD Detection - Implementation Tasks

**Status:** Specification Phase  
**Updated:** 2026-10-03

---

## Implementation Sequence

### Phase 1: DRAEM Dataset Implementation

**Task 1.1: DRAEM Augmenter**
- [ ] Implement Perlin noise generation
- [ ] Implement texture augmentation pipeline
- [ ] Implement mask generation and blending
- [ ] Test mask alignment with augmented regions
- [ ] Verify 50% clean sample ratio

**Task 1.2: DiffAD Dataset**
- [ ] Implement paired clean/corrupted dataset class
- [ ] Verify shapes: image (3,480,640), mask (1,480,640)
- [ ] Test that clean and augmented come from same source
- [ ] Implement train/validation split (80/20, seed=42)
- [ ] Add unit tests

**Task 1.3: DTD Texture Dataset**
- [ ] Download DTD dataset
- [ ] Organize texture paths
- [ ] Verify 5640 images available

---

### Phase 2: Conditional LDM Implementation

**Task 2.1: Conditional U-Net Architecture**
- [ ] Implement 8-channel input U-Net
- [ ] Add timestep embedding (sinusoidal + MLP)
- [ ] Add self-attention at levels 2-3
- [ ] Verify forward pass: (B,8,60,80) → (B,4,60,80)
- [ ] Count parameters (~23M target)
- [ ] Add unit tests

**Task 2.2: Training Loop**
- [ ] Implement conditional training step:
  - Encode clean and corrupted (frozen VQ)
  - Sample independent timesteps
  - Add noise independently
  - Concatenate and predict
  - MSE loss on target noise
- [ ] Add gradient clipping
- [ ] Add validation loop
- [ ] Add checkpointing
- [ ] Add EMA tracking (optional)

**Task 2.3: Configuration**
- [ ] Create ldm_train.yaml config file
- [ ] Implement config loading
- [ ] Make all hyperparameters configurable

**Task 2.4: Testing**
- [ ] Unit test: U-Net forward pass
- [ ] Unit test: Training step executes
- [ ] Unit test: Latent scaling invertible
- [ ] Integration test: Full epoch completes
- [ ] Smoke test: 5 epochs, qualitative check

---

### Phase 3: Conditional Inference

**Task 3.1: DDIM Sampling with Conditioning**
- [ ] Implement conditional DDIM sampler
- [ ] Add condition noise injection (t ∈ [400,500])
- [ ] Implement c_noisy mode (default)
- [ ] Implement c clean mode (ablation)
- [ ] Make configurable

**Task 3.2: Reconstruction Pipeline**
- [ ] Implement end-to-end reconstruction:
  - Encode test image
  - Add condition noise
  - DDIM sampling (50 steps)
  - Unscale and decode
- [ ] Test on clean validation images
- [ ] Measure reconstruction quality (MSE, SSIM)
- [ ] Compare with unconditional LDM baseline

**Task 3.3: Ablation Studies**
- [ ] Test c vs c_noisy
- [ ] Test different t_inf ranges
- [ ] Test different DDIM steps (25, 50, 100)

---

### Phase 4: Discriminative Network

**Task 4.1: Latent Interpolation**
- [ ] Implement interpolation: λ·c + (1-λ)·z_r
- [ ] Test decoder on interpolated latents
- [ ] Check for artifacts
- [ ] Verify visual quality (x_inter between x_a and x_r)
- [ ] Make λ configurable (default 0.5)

**Task 4.2: Discriminator Architecture**
- [ ] Implement 9-channel input U-Net
- [ ] Add encoder-decoder with skip connections
- [ ] Add sigmoid output activation
- [ ] Verify forward: (B,9,480,640) → (B,1,480,640)
- [ ] Count parameters

**Task 4.3: Discriminator Training**
- [ ] Freeze VQ and LDM
- [ ] Implement on-the-fly reconstruction generation
- [ ] Implement interpolation
- [ ] Implement 9-channel concatenation
- [ ] Implement focal loss (or BCE)
- [ ] Add validation AUROC tracking
- [ ] Add checkpointing

**Task 4.4: Configuration**
- [ ] Create discriminator_train.yaml
- [ ] Implement config loading

**Task 4.5: Testing**
- [ ] Unit test: Discriminator forward
- [ ] Unit test: Interpolation works
- [ ] Integration test: Training step executes
- [ ] Smoke test: 50 epochs, check AUROC > 0.5

---

### Phase 5: Evaluation

**Task 5.1: Reconstruction Metrics**
- [ ] Implement MSE, MAE, PSNR, SSIM
- [ ] Implement boundary metrics
- [ ] Compute on clean validation set
- [ ] Compare DiffAD vs unconditional LDM vs VQ vs CAE

**Task 5.2: Detection Metrics**
- [ ] Implement pixel-level AUROC, AP, F1
- [ ] Implement IoU, Dice
- [ ] Compute on synthetic validation set
- [ ] Compare DiffAD discriminator vs residual baseline

**Task 5.3: Visualization**
- [ ] Generate input → reconstruction pairs
- [ ] Generate input → interp → recon triplets
- [ ] Generate probability heatmaps
- [ ] Generate residual maps
- [ ] Overlay boundaries

**Task 5.4: Evaluation Script**
- [ ] Create evaluate_diffad.py
- [ ] Generate qualitative samples
- [ ] Compute quantitative metrics
- [ ] Save results to JSON/CSV
- [ ] Generate visualization grid

---

### Phase 6: Documentation and Reporting

**Task 6.1: Code Documentation**
- [ ] Add docstrings to all classes/functions
- [ ] Add inline comments for complex logic
- [ ] Update README with usage instructions

**Task 6.2: Implementation Report**
- [ ] Document architecture choices
- [ ] List all modified/new files
- [ ] Document deviations from official DiffAD
- [ ] List remaining ambiguities
- [ ] Document test results

**Task 6.3: Results Summary**
- [ ] Reconstruction quality table
- [ ] Detection performance table
- [ ] Baseline comparisons
- [ ] Qualitative samples
- [ ] Limitation statement

---

## Testing Checklist

### Unit Tests
- [ ] DRAEM augmentation
- [ ] DiffAD dataset
- [ ] Conditional U-Net forward
- [ ] Latent scaling
- [ ] Discriminator forward
- [ ] Latent interpolation

### Integration Tests
- [ ] Full LDM training step
- [ ] Full LDM inference
- [ ] Full discriminator training step
- [ ] Full detection pipeline

### Smoke Tests
- [ ] LDM: 5 epochs, visual check
- [ ] Discriminator: 50 epochs, AUROC check

### Validation Tests
- [ ] Reconstruction quality ≈ VQ
- [ ] Conditioning demonstrably works
- [ ] Detection AUROC > 0.7
- [ ] No NaN/Inf values
- [ ] All shapes correct

---

## File Structure

**New files to create:**
```
src/models/diffad/
  ├── __init__.py
  ├── conditional_ldm.py
  ├── discriminator.py
  └── draem.py

src/datasets/
  └── diffad_dataset.py

src/training/
  └── diffad_trainer.py

scripts/
  ├── train_diffad_ldm.py
  ├── train_diffad_discriminator.py
  └── evaluate_diffad.py

configs/diffad/
  ├── ldm_train.yaml
  ├── discriminator_train.yaml
  └── eval.yaml

tests/
  ├── test_diffad_dataset.py
  ├── test_conditional_ldm.py
  ├── test_discriminator.py
  └── test_draem.py

notebooks/
  ├── kaggle_diffad_ldm.ipynb
  └── kaggle_diffad_discriminator.ipynb
```

---

## Dependencies

**Python packages:**
- torch >= 1.12
- diffusers >= 0.30
- imgaug >= 0.4
- scikit-image
- scikit-learn
- numpy
- opencv-python
- pyyaml
- tqdm

**Datasets:**
- DTD (Describable Textures Dataset)
- No_Mixed thermal images (existing)

**Checkpoints:**
- VQ autoencoder (existing)
- Unconditional LDM (existing, for comparison)

---

## Estimated Timeline

**Phase 1:** 2-3 days  
**Phase 2:** 3-4 days  
**Phase 3:** 1-2 days  
**Phase 4:** 3-4 days  
**Phase 5:** 2-3 days  
**Phase 6:** 1-2 days

**Total:** ~2-3 weeks implementation + testing

**Smoke Test:** 1 day  
**Full Training:** 5-7 days (Kaggle)

---

## Approval Gates

**After Phase 2 (LDM Implementation):**
- [ ] User reviews smoke test results
- [ ] User approves proceeding to discriminator

**After Phase 4 (Discriminator Implementation):**
- [ ] User reviews detection metrics
- [ ] User approves full training

**After Phase 5 (Evaluation):**
- [ ] User reviews complete results
- [ ] User decides on next steps

---

## Notes

- Preserve all existing baselines (VQ, CAE, unconditional LDM)
- Save all checkpoints separately
- Document all assumptions
- Stop and ask if anything is unclear
- No full training without approval

