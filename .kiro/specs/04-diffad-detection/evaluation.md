# DiffAD Detection - Evaluation

**Status:** Specification Phase  
**Updated:** 2026-10-03

---

## 1. Evaluation Overview

DiffAD will be evaluated on three criteria:

1. **Reconstruction Quality:** How well does the conditional LDM reconstruct clean normal images?
2. **Detection Performance:** How well does the discriminator detect synthetic anomalies?
3. **Baseline Comparison:** How does DiffAD compare to simpler methods?

**Important:** All evaluation uses synthetic anomalies (DRAEM). No claims about real mixed-pixel detection are permitted without verified annotations.

---

## 2. Datasets

### 2.1 Clean Validation Set

**Source:** No_Mixed held-out validation split  
**Size:** ~414 images (20% of 2068)  
**Purpose:** Reconstruction quality evaluation  
**Ground Truth:** None (clean images only)

### 2.2 Synthetic Validation Set

**Source:** No_Mixed held-out validation + DRAEM augmentation  
**Size:** ~414 image pairs (clean + synthetic corruption)  
**Purpose:** Detection performance evaluation  
**Ground Truth:** Binary DRAEM masks

**Note:** Ensure validation source images are disjoint from training set.

---

## 3. Reconstruction Metrics

### 3.1 Image-Level Quality

**Computed on clean validation set:**

```python
metrics = {
    'MSE': mean_squared_error(x_true, x_recon),
    'MAE': mean_absolute_error(x_true, x_recon),
    'PSNR': peak_signal_noise_ratio(x_true, x_recon),
    'SSIM': structural_similarity(x_true, x_recon)
}
```

**Aggregation:** Mean and std across validation set

**Comparisons:**
- DiffAD (conditional reconstruction)
- Unconditional LDM (generation from noise)
- VQ autoencoder (direct reconstruction)
- CAE (Experiment 0 baseline)

**Target:** DiffAD ≈ VQ ≈ Unconditional LDM

### 3.2 Boundary-Specific Metrics

**Boundary Proxy:** Sobel gradient magnitude > 75th percentile

```python
boundary_metrics = {
    'boundary_MSE': MSE on boundary pixels only,
    'non_boundary_MSE': MSE on non-boundary pixels only,
    'boundary_ratio': boundary_MSE / non_boundary_MSE,
    'gradient_preservation': cosine_similarity(grad_true, grad_recon)
}
```

**Target:** Boundary ratio close to 1.0 (equal reconstruction quality)

**Limitation:** Boundary mask is a proxy, NOT ground-truth mixed pixels

### 3.3 Qualitative Assessment

**Visual inspection:**
- Sample reconstructions (grid of 20-50 images)
- Zoom into boundary regions
- Check for artifacts
- Compare side-by-side with baselines

**Questions:**
- Does reconstruction look normal?
- Are boundaries preserved?
- Are there off-manifold artifacts?
- How does it compare to VQ/LDM baselines?

---

## 4. Detection Metrics

### 4.1 Pixel-Level Metrics

**Computed on synthetic validation set:**

```python
detection_metrics = {
    'AUROC': roc_auc_score(mask_flat, prob_flat),
    'AP': average_precision_score(mask_flat, prob_flat),
    'F1': best F1 score across thresholds,
    'optimal_threshold': threshold that maximizes F1,
    'IoU': intersection_over_union at optimal threshold,
    'Dice': dice_coefficient at optimal threshold
}
```

**Primary Metric:** AUROC (threshold-independent)

**Target:** 
- AUROC > 0.7 (minimum acceptable)
- AUROC > 0.85 (good performance)
- AUROC > 0.9 (excellent performance, may indicate overfitting to DRAEM)

**Comparisons:**
- DiffAD discriminator (primary)
- Reconstruction residual: `|x_test - x_r|` (baseline)
- Per-pixel reconstruction MSE (baseline)

### 4.2 Image-Level Metrics

**Max pooling detection:**

```python
image_score = prob_map.max()  # or .mean(), .95_percentile
image_label = has_anomaly
```

**Metrics:**
- Image-level AUROC
- Image-level AP
- Detection rate at 5% FPR

**Purpose:** Check if discriminator can identify anomalous images at all

### 4.3 Ablation Studies

**Test variations:**

1. **Condition mode:**
   - c_noisy (default, paper)
   - c (clean, official script)
   - **Metric:** AUROC difference

2. **Interpolation lambda:**
   - λ = 0.3, 0.5 (default), 0.7
   - **Metric:** AUROC at each value

3. **Inference timestep range:**
   - [300, 400], [400, 500] (default), [500, 600]
   - **Metric:** AUROC at each range

4. **Number of DDIM steps:**
   - 25, 50 (default), 100
   - **Metric:** AUROC vs inference time trade-off

**Purpose:** Understand robustness and find optimal settings

### 4.4 Qualitative Assessment

**Visualizations:**

```
Input (x_test) | Interpolated (x_inter) | Reconstructed (x_r) | Probability Map
───────────────────────────────────────────────────────────────────────────────
[Anomalous]    | [Between]              | [Normal]             | [Heatmap]
```

**Questions:**
- Do probability maps align with masks?
- Are false positives concentrated anywhere?
- Does interpolation provide useful signal?
- Are reconstructions consistently normal?

---

## 5. Baseline Comparisons

### 5.1 Reconstruction Baselines

**Models to compare:**

| Model | Type | Purpose |
|-------|------|---------|
| CAE | Autoencoder | Experiment 0 baseline |
| VQ | VQ-VAE | First-stage, direct reconstruction |
| Unconditional LDM | Diffusion | LDM Stage 2 baseline |
| DiffAD LDM | Conditional Diffusion | Our implementation |

**Metrics:** All reconstruction metrics (Section 3)

**Expected Ranking:** VQ ≈ DiffAD ≈ Unconditional LDM > CAE

### 5.2 Detection Baselines

**Methods to compare:**

| Method | Description |
|--------|-------------|
| Residual (L1) | `\|x_test - x_r\|` mean across channels |
| Residual (L2) | `(x_test - x_r)^2` mean across channels |
| Per-pixel MSE | Channel-wise MSE retained |
| DiffAD Discriminator | Full method |

**Metrics:** All detection metrics (Section 4.1)

**Expected Ranking:** DiffAD > Per-pixel MSE > Residual

**Target:** DiffAD improves AUROC by ≥5% over best baseline

### 5.3 Statistical Significance

**Test:** Paired t-test or Wilcoxon signed-rank

**Compare:**
- DiffAD AUROC vs Residual AUROC (per image)
- Null hypothesis: No difference
- α = 0.05

**Report:** p-value and confidence interval

---

## 6. Evaluation Protocol

### 6.1 Data Splits

**Training:**
- No_Mixed train split (80%)
- DRAEM augmentation applied
- Used for LDM and discriminator training

**Validation:**
- No_Mixed val split (20%)
- Held-out from training
- Used for all evaluation

**Seed:** 42 (fixed for reproducibility)

**Verification:** No overlap between train and val image paths

### 6.2 Inference Settings

**Conditional LDM:**
- DDIM steps: 50
- Condition noise: t ∈ [400, 500]
- Condition mode: c_noisy (default)
- DDIM eta: 0.0 (deterministic)

**Discriminator:**
- Interpolation: λ = 0.5
- Batch size: 8 (or 1 if memory constrained)

**Device:** GPU (CUDA)

**Precision:** Float32 (same as training)

### 6.3 Reproducibility

**Fixed:**
- Random seed (42)
- Model checkpoints (save exact epoch)
- Dataset split (same seed)
- DRAEM seed (per-image, from index)

**Logged:**
- Exact checkpoint paths
- Configuration files
- Software versions (PyTorch, diffusers, etc.)
- Hardware (GPU model)

---

## 7. Reporting

### 7.1 Quantitative Results

**Table 1: Reconstruction Quality (Clean Validation)**

| Model | MSE ↓ | MAE ↓ | PSNR ↑ | SSIM ↑ | Boundary Ratio |
|-------|-------|-------|--------|--------|----------------|
| CAE | ... | ... | ... | ... | ... |
| VQ | ... | ... | ... | ... | ... |
| Unconditional LDM | ... | ... | ... | ... | ... |
| DiffAD LDM | ... | ... | ... | ... | ... |

**Table 2: Detection Performance (Synthetic Validation)**

| Method | AUROC ↑ | AP ↑ | F1 ↑ | IoU ↑ | Dice ↑ |
|--------|---------|------|------|-------|--------|
| Residual (L1) | ... | ... | ... | ... | ... |
| Residual (L2) | ... | ... | ... | ... | ... |
| Per-pixel MSE | ... | ... | ... | ... | ... |
| DiffAD Discriminator | ... | ... | ... | ... | ... |

**Table 3: Ablation Studies**

| Configuration | AUROC ↑ | Notes |
|---------------|---------|-------|
| c_noisy (default) | ... | Paper method |
| c (clean) | ... | Official script |
| λ = 0.3 | ... | Less reconstruction weight |
| λ = 0.7 | ... | More reconstruction weight |
| t ∈ [300,400] | ... | Less condition noise |
| t ∈ [500,600] | ... | More condition noise |

### 7.2 Qualitative Results

**Figures:**

1. **Reconstruction Grid:** 4×5 grid of input → reconstruction pairs
2. **Detection Visualization:** Input | Interp | Recon | Prob Map | Mask
3. **Boundary Comparison:** Zoomed boundary regions across methods
4. **Error Maps:** Heatmaps of reconstruction error
5. **ROC Curves:** All detection methods on same plot
6. **Precision-Recall Curves:** All detection methods

### 7.3 Limitation Statement

**Required in all reports:**

> "This evaluation uses synthetic anomalies generated via DRAEM texture blending. These synthetic corruptions do not physically model real thermal mixed pixels, which result from multiple thermal contributions within a single camera pixel's field of view. 
>
> The reported detection metrics (AUROC, AP, F1) measure performance on DRAEM-style synthetic anomalies only. These metrics do NOT represent detection accuracy on real mixed pixels without independent validation.
>
> The boundary preservation metrics use a Sobel gradient-based boundary proxy (75th percentile threshold). This proxy is NOT equivalent to ground-truth mixed-pixel annotations. Boundary preservation claims are relative comparisons only.
>
> Generalization to real mixed pixels is unknown and requires:
> 1. Verified ground-truth mixed-pixel annotations, OR
> 2. Physics-based thermal corruption models, OR
> 3. Multi-camera validation experiments"

---

## 8. Success Criteria

### 8.1 Minimum Acceptable Performance

**Reconstruction:**
- ✅ MSE comparable to VQ (within 10%)
- ✅ SSIM > 0.9 on clean validation
- ✅ No catastrophic artifacts

**Detection:**
- ✅ AUROC > 0.7 on synthetic validation
- ✅ Visual alignment of prob maps with masks
- ✅ Better than residual baseline (AUROC)

**Baselines:**
- ✅ DiffAD ≥ Unconditional LDM (reconstruction)
- ✅ DiffAD > Residual baseline (detection)

### 8.2 Good Performance

**Reconstruction:**
- ✅ MSE within 5% of VQ
- ✅ SSIM > 0.95
- ✅ Boundary ratio 0.9-1.1

**Detection:**
- ✅ AUROC > 0.85
- ✅ F1 > 0.7 at optimal threshold
- ✅ >5% AUROC improvement over baseline

### 8.3 Excellent Performance

**Reconstruction:**
- ✅ MSE matches VQ
- ✅ SSIM > 0.97
- ✅ Boundary ratio 0.95-1.05

**Detection:**
- ✅ AUROC > 0.9
- ✅ F1 > 0.8
- ✅ >10% AUROC improvement over baseline

**Note:** Excellent detection may indicate overfitting to DRAEM patterns.

---

## 9. Failure Modes

### 9.1 Reconstruction Failures

**Symptoms:**
- High MSE (worse than VQ)
- Low SSIM (< 0.85)
- Visual artifacts
- Boundary degradation

**Possible Causes:**
- Conditioning not working
- Training diverged
- Latent interpolation issues
- Decoder struggles with off-manifold

**Diagnosis:**
- Check c_noisy vs c ablation
- Visualize latent distributions
- Test decoder directly on interpolations
- Compare with unconditional LDM

### 9.2 Detection Failures

**Symptoms:**
- AUROC < 0.7
- Random probability maps
- No improvement over residual
- High false positive rate

**Possible Causes:**
- Discriminator didn't train
- Interpolation not helpful
- Reconstructions don't remove anomalies
- Training/validation leakage

**Diagnosis:**
- Check discriminator training loss
- Visualize reconstructions (do anomalies remain?)
- Test without interpolation (only x_a, x_r)
- Verify data split integrity

### 9.3 Overfitting to DRAEM

**Symptoms:**
- AUROC > 0.95
- Perfect detection on validation
- Discriminator memorized DRAEM patterns

**Risk:**
- Won't generalize to real anomalies
- Focal loss may be too aggressive

**Mitigation:**
- Use strong DRAEM augmentation
- Include 50% clean samples
- Test on different texture sources
- Reduce discriminator capacity if needed

---

## 10. Evaluation Scripts

### 10.1 Reconstruction Evaluation

**Script:** `scripts/evaluate_diffad_reconstruction.py`

**Usage:**
```bash
python scripts/evaluate_diffad_reconstruction.py \
  --ldm_checkpoint checkpoints/diffad/ldm/best_model.pth \
  --vq_checkpoint checkpoints/experiment1/vq_nomixed/best_model(1).pth \
  --val_dir src/datasets/processed/No_Mixed/No_mixed \
  --output_dir outputs/diffad_eval/reconstruction \
  --num_samples 50
```

**Outputs:**
- `metrics.json` - Quantitative metrics
- `samples/` - Reconstruction visualizations
- `comparison.png` - Side-by-side with baselines

### 10.2 Detection Evaluation

**Script:** `scripts/evaluate_diffad_detection.py`

**Usage:**
```bash
python scripts/evaluate_diffad_detection.py \
  --ldm_checkpoint checkpoints/diffad/ldm/best_model.pth \
  --disc_checkpoint checkpoints/diffad/discriminator/best_model.pth \
  --vq_checkpoint checkpoints/experiment1/vq_nomixed/best_model(1).pth \
  --val_dir src/datasets/processed/No_Mixed/No_mixed \
  --output_dir outputs/diffad_eval/detection \
  --num_samples 100
```

**Outputs:**
- `metrics.json` - Detection metrics
- `roc_curve.png` - ROC curves
- `pr_curve.png` - Precision-recall curves
- `visualizations/` - Detection visualizations

### 10.3 Ablation Evaluation

**Script:** `scripts/evaluate_diffad_ablations.py`

**Usage:**
```bash
python scripts/evaluate_diffad_ablations.py \
  --ldm_checkpoint checkpoints/diffad/ldm/best_model.pth \
  --disc_checkpoint checkpoints/diffad/discriminator/best_model.pth \
  --vq_checkpoint checkpoints/experiment1/vq_nomixed/best_model(1).pth \
  --val_dir src/datasets/processed/No_Mixed/No_mixed \
  --output_dir outputs/diffad_eval/ablations
```

**Outputs:**
- `ablation_results.json` - All ablation metrics
- `ablation_comparison.png` - Bar chart comparison

---

## 11. Timeline

**Reconstruction Evaluation:** 2-4 hours (GPU)  
**Detection Evaluation:** 4-6 hours (GPU, includes reconstruction)  
**Ablation Studies:** 8-12 hours (GPU, multiple runs)  
**Analysis and Reporting:** 1-2 days

**Total:** 2-3 days for complete evaluation

---

## 12. Approval

After evaluation complete:

☐ User reviews quantitative results  
☐ User reviews qualitative samples  
☐ User reviews limitation statement  
☐ User approves conclusions  
☐ User decides on next steps (publish, iterate, new experiments)

