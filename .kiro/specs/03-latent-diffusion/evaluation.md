# Latent Diffusion Model (Stage 2) — Evaluation

## 1. Objective

Evaluate the trained LDM baseline against the VQ first-stage autoencoder baseline to determine:

1. Whether latent diffusion successfully generates realistic agricultural thermal images.
2. How generated sample quality compares to VQ reconstruction baseline.
3. Whether the diffusion model learned meaningful structure in the VQ latent space.
4. Training stability, convergence, and computational efficiency.
5. Suitability of the LDM baseline for future conditional extensions or architectural ablations.

The evaluation does NOT claim:
- Ground-truth mixed-pixel detection accuracy (no annotations available).
- Superiority in downstream classification (not evaluated).
- Input-specific reconstruction (unconditional model).

## 2. Research Questions

### 2.1 Generation Quality

1. Can the LDM generate thermal images that are visually realistic and structurally coherent?
2. Do generated samples exhibit thermal patterns consistent with the training distribution?
3. Are there visible artifacts (noise, blurriness, checkerboard patterns, mode collapse)?

### 2.2 Comparison with VQ Baseline

4. How does the quality of LDM-generated samples compare to VQ reconstruction of validation images?
5. Do LDM samples preserve thermal boundaries as well as VQ reconstructions?
6. What is the computational cost of LDM sampling vs VQ encode/decode?

### 2.3 Training Dynamics

7. Did the diffusion model training converge stably?
8. Was the validation loss trajectory consistent with learning?
9. Were EMA weights beneficial for generation quality?

### 2.4 Latent Space Learning

10. Did the diffusion model learn meaningful structure in the VQ latent space?
11. Are denoised latents consistent with the VQ latent distribution?
12. Do off-manifold latents (interpolations between codebook entries) decode successfully?

## 3. Scope and Constraints

### 3.1 What is Evaluated

- Generated sample quality (unconditional).
- Generated sample diversity.
- Comparison with VQ autoencoder reconstruction baseline.
- Training and validation loss curves.
- Sampling efficiency (time, memory).
- Boundary preservation (using gradient proxy).

### 3.2 What is NOT Evaluated

- Input-specific reconstruction accuracy (model is unconditional).
- Mixed-pixel detection accuracy (no ground-truth masks).
- Downstream classification improvement (not in scope).
- Conditional generation (not implemented in baseline).
- Alternative sampling methods (DDIM, DPM-Solver deferred).
- Alternative latent representations (pre-quant, discrete deferred).

### 3.3 Constraints

- Use same validation set as Experiment 1 (413 images from 80/20 split).
- Use frozen VQ checkpoint from Experiment 1.
- Use measured latent statistics from training set.
- Do not retrain VQ model or modify preprocessing.
- Do not fabricate results or modify evaluation after seeing outcomes.

## 4. Evaluation Data

### 4.1 Validation Set

**Dataset:** No_Mixed thermal images  
**Split:** 80/20 train/val, stratified, seed=42  
**Validation size:** 413 images  
**Same as:** Experiment 1 evaluation set

**Preprocessing:**
- Resize to 640×480 (INTER_AREA)
- ToTensor: [0,255] → [0,1]
- No additional normalization

### 4.2 Generated Samples

**Number:** At least 100 unconditional samples for quantitative analysis, more for qualitative grids.  
**Generation:** Using EMA weights, 50-step DDPM sampling, batch size 8-16.  
**Random seed:** Fixed for reproducibility.

## 5. Evaluation Dimensions

### 5.1 Training Metrics

**Collected during training:**

- Training loss per epoch (mean MSE between predicted and target noise).
- Validation loss per epoch.
- Gradient norms (optional diagnostic).
- EMA update statistics (optional).
- Training time per epoch.
- Peak memory usage.

**Visualize:**
- Loss curves (training and validation).
- Identify convergence epoch.
- Check for overfitting (validation loss increasing while training loss decreases).

**Success indicators:**
- Training loss decreases monotonically or with reasonable fluctuation.
- Validation loss decreases and stabilizes.
- No divergence (loss → ∞).
- No collapse (loss → 0 with poor sample quality).

### 5.2 Generated Sample Quality (Qualitative)

**Visual inspection of sample grids:**

Generate 4×4 or 8×8 grids of unconditional samples at multiple training checkpoints (e.g., epoch 10, 20, 50, 100, best).

**Assess:**
- Realism: Do samples resemble agricultural thermal images?
- Structure: Presence of recognizable spatial patterns (e.g., foliage, soil, thermal gradients).
- Sharpness: Are boundaries and details crisp, or blurred?
- Artifacts: Checkerboard patterns, high-frequency noise, color shifts.
- Diversity: Variation across samples (not mode collapse).

**Compare visually with:**
- Real validation images.
- VQ reconstruction of validation images.

**Limitations:**
- Subjective assessment.
- No ground-truth "ideal" thermal image.

### 5.3 Generated Sample Quality (Quantitative)

**Option 1: Distributional Comparison**

Compare generated sample statistics to validation set statistics:

**Pixel-level statistics:**
- Mean and std of pixel intensities (per channel).
- Histogram comparison (e.g., Earth Mover's Distance, KL divergence).

**Gradient statistics:**
- Mean gradient magnitude.
- Gradient histogram.

**Interpretation:**
- Similar statistics suggest generated samples match the data distribution.
- Does NOT prove perceptual quality or usefulness.

**Option 2: Fréchet Inception Distance (FID) — DEFERRED**

**Requires:**
- Pre-trained feature extractor (ImageNet Inception model or thermal-specific).
- Reference set (validation images).

**Limitations:**
- ImageNet features may not capture thermal-specific structure.
- FID is a proxy, not ground truth.

**Decision:** Defer FID to later evaluation if justified and feature extractor available.

### 5.4 Reconstruction Comparison (VQ Baseline)

**Since the LDM is unconditional, direct input-output reconstruction comparison is not possible.**

**Alternative approach:**

1. **VQ Baseline Reconstruction:**
   - Encode validation images with frozen VQ encoder.
   - Decode to obtain reconstructions.
   - Compute MSE, SSIM, boundary metrics (as in Experiment 1).

2. **LDM Generated Samples:**
   - Generate samples unconditionally.
   - Compute same statistics (mean intensity, gradient magnitude, boundary proxy coverage).
   - Compare distributions, NOT individual image reconstruction errors.

**Metrics for VQ reconstructions (reproduce from Experiment 1):**
- Mean Squared Error (MSE).
- Structural Similarity Index (SSIM).
- Boundary-region MSE.
- Non-boundary-region MSE.
- Boundary/non-boundary error ratio.

**Metrics for LDM generated samples:**
- Mean pixel intensity and std (per channel).
- Mean gradient magnitude.
- Boundary proxy coverage (fraction of pixels identified as boundary).

**Interpretation:**
- If LDM sample statistics closely match validation set, generation captures data distribution.
- If VQ reconstruction error is low, VQ baseline is strong.
- LDM cannot be directly compared to VQ on per-image reconstruction accuracy.

### 5.5 Boundary Preservation

**Gradient-based boundary proxy:**

Apply Sobel gradient filter + 75th percentile threshold (as in Experiment 1) to:
1. Validation images (ground reference).
2. VQ reconstructions.
3. LDM generated samples.

**Metrics:**
- Boundary pixel fraction.
- Mean gradient magnitude within boundary regions.
- Mean gradient magnitude in non-boundary regions.

**Compare:**
- VQ reconstructions vs validation images (boundary preservation in reconstruction).
- LDM samples vs validation images (boundary presence in generation).

**Critical limitation:**
- Proxy is NOT ground-truth mixed-pixel mask.
- High gradient in generated samples does not prove better mixed-pixel detection.
- Interpret as structural quality indicator only.

### 5.6 Sample Diversity

**Assess mode collapse:**

Generate multiple batches of samples and inspect:
- Visual variation across samples.
- Pixel-level variance across generated images.
- Perceptual hashing or feature-space distance (if feature extractor available).

**Success indicators:**
- Generated samples are visually diverse.
- Low pairwise similarity (not generating same image repeatedly).

**Failure indicators:**
- All samples look nearly identical.
- Limited variation in intensity, structure, or pattern.

### 5.7 Latent Space Analysis

**Denoised latent distribution:**

After sampling, before decoding:
1. Collect denoised latents `z_0` (scaled) from multiple samples.
2. Compare statistics to training set latent statistics:
   - Mean and std per channel.
   - Value ranges.
   - Spatial autocorrelation (optional).

**Check:**
- Are denoised latents in the expected range?
- Do they resemble VQ encoder outputs statistically?
- Are there outliers or off-manifold values?

**Decoder robustness test:**

Generate latents with known off-manifold properties (e.g., interpolate between codebook entries, add small perturbations) and decode:
- Do decoded images remain realistic?
- Are there visible artifacts?

**Interpretation:**
- Robust decoder tolerates diffusion-generated latents.
- If decoder fails on off-manifold latents, may indicate codebook underutilization issue.

### 5.8 Computational Efficiency

**Sampling time:**

Measure time to generate a batch of images:
- 50-step DDPM sampling (baseline).
- Include latent unscaling and decoding.
- Report per-image and per-batch time.

**Compare with VQ baseline:**
- VQ encode + decode time (single forward pass).

**Expected:**
- LDM sampling: ~2-5 seconds per image (50 steps × ~0.05s per step).
- VQ encode/decode: ~0.05-0.1 seconds per image.

**Trade-off:**
- LDM is ~20-50× slower than VQ.
- Acceptable if generation quality justifies cost.

**Memory usage:**
- Peak GPU memory during sampling.
- Verify fits within T4 16GB limit.

## 6. Decision Framework

### 6.1 Success Criteria

The LDM baseline is successful if:

1. **Training converged:** Validation loss decreased and stabilized without divergence.
2. **Samples are realistic:** Generated images visually resemble agricultural thermal images.
3. **Samples are diverse:** Multiple samples show variation, no mode collapse.
4. **Latent space learned:** Denoised latents are consistent with VQ latent distribution.
5. **Decoder robustness:** Off-manifold latents decode without catastrophic artifacts.
6. **Computational feasibility:** Sampling completes within reasonable time and memory.

### 6.2 Comparison with VQ Baseline

**Not a competition:** LDM and VQ serve different purposes.

- **VQ baseline:** Reconstruction model (input → encode → decode → output).
- **LDM baseline:** Generative model (noise → denoise → decode → sample).

**Comparison points:**
- **Reconstruction quality:** VQ reconstructs specific inputs. LDM cannot (unconditional).
- **Generation quality:** LDM generates novel samples. VQ cannot (deterministic encoder).
- **Boundary preservation:** Both can be evaluated on gradient proxy, but interpret differently.
- **Computational cost:** VQ is faster. LDM is slower but generates samples.

**Avoid invalid comparisons:**
- Do NOT claim LDM is better because generated samples look good.
- Do NOT claim VQ is better because reconstruction error is lower.
- Do NOT claim either improves mixed-pixel detection without ground truth.

### 6.3 Interpretation Guidelines

**Observations:**
- Measured metrics, loss curves, sample quality, computational cost.

**Interpretations:**
- What metrics indicate about learning, generation quality, and feasibility.
- Trade-offs between generation flexibility and computational cost.

**Hypotheses:**
- Whether conditional LDM would improve reconstruction quality.
- Whether accelerated sampling would reduce cost.
- Whether architectural modifications would improve sample quality.

**Separate findings from speculation.**

## 7. Deliverables

### 7.1 Evaluation Script

**Location:** `scripts/evaluate_ldm_baseline.py`

**Functionality:**
- Load frozen VQ model and verify reconstruction baseline.
- Load trained LDM checkpoint (with EMA weights).
- Generate unconditional samples.
- Compute all quantitative metrics.
- Save sample grids and metric reports.

**Reproducibility:**
- Fixed random seed.
- Documented configuration.
- Version-controlled.

### 7.2 Outputs

**Generated samples:**
- `outputs/experiment_ldm_baseline/samples/sample_grid_epoch_{epoch}.png`
- Multiple grids across training checkpoints.

**Metrics:**
- `outputs/experiment_ldm_baseline/metrics/training_metrics.csv`
- `outputs/experiment_ldm_baseline/metrics/sample_quality_metrics.yaml`
- `outputs/experiment_ldm_baseline/metrics/comparison_vq_vs_ldm.yaml`

**Visualizations:**
- Loss curves (training and validation).
- Sample grids (4×4 or 8×8).
- Comparison panel: validation image | VQ reconstruction | LDM sample.
- Gradient magnitude histograms.
- Boundary proxy overlays.

### 7.3 Evaluation Report

**Location:** `docs/experiment_ldm_baseline_report.md`

**Sections:**

1. **Experiment Summary**
   - Objective, scope, baseline configuration.

2. **Training Results**
   - Loss curves, convergence analysis.
   - EMA impact (if tested).
   - Training time and memory usage.

3. **Generated Sample Quality**
   - Qualitative assessment with sample grids.
   - Quantitative metrics (distributional comparison).
   - Diversity analysis.

4. **Comparison with VQ Baseline**
   - VQ reconstruction metrics (reference).
   - LDM sample statistics.
   - Boundary analysis (with proxy disclaimer).

5. **Latent Space Analysis**
   - Denoised latent distribution.
   - Decoder robustness findings.

6. **Computational Efficiency**
   - Sampling time, memory usage.
   - Comparison with VQ encode/decode.

7. **Limitations**
   - Unconditional generation only.
   - No ground-truth mixed-pixel evaluation.
   - Codebook underutilization impact.
   - Small validation set.

8. **Interpretation**
   - What results indicate about LDM viability.
   - Trade-offs and open questions.

9. **Recommendations**
   - Whether to proceed with conditional LDM.
   - Whether to explore accelerated sampling.
   - Whether to ablate architecture or latent representation.

**Tone:**
- Factual, evidence-based.
- Distinguish observations from interpretations.
- Do not over-claim success or failure.

## 8. Acceptance Criteria

The evaluation is complete when:

1. Frozen VQ checkpoint loaded and reconstruction baseline verified.
2. Trained LDM checkpoint loaded with EMA weights.
3. At least 100 unconditional samples generated.
4. Training and validation loss curves plotted and analyzed.
5. Sample grids produced and qualitatively assessed.
6. Quantitative metrics computed (distributional comparison, boundary analysis).
7. Sampling time and memory usage measured.
8. Latent space analysis conducted (denoised latent statistics, decoder robustness).
9. Comparison with VQ baseline documented (with limitations acknowledged).
10. Evaluation report written with clear separation of observations, interpretations, and hypotheses.
11. No claims of ground-truth mixed-pixel detection or downstream classification improvement.
12. Known limitations documented.

## 9. What NOT to Claim

**Do NOT claim:**

1. **Ground-truth mixed-pixel detection:** No verified masks available. Gradient proxy is not ground truth.

2. **Input-specific reconstruction accuracy:** LDM is unconditional and cannot reconstruct specific inputs like VQ autoencoder.

3. **Downstream classification improvement:** Classification is not evaluated in this experiment.

4. **Superiority over VQ baseline:** LDM and VQ have different purposes (generation vs reconstruction). Direct comparison of "better" is invalid.

5. **Boundary preservation proves detection:** High gradient in generated samples does not prove mixed-pixel detection capability.

6. **Perceptual quality equals utility:** Realistic-looking samples do not guarantee usefulness for mixed-pixel analysis or classification.

7. **Sample quality proves latent space learned meaningful representation:** Visual quality is necessary but not sufficient evidence.

**Instead, report:**
- Measured metrics.
- Observed sample quality.
- Computational costs.
- Limitations and unknowns.
- Recommendations for future experiments.

## 10. Known Limitations

### 10.1 Codebook Underutilization

- VQ codebook: 178 / 1024 active codes (17.4%).
- Diffusion operates over discrete manifold embedded in continuous space.
- Generated latents may interpolate between codes (off-manifold).
- Decoder robustness to off-manifold latents is unknown until tested.

**Mitigation:**
- Empirically test decoder on off-manifold latents.
- Document findings.
- Consider nearest-codebook projection as post-processing (deferred).

### 10.2 Ground Truth Limitations

- No verified mixed-pixel masks.
- Boundary proxy (Sobel + threshold) is structural indicator, not ground truth.
- Cannot evaluate true mixed-pixel detection accuracy.

**Implication:**
- Boundary metrics are relative comparisons only.
- Do not claim detection accuracy.

### 10.3 Unconditional Generation

- Baseline does not condition on input images or attributes.
- Cannot perform input-specific reconstruction.
- Cannot compare to CAE or VQ on matched input/output pairs.

**Implication:**
- Evaluation focuses on distributional quality, not per-image reconstruction.

### 10.4 Small Evaluation Set

- Validation set: 413 images.
- Limited statistical power.
- Avoid hyperparameter tuning on this set.

**Implication:**
- Report confidence intervals where applicable.
- Do not overinterpret small differences.

### 10.5 Provisional Architecture

- U-Net config (128 channels, 4 levels, attention at 2-3) not yet memory-validated.
- Noise schedule (cosine) and sampling method (50-step DDPM) provisional.

**Implication:**
- Architecture may change after smoke test.
- Evaluation may need to be rerun if architecture changes significantly.

## 11. Future Extensions (Deferred)

The following are explicitly out of scope for baseline evaluation but may be considered in future experiments:

1. **Conditional generation:** Class, text, or spatial conditioning.
2. **Accelerated sampling:** DDIM, DPM-Solver, consistency models.
3. **Alternative latent representations:** Pre-quantization, discrete indices.
4. **Architecture ablations:** Multi-head attention, cross-attention, custom modules.
5. **Advanced metrics:** FID with thermal-specific features, perceptual loss.
6. **Codebook projection:** Nearest-neighbor post-processing.
7. **Downstream evaluation:** Mixed-pixel detection, classification.

## 12. Summary

The LDM baseline evaluation determines whether latent diffusion successfully generates realistic agricultural thermal images and whether the approach is viable for future conditional extensions.

Evaluation focuses on:
- Training stability and convergence.
- Generated sample quality and diversity.
- Comparison with VQ reconstruction baseline (with appropriate limitations).
- Computational efficiency.

Evaluation does NOT claim:
- Ground-truth mixed-pixel detection.
- Input-specific reconstruction accuracy.
- Downstream classification improvement.
- Absolute superiority over VQ baseline.

Results will inform whether to proceed with conditional LDM, architectural ablations, or alternative approaches.
