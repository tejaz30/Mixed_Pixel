# KL vs VQ Autoencoder: Latent Representation Evaluation

## 1. Objective

Determine which of the trained KL-regularized and VQ autoencoders provides the more suitable first-stage representation for the project's Latent Diffusion Model (LDM).

The goal is not to select the best standalone mixed-pixel detector. It is to characterize the information retained in each compressed representation, its spatial fidelity, latent utilization, statistical properties, and suitability for subsequent diffusion training.

## 2. Research questions

1. How well does each autoencoder reconstruct held-out clean thermal images?
2. Does either representation lose fine spatial structure or thermal transitions?
3. Are the latent dimensions or codebook entries meaningfully utilized?
4. Are the latent values numerically stable and suitable for diffusion training?
5. Which representation offers the best overall trade-off for the intended LDM?

## 3. Scope and constraints

- Evaluate the existing trained KL and VQ checkpoints.
- Use the same held-out images for both models.
- Preserve the existing dataset split, preprocessing, resolution, compression factor, architecture, and checkpoint-selection decisions.
- Do not retrain either model unless separately approved.
- Do not modify training losses, model architectures, or latent dimensions during evaluation.
- Do not use mixed-pixel detection accuracy or downstream classification as the selection criterion in this experiment.
- Do not interpret reconstruction quality as proof of mixed-pixel detection capability.
- Do not invent results or assume that either model is superior in advance.

## 4. Data and reproducibility

Use the existing no-mixed thermal dataset and the established held-out evaluation split. Do not create a new split without approval.

Load the exact best checkpoints for both models. Record their paths, configuration files, checkpoint metadata, training epochs, and relevant code/configuration versions.

Use identical input images, image ordering, preprocessing, normalization, and intensity conventions. Ensure evaluation runs in inference mode and does not update model parameters or codebooks.

If the current split, checkpoint, or preprocessing configuration cannot be established, stop and report the missing information rather than guessing.

## 5. Evaluation dimensions

### 5.1 Reconstruction fidelity

For every held-out image, compute:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Structural Similarity Index (SSIM)
- Optional Peak Signal-to-Noise Ratio (PSNR), provided its data range is explicitly specified

Aggregate per-image measurements using mean and standard deviation, and report median where useful. Include paired KL-versus-VQ differences on the same images.

Use identical SSIM settings and input intensity ranges. Do not compare raw training losses as if they were equivalent: the KL and VQ objectives differ.

### 5.2 Spatial detail preservation

Compare original images, reconstructions, absolute residual maps, and selected high-detail crops.

Include a gradient-based diagnostic, such as gradient-magnitude error, if it can be implemented consistently for both models.

Assess whether fine spatial patterns and thermal transitions survive compression and reconstruction. Treat this as a structural diagnostic, not evidence of true mixed-pixel detection.

### 5.3 KL latent diagnostics

Inspect the actual representation produced by the trained implementation.

Where applicable, report:

- Latent tensor shape and number of latent channels.
- Per-channel means, standard deviations, and variance.
- Posterior mean and log-variance statistics.
- KL divergence per image and per latent channel.
- Number and fraction of active latent dimensions, using a documented criterion.
- Numerical stability, non-finite values, and unusually large latent values.

Document whether the evaluated representation uses posterior means, posterior samples, or another implementation-specific latent. Use the existing implementation's correct semantics.

If the LDM will use sampled latents during training, also assess the effect of sampling variability where appropriate.

### 5.4 VQ latent diagnostics

Inspect the actual quantized representation and codebook.

Report, where available:

- Pre-quantization and post-quantization latent shapes.
- Quantization error.
- Number and fraction of codebook entries used.
- Codebook perplexity and code-usage distribution.
- Codebook vector norms and latent-value statistics.
- Dead or rarely used codes, using a documented definition.
- Numerical stability and non-finite values.

Calculate utilization over a sufficiently representative held-out set, not just a single batch. Respect the implementation's actual codebook and quantization behavior.

### 5.5 Latent distribution and LDM readiness

For both representations, examine:

- Latent shape and effective compression.
- Per-channel mean, standard deviation, variance, and range.
- Outliers and numerical stability.
- Correlation between latent channels, where useful.
- Spatial autocorrelation or spatial structure, where useful.
- Variability across held-out images.
- Whether the representation requires normalization or scaling before diffusion training.

Distinguish raw encoder outputs from any normalized or scaled representation used by the future diffusion model.

Do not assume KL and VQ latents are interchangeable. In particular, determine whether the current VQ representation is continuous, quantized, or exposed through another interface before recommending a diffusion formulation.

## 6. Visual evaluation

Generate paired visual panels containing:

1. Original clean thermal image.
2. KL reconstruction.
3. VQ reconstruction.
4. KL absolute residual map.
5. VQ absolute residual map.
6. Optional gradient or structural-detail comparison.

Use identical display ranges for original and reconstructed images, and consistent residual scales for both models.

Choose images using a reproducible selection procedure. Include representative examples and difficult cases; do not select images solely because one model looks better.

Visual comparisons supplement quantitative results and must not replace them.

## 7. Decision framework

Do not create an arbitrary weighted aggregate score.

Compare the models across four primary dimensions:

1. **Reconstruction fidelity:** MSE, MAE, and SSIM.
2. **Information and spatial-detail retention:** latent statistics, utilization, and structural diagnostics.
3. **Representation health:** active dimensions or codebook usage, numerical stability, and evidence of collapse or underutilization.
4. **Diffusion readiness:** stable, reproducible latent values and compatibility with the planned diffusion architecture.

Report trade-offs explicitly. Lower reconstruction error alone does not establish a richer latent representation. Higher codebook perplexity or more active dimensions alone also does not establish superiority.

The outcome may be:
- KL preferred;
- VQ preferred;
- no clear winner;
- insufficient evidence pending a targeted additional diagnostic.

Explain the evidence supporting the decision and the limitations.

## 8. Deliverables

Produce:

- A reproducible evaluation script under the existing `scripts/` structure.
- Reusable evaluation functions under the existing `src/` structure, if appropriate.
- Tests under the existing `tests/` structure.
- A thin notebook for launching and inspecting the evaluation, if consistent with project conventions.
- Per-image metrics and aggregate summary tables.
- Visual comparison panels.
- KL-specific and VQ-specific latent diagnostic reports.
- A final Markdown report documenting results, interpretation, caveats, and the recommendation for LDM development.

Use the repository's existing naming and output conventions after inspecting them. Do not commit large datasets, checkpoints, or generated artifacts if project conventions prohibit them.

## 9. Acceptance criteria

- Both models are evaluated on exactly the same held-out images.
- Checkpoint loading and preprocessing are verified.
- Evaluation is deterministic where the implementation permits it.
- Metric calculations are tested on known examples.
- KL and VQ diagnostics reflect their different latent mechanisms.
- Results are reproducible from documented configurations and commands.
- No results are fabricated, and no unapproved methodology changes are introduced.
- The report clearly separates reconstruction quality, latent utilization, and suitability for diffusion.
