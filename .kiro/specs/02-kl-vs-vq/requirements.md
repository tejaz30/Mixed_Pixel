# KL vs VQ First-Stage Autoencoders — Requirements

## 1. Objective

Compare the existing trained KL-regularized and VQ-based autoencoders as first-stage representation models for the project's future Latent Diffusion Model (LDM).

The experiment investigates how the choice of latent representation affects:

- Reconstruction quality.
- Preservation of thermal image structure and boundaries.
- Latent representation characteristics and utilization.
- Computational cost and practical suitability as a first-stage model.

The purpose is to identify which representation provides the most promising starting point for subsequent LDM development.

**This experiment evaluates the autoencoders only. No diffusion model is included.**

## 2. Experimental Question

How does a continuous KL-regularized latent representation compare with a VQ-based discrete latent representation in terms of reconstruction fidelity, thermal boundary preservation, latent characteristics, and suitability for subsequent diffusion modelling?

## 3. Experimental Scope

The experiment shall evaluate the existing best checkpoints for both autoencoders.

The evaluation shall use the project's existing no-mixed thermal image dataset and established held-out evaluation split.

The experiment shall not attempt to establish mixed-pixel detection accuracy, since ground-truth mixed-pixel masks are unavailable.

Downstream classification, mixed-pixel correction, and comparisons against other detection methods are outside the scope of this experiment. They may be investigated in later stages of the project.

## 4. Controlled Evaluation Conditions

Wherever technically possible, both models shall be evaluated under identical conditions:

- Same held-out images.
- Same image resolution.
- Same preprocessing and normalization.
- Same input intensity range.
- Same reconstruction metric implementations.
- Same boundary-proxy generation procedure.
- Same qualitative sample selection.
- Same metric aggregation and reporting procedure.

The existing dataset split, preprocessing pipeline, compression target, model configurations, and checkpoints must be preserved.

Record the available training metadata for each model, including:

- Training configuration.
- Optimizer and learning rate.
- Training duration and completed epochs.
- Early stopping or checkpoint-selection procedure.
- Random seed and relevant reproducibility settings.
- Checkpoint path and associated configuration.

The KL model stopped at epoch 30 and the VQ model trained through epoch 50, according to the current experiment record. Verify these details against the available logs and checkpoint metadata.

Since the models are already trained, this evaluation shall not retrain them or modify their training configurations. Differences in training duration must be documented when interpreting the comparison.

If an essential evaluation setting cannot be verified, report the uncertainty rather than silently assuming a value.

## 5. Independent Variable

The primary independent variable is the autoencoder's latent representation:

1. KL-regularized autoencoder with a continuous stochastic latent representation.
2. VQ-based autoencoder with a learned codebook and quantized latent representation.

All other evaluation conditions shall remain consistent wherever technically possible.

## 6. Latent Compression

The intended input resolution is:

- Width: 640 pixels.
- Height: 480 pixels.

The target latent spatial resolution is:

- Width: 80 positions.
- Height: 60 positions.

This corresponds to approximately 8× compression along each spatial dimension.

Verify the actual latent tensor shapes, channel counts, and spatial compression from the trained implementations. Do not assume that the target was achieved solely from configuration values.

Do not alter the architecture, latent dimensions, or compression ratio as part of this evaluation.

## 7. KL Autoencoder

The existing KL autoencoder is expected to contain:

- Encoder.
- Continuous latent distribution, parameterized according to the implementation.
- Latent sampling or the relevant deterministic inference path.
- Decoder.
- Reconstruction objective.
- KL regularization.

The evaluation shall verify the implementation's actual latent semantics.

Record whether reconstruction evaluation uses posterior means, sampled latents, or another supported inference path. Use a consistent and documented procedure.

Where relevant to the intended LDM workflow, assess the variability introduced by latent sampling separately from the primary deterministic comparison.

Do not modify the KL objective or architecture.

## 8. VQ Autoencoder

The existing VQ autoencoder is expected to contain:

- Encoder.
- Pre-quantization representation.
- Vector quantization.
- Learned codebook.
- Quantized latent representation.
- Decoder.
- Reconstruction objective.
- Codebook-related training objective.

Verify these components against the actual implementation.

The evaluation shall distinguish pre-quantization encoder outputs from quantized latents and document which representation is used for each diagnostic.

Do not modify the codebook, quantization procedure, training objective, or architecture.

## 9. Evaluation Metrics

### 9.1 Global Reconstruction Quality

Evaluate both autoencoders on the same held-out images using:

- Mean Squared Error (MSE).
- Structural Similarity Index (SSIM).

Report per-image results and aggregate statistics, including mean and standard deviation. Include median where useful.

Use identical intensity ranges and consistent metric settings for both models.

Lower MSE indicates lower pixel-wise reconstruction error. Higher SSIM indicates greater structural similarity under the selected SSIM configuration.

These metrics measure reconstruction quality; they do not independently establish latent richness or mixed-pixel detection ability.

### 9.2 Boundary Preservation

Since ground-truth mixed-pixel masks are unavailable, use a documented gradient-based boundary proxy derived from the original input images.

The proxy shall identify relatively high-gradient regions in the original thermal image. Its gradient operator, thresholding rule, and any spatial dilation shall be documented and fixed independently of the reconstruction errors.

The proxy represents structural boundaries or strong thermal transitions. It must not be treated as a true mixed-pixel mask.

For both models, evaluate:

- Boundary-region MSE.
- Non-boundary-region MSE.
- Boundary-to-non-boundary error ratio.
- Gradient preservation between the original image and reconstruction.

Define the boundary-to-non-boundary error ratio as:

Boundary-region MSE / Non-boundary-region MSE.

Document how zero or near-zero denominators are handled.

For gradient preservation, use a consistent gradient operator and report a documented measure of reconstruction gradient error or similarity.

Report the boundary and non-boundary pixel counts or proportions so the metrics can be interpreted in context.

Do not conclude that a model detects mixed pixels merely because its residuals or gradient errors are concentrated near the proxy boundaries.

### 9.3 KL Latent Diagnostics

Where supported by the implementation, record:

- Latent tensor shape.
- Latent mean statistics.
- Latent standard deviation and variance statistics.
- Activation statistics.
- Posterior mean and log-variance statistics.
- KL divergence statistics.
- Active latent dimensions, using a documented criterion.
- Non-finite values and numerical stability.

Distinguish posterior statistics from statistics of sampled latent values. Document the representation used for each measurement.

### 9.4 VQ Latent Diagnostics

Where supported by the implementation, record:

- Pre-quantization and quantized latent shapes.
- Quantization error.
- Number of codebook entries used.
- Fraction of active codebook entries.
- Codebook perplexity.
- Code-usage distribution.
- Rarely used or unused codebook entries.
- Codebook vector norms.
- Non-finite values and numerical stability.

Compute codebook utilization over the held-out evaluation set rather than relying on a single batch.

Document the exact perplexity and utilization definitions used.

KL-specific and VQ-specific diagnostics shall not be treated as directly interchangeable metrics.

## 10. Qualitative Evaluation

Generate reproducible comparison panels containing:

1. Original thermal image.
2. KL reconstruction.
3. VQ reconstruction.
4. KL absolute-error map.
5. VQ absolute-error map.
6. Gradient-based boundary proxy.
7. Selected boundary crops showing the original and both reconstructions.

Use identical display ranges for the original and reconstructed images. Use consistent residual-map scales for KL and VQ.

Select samples through a documented, reproducible procedure. Include representative examples and difficult cases rather than selecting images solely because one model appears better.

Qualitative inspection shall focus on reconstruction artifacts, smoothing, preservation of spatial detail, and behavior around strong thermal transitions.

Visual observations must be reported separately from quantitative findings.

## 11. Computational Statistics

Where practical and measurable, record:

- Parameter count.
- Checkpoint size.
- Inference time per image and/or batch.
- Peak memory usage.
- Evaluation runtime.

Record hardware, batch size, precision mode, and timing procedure where relevant.

Training time may be reported from existing logs. Do not rerun training merely to obtain comparable timing unless separately approved.

## 12. Reproducibility and Validation

The evaluation implementation shall:

- Load the intended best checkpoint for each model.
- Use the same held-out image set for both models.
- Preserve model weights and codebook state during evaluation.
- Use inference mode and disable gradients where appropriate.
- Handle stochastic inference explicitly.
- Check for non-finite inputs, outputs, and metric values.
- Save per-image metrics and aggregate results.
- Record model configurations, checkpoint identifiers, and evaluation settings.
- Include tests for metric calculations, tensor shapes, and boundary-mask generation.
- Produce the same results under repeated deterministic evaluation, subject to documented numerical limitations.

Run smoke tests before executing the full evaluation.

## 13. Research Constraints

Do not introduce any of the following into this experiment:

- Diffusion models or diffusion training.
- Attention-based architectural modifications.
- Edge-aware or boundary-aware training losses.
- Thermal-specific architectural modules.
- Additional conditioning mechanisms.
- New latent dimensions or arbitrary changes to compression.
- Changes to the existing trained model weights or training objectives.

Any such changes must be proposed as separate experiments and explicitly approved before implementation.

## 14. Interpretation and Decision Framework

Report observations, interpretations, and future hypotheses separately.

### Observed results

Present the measured metrics, latent diagnostics, visual comparisons, and computational statistics without assigning unsupported explanations.

### Interpretation

Explain what the results support about reconstruction fidelity, spatial-detail preservation, latent utilization, numerical stability, and practical suitability for the intended LDM.

### Future hypotheses

Identify questions that require subsequent experiments, including whether the selected latent representation improves diffusion modelling or ultimately benefits mixed-pixel detection.

Do not construct an arbitrary weighted score to declare a winner.

Do not declare an architecture superior solely because it has lower MSE, higher SSIM, more active dimensions, or higher codebook perplexity.

If the evidence is mixed, report the trade-offs and recommend further investigation rather than forcing a winner.

The final recommendation shall identify the more promising first-stage representation only when supported by the combined evidence. Otherwise, report that the current evaluation does not establish a clear winner.

## 15. Deliverables

The completed evaluation shall produce:

- A reproducible evaluation script following the existing repository conventions.
- Reusable evaluation functions where appropriate.
- Tests for the principal metric and diagnostic implementations.
- A thin notebook for launching and inspecting results, if consistent with existing project conventions.
- Per-image reconstruction metrics.
- Aggregate comparison tables.
- Boundary-proxy and gradient-preservation results.
- KL and VQ latent diagnostic reports.
- Qualitative comparison panels.
- Available computational statistics.
- A final Markdown report with methodology, observed results, interpretation, limitations, and the recommendation for subsequent LDM development.

Use the existing project directory structure and naming conventions. Do not commit datasets, checkpoints, or large generated artifacts where project policy excludes them.

## 16. Acceptance Criteria

The experiment is complete when:

1. Both existing best checkpoints have been evaluated on the same held-out images.
2. Preprocessing and metric settings are consistent and documented.
3. MSE and SSIM results are available per image and in aggregate.
4. Boundary-proxy metrics are calculated using a documented, model-independent proxy.
5. KL and VQ latent diagnostics reflect their actual implementations.
6. Qualitative comparison panels are generated with consistent display scales.
7. Available computational statistics and relevant training differences are documented.
8. Evaluation code passes its applicable tests and smoke tests.
9. The final report distinguishes measurements, interpretations, and hypotheses.
10. The conclusion addresses first-stage LDM suitability without claiming that mixed-pixel detection or diffusion performance has already been demonstrated.
