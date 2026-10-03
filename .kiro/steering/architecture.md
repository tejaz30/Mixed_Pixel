# System Architecture

## Overall Research Architecture

The project is a benchmarking framework for unsupervised mixed-pixel detection in agricultural thermal imagery.

The common conceptual pipeline is:

Thermal Image
→ Representation / Reconstruction Model
→ Reconstructed Image
→ Reconstruction Analysis
→ Mixed-Pixel Detection
→ Downstream Evaluation

All candidate architectures should be evaluated through a common evaluation framework wherever technically possible.

## Baseline

The CAE is the reference baseline.

The baseline should remain stable while alternative architectures are implemented and evaluated.

Do not modify the baseline solely to improve its comparison with another architecture.

## Candidate Architectures

The project investigates:

1. Convolutional Autoencoder (CAE)
2. Latent Diffusion Model (LDM)
3. Continuous Feature Convolution (CFC)
4. Context Clusters (CoC)

Each architecture should have a modular implementation.

## LDM Architecture

The LDM implementation is developed in stages.

### Stage 1 — First-Stage Autoencoder

The first stage maps:

Image
→ Encoder
→ Latent representation
→ Decoder
→ Reconstruction

Two first-stage approaches are investigated:

- KL-regularized autoencoder
- VQ-based autoencoder

These are evaluated before introducing diffusion.

### Stage 2 — Latent Diffusion

After first-stage evaluation, the diffusion component will operate in the learned latent space.

Conceptually:

Image
→ First-stage Encoder
→ Latent
→ Forward Noising
→ Diffusion Model / U-Net
→ Denoised Latent
→ First-stage Decoder
→ Reconstructed Image

The diffusion stage must remain separable from the first-stage autoencoder.

## Experiment Isolation

Major experiments should isolate the independent variable wherever possible.

For example, a KL-vs-VQ comparison should not simultaneously introduce:

- different datasets
- different preprocessing
- different evaluation protocols
- unrelated architectural modifications

unless explicitly justified.

## Common Evaluation

Candidate architectures should use common evaluation metrics where applicable.

Current evaluation includes:

- MSE
- SSIM
- boundary-region reconstruction error
- non-boundary reconstruction error
- boundary/non-boundary error ratio
- gradient preservation

Additional metrics may be introduced through experiment specifications.

## Boundary Analysis

Thermal boundaries are important because mixed pixels are particularly relevant around transitions between different surfaces.

Where ground-truth mixed-pixel masks are unavailable, boundary analysis must clearly distinguish:

- observed image gradients
- boundary proxies
- actual ground-truth mixed pixels

A gradient-based boundary proxy must never be described as ground-truth mixed-pixel annotation.

## Downstream Evaluation

Where applicable, the project should evaluate whether improvements in reconstruction or representation translate into improvements in:

- mixed-pixel detection
- downstream classification

An improvement in reconstruction quality alone must not automatically be interpreted as an improvement in mixed-pixel detection.

## Reproducibility

Every major experiment should have:

- unique experiment identifier
- configuration
- model version
- dataset information
- training settings
- evaluation settings
- results
- relevant checkpoints
- notes on interpretation

## Research Integrity

Agents must not:

- fabricate results
- modify results to fit expectations
- selectively report only favorable samples
- change evaluation criteria after seeing results without documenting the change
- describe a hypothesis as an established finding

Observed results, interpretation, and future hypotheses must remain distinguishable.