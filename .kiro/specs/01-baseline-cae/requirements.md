# Baseline CAE — Requirements

## Objective

Establish a reproducible Convolutional Autoencoder (CAE) baseline for unsupervised mixed-pixel detection in agricultural thermal imagery.

## Functional Requirements

### R1 — Image Input

The model shall accept the project's standardized thermal image representation.

Current dataset representation:

- Resolution: 640 × 480
- Channels: 3
- Value range: [0, 1]

### R2 — Reconstruction

The CAE shall encode the input image into a compact latent representation and reconstruct the original image.

### R3 — Reconstruction Error

The system shall produce a reconstruction-error map comparing the input image and reconstructed image.

### R4 — Mixed-Pixel Detection

The reconstruction-error map shall be usable as an unsupervised signal for identifying potential mixed pixels.

### R5 — Evaluation

The baseline shall support evaluation using:

- MSE
- SSIM
- reconstruction-error maps
- boundary-region analysis where applicable

### R6 — Reproducibility

The experiment shall record:

- dataset information
- train/validation split
- model configuration
- hyperparameters
- random seed
- software environment
- hardware environment
- experiment identifier

## Non-Goals

This experiment shall not:

- introduce diffusion
- introduce CFC
- introduce CoC
- introduce thermal-specific architectural modifications
- change the existing baseline solely to improve comparison with later models

## Success Criteria

The baseline must:

1. Train successfully.
2. Reconstruct the input images.
3. Produce reproducible metrics.
4. Produce reconstruction-error maps.
5. Provide a stable reference for subsequent experiments.