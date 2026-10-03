# Product / Research Context

## Project

Mixed-Pixel Detection in Agricultural Thermal Imagery

## Project Goal

Develop and benchmark deep-learning architectures for unsupervised mixed-pixel detection in agricultural thermal imagery.

The project investigates whether alternative representation and reconstruction architectures can improve upon a conventional Convolutional Autoencoder (CAE) baseline, particularly in preserving important thermal boundaries and supporting reliable mixed-pixel detection.

## Problem

Thermal cameras have limited spatial resolution. As a result, a single pixel may contain thermal information from multiple physical surfaces, particularly around boundaries such as leaf-soil interfaces.

These pixels are referred to as mixed pixels.

Mixed pixels can introduce inaccurate or blended thermal information that may affect subsequent image analysis and downstream agricultural classification tasks.

## Existing Baseline

The existing baseline is a Convolutional Autoencoder (CAE).

The CAE reconstructs the input thermal image and uses reconstruction error as an unsupervised signal for identifying potential mixed pixels.

The baseline pipeline is:

Input thermal image
→ Encoder
→ Latent representation
→ Decoder
→ Reconstructed image
→ Reconstruction error
→ Potential mixed-pixel detection

## Research Direction

The project benchmarks the existing CAE against alternative architectures:

- Latent Diffusion Models (LDM)
- Continuous Feature Convolution (CFC)
- Context Clusters (CoC)

The goal is not to assume that any particular architecture is superior.

The goal is to conduct a controlled benchmarking study and determine how different architectures behave on this specific problem.

## Current LDM Direction

The LDM investigation is being developed incrementally.

The first stage focuses on the latent autoencoder used by an LDM.

KL-regularized and VQ-based first-stage autoencoders are evaluated before introducing the latent diffusion stage.

The diffusion stage will be implemented only after the first-stage representation experiments have been evaluated.

## Primary Research Questions

1. How well can different architectures reconstruct agricultural thermal imagery?
2. How well do they preserve important thermal and spatial boundary information?
3. How does reconstruction quality affect unsupervised mixed-pixel detection?
4. Do alternative architectures provide measurable advantages over the CAE baseline?
5. Do improvements in representation or reconstruction translate into improvements in downstream classification?

## Research Principles

- Prefer controlled experiments over arbitrary architectural changes.
- Change one major experimental variable at a time where possible.
- Compare models under consistent evaluation conditions.
- Separate observed results from interpretation.
- Do not claim an architecture is better without experimental evidence.
- Preserve reproducibility of all experiments.
- Document important research decisions and their rationale.