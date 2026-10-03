# KL vs VQ — Design

## Overview

This experiment compares two first-stage autoencoder approaches under a controlled setup.

The models share the same overall image-to-latent-to-image objective but differ in how the latent representation is constructed.

## KL Pipeline

Input image

→ Encoder

→ Latent distribution

→ Sample latent

→ Decoder

→ Reconstruction

The KL objective combines reconstruction quality with a KL regularization term.

## VQ Pipeline

Input image

→ Encoder

→ Pre-quantization representation

→ Vector quantization

→ Codebook lookup

→ Quantized latent

→ Decoder

→ Reconstruction

The VQ model learns a discrete latent representation through a learned codebook.

## Common Interface

Both models should expose a consistent interface where practical.

Conceptually:

    encode(x)
    decode(z)
    reconstruct(x)

Training-specific outputs may include additional diagnostics.

## Evaluation Interface

The evaluation framework should operate on reconstructed images rather than depending on architecture-specific internals.

This allows later architectures to be evaluated using the same framework.

## Boundary Evaluation

When ground-truth mixed-pixel annotations are unavailable, a gradient-based boundary proxy is used.

The implementation must clearly label this as a proxy and not as ground truth.

## Reproducibility

The experiment configuration should define:

- seed
- dataset
- preprocessing
- latent dimensions
- optimizer
- learning rate
- batch size
- training epochs
- early stopping
- evaluation configuration

## Outputs

Each experiment should produce:

- configuration
- trained checkpoint
- training metrics
- validation metrics
- reconstruction samples
- error maps
- boundary analysis
- latent diagnostics
- summary results