# KL vs VQ First-Stage Autoencoders — Requirements

## Objective

Compare KL-regularized and VQ-based first-stage autoencoders for representing and reconstructing agricultural thermal imagery.

The experiment is intended to determine how the choice of latent representation affects:

- reconstruction quality
- thermal boundary preservation
- latent representation characteristics

This experiment is a first-stage LDM investigation.

No diffusion model is included.

## Experimental Question

How does a continuous KL-regularized latent representation compare with a VQ-based discrete latent representation when reconstructing the project's thermal imagery?

## Controlled Variables

The following should remain consistent between the two approaches wherever technically possible:

- dataset
- preprocessing
- image resolution
- train/validation split
- compression target
- evaluation procedure
- random seed policy
- optimizer
- learning rate
- training budget
- early stopping policy

## Independent Variable

The primary independent variable is the first-stage latent representation:

1. KL-regularized autoencoder
2. VQ-based autoencoder

## Latent Compression

Target approximately 8× spatial compression.

Input:

640 × 480

Target latent spatial resolution:

80 × 60

## KL Autoencoder

The KL model shall contain:

- encoder
- continuous latent distribution
- latent sampling
- decoder
- reconstruction loss
- KL regularization

## VQ Autoencoder

The VQ model shall contain:

- encoder
- pre-quantization representation
- vector quantization
- learned codebook
- quantized latent representation
- decoder
- reconstruction loss
- codebook-related training objective

## Evaluation

The experiment shall evaluate:

### Global Reconstruction

- MSE
- SSIM

### Boundary Preservation

Use a documented gradient-based boundary proxy where ground-truth mixed-pixel masks are unavailable.

Evaluate:

- boundary MSE
- non-boundary MSE
- boundary/non-boundary error ratio
- gradient preservation

### Latent Diagnostics

For KL:

- latent mean statistics
- latent standard deviation statistics
- activation statistics

For VQ:

- codebook utilization
- active code count
- codebook perplexity

## Qualitative Evaluation

For controlled samples, produce:

- original image
- KL reconstruction
- VQ reconstruction
- absolute error maps
- boundary proxy
- selected boundary crops

## Compute Statistics

Where practical, record:

- parameter count
- training time
- inference time
- memory usage

## Research Constraints

Do not introduce:

- diffusion
- attention-based modifications
- edge-aware loss
- thermal-specific architectural modules
- additional conditioning
- arbitrary compression changes

unless explicitly defined as a separate experiment.

## Interpretation

The experiment must distinguish:

- observed result
- interpretation
- hypothesis for future experiments

No architecture should be declared superior without experimental evidence.