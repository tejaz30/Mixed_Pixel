# Testing and Validation

## Purpose

Testing must verify both:

1. software correctness
2. experimental correctness

A model running without errors does not necessarily mean that the experiment is valid.

## Unit Tests

Reusable components should have tests where practical.

Priority areas include:

- dataset loading
- preprocessing
- tensor shapes
- model forward passes
- loss functions
- evaluation metrics
- checkpoint loading
- configuration handling

## Shape Validation

Model components should validate expected tensor dimensions where practical.

For image models, verify:

- input channels
- image height and width
- latent dimensions
- decoder output dimensions

## Dataset Validation

Before training, verify:

- expected number of samples
- valid image files
- expected image dimensions
- expected number of channels
- valid value ranges
- absence of unexpected NaN/Inf values

## Model Sanity Checks

Before a full training run:

- perform a forward pass
- verify output dimensions
- verify loss can be computed
- verify gradients are produced
- verify optimizer updates parameters
- verify checkpoint saving/loading works

## Training Sanity Checks

A small-scale training run should be possible before launching a full Kaggle experiment.

The sanity run should verify that:

- the loss decreases or behaves plausibly
- validation runs successfully
- checkpoints are produced
- metrics are logged
- the evaluation pipeline works

## Experiment Validation

Before comparing two architectures, verify that the comparison uses consistent:

- dataset
- preprocessing
- train/validation split
- evaluation procedure
- relevant training budget
- random seed policy

Any intentional difference must be documented.

## Evaluation Validation

Metrics must be implemented and tested independently where practical.

Boundary-based evaluation must clearly document the method used to construct the boundary proxy.

Do not treat a proxy as ground truth.

## Regression Testing

Changes to shared components should be checked for unintended effects on existing experiments.

The CAE baseline should remain reproducible after changes to shared infrastructure.

## Reproducibility Checks

Important experiments should be reproducible from their configuration and documented environment.

At minimum, record:

- code version / commit
- experiment configuration
- dataset information
- random seed
- hardware
- dependency versions

## Test Before Expensive Runs

Do not launch expensive GPU/TPU training in Kaggle before basic local or lightweight validation has passed.

The preferred sequence is:

Code
→ Unit tests
→ Small sanity run
→ Kaggle test run
→ Full experiment

## Research Result Integrity

Tests must not be modified simply to make an experiment pass.

If an implementation fails because the expected behavior is incorrect, investigate the underlying issue rather than weakening the test without justification.