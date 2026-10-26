# KL vs VQ — Tasks

## Phase 1 — Repository Inspection

- [ ] Inspect existing dataset pipeline.
- [ ] Inspect existing CAE implementation.
- [ ] Inspect existing training utilities.
- [ ] Inspect existing evaluation utilities.
- [ ] Identify reusable components.
- [ ] Identify missing dependencies.

## Phase 2 — Common Infrastructure

- [ ] Define experiment configuration.
- [ ] Define reproducible seed handling.
- [ ] Define shared training interface.
- [ ] Define shared reconstruction evaluation.
- [ ] Define boundary-proxy evaluation.
- [ ] Define experiment output structure.

## Phase 3 — KL Autoencoder

- [ ] Implement encoder.
- [ ] Implement latent distribution.
- [ ] Implement sampling.
- [ ] Implement decoder.
- [ ] Implement KL regularization.
- [ ] Implement training integration.
- [ ] Implement checkpointing.
- [ ] Add tests.
- [ ] Run sanity training.

## Phase 4 — VQ Autoencoder

- [ ] Implement encoder.
- [ ] Implement pre-quantization projection.
- [ ] Implement vector quantization.
- [ ] Implement codebook.
- [ ] Implement decoder.
- [ ] Implement VQ training objective.
- [ ] Implement codebook diagnostics.
- [ ] Implement checkpointing.
- [ ] Add tests.
- [ ] Run sanity training.

## Phase 5 — Controlled Training

- [ ] Train KL model.
- [ ] Train VQ model.
- [ ] Record configurations.
- [ ] Record training statistics.
- [ ] Preserve checkpoints.

## Phase 6 — Evaluation

- [ ] Calculate MSE.
- [ ] Calculate SSIM.
- [ ] Calculate boundary MSE.
- [ ] Calculate non-boundary MSE.
- [ ] Calculate boundary/non-boundary ratio.
- [ ] Calculate gradient preservation.
- [ ] Calculate latent diagnostics.
- [ ] Generate qualitative comparisons.

## Phase 7 — Analysis

- [ ] Create comparison table.
- [ ] Create reconstruction figures.
- [ ] Create error-map figures.
- [ ] Analyze latent statistics.
- [ ] Document observations.
- [ ] Document limitations.
- [ ] Record research decision based on results.

## Completion Criteria

The experiment is complete only when:

- both models have been trained successfully
- both models have been evaluated using the same protocol
- results are reproducible
- qualitative outputs are available
- quantitative comparison is documented
- observations and interpretations are separated