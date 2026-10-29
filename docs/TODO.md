# TODO

## LDM Stage 2 Baseline

### Immediate (Pending User Approval)
- [ ] Review LDM Stage 2 specification (`.kiro/specs/03-latent-diffusion/`)
- [ ] Approve or modify provisional U-Net architecture (128 base channels, 4 levels, attention at 2-3)
- [ ] Approve or modify noise schedule definition (linear β: 1e-4 → 2e-2)
- [ ] Approve or modify sampling method (DDPM 50-step)
- [ ] Approve smoke test criteria

### After Specification Approval
- [ ] Implement pre-training test suite
  - [ ] Test VQ checkpoint loading and frozen parameters
  - [ ] Test latent scaling/unscaling invertibility
  - [ ] Test decoder with off-manifold latents (smoke test)
- [ ] Implement U-Net architecture
- [ ] Implement noise scheduler
- [ ] Implement training loop with EMA
- [ ] Implement DDPM sampler
- [ ] Run pre-training tests (verify all pass)

### After Pre-Training Tests Pass
- [ ] Run smoke training (1-5 epochs)
- [ ] Review smoke test results:
  - [ ] Verify loss decreases
  - [ ] Verify samples are not blank/NaN
  - [ ] Verify metrics computed successfully
- [ ] Get approval for full training

### After Approval for Full Training
- [ ] Prepare Kaggle notebook for LDM Stage 2
- [ ] Execute full training run (100 epochs with early stopping)
- [ ] Monitor training (loss curves, sample quality every 10 epochs)
- [ ] Download checkpoints and results
- [ ] Run evaluation metrics
- [ ] Document results
- [ ] Compare with VQ baseline reconstruction
- [ ] Write findings report

### After LDM Stage 2 Baseline Complete
- [ ] Decide on ablation experiments:
  - [ ] Pre-quantization continuous embeddings?
  - [ ] Discrete categorical diffusion?
  - [ ] Cosine noise schedule?
  - [ ] DDIM fast sampling?
  - [ ] Conditioning on thermal statistics?
- [ ] Decide on next architecture (CFC or CoC)

---

## Infrastructure

- [ ] Finalize checkpoint management for multi-stage models
- [ ] Add experiment configuration management for diffusion models
- [ ] Update evaluation pipeline to handle diffusion-based reconstruction

---

## Future Experiments

- [ ] Define CFC experiment
- [ ] Define CoC experiment
- [ ] Build final benchmarking pipeline comparing all architectures
- [ ] Investigate downstream classification improvements