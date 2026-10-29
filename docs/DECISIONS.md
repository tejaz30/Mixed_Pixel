# Research and Architectural Decisions

This document records significant research and architectural decisions made during the project, along with their rationale.

---

## Baseline Selection

**Decision:** Use a Convolutional Autoencoder (CAE) as the baseline model.

**Rationale:**
- CAE provides a simple, well-understood reconstruction-based approach
- Reconstruction error serves as an unsupervised signal for mixed-pixel detection
- Establishes performance reference for comparing alternative architectures
- No thermal-specific architectural assumptions

**Date:** Project inception

---

## LDM Staged Approach

**Decision:** Evaluate first-stage autoencoders (KL and VQ) before introducing the latent diffusion stage.

**Rationale:**
- First-stage representation quality directly affects diffusion model performance
- Understanding representation trade-offs informs diffusion architecture design
- Controlled comparison requires isolating the independent variable (KL vs VQ)
- Prevents premature architectural decisions about diffusion conditioning or attention

**Date:** Experiment 02 planning

**Related:** AGENTS.md Section 10 "LDM Research Direction"

---

## Boundary Proxy vs Ground Truth

**Decision:** Use gradient-based boundary detection as a proxy, not as ground-truth annotation.

**Rationale:**
- No verified ground-truth mixed-pixel annotations available for thermal imagery
- Gradient-based detection provides a reasonable but imperfect proxy
- Must clearly distinguish proxy from actual annotation to maintain research integrity
- Prevents over-claiming in analysis and publication

**Date:** Experiment design phase

**Related:** AGENTS.md Section 9 "Boundary Evaluation"

---

## Compression Target

**Decision:** Target approximately 8× spatial compression (640×480 → 80×60).

**Rationale:**
- Balances latent compactness with information preservation
- Matches common practice in latent diffusion models (e.g., Stable Diffusion uses 8× compression)
- Enables efficient diffusion in latent space (Stage 2)
- Provides meaningful latent representation for analysis

**Date:** Experiment 02 design

**Related:** `.kiro/specs/02-kl-vs-vq/requirements.md`

---

## Controlled Experimental Variables

**Decision:** Keep dataset, preprocessing, training conditions, and evaluation consistent between KL and VQ.

**Rationale:**
- Isolates the independent variable (latent representation type)
- Prevents confounding variables that would invalidate comparison
- Ensures any performance difference is attributable to representation choice
- Follows standard scientific experimental design

**Controlled Variables:**
- Dataset: No_Mixed
- Resolution: 640×480
- Train/val split: 80/20
- Seed: 42
- Optimizer: Adam
- Initial LR: 5e-4 (later adjusted to 2.5e-4 for stability)
- Batch size: 8 (per Kaggle 2-GPU setup)
- Training budget: 100 epochs max with early stopping

**Date:** Experiment 02 design

**Related:** AGENTS.md Section 8 "Experimental Control"

---

## Kaggle as Compute Environment

**Decision:** Use Kaggle for GPU/TPU training while maintaining the repository as the source of truth.

**Rationale:**
- Local development environment lacks sufficient GPU resources for full-scale training
- Kaggle provides free GPU access (2x T4)
- Repository-first approach maintains code reproducibility
- Avoids locking project code inside notebook-only implementations

**Workflow:**
1. Develop in Kiro IDE
2. Commit to repository
3. Pull in Kaggle
4. Execute training
5. Upload results to Kaggle Datasets
6. Document findings in repository

**Date:** Project infrastructure setup

**Related:** `.kiro/steering/tech.md`

---

## Hyperparameter Stability Adjustments

**Decision:** Reduce learning rate from 5e-4 to 2.5e-4, gradient clipping from 1.0 to 0.5, and scheduler patience from 5 to 3.

**Rationale:**
- Original hyperparameters caused NaN crashes around epoch 20-30
- More conservative settings enable stable overnight training
- Early stopping still allows model to converge optimally
- Applied to both KL and VQ to maintain experimental consistency

**Evidence:**
- KL training with new hyperparameters: completed 34 epochs without NaN
- Early stopping triggered naturally at epoch 34 (best: epoch 24)

**Date:** 2026-10-02 (during KL training debugging)

**Related:** 
- `configs/experiment1/kl_autoencoder_kaggle_2gpu.yaml`
- `configs/experiment1/vq_autoencoder_kaggle_2gpu.yaml`

---

## Checkpoint Persistence Strategy

**Decision:** Automatically upload checkpoints to Kaggle Datasets after each save, with automatic download and resume on session restart.

**Rationale:**
- Kaggle's `/kaggle/working` directory is ephemeral (lost on session crash/timeout)
- Dataset storage persists across sessions
- Enables automatic crash recovery for long training runs
- Supports overnight training without manual intervention

**Implementation:**
- Upload both `best_model.pth` and `checkpoint_epoch_N.pth` together
- Dataset naming: `{username}/{model-type}-checkpoints`
- Download on startup: check for existing dataset, load if available
- Resume training from latest checkpoint automatically

**Date:** 2026-10-02 (after checkpoint loss incident)

**Related:** `src/training/trainer.py` `_upload_checkpoint_to_kaggle_dataset()`

---

## LDM Stage 2 Baseline Architecture

**Decision:** LDM Stage 2 baseline uses post-quantization embeddings from frozen VQ autoencoder with standard continuous diffusion.

**Rationale:**
- Proven approach in LDM paper (Rombach et al. 2022)
- VQ decoder was trained on quantized inputs (verified via forward pass test)
- Standard continuous diffusion is simpler than discrete categorical diffusion
- Faster iteration to establish working baseline
- Resource constraints favor single baseline first rather than parallel exploration

**Alternatives Considered:**
- Pre-quantization continuous embeddings (deferred to future ablation)
- Discrete categorical diffusion over codebook indices (deferred to future ablation)
- Multi-baseline parallel development (rejected due to resource constraints)

**Evidence:**
- VQ checkpoint verified and loadable (`outputs/experiment1/vq_nomixed/best_model(1).pth`)
- Latent statistics measured: mean=0.0129, std=0.1991, recommended scaling=5.02
- Audit confirmed decoder expects quantized latents (successful forward pass)
- VQ won Experiment 1 comparison (superior reconstruction and boundary preservation)

**Provisional Components (Subject to User Approval):**
- U-Net architecture: 128 base channels, 4 levels, attention at levels 2-3
- Noise schedule: linear β from 1e-4 to 2e-2 over 1000 timesteps
- Sampling: DDPM 50-step with uniform stride
- Training: LR 1e-4, batch size 8, AdamW, 100 epochs, EMA decay 0.9999

**Date:** 2026-10-03

**Related:** `.kiro/specs/03-latent-diffusion/`, `.agents/tasks/vq_baseline_audit_final.md`

---

## LDM Stage 2 U-Net Base Channels

**Decision:** LDM Stage 2 baseline uses `base_channels=64` instead of the originally specified 128.

**Rationale:**
- Original specification: `base_channels=128` targeting 25-30M parameters
- Implementation revealed: 128 channels produces **106M parameters** (4× over target)
- Reduced to 64 channels: produces **23.2M parameters** (within 25-30M target)
- Memory estimate: ~3-4GB per batch (fits Kaggle T4 16GB comfortably)

**Evidence:**
- U-Net with base_channels=128: 106M parameters (measured during Phase 2)
- U-Net with base_channels=64: 23.2M parameters (measured and verified)
- Phase 3 testing: All 21 tests passing with 64-channel implementation
- Parameter count test explicitly validates 20-35M range (23.2M passes)

**Approval:** User explicitly approved on 2026-10-29 after Phase 3 test completion and final audit.

**Scope:** This decision applies to the LDM Stage 2 baseline only. Future experiments may use different channel counts.

**Date:** 2026-10-29

**Related:** 
- `.kiro/specs/03-latent-diffusion/design.md` (updated to reflect 64 channels)
- `src/models/ldm/stage2/unet.py` (implementation)
- `tests/test_ldm_stage2.py` (parameter count test)
- `.agents/tasks/phase3_test_report.md` (deviation documentation)

---

## Template for Future Decisions

When adding new decisions, include:
- **Decision:** What was decided
- **Rationale:** Why this choice was made
- **Evidence:** Supporting data or reasoning
- **Date:** When decided
- **Related:** Links to relevant files or documentation

---

**Note:** Do not record routine implementation choices here. This document is for research-methodology and architectural decisions that affect experimental validity or project direction.
