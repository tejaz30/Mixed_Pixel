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

## Template for Future Decisions

When adding new decisions, include:
- **Decision:** What was decided
- **Rationale:** Why this choice was made
- **Evidence:** Supporting data or reasoning
- **Date:** When decided
- **Related:** Links to relevant files or documentation

---

**Note:** Do not record routine implementation choices here. This document is for research-methodology and architectural decisions that affect experimental validity or project direction.
