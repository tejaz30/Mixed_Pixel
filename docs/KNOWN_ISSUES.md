# Known Issues and Limitations

This document tracks known limitations, uncertainties, and unresolved technical/research issues that may affect interpretation or future work.

---

## Data and Ground Truth

### No Verified Ground-Truth Mixed-Pixel Annotations

**Issue:** The project does not have verified ground-truth labels for which pixels are truly "mixed pixels."

**Impact:**
- Cannot compute true mixed-pixel detection accuracy
- Boundary evaluation relies on gradient-based proxy
- Must clearly distinguish proxy from ground truth in all reporting

**Mitigation:**
- Use gradient-based boundary detection as documented proxy
- Clearly label all boundary metrics as "proxy-based"
- Consider supervised annotation as future work

**Status:** Accepted limitation

**Related:** AGENTS.md Section 9

---

## Compute Environment

### Kaggle Required for Large-Scale Training

**Issue:** Local development environment lacks GPU resources for full-scale training.

**Impact:**
- Cannot run full experiments locally
- Debugging requires either small-scale local runs or Kaggle notebook cells
- Adds latency to development cycle (commit → push → pull in Kaggle)

**Mitigation:**
- Use sanity tests and small-scale runs locally before Kaggle deployment
- Maintain clear separation between source code (repository) and compute (Kaggle)
- Document Kaggle-specific paths and configurations clearly

**Status:** Accepted limitation

**Related:** `.kiro/steering/tech.md`, `scripts/KAGGLE_QUICKSTART.md`

---

### Kaggle Session Instability

**Issue:** Kaggle sessions can time out or crash during long training runs, losing `/kaggle/working` contents.

**Impact:**
- Training progress lost if checkpoints not persisted
- Requires manual restart and checkpoint recovery

**Mitigation:**
- Implemented automatic checkpoint upload to Kaggle Datasets
- Automatic download and resume on session restart
- Checkpoint uploads occur after each save (every 10 epochs)

**Status:** Mitigated via checkpoint persistence system

**Related:** `src/training/trainer.py`, DECISIONS.md "Checkpoint Persistence Strategy"

---

## Model Architecture

### LDM Stage 2 Baseline - Provisional Decisions

**Issue:** Several architectural and training decisions are provisional and require user approval before implementation.

**Provisional Decisions:**
1. **U-Net architecture:** 128 base channels, 4 levels, attention at levels 2-3
   - Subject to memory validation on T4 GPU
   - May need to reduce base channels to 64 if memory exceeds 15GB
2. **Noise schedule:** Linear β from 1e-4 to 2e-2 over 1000 timesteps
   - Requires precise mathematical definition before implementation
   - Alternative schedules (cosine, sigmoid) not yet evaluated
3. **Sampling method:** DDPM 50-step with uniform stride
   - Requires precise stride schedule definition
   - DDIM fast sampling not yet implemented
4. **Training configuration:** LR 1e-4, batch size 8, 100 epochs
   - Provisional pending smoke test results
   - May require adjustment based on empirical convergence

**Impact:**
- Implementation blocked until user reviews and approves specification
- Cannot begin coding until architectural decisions are finalized

**Mitigation:**
- Comprehensive specification created in `.kiro/specs/03-latent-diffusion/`
- Pre-training test suite planned to validate decisions early
- Smoke test (1-5 epochs) planned before full Kaggle training

**Status:** Awaiting user approval

**Related:** `.kiro/specs/03-latent-diffusion/requirements.md`, `design.md`

---

### VQ First-Stage Known Limitations

**Issue:** VQ autoencoder has low codebook utilization and may produce off-manifold latents during diffusion.

**Limitations:**
1. **Codebook underutilization:** Only 178/1024 codes active (17.4%)
   - Most codes never used during training
   - Latent space has large unused regions
2. **Off-manifold latents:** Diffusion may generate interpolations between codebook vectors
   - VQ decoder was trained only on exact codebook embeddings
   - Decoder robustness to off-manifold inputs is unknown
3. **Discrete quantization artifacts:** VQ reconstruction shows slight blocky artifacts compared to KL

**Impact:**
- Diffusion model may generate latents that VQ decoder handles poorly
- May see quality degradation or artifacts in diffusion-based reconstructions
- Requires empirical testing to assess practical severity

**Mitigation:**
- Smoke test will include off-manifold latent testing
- Pre-training test suite includes decoder robustness checks
- Will monitor reconstruction quality during training
- Can consider codebook expansion or continuous-embedding ablation if issues arise

**Status:** Accepted limitation for baseline; may address in future ablations

**Related:** `.kiro/specs/03-latent-diffusion/testing.md`, `docs/exp1_evaluation_summary.md`

---

### Later LDM Architecture Undecided

**Issue:** The latent diffusion model architecture (Stage 2) has not been finalized.

**Impact:**
- Cannot begin diffusion implementation until first-stage experiments complete
- Architectural decisions depend on KL vs VQ findings

**Mitigation:**
- Complete Stage 1 (KL vs VQ) first
- Use findings to inform diffusion architecture design
- Document decision rationale when made

**Status:** Intentional staging; not a bug

**Related:** AGENTS.md Section 10, `.kiro/specs/03-latent-diffusion/`

---

## Training Stability

### Historical NaN Issues (Now Resolved)

**Issue:** Initial training runs experienced NaN losses around epoch 20-30.

**Impact:**
- Training crashes required manual restart
- Lost training time and compute resources

**Resolution:**
- Reduced learning rate from 5e-4 to 2.5e-4
- Reduced gradient clipping from 1.0 to 0.5
- Reduced scheduler patience from 5 to 3
- KL training completed successfully with new hyperparameters (34 epochs)

**Status:** ✅ Resolved

**Related:** DECISIONS.md "Hyperparameter Stability Adjustments"

---

### Multi-GPU Diagnostic Tensor Shapes

**Issue:** Diagnostic wrapper (`diagnose_nan.py`) did not handle DataParallel's per-GPU tensor shapes.

**Impact:**
- `.item()` calls on shape `[2]` tensors failed
- Blocked VQ training in diagnostic mode

**Resolution:**
- Loop through loss dictionary and call `.mean()` before `.item()`
- Handles both scalar and per-GPU shaped tensors

**Status:** ✅ Resolved (commit 8c7bbb2)

**Related:** `scripts/investigation/diagnose_nan.py`

---

## Reproducibility

### Kaggle Notebook Manual Modifications

**Issue:** Some Kaggle-specific paths or settings may require manual adjustment per session.

**Impact:**
- Reproducibility documentation must be Kaggle-aware
- Cannot guarantee bit-exact reproduction without documented manual steps

**Mitigation:**
- Document all Kaggle-specific modifications in notebook cells
- Use configuration files to minimize hard-coded paths
- Provide KAGGLE_QUICKSTART.md with setup instructions

**Status:** Partially mitigated

**Related:** `notebooks/kaggle_kl_vq_minimal.ipynb`, `scripts/KAGGLE_QUICKSTART.md`

---

## Evaluation Metrics

### LPIPS Not Enabled

**Issue:** Learned Perceptual Image Patch Similarity (LPIPS) metric disabled in current evaluation.

**Impact:**
- Relying on MSE, SSIM, and boundary metrics only
- Missing perceptual quality assessment

**Rationale:**
- LPIPS adds dependency and compute overhead
- MSE/SSIM sufficient for initial experiments
- Can enable later if needed

**Status:** Accepted limitation for now

**Related:** `configs/experiment1/kl_autoencoder_kaggle_2gpu.yaml` (line 99: `compute_lpips: false`)

---

## Future Work Items (Not Issues)

These are known areas for future investigation, not current problems:

- Ground-truth mixed-pixel annotation collection
- Diffusion model design (Stage 2)
- CFC architecture implementation
- CoC architecture implementation
- Downstream classification evaluation
- Real agricultural deployment validation

---

## Template for New Issues

When documenting new issues:

```markdown
### Issue Title

**Issue:** Brief description of the problem

**Impact:** How this affects the project or results

**Mitigation:** What we're doing to address it (if anything)

**Status:** Current state (Accepted / Mitigated / Resolved / Under Investigation)

**Related:** Links to relevant files, commits, or documentation
```

---

**Last Updated:** 2026-10-03


---

## 10. Kaggle Notebook Import Path Issues

**Status:** ⚠️ DOCUMENTED SOLUTION

**Description:**
- Kaggle notebooks using GitHub clone/pull require specific import patterns
- `sys.path.insert(0, str(WORK_DIR / 'src'))` adds `src/` to path
- Modules inside `src/` become top-level imports (NO `src.` prefix)

**Correct Import Pattern:**
```python
# Environment setup
sys.path.insert(0, str(WORK_DIR / 'src'))  # Adds src/ to path

# Then imports work WITHOUT src. prefix:
from models.ldm.first_stage.vq.model import VQAutoencoder  # ✅ CORRECT
from models.ldm.stage2.latent_scaler import LatentScaler
from models.ldm.stage2.unet import DiffusionUNet
from datasets.thermal_dataset import ThermalImageDataset

# NOT:
from src.models.vq_autoencoder import VQAutoencoder  # ❌ WRONG
```

**Key Files:**
- VQ Autoencoder: `src/models/ldm/first_stage/vq/model.py` (not `src/models/vq_autoencoder.py`)
- KL Autoencoder: `src/models/ldm/first_stage/kl/model.py`
- LDM Stage 2: `src/models/ldm/stage2/`

**Working Examples:**
- See `notebooks/kaggle_kl_vs_vq_FIXED.ipynb` for reference pattern
- All imports work identically in Kaggle and local Jupyter

**Resolution:** Use this pattern in all future Kaggle notebooks. Do NOT add `src.` prefix when `sys.path` points to `src/` directory.
