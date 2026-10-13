# Running Training with NaN Diagnostics on Kaggle

## Setup (One Time)

Add this as **Cell 2.5** in your Kaggle notebook (between Cell 2 and Cell 3):

```python
# Cell 2.5: Enable NaN Diagnostics
print("Loading diagnostic tools...")

import sys
sys.path.insert(0, '/kaggle/working/Mixed-Pixels')

from scripts.investigation.diagnose_nan import add_diagnostics_to_trainer
add_diagnostics_to_trainer()

print("✅ Diagnostic logging enabled")
print("\nThis will log every 50 batches:")
print("  - Learning rate, logvar range, exp(logvar) max")
print("  - Gradient norms before/after clipping")
print("  - First location of any non-finite values")
```

## Then Run Training Normally

```python
# Cell 3: Train KL Model (now with diagnostics)
!python scripts/training/train_first_stage.py {KL_CONFIG}
```

## What You'll See

### Normal Output (every 50 batches):
```
Batch 0:
  Finite checks: {'images': True, 'reconstruction': True, 'mean': True, 'logvar': True, 'exp_logvar': True}
  logvar range: [-5.23, 3.45]
  max exp(logvar): 3.16e+01
  mean range: [-0.1234, 0.0987]
  Loss components:
    reconstruction: 0.004234
    kl: 8234.123456
    weighted_kl: 0.008234
    total: 0.012468
  Gradients before clip: norm=0.8734, max=0.0234
  Gradients after clip:  norm=0.8734
```

### If Problems Detected:
```
Batch 192:
  Finite checks: {'images': True, 'reconstruction': True, 'mean': True, 'logvar': True, 'exp_logvar': False}
  logvar range: [-5.23, 45.67]
  max exp(logvar): 8.21e+19  ⚠️  WARNING: exp(logvar) = 8.21e+19 is very large!
  ❌ NON-FINITE VALUE DETECTED!
     First non-finite in: ['exp_logvar']
```

This tells us EXACTLY where NaN first appears.

## After Getting Diagnostic Output

Share the last few batches before NaN with me. Look for:
1. When does `max exp(logvar)` start growing rapidly?
2. What's the `logvar range` at failure?
3. Are gradients being clipped (norm > 1.0)?
4. What's the learning rate being used?

Then we'll know whether to:
- Lower learning rate
- Clamp logvar
- Increase gradient clipping
- Something else

## Current Baseline Config

With the fixes committed, your config values are now actually used:
- **Learning Rate**: 5.0e-4 (from config)
- **Gradient Clipping**: 1.0 (config default, can be changed)
- **Optimizer eps**: 1.0e-8 (config default, can be changed)

To test different values, just modify `configs/experiment1/kl_autoencoder_kaggle_2gpu.yaml` and commit.
