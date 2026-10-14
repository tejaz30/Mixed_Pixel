# Kaggle Checkpoint Persistence System

## Overview

This system ensures that training checkpoints **survive Kaggle session crashes and restarts** by automatically uploading them to a persistent Kaggle Dataset and resuming from the latest checkpoint on restart.

**Problem Solved:** In interactive Kaggle sessions, files in `/kaggle/working/` are lost when the session crashes. This system persists checkpoints to Kaggle Datasets, which survive session restarts.

---

## How It Works

### 1. **Checkpoint Saving (During Training)**

Every time a checkpoint is saved:

```
Training Epoch 10 completes
    ↓
Checkpoint saved to /kaggle/working/checkpoints/kl/checkpoint_epoch_10.pth
    ↓
Checkpoint automatically uploaded to Kaggle Dataset
    ↓
Dataset: <username>/mixed-pixels-kl-autoencoder-checkpoints
    ↓
✓ Checkpoint persists forever (independent of session)
```

### 2. **Checkpoint Loading (On Restart)**

When training starts:

```
Training script starts
    ↓
Checks Kaggle Dataset for existing checkpoints
    ↓
Downloads all checkpoints from Dataset to /kaggle/working/checkpoints/
    ↓
Finds latest checkpoint (e.g., checkpoint_epoch_30.pth)
    ↓
Loads model, optimizer, scheduler state
    ↓
✓ Resumes training from epoch 31
```

---

## Architecture

### Components

1. **Trainer (`src/training/trainer.py`)**
   - Detects if running on Kaggle
   - Sets up Kaggle Dataset name based on model type
   - After each checkpoint save, uploads to Kaggle Dataset
   - Stores `best_val_loss` and `best_epoch` in checkpoints

2. **Training Script (`scripts/training/train_first_stage.py`)**
   - On startup, checks for Kaggle Dataset
   - Downloads checkpoints from Dataset if available
   - Auto-detects latest checkpoint
   - Loads checkpoint and resumes training

### Kaggle Dataset Naming

```
<username>/mixed-pixels-<model-type>-checkpoints
```

**Examples:**
- KL Autoencoder: `tejabulusu/mixed-pixels-kl-autoencoder-checkpoints`
- VQ Autoencoder: `tejabulusu/mixed-pixels-vq-autoencoder-checkpoints`

**Note:** Username is automatically detected from `KAGGLE_USERNAME` environment variable.

---

## Implementation Details

### Checkpoint Content

Each checkpoint file contains:

```python
{
    'epoch': int,                      # Current epoch number
    'model_state_dict': dict,          # Model weights
    'optimizer_state_dict': dict,      # Optimizer state
    'scheduler_state_dict': dict,      # LR scheduler state (if enabled)
    'train_losses': dict,              # Training losses
    'val_losses': dict,                # Validation losses
    'best_val_loss': float,            # Best validation loss so far
    'best_epoch': int,                 # Epoch with best validation loss
    'config': dict                     # Full experiment configuration
}
```

### Upload Process

When a checkpoint is saved:

1. **Create temporary dataset directory** at `/kaggle/temp_dataset/`
2. **Copy checkpoint** to dataset directory
3. **Create metadata file** (`dataset-metadata.json`)
4. **Check if dataset exists** using `kaggle datasets status`
5. **Upload:**
   - If dataset exists: Create new version (`kaggle datasets version`)
   - If dataset doesn't exist: Create dataset (`kaggle datasets create`)
6. **Cleanup** temporary directory

**Error Handling:** Upload failures do not stop training. Errors are logged but training continues.

### Download Process

On training startup:

1. **Detect Kaggle environment** (check if `/kaggle/working/` exists)
2. **Construct dataset name** from username and model type
3. **Download dataset** using `kaggle datasets download`
4. **Extract checkpoints** to `/kaggle/working/checkpoints/`
5. **Auto-detect latest** checkpoint by sorting filenames
6. **Load checkpoint** if found, otherwise start fresh

---

## Usage

### No Code Changes Required

The system works automatically. Just run your training script normally:

```python
# In Kaggle notebook
!python scripts/training/train_first_stage.py configs/experiment1/kl_autoencoder.yaml
```

### What You'll See

**First Run (No existing checkpoints):**
```
======================================================================
🔍 CHECKING KAGGLE DATASET FOR CHECKPOINTS
======================================================================
Dataset: tejabulusu/mixed-pixels-kl-autoencoder-checkpoints
  ℹ️  No existing dataset found (starting fresh)

======================================================================
Starting fresh training (no checkpoint found)
======================================================================

📦 Kaggle Dataset Persistence ENABLED
   Dataset: tejabulusu/mixed-pixels-kl-autoencoder-checkpoints
   Checkpoints will be saved to Kaggle Dataset (survives session restarts)

Training starts...
Epoch 1...
  → Checkpoint saved to /kaggle/working/checkpoints/kl/checkpoint_epoch_10.pth
    📤 Created Kaggle Dataset: tejabulusu/mixed-pixels-kl-autoencoder-checkpoints
```

**After Restart (Checkpoints exist):**
```
======================================================================
🔍 CHECKING KAGGLE DATASET FOR CHECKPOINTS
======================================================================
Dataset: tejabulusu/mixed-pixels-kl-autoencoder-checkpoints
  ✓ Downloaded: checkpoint_epoch_10.pth
  ✓ Downloaded: checkpoint_epoch_20.pth
  ✓ Downloaded: checkpoint_epoch_30.pth

======================================================================
🔄 RESUMING FROM CHECKPOINT
======================================================================
Checkpoint: /kaggle/working/checkpoints/kl/checkpoint_epoch_30.pth
✓ Resumed from epoch 30
✓ Best val loss so far: 0.003632
✓ Will continue training from epoch 31

Training continues...
Epoch 31...
```

---

## Crash Recovery Scenarios

### Scenario 1: Session Crashes During Training

```
Run 1: Trains epochs 1-30 → Session crashes
    ↓
Checkpoints uploaded to Dataset: epochs 10, 20, 30
    ↓
Run 2: Restart notebook → Downloads checkpoints → Resumes from epoch 30
    ↓
Trains epochs 31-60 → Session crashes
    ↓
Checkpoints uploaded: epochs 40, 50, 60
    ↓
Run 3: Restart → Resumes from epoch 60 → Completes training
```

**Result:** 100 epochs completed across 3 sessions, no epochs wasted.

### Scenario 2: Manual Stop and Resume

```
Training at epoch 45
    ↓
You manually stop the notebook
    ↓
Session lost, /kaggle/working/ deleted
    ↓
But checkpoint_epoch_40.pth is in Kaggle Dataset
    ↓
Later: Restart notebook → Resumes from epoch 40
```

### Scenario 3: Multiple Restarts

The system handles unlimited restarts. Each time:
1. Downloads all checkpoints from Dataset
2. Finds the latest one
3. Resumes from that point

---

## Manual Operations

### View Your Checkpoints

Go to: `https://www.kaggle.com/datasets/<username>/mixed-pixels-<model>-checkpoints`

Example: `https://www.kaggle.com/datasets/tejabulusu/mixed-pixels-kl-autoencoder-checkpoints`

### Download Checkpoints Locally

```bash
# From your local machine (not Kaggle)
kaggle datasets download -d <username>/mixed-pixels-kl-autoencoder-checkpoints --unzip
```

### Delete All Checkpoints (Fresh Start)

Delete the Kaggle Dataset from the web interface or:

```bash
kaggle datasets delete <username>/mixed-pixels-kl-autoencoder-checkpoints
```

### Force Resume from Specific Checkpoint

```bash
# In Kaggle notebook
!python scripts/training/train_first_stage.py CONFIG.yaml --resume /kaggle/working/checkpoints/kl/checkpoint_epoch_50.pth
```

---

## Configuration

### Required YAML Config Keys

```yaml
paths:
  checkpoint_dir: /kaggle/working/checkpoints/kl  # Where checkpoints are saved locally
  log_dir: /kaggle/working/logs/kl

training:
  save_frequency: 10  # Save checkpoint every N epochs
```

### Environment Variables

- `KAGGLE_USERNAME` - Automatically set by Kaggle (used for dataset naming)

---

## File Structure

### On Kaggle During Training

```
/kaggle/working/
├── checkpoints/
│   └── kl/
│       ├── checkpoint_epoch_10.pth
│       ├── checkpoint_epoch_20.pth
│       ├── checkpoint_epoch_30.pth
│       └── best_model.pth
└── logs/
    └── kl/
```

### In Kaggle Dataset (Persistent)

```
mixed-pixels-kl-autoencoder-checkpoints/
├── checkpoint_epoch_10.pth
├── checkpoint_epoch_20.pth
├── checkpoint_epoch_30.pth
└── best_model.pth
```

**Note:** Dataset versions are updated each time a new checkpoint is uploaded.

---

## Best Practices

### 1. Use Interactive Sessions

This system is designed for **interactive Kaggle sessions** where you run cells manually and can restart as needed.

### 2. Monitor Upload Status

Watch for upload confirmations:
```
📤 Uploaded to Kaggle Dataset (updated)
```

If you see warnings, note them but training will continue.

### 3. Save Frequency

Balance checkpoint frequency with upload overhead:
- **Too frequent (every epoch):** Slower training, many uploads
- **Too infrequent (every 50 epochs):** Risk losing more progress
- **Recommended:** Every 10-20 epochs

Configure in your YAML:
```yaml
training:
  save_frequency: 10
```

### 4. Download Final Model

After training completes, download `best_model.pth` from the Kaggle Dataset for local evaluation.

### 5. Clean Up Old Checkpoints

Periodically delete old checkpoint datasets you no longer need to save storage quota.

---

## Limitations

1. **Kaggle API Required:** System uses `kaggle` CLI commands (available by default on Kaggle)
2. **Upload Time:** Checkpoints are ~200-500MB, uploads take 10-30 seconds
3. **Dataset Quota:** Kaggle has storage limits for datasets (currently generous)
4. **Public vs Private:** Datasets are created as private by default
5. **Version Limit:** Old dataset versions are deleted automatically (`--delete-old-versions`)

---

## Troubleshooting

### "No existing dataset found"

**Expected behavior on first run.** Dataset will be created when first checkpoint saves.

### "Dataset upload failed"

**Causes:**
- Kaggle API not configured (shouldn't happen on Kaggle)
- Network issues
- Storage quota exceeded

**Action:** Training continues. Checkpoint still saved locally. Next successful upload will sync.

### "Failed to load checkpoint"

**Causes:**
- Corrupted checkpoint file
- Incompatible model architecture

**Action:** Training starts from scratch. Check error message for details.

### Training starts from epoch 1 despite checkpoints existing

**Check:**
1. Are you on Kaggle? (System only works on Kaggle)
2. Is `KAGGLE_USERNAME` environment variable set?
3. Does the dataset name match your username?

**Debug:**
```python
import os
print(f"Username: {os.environ.get('KAGGLE_USERNAME')}")
```

---

## Testing

### Local Tests

Run checkpoint logic tests:
```bash
python scripts/test_checkpoint_simple.py
```

Expected output:
```
======================================================================
ALL TESTS PASSED!
======================================================================
Checkpoint functionality verified:
  [OK] Checkpoints save correctly
  [OK] Latest checkpoint auto-detected
  [OK] Resume epoch tracking correct
  [OK] Training continuation logic correct

Ready for Kaggle deployment!
```

### Kaggle Tests

Test the full workflow on Kaggle:

1. Start training for 20 epochs
2. Manually stop the notebook
3. Verify checkpoint appears in Kaggle Dataset
4. Restart notebook
5. Verify training resumes from epoch 20

---

## Integration with Future Training Scripts

### Standard Pattern

All future training scripts should follow this pattern:

1. **Trainer Initialization:**
   - Detect Kaggle environment
   - Setup dataset persistence
   - Include checkpoint metadata (best_val_loss, best_epoch)

2. **Training Script:**
   - Check Kaggle Dataset on startup
   - Download existing checkpoints
   - Auto-detect and load latest checkpoint
   - Resume training loop from loaded epoch

3. **Checkpoint Saving:**
   - Save to local directory first
   - Upload to Kaggle Dataset after save
   - Handle upload failures gracefully

### Template Code

See `src/training/trainer.py` and `scripts/training/train_first_stage.py` as reference implementations.

**Key Methods:**
- `_setup_kaggle_dataset()` - Initialize persistence
- `_upload_checkpoint_to_kaggle_dataset()` - Upload after save
- Dataset download logic in training script startup

---

## Summary

**What This System Does:**
✅ Automatically saves checkpoints to persistent Kaggle Dataset  
✅ Automatically downloads and resumes from latest checkpoint  
✅ Survives session crashes, manual stops, and restarts  
✅ No manual intervention required  
✅ Training never wastes epochs re-training completed work  

**What You Need to Do:**
1. Push code to GitHub
2. Run training on Kaggle normally
3. If session crashes, just restart and run again
4. Training automatically continues from where it left off

**Result:** Reliable, crash-resistant training on Kaggle.
