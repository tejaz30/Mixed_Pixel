# Checkpoint Auto-Resume Guide

## What Was Fixed

Training now **automatically resumes** from the last saved checkpoint if the session crashes or restarts.

## How It Works

1. **Checkpoints are saved** every N epochs (configured in your YAML)
2. **On restart**, the training script automatically:
   - Scans the checkpoint directory
   - Finds the latest `checkpoint_epoch_X.pth`
   - Loads model, optimizer, and training state
   - **Continues from that epoch**

## Verified Functionality

✅ **Tested locally** - all tests passed:
- Checkpoints save correctly
- Latest checkpoint auto-detected
- Resume epoch tracking correct  
- Training continuation logic correct

## Usage on Kaggle

### Interactive Session (Your Preferred Method)

Just run your notebook normally. If Kaggle crashes/restarts:

```python
# Cell 3 remains the same - NO CHANGES NEEDED!
!python scripts/training/train_first_stage.py {KL_CONFIG}
```

The script will:
1. Check for existing checkpoints in `/kaggle/working/checkpoints/kl/`
2. If found, automatically resume from the latest one
3. If not found, start fresh from epoch 1

### What You'll See on Resume

```
======================================================================
🔄 RESUMING FROM CHECKPOINT
======================================================================
Checkpoint: /kaggle/working/checkpoints/kl/checkpoint_epoch_30.pth
✓ Resumed from epoch 30
✓ Best val loss so far: 0.003845
✓ Will continue training from epoch 31
```

## Kaggle Persistence

**IMPORTANT**: In interactive sessions, checkpoints in `/kaggle/working/` are temporary!

To keep them after session ends:
- Download checkpoints during training breaks
- OR use "Save & Run All (Commit)" mode which persists `/kaggle/working/` to Output

## Manual Resume (Optional)

If you want to explicitly resume from a specific checkpoint:

```bash
!python scripts/training/train_first_stage.py {KL_CONFIG} --resume /kaggle/working/checkpoints/kl/checkpoint_epoch_50.pth
```

## Files Modified

- `scripts/training/train_first_stage.py` - Added auto-resume logic
- `src/training/trainer.py` - Fixed epoch loop to continue from resumed epoch

## Test Results

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

Run the test yourself:
```bash
python scripts/test_checkpoint_simple.py
```

## Ready to Deploy!

Your training is now crash-resistant. Push to GitHub and run on Kaggle with confidence.
