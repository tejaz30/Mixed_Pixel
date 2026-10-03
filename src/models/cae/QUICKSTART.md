# CAE Quick Start Guide

Fast track to training the Convolutional Autoencoder for thermal image reconstruction.

## Prerequisites

✅ Dataset preprocessed and available at `datasets/processed/`  
✅ Python environment with PyTorch installed  
✅ GPU recommended (CUDA) but CPU works too

## Quick Start (30 seconds)

### 1. Run Tests (Verify Implementation)

```bash
python scripts/test_cae.py
```

Expected output: `ALL TESTS PASSED ✓`

### 2. Train Model (Default Config)

```bash
python scripts/train_cae.py
```

This will:
- Load 3,778 preprocessed thermal images from `datasets/processed/`
- Split 80/20 train/validation (paper specification)
- Train CAE with Adam optimizer (lr=0.001)
- Run for up to 500 epochs with early stopping (patience=5)
- Save checkpoints to `checkpoints/cae/`
- Log to TensorBoard at `runs/cae/`

### 3. Monitor Training

```bash
tensorboard --logdir runs/cae
```

Open http://localhost:6006 to view training progress.

## Training Options

### Resume Training

```bash
python scripts/train_cae.py --resume checkpoints/cae/checkpoint_epoch_50.pth
```

### Custom Batch Size

```bash
python scripts/train_cae.py --batch-size 64
```

### Custom Learning Rate

```bash
python scripts/train_cae.py --lr 0.0005
```

### Limit Epochs

```bash
python scripts/train_cae.py --epochs 100
```

### Force CPU

```bash
python scripts/train_cae.py --device cpu
```

### Combine Options

```bash
python scripts/train_cae.py --batch-size 64 --epochs 200 --lr 0.0005
```

## Expected Training Time

**GPU (NVIDIA RTX 3090)**:
- ~30-60 seconds per epoch (batch_size=32)
- Total: ~30-90 minutes (early stopping likely triggers before 500 epochs)

**CPU**:
- ~5-10 minutes per epoch
- Total: Several hours (not recommended for full training)

## Output Files

After training completes:

```
checkpoints/cae/
├── best_model.pth              ← Use this for inference
├── checkpoint_epoch_10.pth
├── checkpoint_epoch_20.pth
└── ...

logs/cae/
└── train.log                   ← Training logs

runs/cae/
└── events.out.tfevents.*       ← TensorBoard logs
```

## Quick Inference Test

After training, test the model:

```python
from models.cae import load_model_for_inference
import torch

# Load trained model
inference = load_model_for_inference(
    checkpoint_path='checkpoints/cae/best_model.pth'
)

# Create test image (640×480, normalized to [0, 1])
test_image = torch.rand(3, 480, 640)

# Reconstruct
results = inference.reconstruct_image(test_image)

print(f"Reconstructed shape: {results['reconstructed'].shape}")
print(f"Error map shape: {results['error_map'].shape}")
print(f"Mean reconstruction error: {results['error_map'].mean():.6f}")
```

## Troubleshooting

### Out of Memory Error

```bash
python scripts/train_cae.py --batch-size 16
```

Or edit `configs/cae.yaml` and set `training.batch_size: 16`

### Training Too Slow

- Ensure GPU is detected: Check console for "Device: cuda"
- Reduce `data.num_workers` in config if CPU bottleneck
- Use smaller batch size for faster iterations (but more epochs needed)

### Model Not Improving

- Check data normalization: Images should be in [0, 1]
- Verify dataset loading: Console should show 3,778 images
- Check loss is decreasing: Monitor TensorBoard or console output

### Import Errors

```bash
# Ensure you're in project root
cd "path/to/Mixed Pixels"

# Run from project root
python scripts/train_cae.py
```

## What's Next?

After training:

1. ✅ **Evaluate reconstruction quality** - Compute metrics on test set
2. ✅ **Visualize reconstructions** - Compare original vs reconstructed images  
3. ✅ **Analyze error maps** - Identify where reconstruction fails
4. ✅ **Mixed-pixel detection** - Use error maps to detect mixed pixels
5. ✅ **Replace with newer detectors** - LDM, Context Clusters, CFC

## Configuration

To customize training, edit `configs/cae.yaml`:

```yaml
# Key settings to tune
training:
  batch_size: 32          # Reduce if OOM
  learning_rate: 0.001    # Paper default
  max_epochs: 500         # Paper default
  early_stopping:
    patience: 5           # Paper default

data:
  num_workers: 4          # Parallel data loading
  pin_memory: true        # Faster GPU transfer

hardware:
  device: "auto"          # "cuda", "cpu", or "auto"
```

## Paper Compliance Checklist

✅ Input: 640×480 thermal images normalized to [0, 1]  
✅ Architecture: Encoder (1, 6, 8 filters) + Decoder  
✅ Loss: Mean Squared Error (MSE)  
✅ Optimizer: Adam with lr=0.001  
✅ Max epochs: 500  
✅ Early stopping: Patience=5  
✅ Train/val split: 80/20  
✅ Output: Sigmoid activation, range [0, 1]

## Support

For detailed documentation:
- Full API: `models/cae/README.md`
- Training details: `models/cae/trainer.py`
- Architecture: `models/cae/model.py`
- Configuration: `configs/cae.yaml`
