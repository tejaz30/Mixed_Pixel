# Convolutional Autoencoder (CAE)

Baseline Convolutional Autoencoder for thermal image mixed-pixel detection, implementing the architecture described in **Section 4.1** of the research paper.

## Overview

The CAE learns to compress thermal images into a latent representation while preserving spatial information necessary for mixed-pixel detection. The reconstruction error map is used downstream for identifying mixed pixels.

### Architecture

```
Input (640×480 RGB)
    ↓
Encoder: Conv2D → ReLU → MaxPool (3 blocks)
    ↓
Latent Representation (8×60×80)
    ↓
Decoder: Upsample → Conv2D → ReLU (3 blocks)
    ↓
Output (640×480 RGB) + Sigmoid
```

### Key Specifications (from Paper)

- **Encoder filters**: 1, 6, 8
- **Decoder**: Mirrors encoder architecture
- **Input**: 640×480 thermal images, normalized to [0, 1]
- **Output**: Reconstructed images in [0, 1] (Sigmoid activation)
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam with learning rate 0.001
- **Max epochs**: 500
- **Early stopping**: Enabled with patience=5 epochs
- **Train/val split**: 80/20

## Module Structure

```
models/cae/
├── __init__.py           # Module exports
├── model.py              # Main CAE model
├── encoder.py            # Encoder network
├── decoder.py            # Decoder network
├── loss.py               # Loss functions and metrics
├── trainer.py            # Training loop with early stopping
├── inference.py          # Inference and reconstruction
├── utils.py              # Utility functions
└── README.md             # This file
```

## Usage

### 1. Create Model

```python
from models.cae import ConvolutionalAutoencoder

# Create model with paper specifications
model = ConvolutionalAutoencoder(
    input_channels=3,
    input_height=480,
    input_width=640,
    encoder_filters=(1, 6, 8)
)

print(model)
# Total parameters: ~XXX
```

### 2. Training

#### Using Training Script

```bash
# Train with default config
python scripts/train_cae.py

# Train with custom config
python scripts/train_cae.py --config configs/cae.yaml

# Resume from checkpoint
python scripts/train_cae.py --resume checkpoints/cae/checkpoint_epoch_50.pth

# Override config parameters
python scripts/train_cae.py --batch-size 64 --epochs 300 --lr 0.0005
```

#### Using Trainer Directly

```python
from models.cae import ConvolutionalAutoencoder, CAETrainer, get_device
from torch.utils.data import DataLoader

# Setup
device = get_device()
model = ConvolutionalAutoencoder()

# Create trainer
trainer = CAETrainer(
    model=model,
    device=device,
    checkpoint_dir='checkpoints/cae',
    tensorboard_dir='runs/cae',
    learning_rate=0.001,
    max_epochs=500,
    patience=5
)

# Train
history = trainer.train(train_loader, val_loader)

print(f"Best validation loss: {history['best_val_loss']:.6f}")
print(f"Epochs trained: {history['epochs_trained']}")
```

### 3. Inference

```python
from models.cae import load_model_for_inference
import torch

# Load trained model
inference = load_model_for_inference(
    checkpoint_path='checkpoints/cae/best_model.pth',
    device=device
)

# Predict on batch
images = torch.rand(4, 3, 480, 640)  # Batch of 4 images
results = inference.predict(images)

reconstructed = results['reconstructed']  # (4, 3, 480, 640)
error_map = results['error_map']          # (4, 3, 480, 640)
latent = results['latent']                # (4, 8, 60, 80)

# Reconstruct single image
single_image = torch.rand(3, 480, 640)
single_results = inference.reconstruct_image(single_image)

# Compute reconstruction error with different reductions
error_per_pixel = inference.compute_reconstruction_error(images, reduction='none')
error_per_channel = inference.compute_reconstruction_error(images, reduction='channel')
error_mean = inference.compute_reconstruction_error(images, reduction='mean')
```

### 4. Testing

```bash
# Run test suite to verify implementation
python scripts/test_cae.py
```

Tests verify:
- Model architecture and parameter counts
- Forward pass and shape consistency
- Loss computation
- Inference functionality
- Gradient flow
- Encoder-decoder symmetry

## Configuration

Configuration is managed through `configs/cae.yaml`:

### Model Configuration

```yaml
model:
  input_channels: 3
  input_height: 480
  input_width: 640
  encoder_filters: [1, 6, 8]
  kernel_size: 3
  pool_size: 2
  padding: 1
```

### Training Configuration

```yaml
training:
  optimizer: "adam"
  learning_rate: 0.001
  max_epochs: 500
  batch_size: 32
  
  early_stopping:
    enabled: true
    patience: 5
    min_delta: 0.0
    mode: "min"
  
  validation_split: 0.2
  save_frequency: 10
```

### Data Configuration

```yaml
data:
  dataset_root: "datasets/processed"
  num_workers: 4
  pin_memory: true
  shuffle: true
  train_ratio: 0.8
  val_ratio: 0.2
  seed: 42
```

## Model Architecture Details

### Encoder

Three convolutional blocks, each consisting of:
- Conv2D (kernel=3×3, padding=1)
- ReLU activation
- MaxPool2D (2×2)

**Progressive feature extraction:**
1. **Block 1**: 3 → 1 channels (low-level edges)
2. **Block 2**: 1 → 6 channels (intermediate spatial features)
3. **Block 3**: 6 → 8 channels (higher-level thermal features)

**Spatial compression:**
- Input: 640×480
- After block 1: 320×240
- After block 2: 160×120
- After block 3: 80×60
- **Latent**: 8×60×80

### Decoder

Mirrors encoder architecture:
- Upsample (scale=2)
- Conv2D (kernel=3×3, padding=1)
- ReLU activation

**Final layer:**
- Conv2D to project to RGB (3 channels)
- **Sigmoid activation** to constrain output to [0, 1]

### Loss Function

**Mean Squared Error (MSE)**:
```
L = (1/n) Σ(x_reconstructed - x_original)²
```

Computed per-pixel across all channels and averaged over the batch.

## Outputs

### Training Outputs

```
checkpoints/cae/
├── best_model.pth                # Best model (lowest validation loss)
├── checkpoint_epoch_10.pth       # Periodic checkpoints
├── checkpoint_epoch_20.pth
└── ...

logs/cae/
└── train.log                     # Training logs

runs/cae/
└── events.out.tfevents.*         # TensorBoard logs
```

### Model Checkpoint Contents

```python
checkpoint = {
    'model_state_dict': ...,       # Model weights
    'model_config': {              # Model architecture config
        'input_channels': 3,
        'input_height': 480,
        'input_width': 640,
        'encoder_filters': (1, 6, 8)
    },
    'optimizer_state_dict': ...,   # Optimizer state (if training checkpoint)
    'epoch': ...,                  # Current epoch
    'best_val_loss': ...,          # Best validation loss
    'train_losses': [...],         # Training loss history
    'val_losses': [...]            # Validation loss history
}
```

## Metrics

During training, the following metrics are computed:

- **Training loss**: MSE reconstruction loss on training set
- **Validation loss**: MSE reconstruction loss on validation set
- **MAE**: Mean Absolute Error
- **PSNR**: Peak Signal-to-Noise Ratio (dB)

Metrics are logged to:
- Console output
- Log files (`logs/cae/train.log`)
- TensorBoard (`runs/cae/`)

## TensorBoard

Monitor training progress:

```bash
tensorboard --logdir runs/cae
```

Visualizations include:
- Training and validation loss curves
- Reconstruction metrics (PSNR, MAE)
- Learning rate schedule

## Expected Performance

Based on paper specifications:
- Training will run for up to 500 epochs
- Early stopping typically triggers before max epochs
- Best model is automatically saved based on validation loss
- Reconstruction quality improves progressively during training

## Integration with Dataset

The CAE expects preprocessed thermal images from `datasets/processed/`:
- **Format**: RGB, 640×480
- **Normalization**: Values in [0, 1] (divided by 255)
- **Total samples**: 3,778 images
- **Classes**: 7 categories

Use `ThermalImageDataset` and `ThermalImageDataModule` from `data/` for loading.

## Future Work

This baseline CAE will be replaced with newer architectures:
- **Latent Diffusion Models (LDM)**
- **Context Clusters**
- **Continuous Feedback Control (CFC)**

The modular design allows replacing only the detector while keeping:
- Data loading pipeline
- Training infrastructure
- Evaluation framework
- Mixed-pixel detection logic

## API Reference

### ConvolutionalAutoencoder

```python
model = ConvolutionalAutoencoder(
    input_channels=3,
    input_height=480,
    input_width=640,
    encoder_filters=(1, 6, 8),
    kernel_size=3,
    pool_size=2,
    padding=1
)

# Forward pass
output = model(x)                           # Full reconstruction
latent = model.encode(x)                    # Encode only
decoded = model.decode(latent)              # Decode only
reconstructed, error = model.reconstruct(x) # Reconstruction + error map
```

### CAETrainer

```python
trainer = CAETrainer(
    model=model,
    device=device,
    checkpoint_dir=Path('checkpoints/cae'),
    tensorboard_dir=Path('runs/cae'),
    learning_rate=0.001,
    max_epochs=500,
    patience=5,
    save_frequency=10
)

history = trainer.train(train_loader, val_loader, resume_from=None)
```

### CAEInference

```python
inference = CAEInference(model, device, checkpoint_path)

results = inference.predict(images)
results = inference.reconstruct_image(single_image)
error = inference.compute_reconstruction_error(images, reduction='mean')
latent = inference.encode_batch(images)
reconstructed = inference.decode_batch(latent)
```

## Troubleshooting

### Common Issues

**Issue**: Out of memory during training
- **Solution**: Reduce batch size in config or command line: `--batch-size 16`

**Issue**: Training too slow
- **Solution**: Ensure GPU is being used, check `device` in config

**Issue**: Model not improving
- **Solution**: Check data normalization (should be [0, 1]), verify loss is decreasing

**Issue**: Shape mismatch errors
- **Solution**: Ensure input images are exactly 640×480 (use preprocessing pipeline)

## References

- Paper Section 4.1.2: Model Architecture
- Paper Section 4.1.3: Model Training
- PyTorch Documentation: https://pytorch.org/docs/
