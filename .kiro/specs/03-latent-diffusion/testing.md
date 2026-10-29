# Latent Diffusion Model (Stage 2) — Testing Plan

## Overview

This document specifies testing requirements for the LDM Stage 2 baseline implementation. All tests must pass before training begins.

## Pre-Training Tests

### 1. VQ First-Stage Interface Tests

**Verify frozen VQ model loads and functions correctly.**

```python
def test_vq_loading():
    """Test VQ checkpoint loads with correct architecture."""
    checkpoint_path = 'outputs/experiment1/vq_nomixed/best_model (1).pth'
    config_path = 'configs/experiment1/vq_autoencoder_kaggle_2gpu.yaml'
    
    # Load model
    model = load_vq_model(checkpoint_path, config_path, device='cuda')
    
    # Verify frozen
    assert all(not p.requires_grad for p in model.parameters())
    
    # Verify codebook loaded
    assert hasattr(model.quantizer, 'ema_cluster_size')
    assert model.quantizer.ema_cluster_size.sum() > 0

def test_vq_encode_decode():
    """Test VQ encode/decode with correct shapes."""
    model = load_vq_model(...)
    x = torch.randn(2, 3, 480, 640, device='cuda')
    
    # Encode
    z_quantized, z_continuous, vq_losses = model.encode(x)
    assert z_quantized.shape == (2, 4, 60, 80)
    assert z_continuous.shape == (2, 4, 60, 80)
    
    # Decode
    recon = model.decode(z_quantized)
    assert recon.shape == (2, 3, 480, 640)
    assert torch.isfinite(recon).all()
```

### 2. Latent Scaler Tests

**Verify standardization is invertible.**

```python
def test_latent_scaler_invertibility():
    """Test scale/unscale round-trip."""
    scaler = LatentScaler(mean=0.01289583, std=0.19909434)
    
    z = torch.randn(8, 4, 60, 80)
    z_scaled = scaler.scale(z)
    z_reconstructed = scaler.unscale(z_scaled)
    
    error = (z - z_reconstructed).abs().max().item()
    assert error < 1e-5, f"Invertibility error: {error}"

def test_latent_scaler_statistics():
    """Verify scaled latents have mean≈0, std≈1."""
    scaler = LatentScaler(mean=0.01289583, std=0.19909434)
    
    # Load training set latents
    z_batch = load_training_latents(num_samples=100)
    z_scaled = scaler.scale(z_batch)
    
    mean = z_scaled.mean().item()
    std = z_scaled.std().item()
    
    assert abs(mean) < 0.1, f"Mean not close to 0: {mean}"
    assert abs(std - 1.0) < 0.1, f"Std not close to 1: {std}"
```

### 3. Noise Scheduler Tests

**Verify noise schedule is correctly implemented.**

```python
def test_cosine_beta_schedule():
    """Test cosine schedule produces valid beta values."""
    betas = cosine_beta_schedule(1000)
    
    assert betas.shape == (1000,)
    assert (betas >= 0.0001).all()
    assert (betas <= 0.9999).all()
    assert torch.isfinite(betas).all()

def test_noise_scheduler_forward():
    """Test adding noise produces correct distribution."""
    scheduler = NoiseScheduler(num_timesteps=1000, schedule='cosine')
    
    z_0 = torch.randn(8, 4, 60, 80)
    t = torch.randint(0, 1000, (8,))
    noise = torch.randn_like(z_0)
    
    z_t = scheduler.add_noise(z_0, t, noise)
    
    assert z_t.shape == z_0.shape
    assert torch.isfinite(z_t).all()
```

### 4. U-Net Architecture Tests

**Verify U-Net forward pass produces correct shapes.**

```python
def test_unet_forward():
    """Test U-Net forward pass."""
    unet = DiffusionUNet(
        in_channels=4,
        out_channels=4,
        base_channels=128,
        channel_multipliers=[1, 2, 4, 4],
        num_res_blocks=2,
        attention_levels=[2, 3],
        time_emb_dim=256
    )
    
    z_t = torch.randn(8, 4, 60, 80)
    t = torch.randint(0, 1000, (8,))
    
    noise_pred = unet(z_t, t)
    
    assert noise_pred.shape == (8, 4, 60, 80)
    assert torch.isfinite(noise_pred).all()

def test_unet_parameter_count():
    """Verify parameter count is within expected range."""
    unet = DiffusionUNet(...)
    num_params = sum(p.numel() for p in unet.parameters())
    
    assert 20e6 < num_params < 35e6, f"Unexpected param count: {num_params/1e6:.1f}M"
```

### 5. EMA Tests

**Verify EMA updates correctly.**

```python
def test_ema_update():
    """Test EMA shadow parameters update."""
    model = DiffusionUNet(...)
    ema = EMA(model, decay=0.9999)
    
    # Initial state
    initial_shadow = {name: param.clone() for name, param in ema.shadow.items()}
    
    # Simulate training step
    for param in model.parameters():
        param.data += 0.1 * torch.randn_like(param)
    
    # Update EMA
    ema.update()
    
    # Verify shadow changed
    for name, param in ema.shadow.items():
        assert not torch.equal(param, initial_shadow[name])
```

### 6. Training Loop Sanity Tests

**Verify training loop runs without errors.**

```python
def test_single_training_step():
    """Test one training step completes."""
    # Setup
    vq_model = load_vq_model(...)
    unet = DiffusionUNet(...)
    scheduler = NoiseScheduler(...)
    scaler = LatentScaler(...)
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
    
    # Load batch
    images = load_batch(batch_size=8)
    
    # Encode
    with torch.no_grad():
        z_quantized, _, _ = vq_model.encode(images)
    z_0 = scaler.scale(z_quantized)
    
    # Sample timestep and noise
    t = torch.randint(0, 1000, (8,), device=z_0.device)
    noise = torch.randn_like(z_0)
    
    # Add noise
    z_t = scheduler.add_noise(z_0, t, noise)
    
    # Predict noise
    noise_pred = unet(z_t, t)
    
    # Compute loss
    loss = F.mse_loss(noise_pred, noise)
    
    # Backward
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    assert torch.isfinite(loss)
    assert loss.item() > 0
```

### 7. Sampling Tests

**Verify sampling produces finite outputs.**

```python
def test_sampling():
    """Test 50-step DDPM sampling."""
    unet = load_trained_unet(...)  # or random init for shape test
    scheduler = NoiseScheduler(...)
    scaler = LatentScaler(...)
    vq_decoder = load_vq_decoder(...)
    
    images = sample(
        unet=unet,
        scheduler=scheduler,
        scaler=scaler,
        vq_decoder=vq_decoder,
        batch_size=4,
        device='cuda'
    )
    
    assert images.shape == (4, 3, 480, 640)
    assert torch.isfinite(images).all()
    assert images.min() >= -0.5  # Rough sanity check
    assert images.max() <= 1.5
```

## Smoke Training Test

**After pre-training tests pass, run 1-5 epoch smoke test.**

### Acceptance Criteria

1. **No crashes** - Training completes without errors
2. **Finite outputs** - No NaN or Inf in loss, gradients, or activations
3. **Loss decreases** - Training loss shows downward trend
4. **Memory fits** - Peak memory usage < 14GB on T4 (2GB safety margin)
5. **Checkpoints save** - Checkpoint file created with correct keys

### Smoke Test Script

```python
def smoke_test_ldm_training():
    """Run 1-5 epoch smoke training test."""
    config = load_config('configs/ldm_stage2_baseline.yaml')
    config['training']['max_epochs'] = 5
    config['training']['save_frequency'] = 1
    
    trainer = LDMTrainer(config)
    history = trainer.train()
    
    # Verify loss decreased
    initial_loss = history['train_loss'][0]
    final_loss = history['train_loss'][-1]
    assert final_loss < initial_loss, "Loss did not decrease"
    
    # Verify checkpoint exists
    checkpoint_path = Path(config['paths']['checkpoint_dir']) / 'checkpoint_epoch_5.pth'
    assert checkpoint_path.exists()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    assert 'model_state_dict' in checkpoint
    assert 'ema_shadow' in checkpoint
    assert 'optimizer_state_dict' in checkpoint
```

## Test Execution Order

1. **VQ interface tests** - Verify frozen first-stage works
2. **Latent scaler tests** - Verify standardization correct
3. **Noise scheduler tests** - Verify diffusion math correct
4. **U-Net tests** - Verify architecture correct
5. **EMA tests** - Verify EMA module works
6. **Training step test** - Verify single step runs
7. **Sampling test** - Verify generation pipeline works
8. **Smoke training test** - Verify full training works for 1-5 epochs

**All tests must pass before launching full Kaggle training.**
