# VQ Checkpoint Architecture Mismatch Investigation

**Date:** 2026-10-28  
**Investigation Context:** Attempting to load VQ checkpoint from Experiment 1 (KL vs VQ comparison) failed with state_dict key mismatch. This investigation resolves whether the model architecture was changed after training.

---

## Executive Summary

**FINDING: NO ARCHITECTURE CHANGE DETECTED**

The VQ checkpoint (`best_model (1).pth`) was trained with the **CURRENT** VQ architecture. The checkpoint contains the expected `encoder.conv_in`, `encoder.down.*.blocks`, `decoder.conv_out`, `decoder.up.*.blocks` structure that matches the current `src/models/ldm/first_stage/vq/model.py` implementation.

**The block1/block2/block3 naming pattern observed during debugging comes from the CAE model, NOT the VQ model.**

---

## Critical Questions Answered

### 1. Has the VQ model architecture file been modified since Experiment 1?

**Answer: NO**

**Evidence:**
- VQ model.py was created on **October 3, 2026** in commit `436cc8f5` ("Restructure")
- VQ checkpoint was saved on **October 28, 2026** (25 days AFTER the model was created)
- The September 12 commit (`fb035f5b`) that initially created LDM foundations had an IDENTICAL VQ model architecture at path `models/ldm/first_stage/vq/model.py` (moved to `src/models/ldm/first_stage/vq/model.py` in the October 3 restructure)
- No commits between October 3 and October 28 modified the VQ model architecture

**Git History:**
```
436cc8f5 | 2026-10-03 22:03:16 | Restructure
  - Created src/models/ldm/first_stage/vq/model.py (new path)
  - Architecture: Encoder → quant_conv → VectorQuantizer → post_quant_conv → Decoder

fb035f5b | 2026-09-12 18:21:25 | LDM beginnings: Encoder and Decoder of VAE is built...
  - Created models/ldm/first_stage/vq/model.py (old path)
  - IDENTICAL architecture to current version
```

**Conclusion:** The VQ model has been stable since creation. The checkpoint was trained with the current architecture.

---

### 2. Are there multiple VQ implementations in the codebase?

**Answer: NO**

**Evidence:**
- Only ONE VQ autoencoder implementation exists: `src/models/ldm/first_stage/vq/model.py`
- The `VQAutoencoder` class is the only VQ model class in the repository
- No alternative implementations, backup files, or `_old` variants found

**Search Results:**
```
src/models/ldm/first_stage/vq/
├── __init__.py
├── model.py          ← ONLY VQ autoencoder implementation
├── loss.py           ← Loss function
└── vector_quantizer.py  ← Quantizer module
```

**Confusion Source Identified:**
The `block1`, `block2`, `block3` naming pattern comes from the **CAE model**, not VQ:
- `src/models/cae/encoder.py` uses `self.block1`, `self.block2`, `self.block3`
- `src/models/cae/decoder.py` uses `self.block1`, `self.block2`, `self.block3`

This simple naming is appropriate for the CAE baseline but is NOT used in the VQ/KL first-stage models.

---

### 3. What is the actual structure saved in the checkpoint?

**Answer: Current VQ architecture with correct Encoder/Decoder structure**

**Checkpoint Structure Analysis:**
```
File: outputs/experiment1/vq_nomixed/best_model (1).pth
Size: 58.88 MB
Date: October 28, 2026 12:39:41 PM
Epoch: 43
Best Val Loss: 0.001910

Top-level keys:
  - model_state_dict    ← Model weights
  - optimizer_state_dict
  - scheduler_state_dict
  - epoch
  - val_loss
  - best_val_loss
  - best_epoch
  - config              ← Full experiment config preserved
```

**Model State Dict (231 parameters total):**

#### Encoder Keys (96 parameters)
```
encoder.conv_in.weight          [32, 3, 3, 3]
encoder.conv_in.bias            [32]

encoder.down.0.blocks.0.norm1.weight    [32]
encoder.down.0.blocks.0.norm1.bias      [32]
encoder.down.0.blocks.0.conv1.weight    [32, 32, 3, 3]
encoder.down.0.blocks.0.conv1.bias      [32]
encoder.down.0.blocks.0.norm2.weight    [32]
encoder.down.0.blocks.0.norm2.bias      [32]
encoder.down.0.blocks.0.conv2.weight    [32, 32, 3, 3]
encoder.down.0.blocks.0.conv2.bias      [32]
encoder.down.0.blocks.1.*              (second ResBlock)
encoder.down.0.downsample.conv.weight   [32, 32, 3, 3]
encoder.down.0.downsample.conv.bias     [32]

encoder.down.1.*                       (second resolution level)
encoder.down.2.*                       (third resolution level)
encoder.down.3.*                       (fourth resolution level)

encoder.conv_out.weight         [4, 128, 3, 3]
encoder.conv_out.bias           [4]
```

#### Decoder Keys (128 parameters)
```
decoder.conv_in.weight          [128, 4, 3, 3]
decoder.conv_in.bias            [128]

decoder.mid.0.*                 (middle ResBlocks)
decoder.mid.1.*

decoder.up.0.*                  (first upsampling level)
decoder.up.1.*                  (second upsampling level)
decoder.up.2.*                  (third upsampling level)

decoder.norm_out.weight         [32]
decoder.norm_out.bias           [32]
decoder.conv_out.weight         [3, 32, 3, 3]
decoder.conv_out.bias           [3]
```

#### Quantizer Keys (3 parameters)
```
quantizer.embedding.weight      [1024, 4]    ← Codebook
quantizer.ema_cluster_size      [1024]       ← EMA cluster tracking
quantizer.ema_embed_avg         [1024, 4]    ← EMA embedding average
```

#### Other Keys (4 parameters)
```
quant_conv.weight               [4, 4, 1, 1]
quant_conv.bias                 [4]
post_quant_conv.weight          [4, 4, 1, 1]
post_quant_conv.bias            [4]
```

**Architecture Preserved in Config:**
```yaml
model:
  type: "vq_autoencoder"
  in_channels: 3
  latent_channels: 4
  base_channels: 32
  channel_multipliers: [1, 2, 4, 4]
  num_res_blocks: 2
  num_embeddings: 1024
  commitment_cost: 0.25
  ema_decay: 0.99
```

**Conclusion:** The checkpoint contains the exact architecture currently implemented in `src/models/ldm/first_stage/vq/model.py`.

---

### 4. Which VQ implementation was used in Experiment 1 training?

**Answer: The CURRENT implementation (only one exists)**

**Evidence:**
1. **Timeline confirms continuity:**
   - Sep 12: VQ model created at `models/ldm/first_stage/vq/model.py`
   - Oct 3: Restructure moved to `src/models/ldm/first_stage/vq/model.py` (no arch change)
   - Oct 28: Checkpoint saved from Experiment 1 training

2. **Checkpoint config matches current implementation:**
   - Config preserves `base_channels: 32`, `channel_multipliers: [1, 2, 4, 4]`
   - Matches `configs/experiment1/vq_autoencoder_kaggle_2gpu.yaml`

3. **Training was successful:**
   - Trained for 43 epochs
   - Achieved val_loss: 0.001910
   - No architectural errors during training

**Conclusion:** The training used the current VQ implementation. The architecture has not changed.

---

## Root Cause Diagnosis

### The Mismatch is NOT Due to Architecture Change

The checkpoint architecture is **correct** and matches the current code.

### Likely Root Causes of Loading Failure

Based on the investigation, the actual loading failure is likely due to one of these issues:

1. **Incorrect Model Instantiation**
   - Loading code may be instantiating the model with wrong parameters
   - Example: Using `base_channels=128` instead of `base_channels=32`
   - The checkpoint config shows `base_channels: 32` was used during training

2. **DataParallel/DDP Wrapper Issue**
   - If trained with multi-GPU (2x T4 confirmed in config), state_dict may have `module.` prefix
   - Loading code may need to strip or add the prefix

3. **Wrong Model Class**
   - Loading code may be trying to load VQ checkpoint into CAE model (which has block1/block2/block3)
   - Or vice versa

4. **State Dict Key Extraction**
   - Checkpoint is wrapped: `checkpoint['model_state_dict']`
   - Loading code must extract: `state_dict = torch.load(...)['model_state_dict']`
   - Not: `state_dict = torch.load(...)`

---

## Recommended Next Steps

### 1. Verify Loading Code Parameters

Check that the model is instantiated with **EXACT** config from checkpoint:
```python
config = checkpoint['config']['model']
model = VQAutoencoder(
    in_channels=config['in_channels'],          # 3
    latent_channels=config['latent_channels'],  # 4
    base_channels=config['base_channels'],      # 32 (NOT 128!)
    channel_multipliers=config['channel_multipliers'],  # [1, 2, 4, 4]
    num_res_blocks=config['num_res_blocks'],    # 2
    num_embeddings=config['num_embeddings'],    # 1024
    commitment_cost=config['commitment_cost']   # 0.25
)
```

### 2. Check for Multi-GPU Wrapper

If state_dict keys have `module.` prefix:
```python
state_dict = checkpoint['model_state_dict']

# Remove 'module.' prefix if present
if any(k.startswith('module.') for k in state_dict.keys()):
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
```

### 3. Verify Model Class

Ensure loading the correct model:
```python
# CORRECT
from src.models.ldm.first_stage.vq.model import VQAutoencoder

# INCORRECT (would have block1/block2/block3)
from src.models.cae.model import CAE  # Wrong model!
```

### 4. Print Key Comparison

Debug by comparing expected vs actual keys:
```python
model_keys = set(model.state_dict().keys())
checkpoint_keys = set(checkpoint['model_state_dict'].keys())

missing = model_keys - checkpoint_keys
unexpected = checkpoint_keys - model_keys

print(f"Missing keys: {missing}")
print(f"Unexpected keys: {unexpected}")
```

---

## Research Integrity Status

✅ **EXPERIMENT 1 INTEGRITY PRESERVED**

- The VQ model architecture has been stable since creation
- The checkpoint was trained with the current implementation
- No post-training architecture modifications occurred
- The Experiment 1 results remain valid and reproducible

**The loading failure is an implementation/configuration issue, not an architecture change.**

---

## Files Examined

### Git History
- `src/models/ldm/first_stage/vq/model.py` (commit history)
- `src/models/ldm/first_stage/vq/vector_quantizer.py` (commit history)
- `src/models/ldm/first_stage/encoder.py` (commit history)
- `models/ldm/first_stage/vq/model.py` (old path, September commit)

### Checkpoint
- `outputs/experiment1/vq_nomixed/best_model (1).pth` (direct inspection)

### Configuration
- `configs/experiment1/vq_autoencoder_kaggle_2gpu.yaml` (training config)

### Source Code
- `src/models/ldm/first_stage/vq/model.py` (current VQ implementation)
- `src/models/ldm/first_stage/encoder.py` (current encoder)
- `src/models/ldm/first_stage/decoder.py` (current decoder)
- `src/models/cae/encoder.py` (CAE with block1/block2/block3)
- `src/models/cae/decoder.py` (CAE with block1/block2/block3)

---

## Appendix: Timeline

```
2026-09-12: VQ model created (initial LDM work)
  - Path: models/ldm/first_stage/vq/model.py
  - Architecture: Encoder → Quantizer → Decoder

2026-10-03: Repository restructured
  - Path changed to: src/models/ldm/first_stage/vq/model.py
  - Architecture: UNCHANGED

2026-10-16: Project documentation initialized
  - No model changes

2026-10-26: Multiple evaluation script fixes
  - No model architecture changes
  - Only fixing import paths and interface issues

2026-10-28: Checkpoint saved
  - File: outputs/experiment1/vq_nomixed/best_model (1).pth
  - Architecture: CURRENT VQ implementation
  - Training completed successfully at epoch 43
```

---

## Investigator Notes

This investigation was conducted to resolve a research-integrity concern before proceeding with LDM baseline specification. The concern was whether the VQ checkpoint was trained with a different architecture than currently exists in the codebase.

**Conclusion:** No architecture change detected. The checkpoint is valid and was trained with the current implementation. The loading failure is a separate implementation issue that needs debugging in the loading code, not the model architecture.
