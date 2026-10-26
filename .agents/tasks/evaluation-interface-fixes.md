# KL vs VQ Evaluation - Complete Interface Audit

**Date:** 2026-10-03  
**Issue:** Evaluation script had wrong assumptions about model return formats  
**Commits:** 5d5e797, c35030b

---

## Root Cause

The evaluation script and diagnostic classes were written assuming models return **dictionaries**, but the actual models return **tuples**.

---

## Ground Truth (from actual source code)

### KL Autoencoder (`src/models/ldm/first_stage/kl/model.py`)

**Forward signature:**
```python
def forward(self, x: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns: (reconstruction, mean, logvar)"""
```

**Training code usage:**
```python
reconstruction, mean, logvar = self.model(images, sample=True)
```

### VQ Autoencoder (`src/models/ldm/first_stage/vq/model.py`)

**Forward signature:**
```python
def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Returns: (reconstruction, loss_dict)"""
```

Where `loss_dict` contains:
- `z_quantized`: Post-quantization latent
- `z_continuous`: Pre-quantization latent  
- `vq_loss`, `codebook_loss`, `commitment_loss`: VQ losses

**Encode signature:**
```python
def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Returns: (z_quantized, h_pre_quant, vq_losses)"""
```

**Training code usage:**
```python
reconstruction, vq_outputs = self.model(images)
```

---

## Fixes Applied

### 1. Evaluation Script (`scripts/evaluation/evaluate_kl_vs_vq.py`)

#### KL evaluation function (Line ~290)

**Wrong:**
```python
outputs = model(images)
reconstructed = outputs['reconstruction']
latents = outputs['latent']
```

**Fixed:**
```python
# KL model returns tuple: (reconstruction, mean, logvar)
reconstruction, mean, logvar = model(images, sample=False)
# Use mean for latent analysis
```

#### VQ evaluation function (Line ~390)

**Wrong:**
```python
outputs = model(images)
reconstructed = outputs['reconstruction']
latents = outputs['post_quant']
```

**Fixed:**
```python
# VQ model returns tuple: (reconstruction, loss_dict)
reconstruction, loss_dict = model(images)

# Extract latents from loss_dict
z_quantized = loss_dict['z_quantized']  
z_continuous = loss_dict['z_continuous']
```

### 2. KL Diagnostics (`src/evaluation/kl_diagnostics.py`)

#### Line ~465

**Wrong:**
```python
outputs = self.model(x)
mean = outputs['mean']
logvar = outputs['logvar']
```

**Fixed:**
```python
# KL model returns tuple: (reconstruction, mean, logvar)
_, mean, logvar = self.model(x, sample=False)
```

### 3. VQ Diagnostics (`src/evaluation/vq_diagnostics.py`)

#### Line ~85

**Wrong:**
```python
outputs = self.model.encode_with_vq_diagnostics(x)  # Method doesn't exist!
```

**Fixed:**
```python
# VQ model encode() returns: (z_quantized, h_pre_quant, vq_losses)
z_quantized, h_pre_quant, vq_losses = self.model.encode(x)

# Compute indices manually (not in vq_losses dict)
# This matches what the quantizer does internally
h_flattened = h_pre_quant.permute(0, 2, 3, 1).contiguous().view(-1, h_pre_quant.shape[1])

distances = (
    torch.sum(h_flattened ** 2, dim=1, keepdim=True)
    + torch.sum(self.model.quantizer.embedding.weight ** 2, dim=1)
    - 2 * torch.matmul(h_flattened, self.model.quantizer.embedding.weight.t())
)

encoding_indices = torch.argmin(distances, dim=1)
B, C, H, W = h_pre_quant.shape
indices = encoding_indices.view(B, H, W)

outputs = {
    'post_quant': z_quantized,
    'pre_quant': h_pre_quant,
    'indices': indices,
    'vq_loss': vq_losses.get('vq_loss', None)
}
```

**Why indices needed to be computed:**
- VQ diagnostics needs indices for codebook usage analysis
- The quantizer computes indices internally but doesn't return them
- Solution: Replicate the quantizer's distance computation to get indices

---

## Verification

**Local tests:** ✅ 21/21 passing

```bash
pytest tests/test_kl_vq_evaluation.py -v
```

All diagnostic classes now correctly interface with actual model implementations.

---

## Lesson Learned

**ALWAYS read the actual source code:**
1. Don't assume model return formats
2. Check training code to see verified working patterns
3. Verify method existence before calling
4. Match exact tuple unpacking to actual returns

**The training code is the ground truth** - it's been tested and working. Match its patterns exactly.
