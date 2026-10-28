"""
Inspect VQ checkpoint structure
"""
import torch
from pathlib import Path

checkpoint_path = r"c:\Users\tejab\OneDrive\Desktop\Mixed Pixels\outputs\experiment1\vq_nomixed\best_model (1).pth"

print("=" * 80)
print(f"Inspecting checkpoint: {checkpoint_path}")
print("=" * 80)

# Load checkpoint with weights_only for safety
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print(f"\nTop-level keys in checkpoint:")
for key in checkpoint.keys():
    print(f"  - {key}")

# Check if it's wrapped
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print("\nCheckpoint structure: Wrapped (has 'model_state_dict' key)")
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    print("\nCheckpoint structure: Wrapped (has 'state_dict' key)")
else:
    state_dict = checkpoint
    print("\nCheckpoint structure: Direct state_dict")

print(f"\nNumber of model parameters: {len(state_dict)}")

print("\n" + "=" * 80)
print("All model parameter keys:")
print("=" * 80)

encoder_keys = []
decoder_keys = []
quantizer_keys = []
other_keys = []

for key in sorted(state_dict.keys()):
    if key.startswith('encoder.'):
        encoder_keys.append(key)
    elif key.startswith('decoder.'):
        decoder_keys.append(key)
    elif key.startswith('quantizer.'):
        quantizer_keys.append(key)
    else:
        other_keys.append(key)

print(f"\n### ENCODER KEYS ({len(encoder_keys)}):")
for key in encoder_keys[:50]:  # First 50
    shape = state_dict[key].shape
    print(f"  {key:60s} {str(shape):30s}")
if len(encoder_keys) > 50:
    print(f"  ... and {len(encoder_keys) - 50} more")

print(f"\n### DECODER KEYS ({len(decoder_keys)}):")
for key in decoder_keys[:50]:  # First 50
    shape = state_dict[key].shape
    print(f"  {key:60s} {str(shape):30s}")
if len(decoder_keys) > 50:
    print(f"  ... and {len(decoder_keys) - 50} more")

print(f"\n### QUANTIZER KEYS ({len(quantizer_keys)}):")
for key in quantizer_keys:
    shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'scalar'
    print(f"  {key:60s} {str(shape):30s}")

if other_keys:
    print(f"\n### OTHER KEYS ({len(other_keys)}):")
    for key in other_keys:
        shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'scalar'
        print(f"  {key:60s} {str(shape):30s}")

# Check for metadata
print("\n" + "=" * 80)
print("Checkpoint Metadata:")
print("=" * 80)

if 'epoch' in checkpoint:
    print(f"Epoch: {checkpoint['epoch']}")
if 'best_val_loss' in checkpoint:
    print(f"Best val loss: {checkpoint['best_val_loss']:.6f}")
if 'optimizer_state_dict' in checkpoint:
    print("Optimizer state: Present")
if 'config' in checkpoint:
    print("Config: Present")
    print(checkpoint['config'])
