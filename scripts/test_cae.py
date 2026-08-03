"""
Test script for Convolutional Autoencoder (CAE).

Verifies:
- Model architecture
- Forward pass
- Loss computation
- Inference
- Input/output shapes

Usage:
    python scripts/test_cae.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from models.cae import (
    ConvolutionalAutoencoder,
    CAELoss,
    CAEInference,
    get_device,
    count_parameters,
    validate_input_shape
)


def test_model_creation():
    """Test model creation and architecture."""
    print("\n" + "="*70)
    print("TEST 1: Model Creation")
    print("="*70)
    
    # Create model with paper specifications
    model = ConvolutionalAutoencoder(
        input_channels=3,
        input_height=480,
        input_width=640,
        encoder_filters=(1, 6, 8)
    )
    
    print(model)
    print()
    
    # Check parameters
    params = count_parameters(model)
    print(f"✓ Total parameters: {params['total']:,}")
    print(f"✓ Trainable parameters: {params['trainable']:,}")
    
    # Check latent shape
    latent_shape = model.get_latent_shape()
    print(f"✓ Latent shape: {latent_shape}")
    
    expected_latent = (8, 60, 80)  # After 3x pooling from 480x640
    assert latent_shape == expected_latent, f"Expected {expected_latent}, got {latent_shape}"
    
    print("✓ Model creation test PASSED")
    return model


def test_forward_pass(model):
    """Test forward pass with dummy data."""
    print("\n" + "="*70)
    print("TEST 2: Forward Pass")
    print("="*70)
    
    # Create dummy input (batch_size=2, channels=3, height=480, width=640)
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 480, 640)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        # Full reconstruction
        output = model(input_tensor)
        print(f"Output shape: {output.shape}")
        
        # Encode
        latent = model.encode(input_tensor)
        print(f"Latent shape: {latent.shape}")
        
        # Decode
        decoded = model.decode(latent)
        print(f"Decoded shape: {decoded.shape}")
        
        # Reconstruct with error map
        reconstructed, error_map = model.reconstruct(input_tensor)
        print(f"Reconstructed shape: {reconstructed.shape}")
        print(f"Error map shape: {error_map.shape}")
    
    # Verify shapes
    assert output.shape == (batch_size, 3, 480, 640), f"Output shape mismatch"
    assert latent.shape == (batch_size, 8, 60, 80), f"Latent shape mismatch"
    assert decoded.shape == (batch_size, 3, 480, 640), f"Decoded shape mismatch"
    assert reconstructed.shape == (batch_size, 3, 480, 640), f"Reconstructed shape mismatch"
    assert error_map.shape == (batch_size, 3, 480, 640), f"Error map shape mismatch"
    
    # Verify output is in [0, 1] (Sigmoid activation)
    assert output.min() >= 0 and output.max() <= 1, "Output not in [0, 1] range"
    
    print("✓ All shapes correct")
    print("✓ Output in valid range [0, 1]")
    print("✓ Forward pass test PASSED")


def test_loss_computation(model):
    """Test loss computation."""
    print("\n" + "="*70)
    print("TEST 3: Loss Computation")
    print("="*70)
    
    # Create dummy input normalized to [0, 1]
    batch_size = 4
    input_tensor = torch.rand(batch_size, 3, 480, 640)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        reconstructed = model(input_tensor)
    
    # Compute loss
    criterion = CAELoss()
    losses = criterion(reconstructed, input_tensor)
    
    print(f"Total loss: {losses['total'].item():.6f}")
    print(f"Reconstruction loss: {losses['reconstruction'].item():.6f}")
    
    # Verify loss is scalar
    assert losses['total'].dim() == 0, "Loss should be scalar"
    assert losses['reconstruction'].dim() == 0, "Reconstruction loss should be scalar"
    
    # Verify loss is non-negative
    assert losses['total'].item() >= 0, "Loss should be non-negative"
    
    print("✓ Loss computation test PASSED")


def test_inference(model):
    """Test inference functionality."""
    print("\n" + "="*70)
    print("TEST 4: Inference")
    print("="*70)
    
    device = get_device()
    inference = CAEInference(model=model, device=device)
    
    # Create test images
    batch_size = 3
    images = torch.rand(batch_size, 3, 480, 640)
    
    # Test batch prediction
    results = inference.predict(images)
    
    print(f"Reconstructed shape: {results['reconstructed'].shape}")
    print(f"Error map shape: {results['error_map'].shape}")
    print(f"Latent shape: {results['latent'].shape}")
    
    assert results['reconstructed'].shape == (batch_size, 3, 480, 640)
    assert results['error_map'].shape == (batch_size, 3, 480, 640)
    assert results['latent'].shape == (batch_size, 8, 60, 80)
    
    # Test single image reconstruction
    single_image = torch.rand(3, 480, 640)
    single_results = inference.reconstruct_image(single_image)
    
    print(f"Single reconstructed shape: {single_results['reconstructed'].shape}")
    assert single_results['reconstructed'].shape == (3, 480, 640)
    
    # Test reconstruction error with different reductions
    error_none = inference.compute_reconstruction_error(images, reduction='none')
    error_channel = inference.compute_reconstruction_error(images, reduction='channel')
    error_mean = inference.compute_reconstruction_error(images, reduction='mean')
    
    print(f"Error (none) shape: {error_none.shape}")
    print(f"Error (channel) shape: {error_channel.shape}")
    print(f"Error (mean) shape: {error_mean.shape}")
    
    assert error_none.shape == (batch_size, 3, 480, 640)
    assert error_channel.shape == (batch_size, 480, 640)
    assert error_mean.shape == (batch_size,)
    
    print("✓ Inference test PASSED")


def test_input_validation():
    """Test input shape validation."""
    print("\n" + "="*70)
    print("TEST 5: Input Validation")
    print("="*70)
    
    # Correct shape
    correct = torch.randn(2, 3, 480, 640)
    assert validate_input_shape(correct, (2, 3, 480, 640)), "Should validate correct shape"
    
    # Wrong height
    wrong_height = torch.randn(2, 3, 224, 640)
    assert not validate_input_shape(wrong_height, (2, 3, 480, 640)), "Should reject wrong height"
    
    # Wrong channels
    wrong_channels = torch.randn(2, 1, 480, 640)
    assert not validate_input_shape(wrong_channels, (2, 3, 480, 640)), "Should reject wrong channels"
    
    # Variable batch size (use -1)
    variable_batch = torch.randn(5, 3, 480, 640)
    assert validate_input_shape(variable_batch, (-1, 3, 480, 640)), "Should accept variable batch size"
    
    print("✓ Input validation test PASSED")


def test_gradient_flow(model):
    """Test gradient flow through the model."""
    print("\n" + "="*70)
    print("TEST 6: Gradient Flow")
    print("="*70)
    
    # Move model to CPU for gradient test
    device = torch.device('cpu')
    model = model.to(device)
    model.train()
    
    # Create dummy input and target
    input_tensor = torch.rand(2, 3, 480, 640, requires_grad=True)
    
    # Forward pass
    output = model(input_tensor)
    
    # Compute loss
    criterion = CAELoss()
    losses = criterion(output, input_tensor)
    loss = losses['total']
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    params_with_grad = 0
    params_without_grad = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                if param.grad.abs().sum() > 0:
                    params_with_grad += 1
                else:
                    params_without_grad += 1
                    print(f"  Warning: {name} has zero gradient")
            else:
                params_without_grad += 1
                print(f"  Warning: {name} has no gradient")
    
    print(f"Parameters with gradients: {params_with_grad}")
    print(f"Parameters without gradients: {params_without_grad}")
    
    # At least most parameters should have gradients
    assert params_with_grad > 0, "No parameters have gradients"
    assert params_with_grad > params_without_grad, "Most parameters don't have gradients"
    
    print("✓ Gradient flow test PASSED")


def test_encoder_decoder_symmetry(model):
    """Test encoder-decoder symmetry."""
    print("\n" + "="*70)
    print("TEST 7: Encoder-Decoder Symmetry")
    print("="*70)
    
    # Move model to CPU for consistency
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    # Test shape preservation
    input_tensor = torch.rand(1, 3, 480, 640)
    
    with torch.no_grad():
        # Encode and decode
        latent = model.encode(input_tensor)
        reconstructed = model.decode(latent)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    assert input_tensor.shape == reconstructed.shape, "Input and output shapes don't match"
    
    # Test that encoder output matches expected dimensions
    # After 3 pooling operations (each divides by 2):
    # Height: 480 / 8 = 60
    # Width: 640 / 8 = 80
    # Channels: 8 (last encoder filter)
    expected_latent_shape = (1, 8, 60, 80)
    assert latent.shape == expected_latent_shape, f"Expected {expected_latent_shape}, got {latent.shape}"
    
    print("✓ Encoder-decoder symmetry test PASSED")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("CONVOLUTIONAL AUTOENCODER TEST SUITE")
    print("="*70)
    
    try:
        # Test 1: Model creation
        model = test_model_creation()
        
        # Test 2: Forward pass
        test_forward_pass(model)
        
        # Test 3: Loss computation
        test_loss_computation(model)
        
        # Test 4: Inference
        test_inference(model)
        
        # Test 5: Input validation
        test_input_validation()
        
        # Test 6: Gradient flow
        test_gradient_flow(model)
        
        # Test 7: Encoder-decoder symmetry
        test_encoder_decoder_symmetry(model)
        
        # Summary
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nThe CAE implementation is working correctly!")
        print("Ready for training on thermal image dataset.")
        print()
        
        return 0
    
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
