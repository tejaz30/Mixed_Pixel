"""
Test script to verify Experiment 1 models can be instantiated and run.

Usage:
    python scripts/test_experiment1_models.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F

from models.ldm.first_stage.kl import KLAutoencoder
from models.ldm.first_stage.vq import VQAutoencoder


def test_kl_model():
    """Test KL autoencoder."""
    print("\n" + "="*70)
    print("Testing KL Autoencoder")
    print("="*70)
    
    # Create model
    model = KLAutoencoder(
        in_channels=3,
        latent_channels=4,
        base_channels=128,
        channel_multipliers=(1, 2, 4, 4),
        num_res_blocks=2
    )
    
    print(model)
    print()
    
    params = model.count_parameters()
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Encoder parameters: {params['encoder']:,}")
    print(f"Decoder parameters: {params['decoder']:,}")
    print()
    
    # Test forward pass
    print("Testing forward pass...")
    model.eval()
    
    with torch.no_grad():
        # Create dummy input (B=2, C=3, H=480, W=640)
        x = torch.randn(2, 3, 480, 640)
        print(f"Input shape: {x.shape}")
        
        # Encode
        posterior = model.encode(x)
        print(f"Posterior mean shape: {posterior.mean.shape}")
        print(f"Posterior logvar shape: {posterior.logvar.shape}")
        
        # Sample
        z = posterior.sample()
        print(f"Sampled latent shape: {z.shape}")
        
        # Decode
        reconstruction = model.decode(z)
        print(f"Reconstruction shape: {reconstruction.shape}")
        
        # Full forward
        reconstruction2, posterior2 = model(x, sample=False)
        print(f"Full forward reconstruction shape: {reconstruction2.shape}")
        
        # Compute loss
        mse = F.mse_loss(reconstruction2, x)
        kl = posterior2.kl().mean()
        print(f"\nMSE: {mse.item():.6f}")
        print(f"KL: {kl.item():.6f}")
    
    print("\n✓ KL Autoencoder test passed!")
    return True


def test_vq_model():
    """Test VQ autoencoder."""
    print("\n" + "="*70)
    print("Testing VQ Autoencoder")
    print("="*70)
    
    # Create model
    model = VQAutoencoder(
        in_channels=3,
        latent_channels=4,
        base_channels=128,
        channel_multipliers=(1, 2, 4, 4),
        num_res_blocks=2,
        num_embeddings=1024,
        commitment_cost=0.25
    )
    
    print(model)
    print()
    
    params = model.count_parameters()
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Encoder parameters: {params['encoder']:,}")
    print(f"Decoder parameters: {params['decoder']:,}")
    print(f"Quantizer parameters: {params['quantizer']:,}")
    print()
    
    # Test forward pass
    print("Testing forward pass...")
    model.eval()
    
    with torch.no_grad():
        # Create dummy input
        x = torch.randn(2, 3, 480, 640)
        print(f"Input shape: {x.shape}")
        
        # Encode and quantize
        z_quantized, z_continuous, vq_losses = model.encode(x)
        print(f"Continuous latent shape: {z_continuous.shape}")
        print(f"Quantized latent shape: {z_quantized.shape}")
        
        # Decode
        reconstruction = model.decode(z_quantized)
        print(f"Reconstruction shape: {reconstruction.shape}")
        
        # Full forward
        reconstruction2, vq_outputs = model(x)
        print(f"Full forward reconstruction shape: {reconstruction2.shape}")
        
        # Loss components
        print(f"\nVQ Loss: {vq_losses['vq_loss'].item():.6f}")
        print(f"Codebook Loss: {vq_losses['codebook_loss'].item():.6f}")
        print(f"Commitment Loss: {vq_losses['commitment_loss'].item():.6f}")
        print(f"Perplexity: {vq_losses['perplexity'].item():.2f}")
        print(f"Codebook Utilization: {vq_losses['codebook_utilization'].item():.4f}")
        
        mse = F.mse_loss(reconstruction2, x)
        print(f"MSE: {mse.item():.6f}")
    
    # Codebook usage
    usage = model.get_codebook_usage()
    print(f"\nCodebook Usage:")
    print(f"  Total codes: {usage['total_codes']}")
    print(f"  Active codes: {usage['active_codes']}")
    print(f"  Utilization: {usage['utilization']:.4f}")
    
    print("\n✓ VQ Autoencoder test passed!")
    return True


def main():
    print("="*70)
    print("EXPERIMENT 1 MODEL TESTS")
    print("="*70)
    
    try:
        # Test KL
        kl_success = test_kl_model()
        
        # Test VQ
        vq_success = test_vq_model()
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"KL Autoencoder: {'✓ PASS' if kl_success else '✗ FAIL'}")
        print(f"VQ Autoencoder: {'✓ PASS' if vq_success else '✗ FAIL'}")
        print("="*70)
        
        if kl_success and vq_success:
            print("\n✓ All tests passed! Models are ready for training.")
            return 0
        else:
            print("\n✗ Some tests failed. Check errors above.")
            return 1
    
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
