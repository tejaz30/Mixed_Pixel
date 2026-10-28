"""
Tests for diagnostic sample count validation.

Verifies that diagnostic modules correctly validate and report sample count mismatches,
preventing contamination from multiple evaluation runs.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from evaluation.kl_diagnostics import KLDiagnostics
from evaluation.vq_diagnostics import VQDiagnostics
from evaluation.latent_distribution import LatentDistributionAnalyzer


class MockKLModel:
    """Mock KL model for testing."""
    def __init__(self):
        self.latent_channels = 4
    
    def eval(self):
        """No-op eval mode for testing."""
        return self
    
    def __call__(self, x, sample=False):
        B = x.shape[0]
        mean = torch.randn(B, 4, 60, 80)
        logvar = torch.randn(B, 4, 60, 80)
        return None, mean, logvar
    
    def encode(self, x):
        class Posterior:
            def __init__(self, mean, logvar):
                self.mean = mean
                self.logvar = logvar
            def mode(self):
                return self.mean
        B = x.shape[0]
        mean = torch.randn(B, 4, 60, 80)
        logvar = torch.randn(B, 4, 60, 80)
        return Posterior(mean, logvar)


class MockVQModel:
    """Mock VQ model for testing."""
    def __init__(self):
        self.latent_channels = 4
        self.num_embeddings = 1024
        self.quantizer = type('obj', (object,), {
            'embedding': type('obj', (object,), {
                'weight': torch.randn(1024, 4)
            })()
        })()
    
    def eval(self):
        """No-op eval mode for testing."""
        return self
    
    def encode(self, x):
        B = x.shape[0]
        z_quantized = torch.randn(B, 4, 60, 80)
        z_continuous = torch.randn(B, 4, 60, 80)
        vq_losses = {'vq_loss': torch.tensor(0.1)}
        return z_quantized, z_continuous, vq_losses


def test_kl_diagnostics_correct_sample_count():
    """Test KL diagnostics with correct sample count."""
    model = MockKLModel()
    device = torch.device('cpu')
    diagnostics = KLDiagnostics(model, device, latent_channels=4)
    
    # Process 52 samples
    expected_samples = 52
    batch_size = 8
    for i in range(7):  # 7 batches
        if i < 6:
            batch = torch.randn(batch_size, 3, 480, 640)
        else:
            batch = torch.randn(4, 3, 480, 640)  # Last batch smaller
        diagnostics.process_batch(batch)
    
    # Save with validation
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir)
        diagnostics.save_diagnostics(save_path, expected_samples=expected_samples)
        
        # Check saved arrays
        means = np.load(save_path / 'posterior_means.npy')
        assert means.shape[0] == expected_samples, \
            f"Expected {expected_samples}, got {means.shape[0]}"


def test_kl_diagnostics_sample_count_mismatch():
    """Test KL diagnostics detects and handles sample count mismatch."""
    model = MockKLModel()
    device = torch.device('cpu')
    diagnostics = KLDiagnostics(model, device, latent_channels=4)
    
    # Simulate contamination: process 60 samples but expect 52
    expected_samples = 52
    batch_size = 8
    for i in range(8):  # 8 batches = 60 samples (simulating extra data)
        if i < 7:
            batch = torch.randn(batch_size, 3, 480, 640)
        else:
            batch = torch.randn(4, 3, 480, 640)
        diagnostics.process_batch(batch)
    
    # Save with validation - should truncate
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir)
        diagnostics.save_diagnostics(save_path, expected_samples=expected_samples)
        
        # Check saved arrays were truncated
        means = np.load(save_path / 'posterior_means.npy')
        assert means.shape[0] == expected_samples, \
            f"Should truncate to {expected_samples}, got {means.shape[0]}"


def test_vq_diagnostics_sample_count_validation():
    """Test VQ diagnostics validates sample counts."""
    model = MockVQModel()
    device = torch.device('cpu')
    diagnostics = VQDiagnostics(model, device, codebook_size=1024, latent_channels=4)
    
    # Process 52 samples across 7 batches
    expected_samples = 52
    n_batches = 7
    batch_size = 8
    for i in range(n_batches):
        if i < 6:
            batch = torch.randn(batch_size, 3, 480, 640)
        else:
            batch = torch.randn(4, 3, 480, 640)
        diagnostics.process_batch(batch)
    
    # Save with validation
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir)
        diagnostics.save_diagnostics(save_path, expected_samples=expected_samples)
        
        # Check trajectory lengths - VQ tracks per-batch, not per-sample
        # So we expect n_batches entries, and the validation should warn about the mismatch
        perplexity = np.load(save_path / 'perplexity_trajectory.npy')
        quant_error = np.load(save_path / 'quantization_error_trajectory.npy')
        
        assert len(perplexity) == n_batches, \
            f"VQ tracks per-batch: expected {n_batches} entries, got {len(perplexity)}"
        assert len(quant_error) == n_batches, \
            f"VQ tracks per-batch: expected {n_batches} entries, got {len(quant_error)}"


def test_latent_analyzer_trajectory_no_double_counting():
    """Test LatentDistributionAnalyzer doesn't double-count trajectories."""
    device = torch.device('cpu')
    analyzer = LatentDistributionAnalyzer(device)
    
    # Process 7 batches (52 total samples)
    n_batches = 7
    expected_trajectory_length = n_batches  # One trajectory entry per batch, NOT per sample
    
    for i in range(n_batches):
        if i < 6:
            latents = torch.randn(8, 4, 60, 80)
        else:
            latents = torch.randn(4, 4, 60, 80)
        # This calls characterize_distribution with store_trajectory=True (default)
        # Each call adds ONE entry to the trajectory (per-batch tracking)
        analyzer.assess_ldm_readiness(latents)
    
    # Compute global stats (should NOT add to trajectory due to store_trajectory=False)
    global_stats = analyzer.compute_global_statistics()
    
    # Check trajectory length - should be n_batches, NOT n_batches + 1
    assert len(analyzer.latent_means) == expected_trajectory_length, \
        f"Expected {expected_trajectory_length} trajectory points (one per batch), got {len(analyzer.latent_means)}"
    
    # Save and verify
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir)
        analyzer.save_distribution_data(save_path)
        
        means_traj = np.load(save_path / 'latent_means_trajectory.npy')
        assert len(means_traj) == expected_trajectory_length, \
            f"Trajectory file should have {expected_trajectory_length} entries, got {len(means_traj)}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
