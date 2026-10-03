"""
Reconstruction Evaluation Pipeline

Loads trained CAE models, performs inference on test images,
and computes per-image MSE and SSIM metrics.

This implements the reconstruction quality evaluation as specified in the paper.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from .evaluation_types import EvaluationResult
from .reconstruction_metrics import ReconstructionMetrics, SSIMConfig
from .utils import tensor_to_numpy, create_directories, get_checkpoint_info


class ReconstructionEvaluator:
    """
    Evaluator for reconstruction quality of trained autoencoder models.
    
    This class:
    1. Loads trained model checkpoints
    2. Runs inference on test images
    3. Computes MSE and SSIM for each image
    4. Saves per-image metrics and visualizations
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        checkpoint_path: Path,
        device: torch.device,
        data_range: float = 1.0,
        ssim_config: Optional[SSIMConfig] = None,
        save_error_maps: bool = True
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Autoencoder model instance (architecture only).
            checkpoint_path: Path to trained model checkpoint.
            device: Device to run inference on.
            data_range: Data range of images (1.0 for normalized [0,1] images).
            ssim_config: SSIM configuration. If None, uses default.
            save_error_maps: Whether to save error maps for all images.
        """
        self.model = model
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.save_error_maps = save_error_maps
        
        # Initialize metrics computer
        self.metrics = ReconstructionMetrics(
            data_range=data_range,
            ssim_config=ssim_config
        )
        
        # Load checkpoint
        self._load_checkpoint()
        
        # Set model to evaluation mode
        self.model.eval()
        
        logging.info("ReconstructionEvaluator initialized")
        logging.info(f"Checkpoint: {self.checkpoint_path}")
        logging.info(f"Device: {self.device}")
        logging.info(f"Data range: {data_range}")
    
    def _load_checkpoint(self) -> None:
        """Load model weights from checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        logging.info(f"Loading checkpoint: {self.checkpoint_path}")
        
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume checkpoint is just state dict
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        logging.info("Checkpoint loaded successfully")
    
    def evaluate_batch(
        self,
        batch: Dict[str, torch.Tensor],
        dataset_name: str
    ) -> List[EvaluationResult]:
        """
        Evaluate a batch of images.
        
        Args:
            batch: Dictionary with 'image' tensor and 'metadata' list.
            dataset_name: Name of the dataset being evaluated.
        
        Returns:
            List of EvaluationResult for each image in the batch.
        """
        images = batch['image'].to(self.device)
        metadata_list = batch['metadata']  # This is a list of metadata dicts
        
        # Run inference
        with torch.no_grad():
            reconstructed = self.model(images)
        
        # Convert to numpy for metric computation
        original_np = tensor_to_numpy(images)  # (B, H, W, C)
        reconstructed_np = tensor_to_numpy(reconstructed)  # (B, H, W, C)
        
        # Compute metrics for each image in batch
        results = []
        batch_size = images.size(0)
        
        for i in range(batch_size):
            orig = original_np[i]  # (H, W, C)
            recon = reconstructed_np[i]  # (H, W, C)
            
            # Compute metrics
            metrics = self.metrics.compute_all(
                orig,
                recon,
                compute_error=self.save_error_maps
            )
            
            # Get image path from metadata list
            if isinstance(metadata_list, list) and i < len(metadata_list):
                meta = metadata_list[i]
                image_path = meta.get('path', meta.get('image_path', f'unknown_{i}'))
            else:
                image_path = f'unknown_{i}'
            
            # Create result
            result = EvaluationResult(
                dataset=dataset_name,
                image_path=str(image_path),
                mse=metrics['mse'],
                ssim=metrics['ssim'],
                error_map=metrics.get('error_map')
            )
            
            results.append(result)
        
        return results
    
    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        dataset_name: str,
        output_dir: Optional[Path] = None
    ) -> List[EvaluationResult]:
        """
        Evaluate entire dataset.
        
        Args:
            dataloader: DataLoader for the dataset.
            dataset_name: Name of the dataset.
            output_dir: Optional directory to save intermediate results.
        
        Returns:
            List of EvaluationResult for all images.
        """
        logging.info(f"Evaluating dataset: {dataset_name}")
        logging.info(f"Number of batches: {len(dataloader)}")
        
        all_results = []
        
        # Iterate through batches with progress bar
        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            batch_results = self.evaluate_batch(batch, dataset_name)
            all_results.extend(batch_results)
        
        logging.info(f"Evaluated {len(all_results)} images from {dataset_name}")
        
        # Compute summary statistics
        mse_values = [r.mse for r in all_results]
        ssim_values = [r.ssim for r in all_results]
        
        logging.info(f"{dataset_name} Summary:")
        logging.info(f"  MSE:  mean={np.mean(mse_values):.6f}, std={np.std(mse_values):.6f}")
        logging.info(f"  SSIM: mean={np.mean(ssim_values):.4f}, std={np.std(ssim_values):.4f}")
        
        return all_results
    
    def save_results(
        self,
        results: List[EvaluationResult],
        output_path: Path
    ) -> None:
        """
        Save per-image results to CSV.
        
        Args:
            results: List of evaluation results.
            output_path: Path to save CSV file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        data = [r.to_dict(include_error_map=False) for r in results]
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logging.info(f"Per-image metrics saved to: {output_path}")
        
        # Log summary
        logging.info(f"Saved {len(results)} evaluation results")
        logging.info(f"Columns: {list(df.columns)}")
    
    def get_checkpoint_metadata(self) -> dict:
        """Get metadata about the checkpoint used."""
        return get_checkpoint_info(self.checkpoint_path)


def evaluate_model(
    config: dict,
    model: torch.nn.Module,
    dataset_name: str,
    dataloader: DataLoader,
    output_dir: Path
) -> Tuple[List[EvaluationResult], dict]:
    """
    Evaluate a trained model on a dataset.
    
    This is the main entry point for running evaluation.
    
    Args:
        config: Evaluation configuration dictionary.
        model: Model architecture instance.
        dataset_name: Name of dataset being evaluated.
        dataloader: DataLoader for the dataset.
        output_dir: Directory to save results.
    
    Returns:
        Tuple of (results_list, metadata_dict).
    """
    # Get device
    device = torch.device(config['hardware']['device'])
    
    # Get checkpoint path
    checkpoint_path = Path(config['checkpoints'][dataset_name])
    
    # Create SSIM config
    ssim_params = config['metrics']['ssim']
    ssim_config = SSIMConfig(
        data_range=config['metrics']['data_range'],
        win_size=ssim_params['win_size'],
        channel_axis=ssim_params['channel_axis'],
        gaussian_weights=ssim_params['gaussian_weights'],
        sigma=ssim_params['sigma']
    )
    
    # Create evaluator
    evaluator = ReconstructionEvaluator(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
        data_range=config['metrics']['data_range'],
        ssim_config=ssim_config,
        save_error_maps=config['output']['save_error_maps']
    )
    
    # Evaluate dataset
    results = evaluator.evaluate_dataset(
        dataloader=dataloader,
        dataset_name=dataset_name,
        output_dir=output_dir
    )
    
    # Prepare metadata
    metadata = {
        'dataset': dataset_name,
        'num_images': len(results),
        'checkpoint': evaluator.get_checkpoint_metadata(),
        'metrics_config': evaluator.metrics.get_config()
    }
    
    return results, metadata


def load_model_for_evaluation(config: dict) -> torch.nn.Module:
    """
    Load model architecture for evaluation.
    
    Args:
        config: Configuration dictionary with model parameters.
    
    Returns:
        Model instance (not loaded with weights yet).
    """
    # Import model
    from models.cae import ConvolutionalAutoencoder
    
    model_config = config['model']
    
    model = ConvolutionalAutoencoder(
        input_channels=model_config['input_channels'],
        input_height=model_config['input_height'],
        input_width=model_config['input_width'],
        encoder_filters=tuple(model_config['encoder_filters']),
        kernel_size=model_config['kernel_size'],
        pool_size=model_config['pool_size'],
        padding=model_config['padding']
    )
    
    logging.info("Model architecture loaded for evaluation")
    
    return model
