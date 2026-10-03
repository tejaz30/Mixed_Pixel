"""
Main script for running reconstruction quality evaluation.

Usage:
    python -m evaluation.run_evaluation
    python -m evaluation.run_evaluation --config configs/evaluation.yaml
    python -m evaluation.run_evaluation --dataset Chilli_Leaves
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List

import yaml
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.evaluate_reconstruction import (
    load_model_for_evaluation,
    evaluate_model,
    EvaluationResult,
    ReconstructionEvaluator
)
from evaluation.aggregation import (
    aggregate_all_datasets,
    save_aggregated_metrics,
    compare_with_paper_results
)
from evaluation.visualization import ReconstructionVisualizer
from evaluation.reconstruction_metrics import SSIMConfig
from evaluation.utils import (
    setup_logging,
    save_evaluation_config,
    create_directories,
    tensor_to_numpy
)
from data import ThermalImageDataset, get_train_transforms


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloader(
    dataset_root: Path,
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool
) -> DataLoader:
    """
    Create DataLoader for a dataset.
    
    Args:
        dataset_root: Root directory containing datasets.
        dataset_name: Name of specific dataset.
        batch_size: Batch size for evaluation.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory for GPU.
    
    Returns:
        DataLoader for the dataset.
    """
    dataset_path = dataset_root / dataset_name
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # For evaluation, we need images in [0, 1] range without normalization
    # This is required for MSE and SSIM computation
    from data.transforms import Compose, ToTensor
    transforms = Compose([ToTensor()])  # Only convert to tensor, no normalization
    
    # Create dataset
    dataset = ThermalImageDataset(
        root_dir=dataset_path,
        transform=transforms,
        return_path=True  # Need paths for saving results
    )
    
    logging.info(f"Loaded dataset {dataset_name}: {len(dataset)} images")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Deterministic order for evaluation
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return dataloader


def collect_reconstruction_images(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> tuple:
    """
    Collect original and reconstructed images for visualization.
    
    Args:
        model: Trained model.
        dataloader: DataLoader for dataset.
        device: Device to run on.
    
    Returns:
        Tuple of (original_images, reconstructed_images) as lists of numpy arrays.
    """
    model.eval()
    
    original_images = []
    reconstructed_images = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            reconstructed = model(images)
            
            # Convert to numpy
            orig_np = tensor_to_numpy(images)
            recon_np = tensor_to_numpy(reconstructed)
            
            # Append individual images
            for i in range(orig_np.shape[0]):
                original_images.append(orig_np[i])
                reconstructed_images.append(recon_np[i])
    
    return original_images, reconstructed_images


def run_evaluation(config: dict, dataset_filter: str = None) -> None:
    """
    Run complete evaluation pipeline.
    
    Args:
        config: Evaluation configuration.
        dataset_filter: Optional dataset name to evaluate only that dataset.
    """
    logging.info("="*70)
    logging.info("RECONSTRUCTION QUALITY EVALUATION")
    logging.info("="*70)
    logging.info(f"Experiment: {config['experiment']['name']}")
    logging.info(f"Architecture: {config['experiment']['architecture']}")
    logging.info("")
    
    # Setup output directory
    output_dir = Path(config['output']['output_dir'])
    create_directories([output_dir])
    
    # Save evaluation config
    if config['output']['save_config']:
        config_path = output_dir / config['output']['config_filename']
        save_evaluation_config(config, config_path)
    
    # Get device
    device = torch.device(config['hardware']['device'])
    logging.info(f"Device: {device}")
    
    # Load model architecture
    logging.info("Loading model architecture...")
    model = load_model_for_evaluation(config)
    
    # Determine which datasets to evaluate
    datasets_to_eval = config['data']['datasets']
    if dataset_filter:
        if dataset_filter in datasets_to_eval:
            datasets_to_eval = [dataset_filter]
            logging.info(f"Evaluating single dataset: {dataset_filter}")
        else:
            raise ValueError(f"Dataset {dataset_filter} not in config")
    
    logging.info(f"Datasets to evaluate: {datasets_to_eval}")
    logging.info("")
    
    # Storage for all results
    all_results: Dict[str, List[EvaluationResult]] = {}
    all_metadata: Dict[str, dict] = {}
    
    # Evaluate each dataset
    dataset_root = Path(config['data']['dataset_root'])
    
    for dataset_name in datasets_to_eval:
        logging.info("="*70)
        logging.info(f"EVALUATING: {dataset_name}")
        logging.info("="*70)
        
        # Check if checkpoint exists
        checkpoint_path = Path(config['checkpoints'][dataset_name])
        if not checkpoint_path.exists():
            logging.error(f"Checkpoint not found: {checkpoint_path}")
            logging.error(f"Skipping {dataset_name}")
            continue
        
        # Create dataloader
        dataloader = create_dataloader(
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory']
        )
        
        # Run evaluation
        results, metadata = evaluate_model(
            config=config,
            model=model,
            dataset_name=dataset_name,
            dataloader=dataloader,
            output_dir=output_dir
        )
        
        all_results[dataset_name] = results
        all_metadata[dataset_name] = metadata
        
        logging.info("")
    
    # Check if we have any results
    if not all_results:
        logging.error("No datasets were successfully evaluated!")
        return
    
    # Save per-image results
    if config['output']['save_per_image']:
        logging.info("="*70)
        logging.info("SAVING PER-IMAGE METRICS")
        logging.info("="*70)
        
        # Combine all results
        combined_results = []
        for results in all_results.values():
            combined_results.extend(results)
        
        # Save results to CSV
        output_path = output_dir / config['output']['per_image_filename']
        
        import pandas as pd
        data = [r.to_dict(include_error_map=False) for r in combined_results]
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        logging.info(f"Per-image metrics saved: {output_path}")
        logging.info(f"Total images evaluated: {len(combined_results)}")
        logging.info("")
    
    # Aggregate metrics
    if config['output']['save_aggregated']:
        logging.info("="*70)
        logging.info("AGGREGATING DATASET METRICS")
        logging.info("="*70)
        
        aggregated_df = aggregate_all_datasets(all_results)
        
        output_path = output_dir / config['output']['aggregated_filename']
        save_aggregated_metrics(aggregated_df, output_path)
        
        logging.info("")
        
        # Compare with paper if No_Mixed was evaluated
        if 'No_Mixed' in all_results:
            compare_with_paper_results(aggregated_df, 'No_Mixed')
    
    # Create visualizations
    if config['output']['visualization']['enabled']:
        logging.info("="*70)
        logging.info("CREATING VISUALIZATIONS")
        logging.info("="*70)
        
        viz_config = config['output']['visualization']
        visualizer = ReconstructionVisualizer(
            output_dir=output_dir,
            dpi=viz_config['dpi'],
            figsize=tuple(viz_config['figsize'])
        )
        
        # Sample reconstructions
        if viz_config['num_samples'] > 0:
            logging.info("Creating sample reconstruction visualizations...")
            
            for dataset_name in all_results.keys():
                # Reload dataloader to get images
                dataloader = create_dataloader(
                    dataset_root=dataset_root,
                    dataset_name=dataset_name,
                    batch_size=config['data']['batch_size'],
                    num_workers=config['data']['num_workers'],
                    pin_memory=config['data']['pin_memory']
                )
                
                # Reload model for this dataset
                checkpoint_path = Path(config['checkpoints'][dataset_name])
                ssim_params = config['metrics']['ssim']
                ssim_config = SSIMConfig(
                    data_range=config['metrics']['data_range'],
                    win_size=ssim_params['win_size'],
                    channel_axis=ssim_params['channel_axis'],
                    gaussian_weights=ssim_params['gaussian_weights'],
                    sigma=ssim_params['sigma']
                )
                
                evaluator = ReconstructionEvaluator(
                    model=load_model_for_evaluation(config),
                    checkpoint_path=checkpoint_path,
                    device=device,
                    data_range=config['metrics']['data_range'],
                    ssim_config=ssim_config,
                    save_error_maps=True
                )
                
                # Collect images
                original_images, reconstructed_images = collect_reconstruction_images(
                    evaluator.model,
                    dataloader,
                    device
                )
                
                # Create visualizations
                visualizer.visualize_sample_reconstructions(
                    results=all_results[dataset_name],
                    original_images=original_images,
                    reconstructed_images=reconstructed_images,
                    dataset_name=dataset_name,
                    num_samples=viz_config['num_samples']
                )
        
        # Distribution plots
        if viz_config['create_distributions']:
            logging.info("Creating distribution plots...")
            visualizer.plot_all_distributions(all_results)
        
        # Boxplots
        if viz_config['create_boxplots']:
            logging.info("Creating boxplots...")
            visualizer.plot_boxplots(all_results)
        
        # Comparison charts
        if viz_config['create_comparisons'] and 'aggregated_df' in locals():
            logging.info("Creating comparison charts...")
            visualizer.plot_metric_comparison(aggregated_df)
        
        # Summary figure
        if viz_config['create_summary'] and 'aggregated_df' in locals():
            logging.info("Creating summary figure...")
            visualizer.create_summary_figure(aggregated_df)
        
        logging.info("")
    
    # Final summary
    logging.info("="*70)
    logging.info("EVALUATION COMPLETE")
    logging.info("="*70)
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Datasets evaluated: {len(all_results)}")
    logging.info(f"Total images: {sum(len(r) for r in all_results.values())}")
    logging.info("")
    
    if 'aggregated_df' in locals():
        logging.info("Summary Results:")
        for _, row in aggregated_df.iterrows():
            logging.info(f"  {row['dataset']:<15} | "
                        f"MSE: {row['mse_mean']:.6f} ± {row['mse_std']:.6f} | "
                        f"SSIM: {row['ssim_mean']:.4f} ± {row['ssim_std']:.4f}")
    
    logging.info("="*70)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate reconstruction quality of trained CAE models"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/evaluation.yaml',
        help='Path to evaluation configuration file'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Evaluate only specific dataset (e.g., Chilli_Leaves, Okra, Paddy_Leaves, No_Mixed)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Setup logging
    output_dir = Path(config['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = None
    if config['logging']['log_to_file']:
        log_file = output_dir / config['logging']['log_filename']
    
    setup_logging(
        log_level=config['logging']['level'],
        log_file=log_file
    )
    
    # Run evaluation
    try:
        run_evaluation(config, dataset_filter=args.dataset)
        return 0
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
