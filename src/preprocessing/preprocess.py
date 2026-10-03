"""
Main preprocessing script for thermal image datasets.

This script processes deduplicated datasets and generates preprocessed images
ready for training. The primary operation is resizing all images to 640x480
as specified in the research paper.

Usage:
    python preprocessing/preprocess.py --config configs/preprocessing.yaml
    python preprocessing/preprocess.py --config configs/preprocessing.yaml --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.utils import load_config, setup_logging, get_relative_path
from preprocessing.image_loader import ImageLoader
from preprocessing.pipeline import PreprocessingPipeline, ConfigurableImageProcessor
from preprocessing.metadata import MetadataManager


logger = logging.getLogger("preprocessing")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess thermal image datasets for mixed-pixel detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run preprocessing
  python preprocessing/preprocess.py --config configs/preprocessing.yaml
  
  # Dry run (analyze only, don't process images)
  python preprocessing/preprocess.py --config configs/preprocessing.yaml --dry-run
  
  # Custom log level
  python preprocessing/preprocess.py --config configs/preprocessing.yaml --log-level DEBUG
        """
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/preprocessing.yaml'),
        help='Path to preprocessing configuration YAML file'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Analyze inputs without processing images'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default=None,
        help='Override log level from config'
    )
    
    return parser.parse_args()


def process_single_image(args):
    """
    Process a single image (for parallel processing).
    
    Args:
        args: Tuple of (input_path, output_path, config, overwrite)
        
    Returns:
        Tuple of (success, metadata or None, error_message or None)
    """
    input_path, output_path, preprocess_config, loader_config, verify, overwrite = args
    
    try:
        # Reconstruct components (needed for multiprocessing)
        loader = ImageLoader(**loader_config)
        pipeline = PreprocessingPipeline(preprocess_config)
        processor = ConfigurableImageProcessor(loader, pipeline, verify=verify)
        
        metadata = processor.process_image(input_path, output_path, overwrite)
        
        if metadata is None:
            return False, None, "Processing failed"
        
        return True, metadata, None
        
    except Exception as e:
        return False, None, str(e)


def main():
    """Main preprocessing workflow."""
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Setup logging
    log_level_str = args.log_level or config.get('logging', {}).get('level', 'INFO')
    log_level = getattr(logging, log_level_str)
    
    reports_output = Path(config['paths']['reports_output'])
    log_file = reports_output / config['logging'].get('log_file', 'preprocessing.log')
    
    logger = setup_logging(
        log_file=log_file,
        level=log_level,
        logger_name="preprocessing"
    )
    
    logger.info("=" * 70)
    logger.info("THERMAL IMAGE PREPROCESSING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Configuration: {args.config}")
    
    if args.dry_run:
        logger.warning("DRY RUN MODE: No images will be processed")
    
    # Get paths from config
    input_root = Path(config['paths']['input_root'])
    output_root = Path(config['paths']['output_root'])
    metadata_output = Path(config['paths']['metadata_output'])
    
    # Initialize components
    loader_config = config.get('image_loading', {})
    loader = ImageLoader(
        valid_extensions=loader_config.get('valid_extensions', ['.jpg', '.jpeg', '.png']),
        load_mode=loader_config.get('load_mode', 'unchanged'),
        skip_corrupted=loader_config.get('skip_corrupted', True)
    )
    
    # Create preprocessing pipeline
    preprocess_config = config.get('preprocessing', {})
    pipeline = PreprocessingPipeline(preprocess_config)
    
    # Log enabled operations
    enabled_ops = pipeline.get_enabled_operations()
    logger.info(f"Enabled preprocessing operations: {', '.join(enabled_ops)}")
    
    processing_config = config.get('processing', {})
    processor = ConfigurableImageProcessor(
        image_loader=loader,
        pipeline=pipeline,
        verify=processing_config.get('verify', True)
    )
    
    # Initialize metadata manager
    metadata_manager = MetadataManager(metadata_output)
    
    # Process each dataset
    datasets = config.get('datasets', [])
    enabled_datasets = [ds for ds in datasets if ds.get('enabled', True)]
    
    logger.info(f"Processing {len(enabled_datasets)} datasets")
    
    # Log target resolution if resize is enabled
    resize_config = preprocess_config.get('resize', {})
    if resize_config.get('enabled', True):
        logger.info(
            f"Target resolution: {resize_config.get('width', 640)}x"
            f"{resize_config.get('height', 480)}"
        )
    
    logger.info("")
    
    total_processed = 0
    total_failed = 0
    total_skipped = 0
    
    for i, dataset_config in enumerate(enabled_datasets, 1):
        dataset_name = dataset_config['name']
        
        logger.info(f"[{i}/{len(enabled_datasets)}] Processing {dataset_name}")
        logger.info("-" * 70)
        
        # Find dataset directory
        dataset_input = input_root / dataset_name
        dataset_output = output_root / dataset_name
        
        if not dataset_input.exists():
            logger.error(f"Dataset directory not found: {dataset_input}")
            continue
        
        # Find all images
        logger.info(f"Scanning for images in {dataset_input}...")
        image_files = loader.find_images(dataset_input, recursive=True)
        logger.info(f"Found {len(image_files)} images")
        
        if len(image_files) == 0:
            logger.warning(f"No images found in {dataset_input}")
            continue
        
        if args.dry_run:
            logger.info(f"DRY RUN: Would process {len(image_files)} images")
            total_processed += len(image_files)
            continue
        
        # Prepare processing tasks
        tasks = []
        for input_path in image_files:
            # Preserve directory hierarchy
            if preprocess_config.get('preserve_hierarchy', True):
                rel_path = get_relative_path(input_path, dataset_input)
                output_path = dataset_output / rel_path
            else:
                output_path = dataset_output / input_path.name
            
            tasks.append((
                input_path,
                output_path,
                preprocess_config,
                {
                    'valid_extensions': loader_config.get('valid_extensions'),
                    'load_mode': loader_config.get('load_mode', 'unchanged'),
                    'skip_corrupted': loader_config.get('skip_corrupted', True)
                },
                processing_config.get('verify', True),
                processing_config.get('overwrite', False)
            ))
        
        # Process images (with multiprocessing)
        max_workers = processing_config.get('max_workers', 4)
        show_progress = processing_config.get('show_progress', True)
        
        logger.info(f"Processing images with {max_workers} workers...")
        
        processed = 0
        failed = 0
        skipped = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_image, task) for task in tasks]
            
            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Processing {dataset_name}",
                disable=not show_progress
            )
            
            for future in pbar:
                try:
                    success, metadata, error = future.result()
                    
                    if success and metadata:
                        # Extract class name from path
                        input_path = Path(metadata['input_path'])
                        class_name = input_path.parent.name
                        
                        metadata_manager.add_record(metadata, dataset_name, class_name)
                        processed += 1
                    elif metadata is None:
                        skipped += 1
                    else:
                        failed += 1
                        if error:
                            logger.error(f"Processing failed: {error}")
                    
                except Exception as e:
                    logger.error(f"Task failed with exception: {e}")
                    failed += 1
            
            pbar.close()
        
        logger.info(f"Processed: {processed}, Failed: {failed}, Skipped: {skipped}")
        logger.info("")
        
        total_processed += processed
        total_failed += failed
        total_skipped += skipped
    
    # Generate reports
    if not args.dry_run and total_processed > 0:
        logger.info("=" * 70)
        logger.info("GENERATING METADATA AND REPORTS")
        logger.info("=" * 70)
        
        try:
            metadata_config = config.get('metadata', {})
            metadata_manager.save_metadata(
                generate_csv=metadata_config.get('generate_csv', True),
                generate_json=metadata_config.get('generate_json', True)
            )
            metadata_manager.save_summary()
        except Exception as e:
            logger.error(f"Error generating reports: {e}", exc_info=True)
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total images processed: {total_processed:,}")
    logger.info(f"Total failed: {total_failed:,}")
    logger.info(f"Total skipped: {total_skipped:,}")
    
    if not args.dry_run:
        logger.info(f"\nProcessed datasets saved to: {output_root}")
        logger.info(f"Metadata saved to: {metadata_output}")
    else:
        logger.info("\nDRY RUN: No images were processed")
    
    logger.info(f"Reports saved to: {reports_output}")
    logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0 if total_failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
