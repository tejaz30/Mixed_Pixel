"""
Main script for dataset deduplication.

Usage:
    python -m preprocessing.deduplicate --config configs/deduplication.yaml
    python -m preprocessing.deduplicate --config configs/deduplication.yaml --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

from .utils import setup_logging, load_config
from .deduplicator import DatasetDeduplicator
from .deduplication_report import DeduplicationReportGenerator


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deduplicate thermal image datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run deduplication
  python -m preprocessing.deduplicate --config configs/deduplication.yaml
  
  # Dry run (analyze only, don't copy files)
  python -m preprocessing.deduplicate --config configs/deduplication.yaml --dry-run
  
  # Custom log level
  python -m preprocessing.deduplicate --config configs/deduplication.yaml --log-level DEBUG
        """
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to deduplication configuration YAML file'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Analyze duplicates without copying files'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default=None,
        help='Override log level from config'
    )
    
    return parser.parse_args()


def main():
    """Main deduplication workflow."""
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
    log_file = reports_output / config['logging'].get('log_file', 'deduplication.log')
    
    logger = setup_logging(
        log_file=log_file,
        level=log_level,
        logger_name="preprocessing"
    )
    
    logger.info("=" * 70)
    logger.info("THERMAL IMAGE DATASET DEDUPLICATION")
    logger.info("=" * 70)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Configuration: {args.config}")
    
    if args.dry_run:
        logger.warning("DRY RUN MODE: No files will be copied")
    
    # Get paths from config
    raw_root = Path(config['paths'].get('raw_datasets_root', 'datasets/raw'))
    dedup_output = Path(config['paths']['deduplicated_output'])
    metadata_output = Path(config['paths']['metadata_output'])
    reports_output = Path(config['paths']['reports_output'])
    
    # Processing options
    processing = config.get('processing', {})
    duplicate_detection = config.get('duplicate_detection', {})
    reporting = config.get('reporting', {})
    safety = config.get('safety', {})
    
    # Override dry-run from command line
    dry_run = args.dry_run or safety.get('dry_run', False)
    
    # Initialize deduplicator
    deduplicator = DatasetDeduplicator(
        hash_algorithm=processing.get('hash_algorithm', 'sha256'),
        chunk_size=processing.get('chunk_size', 4096),
        max_workers=processing.get('max_workers'),
        retention_strategy=processing.get('retention_strategy', 'first'),
        verify_copies=safety.get('verify_copies', True),
        dry_run=dry_run
    )
    
    # Process each dataset
    results = []
    datasets = config.get('datasets', [])
    enabled_datasets = [ds for ds in datasets if ds.get('enabled', True)]
    
    logger.info(f"Processing {len(enabled_datasets)} datasets")
    logger.info("")
    
    for i, dataset_config in enumerate(enabled_datasets, 1):
        dataset_name = dataset_config['name']
        dataset_path = dataset_config['path']
        
        logger.info(f"[{i}/{len(enabled_datasets)}] Processing {dataset_name}")
        logger.info("-" * 70)
        
        # Resolve dataset path (could be relative to raw_root or absolute)
        if Path(dataset_path).is_absolute():
            raw_path = Path(dataset_path)
        else:
            raw_path = raw_root / dataset_path
        
        if not raw_path.exists():
            logger.error(f"Dataset path does not exist: {raw_path}")
            continue
        
        # Output path
        output_path = dedup_output / dataset_name
        
        # Run deduplication
        try:
            result = deduplicator.deduplicate_dataset(
                dataset_name=dataset_name,
                raw_path=raw_path,
                output_path=output_path,
                image_extensions=duplicate_detection.get('image_extensions', ['.jpg', '.jpeg', '.png']),
                skip_hidden=duplicate_detection.get('skip_hidden', True),
                preserve_hierarchy=processing.get('preserve_hierarchy', True)
            )
            
            results.append(result)
            logger.info("")
            
        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {e}", exc_info=True)
            logger.info("")
            continue
    
    # Generate reports
    if results:
        logger.info("=" * 70)
        logger.info("GENERATING REPORTS")
        logger.info("=" * 70)
        
        report_generator = DeduplicationReportGenerator(reports_output)
        
        try:
            report_generator.generate_all_reports(
                results=results,
                generate_csv=reporting.get('generate_csv', True),
                generate_json=reporting.get('generate_json', True),
                generate_markdown=reporting.get('generate_markdown', True)
            )
        except Exception as e:
            logger.error(f"Error generating reports: {e}", exc_info=True)
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("DEDUPLICATION COMPLETE")
    logger.info("=" * 70)
    
    total_original = sum(r.original_count for r in results)
    total_duplicates = sum(r.duplicate_count for r in results)
    total_retained = sum(r.retained_count for r in results)
    
    logger.info(f"Total images processed: {total_original:,}")
    logger.info(f"Duplicates removed: {total_duplicates:,}")
    logger.info(f"Images retained: {total_retained:,}")
    logger.info(f"Deduplication rate: {(total_duplicates/total_original*100 if total_original > 0 else 0):.2f}%")
    
    if not dry_run:
        logger.info(f"\nDeduplicated datasets saved to: {dedup_output}")
    else:
        logger.info("\nDRY RUN: No files were copied")
    
    logger.info(f"Reports saved to: {reports_output}")
    logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
