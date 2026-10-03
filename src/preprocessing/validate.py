"""
Dataset validation module.

Validates processed datasets to ensure correctness and completeness.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import cv2
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.utils import load_config, setup_logging, get_relative_path


logger = logging.getLogger("preprocessing.validation")


@dataclass
class ValidationResult:
    """Container for validation results."""
    
    # Counts
    total_input_images: int = 0
    total_processed_images: int = 0
    
    # Valid images
    valid_images: int = 0
    
    # Issues
    missing_images: List[str] = field(default_factory=list)
    incorrect_resolution: List[Dict[str, Any]] = field(default_factory=list)
    corrupted_images: List[Dict[str, Any]] = field(default_factory=list)
    hierarchy_violations: List[Dict[str, Any]] = field(default_factory=list)
    metadata_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata validation
    metadata_found: bool = False
    metadata_records: int = 0
    
    # Success flag
    passed: bool = True
    
    def has_issues(self) -> bool:
        """Check if any validation issues were found."""
        return (
            len(self.missing_images) > 0 or
            len(self.incorrect_resolution) > 0 or
            len(self.corrupted_images) > 0 or
            len(self.hierarchy_violations) > 0 or
            len(self.metadata_mismatches) > 0 or
            self.total_input_images != self.total_processed_images
        )


def validate_single_image(args) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Validate a single processed image (for parallel processing).
    
    Args:
        args: Tuple of (image_path, expected_width, expected_height)
        
    Returns:
        Tuple of (is_valid, issue_dict or None)
    """
    image_path, expected_width, expected_height = args
    
    try:
        # Check if file exists
        if not image_path.exists():
            return False, {
                'path': str(image_path),
                'issue': 'missing',
                'message': 'File does not exist'
            }
        
        # Try to read the image
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        
        if image is None:
            return False, {
                'path': str(image_path),
                'issue': 'corrupted',
                'message': 'Cannot read image with OpenCV'
            }
        
        # Check resolution
        height, width = image.shape[:2]
        
        if width != expected_width or height != expected_height:
            return False, {
                'path': str(image_path),
                'issue': 'resolution',
                'expected': f"{expected_width}x{expected_height}",
                'actual': f"{width}x{height}",
                'message': f'Expected {expected_width}x{expected_height}, got {width}x{height}'
            }
        
        # Valid image
        return True, None
        
    except Exception as e:
        return False, {
            'path': str(image_path),
            'issue': 'error',
            'message': str(e)
        }


class DatasetValidator:
    """
    Validates processed datasets against source datasets.
    """
    
    def __init__(
        self,
        input_root: Path,
        output_root: Path,
        metadata_path: Optional[Path] = None,
        expected_width: int = 640,
        expected_height: int = 480,
        max_workers: int = 4
    ):
        """
        Initialize dataset validator.
        
        Args:
            input_root: Path to input (deduplicated) datasets.
            output_root: Path to output (processed) datasets.
            metadata_path: Path to metadata CSV file.
            expected_width: Expected image width.
            expected_height: Expected image height.
            max_workers: Number of parallel workers.
        """
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.expected_width = expected_width
        self.expected_height = expected_height
        self.max_workers = max_workers
        
        self.result = ValidationResult()
    
    def find_all_images(self, root: Path) -> List[Path]:
        """
        Find all images in a directory tree.
        
        Args:
            root: Root directory to search.
            
        Returns:
            List of image paths.
        """
        extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        images = []
        
        for ext in extensions:
            images.extend(root.rglob(f"*{ext}"))
            images.extend(root.rglob(f"*{ext.upper()}"))
        
        return sorted(images)
    
    def validate_hierarchy(
        self,
        input_images: List[Path],
        output_images: List[Path]
    ):
        """
        Validate that directory hierarchy is preserved.
        
        Args:
            input_images: List of input image paths.
            output_images: List of output image paths.
        """
        logger.info("Validating directory hierarchy...")
        
        # Build set of relative paths for output images
        output_rel_paths = set()
        for img in output_images:
            rel_path = get_relative_path(img, self.output_root)
            output_rel_paths.add(rel_path)
        
        # Check each input image has corresponding output
        for input_img in input_images:
            input_rel = get_relative_path(input_img, self.input_root)
            
            # Check if same relative path exists in output
            if input_rel not in output_rel_paths:
                # Check if path exists but with different structure
                output_name = input_img.name
                matching_outputs = [
                    o for o in output_images 
                    if o.name == output_name
                ]
                
                if matching_outputs:
                    # Found image but hierarchy is different
                    self.result.hierarchy_violations.append({
                        'input_path': str(input_img),
                        'expected_path': str(self.output_root / input_rel),
                        'actual_path': str(matching_outputs[0]),
                        'message': 'Directory hierarchy not preserved'
                    })
        
        if self.result.hierarchy_violations:
            logger.warning(
                f"Found {len(self.result.hierarchy_violations)} hierarchy violations"
            )
    
    def validate_images(self, images: List[Path]) -> Tuple[int, int, int]:
        """
        Validate a list of images in parallel.
        
        Args:
            images: List of image paths to validate.
            
        Returns:
            Tuple of (valid_count, missing_count, issue_count)
        """
        logger.info(f"Validating {len(images)} images...")
        
        tasks = [
            (img, self.expected_width, self.expected_height)
            for img in images
        ]
        
        valid_count = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(validate_single_image, task) for task in tasks]
            
            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Validating images"
            )
            
            for future in pbar:
                try:
                    is_valid, issue = future.result()
                    
                    if is_valid:
                        valid_count += 1
                    else:
                        # Categorize issue
                        if issue['issue'] == 'missing':
                            self.result.missing_images.append(issue['path'])
                        elif issue['issue'] == 'resolution':
                            self.result.incorrect_resolution.append(issue)
                        elif issue['issue'] in ['corrupted', 'error']:
                            self.result.corrupted_images.append(issue)
                    
                except Exception as e:
                    logger.error(f"Validation task failed: {e}")
            
            pbar.close()
        
        return valid_count, len(self.result.missing_images), len(self.result.corrupted_images)
    
    def validate_metadata(self):
        """Validate metadata CSV against processed images."""
        if self.metadata_path is None or not self.metadata_path.exists():
            logger.warning("Metadata file not found, skipping metadata validation")
            self.result.metadata_found = False
            return
        
        logger.info(f"Validating metadata: {self.metadata_path}")
        
        try:
            # Load metadata
            df = pd.read_csv(self.metadata_path)
            self.result.metadata_found = True
            self.result.metadata_records = len(df)
            
            logger.info(f"Metadata contains {len(df)} records")
            
            # Check each metadata record
            for idx, row in df.iterrows():
                output_path = Path(row['processed_path'])
                
                # Check if processed file exists
                if not output_path.exists():
                    self.result.metadata_mismatches.append({
                        'metadata_row': idx,
                        'processed_path': str(output_path),
                        'issue': 'file_missing',
                        'message': 'File in metadata does not exist'
                    })
                    continue
                
                # Check resolution matches metadata
                expected_w = row.get('processed_width', self.expected_width)
                expected_h = row.get('processed_height', self.expected_height)
                
                try:
                    img = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        h, w = img.shape[:2]
                        if w != expected_w or h != expected_h:
                            self.result.metadata_mismatches.append({
                                'metadata_row': idx,
                                'processed_path': str(output_path),
                                'issue': 'resolution_mismatch',
                                'expected': f"{expected_w}x{expected_h}",
                                'actual': f"{w}x{h}",
                                'message': f'Resolution mismatch with metadata'
                            })
                except Exception as e:
                    self.result.metadata_mismatches.append({
                        'metadata_row': idx,
                        'processed_path': str(output_path),
                        'issue': 'read_error',
                        'message': str(e)
                    })
            
            if self.result.metadata_mismatches:
                logger.warning(
                    f"Found {len(self.result.metadata_mismatches)} metadata mismatches"
                )
            else:
                logger.info("Metadata validation passed")
        
        except Exception as e:
            logger.error(f"Error validating metadata: {e}")
            self.result.metadata_found = False
    
    def validate(self) -> ValidationResult:
        """
        Run complete validation.
        
        Returns:
            ValidationResult object with all findings.
        """
        logger.info("="*70)
        logger.info("DATASET VALIDATION")
        logger.info("="*70)
        logger.info(f"Input root: {self.input_root}")
        logger.info(f"Output root: {self.output_root}")
        logger.info(f"Expected resolution: {self.expected_width}x{self.expected_height}")
        logger.info("")
        
        # Find all input images
        logger.info("Scanning input datasets...")
        input_images = self.find_all_images(self.input_root)
        self.result.total_input_images = len(input_images)
        logger.info(f"Found {self.result.total_input_images} input images")
        
        # Find all output images
        logger.info("Scanning output datasets...")
        output_images = self.find_all_images(self.output_root)
        self.result.total_processed_images = len(output_images)
        logger.info(f"Found {self.result.total_processed_images} processed images")
        
        # Check counts match
        if self.result.total_input_images != self.result.total_processed_images:
            logger.error(
                f"Image count mismatch: {self.result.total_input_images} input, "
                f"{self.result.total_processed_images} processed"
            )
        else:
            logger.info("✓ Image counts match")
        
        logger.info("")
        
        # Validate hierarchy
        self.validate_hierarchy(input_images, output_images)
        logger.info("")
        
        # Validate processed images
        valid_count, missing_count, corrupted_count = self.validate_images(output_images)
        self.result.valid_images = valid_count
        
        logger.info("")
        logger.info("Image validation results:")
        logger.info(f"  Valid: {valid_count}")
        logger.info(f"  Missing: {missing_count}")
        logger.info(f"  Incorrect resolution: {len(self.result.incorrect_resolution)}")
        logger.info(f"  Corrupted: {corrupted_count}")
        
        # Validate metadata
        logger.info("")
        self.validate_metadata()
        
        # Determine overall result
        self.result.passed = not self.result.has_issues()
        
        logger.info("")
        logger.info("="*70)
        if self.result.passed:
            logger.info("✓ VALIDATION PASSED")
        else:
            logger.error("✗ VALIDATION FAILED")
        logger.info("="*70)
        
        return self.result


class ValidationReportGenerator:
    """
    Generates validation reports in various formats.
    """
    
    def __init__(self, result: ValidationResult, output_dir: Path):
        """
        Initialize report generator.
        
        Args:
            result: ValidationResult object.
            output_dir: Directory to save reports.
        """
        self.result = result
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_markdown_report(self, filename: str = "preprocessing_validation.md"):
        """
        Generate human-readable markdown report.
        
        Args:
            filename: Output filename.
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Generating markdown report: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Preprocessing Validation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # Overall status
            if self.result.passed:
                f.write("## Status: ✓ PASSED\n\n")
            else:
                f.write("## Status: ✗ FAILED\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Input images:** {self.result.total_input_images:,}\n")
            f.write(f"- **Processed images:** {self.result.total_processed_images:,}\n")
            f.write(f"- **Valid images:** {self.result.valid_images:,}\n")
            f.write(f"- **Missing images:** {len(self.result.missing_images):,}\n")
            f.write(f"- **Incorrect resolution:** {len(self.result.incorrect_resolution):,}\n")
            f.write(f"- **Corrupted images:** {len(self.result.corrupted_images):,}\n")
            f.write(f"- **Hierarchy violations:** {len(self.result.hierarchy_violations):,}\n")
            
            if self.result.metadata_found:
                f.write(f"- **Metadata records:** {self.result.metadata_records:,}\n")
                f.write(f"- **Metadata mismatches:** {len(self.result.metadata_mismatches):,}\n")
            else:
                f.write("- **Metadata:** Not found or not validated\n")
            
            f.write("\n")
            
            # Image count verification
            f.write("## Image Count Verification\n\n")
            if self.result.total_input_images == self.result.total_processed_images:
                f.write("✓ **Image counts match**\n\n")
                f.write(f"Both input and output contain {self.result.total_input_images:,} images.\n\n")
            else:
                f.write("✗ **Image count mismatch detected**\n\n")
                diff = self.result.total_input_images - self.result.total_processed_images
                f.write(f"- Input: {self.result.total_input_images:,} images\n")
                f.write(f"- Processed: {self.result.total_processed_images:,} images\n")
                f.write(f"- Difference: {diff:+,} images\n\n")
            
            # Resolution verification
            f.write("## Resolution Verification\n\n")
            if len(self.result.incorrect_resolution) == 0:
                f.write("✓ **All images have correct resolution**\n\n")
            else:
                f.write(f"✗ **{len(self.result.incorrect_resolution)} images have incorrect resolution**\n\n")
                f.write("| Image | Expected | Actual |\n")
                f.write("|-------|----------|--------|\n")
                for issue in self.result.incorrect_resolution[:50]:  # Limit to first 50
                    path = Path(issue['path']).name
                    f.write(f"| {path} | {issue['expected']} | {issue['actual']} |\n")
                if len(self.result.incorrect_resolution) > 50:
                    f.write(f"\n*... and {len(self.result.incorrect_resolution) - 50} more*\n")
                f.write("\n")
            
            # Corrupted images
            f.write("## Corrupted Images\n\n")
            if len(self.result.corrupted_images) == 0:
                f.write("✓ **No corrupted images detected**\n\n")
            else:
                f.write(f"✗ **{len(self.result.corrupted_images)} corrupted images detected**\n\n")
                for issue in self.result.corrupted_images[:50]:
                    f.write(f"- `{Path(issue['path']).name}`: {issue['message']}\n")
                if len(self.result.corrupted_images) > 50:
                    f.write(f"\n*... and {len(self.result.corrupted_images) - 50} more*\n")
                f.write("\n")
            
            # Missing images
            if len(self.result.missing_images) > 0:
                f.write("## Missing Images\n\n")
                f.write(f"✗ **{len(self.result.missing_images)} images are missing**\n\n")
                for path in self.result.missing_images[:50]:
                    f.write(f"- `{Path(path).name}`\n")
                if len(self.result.missing_images) > 50:
                    f.write(f"\n*... and {len(self.result.missing_images) - 50} more*\n")
                f.write("\n")
            
            # Hierarchy violations
            if len(self.result.hierarchy_violations) > 0:
                f.write("## Hierarchy Violations\n\n")
                f.write(f"✗ **{len(self.result.hierarchy_violations)} hierarchy violations detected**\n\n")
                for violation in self.result.hierarchy_violations[:20]:
                    f.write(f"- **Expected:** `{violation['expected_path']}`\n")
                    f.write(f"  **Actual:** `{violation['actual_path']}`\n\n")
                if len(self.result.hierarchy_violations) > 20:
                    f.write(f"*... and {len(self.result.hierarchy_violations) - 20} more*\n\n")
            
            # Metadata validation
            if self.result.metadata_found:
                f.write("## Metadata Validation\n\n")
                if len(self.result.metadata_mismatches) == 0:
                    f.write("✓ **Metadata validation passed**\n\n")
                    f.write(f"All {self.result.metadata_records:,} metadata records are consistent.\n\n")
                else:
                    f.write(f"✗ **{len(self.result.metadata_mismatches)} metadata mismatches detected**\n\n")
                    for mismatch in self.result.metadata_mismatches[:20]:
                        f.write(f"- Row {mismatch['metadata_row']}: {mismatch['message']}\n")
                    if len(self.result.metadata_mismatches) > 20:
                        f.write(f"\n*... and {len(self.result.metadata_mismatches) - 20} more*\n")
                    f.write("\n")
            
            # Conclusion
            f.write("---\n\n")
            f.write("## Conclusion\n\n")
            
            if self.result.passed:
                f.write("✓ **All validation checks passed successfully.**\n\n")
                f.write("The processed dataset is ready for the next stage.\n\n")
            else:
                f.write("✗ **Validation failed with issues detected.**\n\n")
                f.write("Please review the issues above and reprocess the dataset.\n\n")
        
        logger.info(f"Markdown report saved to: {output_path}")
    
    def generate_json_report(self, filename: str = "validation_result.json"):
        """
        Generate machine-readable JSON report.
        
        Args:
            filename: Output filename.
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Generating JSON report: {output_path}")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'passed': self.result.passed,
            'summary': {
                'total_input_images': self.result.total_input_images,
                'total_processed_images': self.result.total_processed_images,
                'valid_images': self.result.valid_images,
                'missing_images': len(self.result.missing_images),
                'incorrect_resolution': len(self.result.incorrect_resolution),
                'corrupted_images': len(self.result.corrupted_images),
                'hierarchy_violations': len(self.result.hierarchy_violations),
                'metadata_found': self.result.metadata_found,
                'metadata_records': self.result.metadata_records,
                'metadata_mismatches': len(self.result.metadata_mismatches)
            },
            'issues': {
                'missing_images': self.result.missing_images,
                'incorrect_resolution': self.result.incorrect_resolution,
                'corrupted_images': self.result.corrupted_images,
                'hierarchy_violations': self.result.hierarchy_violations,
                'metadata_mismatches': self.result.metadata_mismatches
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"JSON report saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate processed thermal image datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/preprocessing.yaml'),
        help='Path to preprocessing configuration YAML file'
    )
    
    parser.add_argument(
        '--input-root',
        type=Path,
        help='Override input root from config'
    )
    
    parser.add_argument(
        '--output-root',
        type=Path,
        help='Override output root from config'
    )
    
    parser.add_argument(
        '--metadata',
        type=Path,
        help='Path to metadata CSV file'
    )
    
    parser.add_argument(
        '--expected-width',
        type=int,
        default=640,
        help='Expected image width (default: 640)'
    )
    
    parser.add_argument(
        '--expected-height',
        type=int,
        default=480,
        help='Expected image height (default: 480)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Log level'
    )
    
    return parser.parse_args()


def main():
    """Main validation workflow."""
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Setup logging
    reports_output = Path(config['paths']['reports_output'])
    log_file = reports_output / 'validation.log'
    
    logger = setup_logging(
        log_file=log_file,
        level=getattr(logging, args.log_level),
        logger_name="preprocessing.validation"
    )
    
    logger.info("="*70)
    logger.info("PREPROCESSING VALIDATION")
    logger.info("="*70)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    # Get paths
    input_root = args.input_root or Path(config['paths']['input_root'])
    output_root = args.output_root or Path(config['paths']['output_root'])
    
    # Get metadata path
    if args.metadata:
        metadata_path = args.metadata
    else:
        metadata_output = Path(config['paths']['metadata_output'])
        metadata_path = metadata_output / 'metadata.csv'
    
    # Get expected resolution from config
    preprocess_config = config.get('preprocessing', {})
    resize_config = preprocess_config.get('resize', {})
    
    if resize_config.get('enabled', True):
        expected_width = resize_config.get('width', args.expected_width)
        expected_height = resize_config.get('height', args.expected_height)
    else:
        expected_width = args.expected_width
        expected_height = args.expected_height
    
    # Create validator
    validator = DatasetValidator(
        input_root=input_root,
        output_root=output_root,
        metadata_path=metadata_path,
        expected_width=expected_width,
        expected_height=expected_height,
        max_workers=args.max_workers
    )
    
    # Run validation
    result = validator.validate()
    
    # Generate reports
    logger.info("")
    logger.info("Generating validation reports...")
    
    report_gen = ValidationReportGenerator(result, reports_output)
    report_gen.generate_markdown_report()
    report_gen.generate_json_report()
    
    logger.info("")
    logger.info(f"Reports saved to: {reports_output}")
    logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Exit with appropriate code
    return 0 if result.passed else 1


if __name__ == '__main__':
    sys.exit(main())
