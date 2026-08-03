"""
Metadata generation and management for preprocessing pipeline.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

logger = logging.getLogger("preprocessing.metadata")


class MetadataManager:
    """
    Manages metadata collection and export for processed images.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize metadata manager.
        
        Args:
            output_dir: Directory to save metadata files.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict[str, Any]] = []
    
    def add_record(
        self,
        metadata: Dict[str, Any],
        dataset_name: str,
        class_name: Optional[str] = None
    ) -> None:
        """
        Add a processing record to metadata.
        
        Args:
            metadata: Processing metadata from ImageProcessor.
            dataset_name: Name of the dataset.
            class_name: Class/category name (optional).
        """
        # Extract paths
        input_path = Path(metadata['input_path'])
        output_path = Path(metadata['output_path'])
        
        # Build record
        record = {
            'original_path': str(input_path),
            'processed_path': str(output_path),
            'dataset': dataset_name,
            'class': class_name if class_name else self._extract_class(input_path),
            'filename': input_path.name,
            'original_width': metadata.get('original_width'),
            'original_height': metadata.get('original_height'),
            'original_resolution': metadata.get('original_resolution'),
            'processed_width': metadata.get('target_width'),
            'processed_height': metadata.get('target_height'),
            'processed_resolution': metadata.get('target_resolution'),
            'file_size_original': metadata.get('file_size_original'),
            'file_size_processed': metadata.get('file_size_processed'),
            'processing_time': metadata.get('processing_time'),
            'was_downsampled': metadata.get('was_downsampled'),
            'was_upsampled': metadata.get('was_upsampled'),
            'resize_method': metadata.get('resize_method'),
            'interpolation': metadata.get('interpolation'),
            'bit_depth': metadata.get('bit_depth'),
            'channels': metadata.get('channels'),
            'dtype': metadata.get('dtype'),
        }
        
        self.records.append(record)
    
    def _extract_class(self, file_path: Path) -> str:
        """
        Extract class/category from file path.
        
        Assumes structure: dataset/class/image.jpg
        
        Args:
            file_path: Path to image file.
            
        Returns:
            Class name or "unknown".
        """
        try:
            # Get parent directory name
            return file_path.parent.name
        except:
            return "unknown"
    
    def save_metadata(
        self,
        generate_csv: bool = True,
        generate_json: bool = True
    ) -> None:
        """
        Save metadata to files.
        
        Args:
            generate_csv: If True, generate CSV file.
            generate_json: If True, generate JSON file.
        """
        if not self.records:
            logger.warning("No metadata records to save")
            return
        
        logger.info(f"Saving metadata for {len(self.records)} processed images...")
        
        # Create DataFrame
        df = pd.DataFrame(self.records)
        
        # Save CSV
        if generate_csv:
            csv_path = self.output_dir / 'metadata.csv'
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV metadata: {csv_path}")
        
        # Save JSON
        if generate_json:
            json_path = self.output_dir / 'metadata.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.records, f, indent=2, default=str)
            logger.info(f"Saved JSON metadata: {json_path}")
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics from metadata.
        
        Returns:
            Dictionary with summary statistics.
        """
        if not self.records:
            return {}
        
        df = pd.DataFrame(self.records)
        
        summary = {
            'total_images': len(self.records),
            'datasets': df['dataset'].unique().tolist(),
            'num_datasets': df['dataset'].nunique(),
            'num_classes': df['class'].nunique(),
            'classes': df['class'].unique().tolist(),
            'total_processing_time': df['processing_time'].sum(),
            'avg_processing_time': df['processing_time'].mean(),
            'original_resolutions': df['original_resolution'].value_counts().to_dict(),
            'processed_resolution': df['processed_resolution'].iloc[0] if len(df) > 0 else None,
            'total_size_original': df['file_size_original'].sum(),
            'total_size_processed': df['file_size_processed'].sum(),
            'space_difference': df['file_size_processed'].sum() - df['file_size_original'].sum(),
            'downsampled_count': df['was_downsampled'].sum(),
            'upsampled_count': df['was_upsampled'].sum(),
        }
        
        return summary
    
    def save_summary(self) -> None:
        """Save summary statistics to file."""
        summary = self.generate_summary()
        
        if not summary:
            logger.warning("No summary to save")
            return
        
        summary_path = self.output_dir / 'preprocessing_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved summary: {summary_path}")
        
        # Also save as markdown for readability
        self._save_summary_markdown(summary)
    
    def _save_summary_markdown(self, summary: Dict[str, Any]) -> None:
        """
        Save summary as markdown file.
        
        Args:
            summary: Summary dictionary.
        """
        md_path = self.output_dir / 'preprocessing_summary.md'
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Preprocessing Summary\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            f.write("## Overview\n\n")
            f.write(f"- **Total images processed:** {summary['total_images']:,}\n")
            f.write(f"- **Number of datasets:** {summary['num_datasets']}\n")
            f.write(f"- **Number of classes:** {summary['num_classes']}\n")
            f.write(f"- **Target resolution:** {summary['processed_resolution']}\n\n")
            
            f.write("## Processing Time\n\n")
            f.write(f"- **Total:** {summary['total_processing_time']:.2f} seconds\n")
            f.write(f"- **Average per image:** {summary['avg_processing_time']:.4f} seconds\n\n")
            
            f.write("## Resizing Statistics\n\n")
            f.write(f"- **Downsampled:** {summary['downsampled_count']:,} images\n")
            f.write(f"- **Upsampled:** {summary['upsampled_count']:,} images\n\n")
            
            f.write("## Storage\n\n")
            size_original_mb = summary['total_size_original'] / (1024**2)
            size_processed_mb = summary['total_size_processed'] / (1024**2)
            size_diff_mb = summary['space_difference'] / (1024**2)
            
            f.write(f"- **Original size:** {size_original_mb:.2f} MB\n")
            f.write(f"- **Processed size:** {size_processed_mb:.2f} MB\n")
            f.write(f"- **Difference:** {size_diff_mb:+.2f} MB\n\n")
            
            f.write("## Original Resolutions\n\n")
            for resolution, count in sorted(summary['original_resolutions'].items(), 
                                           key=lambda x: x[1], reverse=True):
                f.write(f"- `{resolution}`: {count:,} images\n")
            f.write("\n")
            
            f.write("## Datasets\n\n")
            for dataset in summary['datasets']:
                f.write(f"- {dataset}\n")
            f.write("\n")
            
            f.write("## Classes\n\n")
            for class_name in sorted(summary['classes']):
                f.write(f"- {class_name}\n")
        
        logger.info(f"Saved markdown summary: {md_path}")
