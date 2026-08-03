"""
Report generation for deduplication process.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd

from .deduplicator import DeduplicationResult
from .utils import format_bytes


logger = logging.getLogger("preprocessing.report")


class DeduplicationReportGenerator:
    """
    Generates comprehensive reports from deduplication results.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all_reports(
        self,
        results: List[DeduplicationResult],
        generate_csv: bool = True,
        generate_json: bool = True,
        generate_markdown: bool = True
    ) -> None:
        """
        Generate all report formats.
        
        Args:
            results: List of deduplication results.
            generate_csv: Generate CSV reports.
            generate_json: Generate JSON reports.
            generate_markdown: Generate markdown summary.
        """
        logger.info("Generating deduplication reports...")
        
        if generate_csv:
            self._generate_csv_reports(results)
        
        if generate_json:
            self._generate_json_reports(results)
        
        if generate_markdown:
            self._generate_markdown_summary(results)
        
        logger.info(f"Reports saved to {self.output_dir}")
    
    def _generate_csv_reports(self, results: List[DeduplicationResult]) -> None:
        """Generate CSV format reports."""
        
        # 1. Deduplication report (all file records)
        all_records = []
        for result in results:
            all_records.extend(result.file_records)
        
        if all_records:
            df = pd.DataFrame(all_records)
            csv_path = self.output_dir / 'deduplication_report.csv'
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved deduplication report: {csv_path}")
        
        # 2. Statistics before/after
        stats_data = []
        for result in results:
            stats_data.append({
                'Dataset': result.dataset_name,
                'Original_Images': result.original_count,
                'Duplicates_Removed': result.duplicate_count,
                'Remaining_Images': result.retained_count,
                'Percentage_Removed': f"{(result.duplicate_count / result.original_count * 100 if result.original_count > 0 else 0):.2f}%",
                'Original_Size_MB': f"{result.original_size / (1024**2):.2f}",
                'Deduplicated_Size_MB': f"{result.deduplicated_size / (1024**2):.2f}",
                'Space_Saved_MB': f"{(result.original_size - result.deduplicated_size) / (1024**2):.2f}",
                'Processing_Time_Seconds': f"{result.processing_time:.2f}"
            })
        
        df_stats = pd.DataFrame(stats_data)
        
        # Before statistics
        before_path = self.output_dir / 'dataset_statistics_before.csv'
        df_before = df_stats[['Dataset', 'Original_Images', 'Original_Size_MB']].copy()
        df_before.to_csv(before_path, index=False)
        logger.info(f"Saved before statistics: {before_path}")
        
        # After statistics
        after_path = self.output_dir / 'dataset_statistics_after.csv'
        df_after = df_stats[['Dataset', 'Remaining_Images', 'Deduplicated_Size_MB', 
                             'Duplicates_Removed', 'Percentage_Removed', 'Space_Saved_MB']].copy()
        df_after.to_csv(after_path, index=False)
        logger.info(f"Saved after statistics: {after_path}")
    
    def _generate_json_reports(self, results: List[DeduplicationResult]) -> None:
        """Generate JSON format reports."""
        
        # 1. Duplicate groups
        all_duplicate_groups = {}
        group_counter = 0
        
        for result in results:
            for file_hash, files in result.duplicate_groups.items():
                group_counter += 1
                group_name = f"group_{group_counter:04d}"
                
                # Select retained file (first in list based on strategy)
                retained = str(files[0])
                duplicates = [str(f) for f in files[1:]]
                
                all_duplicate_groups[group_name] = {
                    'dataset': result.dataset_name,
                    'hash': file_hash,
                    'retained': retained,
                    'duplicates': duplicates,
                    'duplicate_count': len(duplicates)
                }
        
        if all_duplicate_groups:
            json_path = self.output_dir / 'duplicate_groups.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_duplicate_groups, f, indent=2)
            logger.info(f"Saved duplicate groups: {json_path}")
        
        # 2. Summary statistics
        summary = {
            'generated_at': datetime.now().isoformat(),
            'datasets': [],
            'overall': {}
        }
        
        total_original = 0
        total_duplicates = 0
        total_retained = 0
        total_original_size = 0
        total_deduplicated_size = 0
        
        for result in results:
            summary['datasets'].append({
                'name': result.dataset_name,
                'original_count': result.original_count,
                'duplicate_count': result.duplicate_count,
                'retained_count': result.retained_count,
                'original_size_bytes': result.original_size,
                'deduplicated_size_bytes': result.deduplicated_size,
                'processing_time_seconds': result.processing_time
            })
            
            total_original += result.original_count
            total_duplicates += result.duplicate_count
            total_retained += result.retained_count
            total_original_size += result.original_size
            total_deduplicated_size += result.deduplicated_size
        
        summary['overall'] = {
            'total_original_images': total_original,
            'total_duplicates_removed': total_duplicates,
            'total_retained_images': total_retained,
            'percentage_removed': (total_duplicates / total_original * 100 if total_original > 0 else 0),
            'original_size_bytes': total_original_size,
            'deduplicated_size_bytes': total_deduplicated_size,
            'space_saved_bytes': total_original_size - total_deduplicated_size
        }
        
        summary_path = self.output_dir / 'deduplication_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary: {summary_path}")
    
    def _generate_markdown_summary(self, results: List[DeduplicationResult]) -> None:
        """Generate markdown summary report."""
        
        md_path = self.output_dir / 'deduplication_summary.md'
        
        with open(md_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("# Dataset Deduplication Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            
            total_original = sum(r.original_count for r in results)
            total_duplicates = sum(r.duplicate_count for r in results)
            total_retained = sum(r.retained_count for r in results)
            total_original_size = sum(r.original_size for r in results)
            total_deduplicated_size = sum(r.deduplicated_size for r in results)
            
            f.write(f"This report summarizes the deduplication of {len(results)} thermal image datasets.\n\n")
            f.write(f"- **Total images processed:** {total_original:,}\n")
            f.write(f"- **Duplicates removed:** {total_duplicates:,}\n")
            f.write(f"- **Images retained:** {total_retained:,}\n")
            f.write(f"- **Deduplication rate:** {(total_duplicates/total_original*100 if total_original > 0 else 0):.2f}%\n")
            f.write(f"- **Original size:** {format_bytes(total_original_size)}\n")
            f.write(f"- **Deduplicated size:** {format_bytes(total_deduplicated_size)}\n")
            f.write(f"- **Space saved:** {format_bytes(total_original_size - total_deduplicated_size)}\n\n")
            
            # Dataset Comparison Table
            f.write("## Dataset Summary\n\n")
            f.write("| Dataset | Original | Removed | Remaining | % Removed | Space Saved |\n")
            f.write("|---------|----------|---------|-----------|-----------|-------------|\n")
            
            for result in results:
                pct_removed = (result.duplicate_count / result.original_count * 100 
                              if result.original_count > 0 else 0)
                space_saved = result.original_size - result.deduplicated_size
                
                f.write(f"| {result.dataset_name} | {result.original_count:,} | "
                       f"{result.duplicate_count:,} | {result.retained_count:,} | "
                       f"{pct_removed:.2f}% | {format_bytes(space_saved)} |\n")
            
            # Overall totals
            overall_pct = (total_duplicates / total_original * 100 if total_original > 0 else 0)
            overall_saved = total_original_size - total_deduplicated_size
            f.write(f"| **TOTAL** | **{total_original:,}** | **{total_duplicates:,}** | "
                   f"**{total_retained:,}** | **{overall_pct:.2f}%** | "
                   f"**{format_bytes(overall_saved)}** |\n\n")
            
            # Detailed Statistics
            f.write("---\n\n")
            f.write("## Detailed Statistics by Dataset\n\n")
            
            for result in results:
                f.write(f"### {result.dataset_name}\n\n")
                f.write(f"**Processing Time:** {result.processing_time:.2f} seconds\n\n")
                f.write(f"**Image Count:**\n")
                f.write(f"- Original: {result.original_count:,}\n")
                f.write(f"- Duplicates: {result.duplicate_count:,}\n")
                f.write(f"- Retained: {result.retained_count:,}\n\n")
                
                f.write(f"**Storage:**\n")
                f.write(f"- Original size: {format_bytes(result.original_size)}\n")
                f.write(f"- Deduplicated size: {format_bytes(result.deduplicated_size)}\n")
                f.write(f"- Space saved: {format_bytes(result.original_size - result.deduplicated_size)}\n\n")
                
                f.write(f"**Duplicate Groups:** {len(result.duplicate_groups)}\n\n")
                
                if result.duplicate_groups:
                    f.write("<details>\n<summary>Show duplicate group sizes</summary>\n\n")
                    group_sizes = [len(files) for files in result.duplicate_groups.values()]
                    f.write(f"- Minimum group size: {min(group_sizes)}\n")
                    f.write(f"- Maximum group size: {max(group_sizes)}\n")
                    f.write(f"- Average group size: {sum(group_sizes)/len(group_sizes):.2f}\n")
                    f.write("\n</details>\n\n")
                
                f.write("---\n\n")
            
            # Next Steps
            f.write("## Next Steps\n\n")
            f.write("1. Review this deduplication report\n")
            f.write("2. Verify deduplicated datasets in `datasets/deduplicated/`\n")
            f.write("3. Proceed to next preprocessing stage (normalization/augmentation)\n")
            f.write("4. All removed duplicates are traceable via `deduplication_report.csv`\n\n")
            
            # Traceability Note
            f.write("## Traceability\n\n")
            f.write("Every removed duplicate is fully traceable:\n\n")
            f.write("- `deduplication_report.csv` contains all file operations\n")
            f.write("- `duplicate_groups.json` maps duplicates to retained files\n")
            f.write("- Original datasets in `datasets/raw/` remain unmodified\n\n")
        
        logger.info(f"Saved markdown summary: {md_path}")
