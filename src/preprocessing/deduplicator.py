"""
Dataset deduplication module.

Removes duplicate images while preserving complete traceability.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm

from .utils import (
    find_all_images,
    get_relative_path,
    safe_copy_file,
    ensure_directory,
    format_bytes
)
from .hash_computer import HashComputer


logger = logging.getLogger("preprocessing.deduplicator")


@dataclass
class DeduplicationResult:
    """Results from deduplication process."""
    dataset_name: str
    original_count: int
    retained_count: int
    duplicate_count: int
    original_size: int
    deduplicated_size: int
    duplicate_groups: Dict[str, List[Path]]
    file_records: List[Dict[str, Any]]
    processing_time: float


class DatasetDeduplicator:
    """
    Deduplicates datasets while preserving folder structure and traceability.
    """
    
    def __init__(
        self,
        hash_algorithm: str = "sha256",
        chunk_size: int = 4096,
        max_workers: Optional[int] = None,
        retention_strategy: str = "first",
        verify_copies: bool = True,
        dry_run: bool = False
    ):
        """
        Initialize deduplicator.
        
        Args:
            hash_algorithm: Hash algorithm for duplicate detection.
            chunk_size: Chunk size for file reading.
            max_workers: Number of parallel workers.
            retention_strategy: Strategy for choosing which duplicate to keep
                ("first", "smallest", "largest", "shortest_path").
            verify_copies: Verify file copies by hash comparison.
            dry_run: If True, analyze only without copying files.
        """
        self.hash_computer = HashComputer(
            algorithm=hash_algorithm,
            chunk_size=chunk_size,
            max_workers=max_workers
        )
        self.retention_strategy = retention_strategy
        self.verify_copies = verify_copies
        self.dry_run = dry_run
    
    def deduplicate_dataset(
        self,
        dataset_name: str,
        raw_path: Path,
        output_path: Path,
        image_extensions: List[str],
        skip_hidden: bool = True,
        preserve_hierarchy: bool = True
    ) -> DeduplicationResult:
        """
        Deduplicate a single dataset.
        
        Args:
            dataset_name: Name of the dataset.
            raw_path: Path to raw dataset directory.
            output_path: Path to output deduplicated dataset.
            image_extensions: List of valid image extensions.
            skip_hidden: Skip hidden files.
            preserve_hierarchy: Preserve folder structure in output.
            
        Returns:
            DeduplicationResult object with statistics and metadata.
        """
        start_time = datetime.now()
        
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"Source: {raw_path}")
        logger.info(f"Output: {output_path}")
        
        # Find all images
        logger.info("Scanning for image files...")
        image_files = find_all_images(raw_path, image_extensions, skip_hidden)
        logger.info(f"Found {len(image_files)} images")
        
        if len(image_files) == 0:
            logger.warning(f"No images found in {raw_path}")
            return self._empty_result(dataset_name)
        
        # Compute original dataset size
        original_size = sum(f.stat().st_size for f in image_files)
        logger.info(f"Original dataset size: {format_bytes(original_size)}")
        
        # Compute hashes and identify duplicates
        hash_to_files, duplicate_groups, total_duplicates = \
            self.hash_computer.compute_and_identify(image_files)
        
        # Select files to retain
        retained_files, file_records = self._select_retained_files(
            hash_to_files,
            duplicate_groups,
            dataset_name
        )
        
        logger.info(f"Retaining {len(retained_files)} unique images")
        logger.info(f"Removing {total_duplicates} duplicates")
        
        # Copy retained files to output
        if not self.dry_run:
            copied_size = self._copy_files(
                retained_files,
                raw_path,
                output_path,
                preserve_hierarchy
            )
        else:
            logger.info("DRY RUN: No files copied")
            copied_size = sum(f.stat().st_size for f in retained_files)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Deduplication complete in {processing_time:.2f}s")
        logger.info(f"Deduplicated dataset size: {format_bytes(copied_size)}")
        logger.info(f"Space saved: {format_bytes(original_size - copied_size)}")
        
        return DeduplicationResult(
            dataset_name=dataset_name,
            original_count=len(image_files),
            retained_count=len(retained_files),
            duplicate_count=total_duplicates,
            original_size=original_size,
            deduplicated_size=copied_size,
            duplicate_groups=duplicate_groups,
            file_records=file_records,
            processing_time=processing_time
        )
    
    def _select_retained_files(
        self,
        hash_to_files: Dict[str, List[Path]],
        duplicate_groups: Dict[str, List[Path]],
        dataset_name: str
    ) -> Tuple[List[Path], List[Dict[str, Any]]]:
        """
        Select which files to retain from each hash group.
        
        Args:
            hash_to_files: Mapping of hashes to file paths.
            duplicate_groups: Only groups with duplicates.
            dataset_name: Name of the dataset.
            
        Returns:
            Tuple of (retained_files, file_records).
        """
        retained_files = []
        file_records = []
        duplicate_group_id = 0
        
        for file_hash, files in hash_to_files.items():
            if len(files) == 1:
                # No duplicates, keep the file
                retained_file = files[0]
                retained_files.append(retained_file)
                
                file_records.append({
                    'original_path': str(retained_file),
                    'retained_path': str(retained_file),
                    'hash': file_hash,
                    'status': 'retained',
                    'duplicate_group': None,
                    'dataset': dataset_name,
                    'reason': 'unique'
                })
            else:
                # Multiple files with same hash - duplicates found
                duplicate_group_id += 1
                group_name = f"group_{duplicate_group_id:04d}"
                
                # Select which file to retain based on strategy
                retained_file = self._select_by_strategy(files)
                retained_files.append(retained_file)
                
                # Record retained file
                file_records.append({
                    'original_path': str(retained_file),
                    'retained_path': str(retained_file),
                    'hash': file_hash,
                    'status': 'retained',
                    'duplicate_group': group_name,
                    'dataset': dataset_name,
                    'reason': f'selected_by_{self.retention_strategy}'
                })
                
                # Record removed duplicates
                for dup_file in files:
                    if dup_file != retained_file:
                        file_records.append({
                            'original_path': str(dup_file),
                            'retained_path': str(retained_file),
                            'hash': file_hash,
                            'status': 'removed_duplicate',
                            'duplicate_group': group_name,
                            'dataset': dataset_name,
                            'reason': f'duplicate_of_{retained_file.name}'
                        })
        
        return retained_files, file_records
    
    def _select_by_strategy(self, files: List[Path]) -> Path:
        """
        Select one file from duplicates based on retention strategy.
        
        Args:
            files: List of duplicate file paths.
            
        Returns:
            Selected file path.
        """
        if self.retention_strategy == "first":
            return files[0]
        
        elif self.retention_strategy == "smallest":
            return min(files, key=lambda f: f.stat().st_size)
        
        elif self.retention_strategy == "largest":
            return max(files, key=lambda f: f.stat().st_size)
        
        elif self.retention_strategy == "shortest_path":
            return min(files, key=lambda f: len(str(f)))
        
        else:
            logger.warning(f"Unknown strategy '{self.retention_strategy}', using 'first'")
            return files[0]
    
    def _copy_files(
        self,
        files: List[Path],
        source_root: Path,
        dest_root: Path,
        preserve_hierarchy: bool
    ) -> int:
        """
        Copy files to destination directory.
        
        Args:
            files: List of files to copy.
            source_root: Source root directory.
            dest_root: Destination root directory.
            preserve_hierarchy: Preserve folder structure.
            
        Returns:
            Total size of copied files in bytes.
        """
        logger.info("Copying retained files...")
        
        ensure_directory(dest_root)
        total_size = 0
        
        for file_path in tqdm(files, desc="Copying files"):
            try:
                if preserve_hierarchy:
                    # Preserve folder structure
                    rel_path = get_relative_path(file_path, source_root)
                    dest_path = dest_root / rel_path
                else:
                    # Flat structure
                    dest_path = dest_root / file_path.name
                
                # Copy file
                safe_copy_file(
                    file_path,
                    dest_path,
                    verify=self.verify_copies,
                    hash_algorithm=self.hash_computer.algorithm
                )
                
                total_size += dest_path.stat().st_size
                
            except Exception as e:
                logger.error(f"Error copying {file_path}: {e}")
        
        return total_size
    
    def _empty_result(self, dataset_name: str) -> DeduplicationResult:
        """Create empty result for dataset with no images."""
        return DeduplicationResult(
            dataset_name=dataset_name,
            original_count=0,
            retained_count=0,
            duplicate_count=0,
            original_size=0,
            deduplicated_size=0,
            duplicate_groups={},
            file_records=[],
            processing_time=0.0
        )
