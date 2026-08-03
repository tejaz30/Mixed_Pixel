"""
Parallel hash computation for large datasets.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from .utils import compute_file_hash


logger = logging.getLogger("preprocessing.hash")


class HashComputer:
    """
    Efficiently compute file hashes using parallel processing.
    """
    
    def __init__(
        self,
        algorithm: str = "sha256",
        chunk_size: int = 4096,
        max_workers: Optional[int] = None
    ):
        """
        Initialize hash computer.
        
        Args:
            algorithm: Hash algorithm to use.
            chunk_size: Chunk size for reading files (bytes).
            max_workers: Maximum number of parallel workers.
        """
        self.algorithm = algorithm
        self.chunk_size = chunk_size
        self.max_workers = max_workers
    
    def compute_hashes(
        self,
        file_paths: List[Path],
        show_progress: bool = True
    ) -> Dict[str, List[Path]]:
        """
        Compute hashes for a list of files.
        
        Args:
            file_paths: List of file paths to hash.
            show_progress: Show progress bar.
            
        Returns:
            Dictionary mapping hashes to lists of file paths.
            Files with identical hashes are duplicates.
        """
        logger.info(f"Computing {self.algorithm} hashes for {len(file_paths)} files...")
        
        hash_to_files: Dict[str, List[Path]] = {}
        
        # Use ProcessPoolExecutor for CPU-intensive hashing
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    compute_file_hash,
                    file_path,
                    self.algorithm,
                    self.chunk_size
                ): file_path
                for file_path in file_paths
            }
            
            # Collect results with progress bar
            pbar = tqdm(
                as_completed(future_to_file),
                total=len(file_paths),
                desc="Computing hashes",
                disable=not show_progress
            )
            
            for future in pbar:
                file_path = future_to_file[future]
                try:
                    file_hash = future.result()
                    
                    if file_hash not in hash_to_files:
                        hash_to_files[file_hash] = []
                    hash_to_files[file_hash].append(file_path)
                    
                except Exception as e:
                    logger.error(f"Error hashing {file_path}: {e}")
            
            pbar.close()
        
        logger.info(f"Computed hashes for {len(file_paths)} files")
        logger.info(f"Found {len(hash_to_files)} unique hashes")
        
        return hash_to_files
    
    def identify_duplicates(
        self,
        hash_to_files: Dict[str, List[Path]]
    ) -> Tuple[Dict[str, List[Path]], int]:
        """
        Identify duplicate files from hash mapping.
        
        Args:
            hash_to_files: Dictionary mapping hashes to file paths.
            
        Returns:
            Tuple of (duplicate_groups, total_duplicates).
            duplicate_groups: Only hashes with multiple files.
            total_duplicates: Total number of duplicate files found.
        """
        duplicate_groups = {
            file_hash: files
            for file_hash, files in hash_to_files.items()
            if len(files) > 1
        }
        
        # Count total duplicate files (excluding one copy from each group)
        total_duplicates = sum(len(files) - 1 for files in duplicate_groups.values())
        
        logger.info(f"Found {len(duplicate_groups)} duplicate groups")
        logger.info(f"Total duplicate files: {total_duplicates}")
        
        return duplicate_groups, total_duplicates
    
    def compute_and_identify(
        self,
        file_paths: List[Path],
        show_progress: bool = True
    ) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]], int]:
        """
        Compute hashes and identify duplicates in one step.
        
        Args:
            file_paths: List of file paths to process.
            show_progress: Show progress bar.
            
        Returns:
            Tuple of (hash_to_files, duplicate_groups, total_duplicates).
        """
        hash_to_files = self.compute_hashes(file_paths, show_progress)
        duplicate_groups, total_duplicates = self.identify_duplicates(hash_to_files)
        
        return hash_to_files, duplicate_groups, total_duplicates
