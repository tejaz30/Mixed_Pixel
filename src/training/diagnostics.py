"""
Training diagnostics and crash investigation.
"""

import time
import torch
import gc
from pathlib import Path
from typing import Optional, Dict, List, Tuple


class TrainingDiagnostics:
    """Monitor training for memory leaks and crashes."""
    
    def __init__(self, log_path: Optional[Path] = None, enabled: bool = True):
        """
        Initialize diagnostics.
        
        Args:
            log_path: Path to write diagnostic log (None = console only)
            enabled: Whether diagnostics are enabled
        """
        self.enabled = enabled
        if not enabled:
            return
            
        self.log_path = log_path
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # Clear existing log
            if log_path.exists():
                log_path.unlink()
        
        self.start_time = time.time()
        self.epoch_memories: List[Tuple[int, float, float, float]] = []
        self.batch_memories: List[Tuple[int, int, float]] = []
    
    def log(self, message: str) -> None:
        """Write message to log and console."""
        if not self.enabled:
            return
            
        timestamp = time.time() - self.start_time
        log_msg = f"[{timestamp:.1f}s] {message}"
        print(log_msg)
        
        if self.log_path:
            with open(self.log_path, 'a') as f:
                f.write(log_msg + '\n')
    
    def check_memory(self, location: str) -> Tuple[float, float, float]:
        """
        Check GPU memory at specific location.
        
        Args:
            location: Description of where this check is happening
        
        Returns:
            Tuple of (allocated_gb, reserved_gb, free_gb)
        """
        if not self.enabled or not torch.cuda.is_available():
            return 0.0, 0.0, 0.0
        
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = total - allocated
        
        self.log(f"MEMORY [{location}]: "
                f"Allocated={allocated:.2f}GB, "
                f"Reserved={reserved:.2f}GB, "
                f"Free={free:.2f}GB")
        
        return allocated, reserved, free
    
    def check_batch_data(self, batch: torch.Tensor, batch_idx: int) -> bool:
        """
        Check if batch data has issues (NaN, Inf, invalid range).
        
        Args:
            batch: Batch tensor
            batch_idx: Batch index
        
        Returns:
            True if data is valid, False if issues found
        """
        if not self.enabled:
            return True
        
        # Check for NaN or Inf
        has_nan = torch.isnan(batch).any().item()
        has_inf = torch.isinf(batch).any().item()
        
        # Check value range
        img_min = batch.min().item()
        img_max = batch.max().item()
        
        if has_nan or has_inf or img_min < -1 or img_max > 2:
            self.log(f"⚠️  BATCH {batch_idx} DATA ISSUE: "
                    f"nan={has_nan}, inf={has_inf}, "
                    f"min={img_min:.4f}, max={img_max:.4f}")
            return False
        
        return True
    
    def epoch_start(self, epoch: int) -> None:
        """Log epoch start and check memory."""
        if not self.enabled:
            return
            
        self.log(f"\n{'='*70}")
        self.log(f"EPOCH {epoch} START")
        self.log(f"{'='*70}")
        self.check_memory(f"epoch_{epoch}_start")
        
        # Cleanup before epoch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def epoch_end(self, epoch: int, train_loss: float, val_loss: float) -> None:
        """Log epoch end and check for memory growth."""
        if not self.enabled:
            return
            
        allocated, reserved, free = self.check_memory(f"epoch_{epoch}_end")
        self.epoch_memories.append((epoch, allocated, reserved, free))
        
        self.log(f"EPOCH {epoch} END: "
                f"train_loss={train_loss:.6f}, "
                f"val_loss={val_loss:.6f}")
        
        # Check for memory growth
        if len(self.epoch_memories) > 1:
            prev_alloc = self.epoch_memories[-2][1]
            growth = allocated - prev_alloc
            self.log(f"Memory growth since last epoch: {growth:.3f}GB")
            
            if growth > 0.5:
                self.log(f"⚠️  WARNING: Large memory growth detected!")
    
    def batch_checkpoint(self, epoch: int, batch_idx: int, loss_value: float) -> None:
        """Log every N batches."""
        if not self.enabled or batch_idx % 100 != 0:
            return
            
        allocated, _, _ = self.check_memory(f"epoch_{epoch}_batch_{batch_idx}")
        self.batch_memories.append((epoch, batch_idx, allocated))
        self.log(f"Batch {batch_idx}: loss={loss_value:.6f}")
    
    def critical_batch_check(
        self,
        epoch: int,
        batch_idx: int,
        batch: torch.Tensor,
        loss_value: float,
        critical_range: Tuple[int, int] = (735, 745)
    ) -> None:
        """
        Detailed check around critical batch range (e.g., crash zone).
        
        Args:
            epoch: Current epoch
            batch_idx: Current batch index
            batch: Batch tensor
            loss_value: Current loss value
            critical_range: Range of batch indices to monitor closely
        """
        if not self.enabled:
            return
            
        start, end = critical_range
        if start <= batch_idx <= end:
            self.log(f"\n🔍 CRITICAL BATCH {batch_idx} (CRASH ZONE)")
            self.check_memory(f"critical_batch_{batch_idx}")
            self.check_batch_data(batch, batch_idx)
            self.log(f"Loss: {loss_value:.6f}")
            
            # Force cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.log(f"Cleanup performed")
    
    def save_report(self) -> None:
        """Save final diagnostic report."""
        if not self.enabled or not self.log_path:
            return
            
        report_path = self.log_path.parent / 'diagnostic_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("TRAINING DIAGNOSTIC REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("EPOCH MEMORY PROGRESSION:\n")
            for epoch, alloc, res, free in self.epoch_memories:
                f.write(f"  Epoch {epoch}: "
                       f"{alloc:.2f}GB allocated, "
                       f"{free:.2f}GB free\n")
            
            f.write("\n\nBATCH MEMORY SAMPLES (last 20):\n")
            for epoch, batch, alloc in self.batch_memories[-20:]:
                f.write(f"  Epoch {epoch}, Batch {batch}: {alloc:.2f}GB\n")
        
        self.log(f"\n📊 Report saved to: {report_path}")
