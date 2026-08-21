"""
Quick script to check training status of all CAE models.

Usage:
    python scripts/check_training_status.py
"""

from pathlib import Path
from datetime import datetime
import json

# Dataset names
DATASETS = ['Chilli_Leaves', 'Okra', 'Paddy_Leaves', 'No_Mixed']

def check_status():
    """Check and display training status for all datasets."""
    
    print("\n" + "=" * 80)
    print("CAE TRAINING STATUS")
    print("=" * 80)
    print(f"Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    for i, dataset in enumerate(DATASETS, 1):
        print(f"[{i}/4] {dataset}")
        print("-" * 80)
        
        log_file = Path(f"logs/cae/{dataset}/train.log")
        checkpoint = Path(f"checkpoints/cae/{dataset}/best_model.pth")
        
        if not log_file.exists():
            print("  Status: ⏸️  NOT STARTED")
        else:
            # Read last few lines of log
            with open(log_file, 'r') as f:
                lines = f.readlines()
                last_lines = [line.strip() for line in lines[-3:] if line.strip()]
            
            if checkpoint.exists():
                print("  Status: ✅ COMPLETED")
                # Get checkpoint info
                checkpoint_size = checkpoint.stat().st_size / (1024 * 1024)
                print(f"  Model: {checkpoint} ({checkpoint_size:.2f} MB)")
            else:
                print("  Status: 🔄 TRAINING...")
            
            print("  Recent logs:")
            for line in last_lines:
                # Extract just the message part (after log level)
                if ' - ' in line:
                    parts = line.split(' - ', 3)
                    if len(parts) >= 4:
                        msg = parts[3]
                    else:
                        msg = line
                else:
                    msg = line
                print(f"    {msg}")
        
        print()
    
    # Summary
    completed = sum(1 for d in DATASETS if Path(f"checkpoints/cae/{d}/best_model.pth").exists())
    training = sum(1 for d in DATASETS if Path(f"logs/cae/{d}/train.log").exists() 
                   and not Path(f"checkpoints/cae/{d}/best_model.pth").exists())
    pending = len(DATASETS) - completed - training
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Completed: {completed}/{len(DATASETS)}")
    print(f"🔄 Training:  {training}/{len(DATASETS)}")
    print(f"⏸️  Pending:   {pending}/{len(DATASETS)}")
    print("=" * 80)
    print()
    
    if completed == len(DATASETS):
        print("🎉 ALL MODELS TRAINED!")
        print()
        print("Next steps:")
        print("  1. View training curves: tensorboard --logdir runs/cae")
        print("  2. Evaluate models on test set")
        print("  3. Generate reconstruction visualizations")
    elif training > 0:
        print("💡 To monitor in real-time:")
        print("   tensorboard --logdir runs/cae")
        print("   Then open: http://localhost:6006")
    
    print()

if __name__ == "__main__":
    check_status()
