"""
Complete runner for Experiment 1: KL vs VQ first-stage comparison.

This script runs the entire experiment pipeline:
1. Train KL autoencoder
2. Train VQ autoencoder
3. Evaluate both models
4. Generate comparison report

Usage:
    python scripts/run_experiment1.py --train --evaluate --compare
    python scripts/run_experiment1.py --evaluate-only  # Skip training
"""

import sys
import argparse
import subprocess
from pathlib import Path
import time

project_root = Path(__file__).parent.parent


def run_command(cmd: list, description: str) -> bool:
    """Run a command and report success/failure."""
    print("\n" + "="*70)
    print(f"STEP: {description}")
    print("="*70)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        elapsed = time.time() - start_time
        print(f"\n✓ {description} completed in {elapsed:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {description} failed after {elapsed:.2f}s")
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 1")
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation')
    parser.add_argument('--compare', action='store_true', help='Run comparison')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    parser.add_argument('--evaluate-only', action='store_true', help='Skip training, only evaluate')
    parser.add_argument('--skip-kl', action='store_true', help='Skip KL model')
    parser.add_argument('--skip-vq', action='store_true', help='Skip VQ model')
    args = parser.parse_args()
    
    # Default to all if no flags specified
    if not any([args.train, args.evaluate, args.compare, args.all, args.evaluate_only]):
        args.all = True
    
    if args.all:
        args.train = True
        args.evaluate = True
        args.compare = True
    
    if args.evaluate_only:
        args.evaluate = True
        args.compare = True
    
    print("="*70)
    print("EXPERIMENT 1: KL vs VQ FIRST-STAGE AUTOENCODER COMPARISON")
    print("="*70)
    print("\nExperiment Steps:")
    if args.train and not args.skip_kl:
        print("  1. Train KL autoencoder")
    if args.train and not args.skip_vq:
        print("  2. Train VQ autoencoder")
    if args.evaluate:
        print("  3. Evaluate models")
    if args.compare:
        print("  4. Generate comparison report")
    print()
    
    input("Press Enter to start...")
    
    start_time_total = time.time()
    
    # Training
    if args.train:
        if not args.skip_kl:
            success = run_command(
                ['python', 'scripts/train_first_stage.py',
                 '--config', 'configs/experiment1/kl_autoencoder.yaml'],
                "Training KL Autoencoder"
            )
            if not success:
                print("\nTraining failed. Aborting experiment.")
                return 1
        
        if not args.skip_vq:
            success = run_command(
                ['python', 'scripts/train_first_stage.py',
                 '--config', 'configs/experiment1/vq_autoencoder.yaml'],
                "Training VQ Autoencoder"
            )
            if not success:
                print("\nTraining failed. Aborting experiment.")
                return 1
    
    # Evaluation
    if args.evaluate:
        if not args.skip_kl:
            # Check if checkpoint exists
            kl_checkpoint = project_root / 'checkpoints/experiment1/kl/best_model.pth'
            if not kl_checkpoint.exists():
                print(f"\n✗ KL checkpoint not found: {kl_checkpoint}")
                print("  Run training first or use --skip-kl")
                return 1
            
            success = run_command(
                ['python', 'scripts/evaluate_first_stage.py',
                 '--config', 'configs/experiment1/kl_autoencoder.yaml',
                 '--checkpoint', str(kl_checkpoint)],
                "Evaluating KL Autoencoder"
            )
            if not success:
                print("\nEvaluation failed. Continuing anyway...")
        
        if not args.skip_vq:
            # Check if checkpoint exists
            vq_checkpoint = project_root / 'checkpoints/experiment1/vq/best_model.pth'
            if not vq_checkpoint.exists():
                print(f"\n✗ VQ checkpoint not found: {vq_checkpoint}")
                print("  Run training first or use --skip-vq")
                return 1
            
            success = run_command(
                ['python', 'scripts/evaluate_first_stage.py',
                 '--config', 'configs/experiment1/vq_autoencoder.yaml',
                 '--checkpoint', str(vq_checkpoint)],
                "Evaluating VQ Autoencoder"
            )
            if not success:
                print("\nEvaluation failed. Continuing anyway...")
    
    # Comparison
    if args.compare:
        # Check if evaluation results exist
        kl_results = project_root / 'outputs/experiment1/kl/evaluation_results.json'
        vq_results = project_root / 'outputs/experiment1/vq/evaluation_results.json'
        
        if not kl_results.exists() or not vq_results.exists():
            print("\n✗ Evaluation results not found. Run evaluation first.")
            return 1
        
        success = run_command(
            ['python', 'scripts/compare_experiment1.py'],
            "Generating Comparison Report"
        )
        if not success:
            print("\nComparison failed.")
            return 1
    
    elapsed_total = time.time() - start_time_total
    
    print("\n" + "="*70)
    print("EXPERIMENT 1 COMPLETE")
    print("="*70)
    print(f"Total time: {elapsed_total:.2f}s ({elapsed_total/60:.2f}m)")
    print("\nResults available at:")
    print("  - KL outputs: outputs/experiment1/kl/")
    print("  - VQ outputs: outputs/experiment1/vq/")
    print("  - Comparison: outputs/experiment1/comparison/")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
