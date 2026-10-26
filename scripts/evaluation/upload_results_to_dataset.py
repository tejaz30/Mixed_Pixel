"""
Upload evaluation results to Kaggle Dataset

This script packages evaluation results and uploads them as a Kaggle Dataset.
Run this AFTER the evaluation completes in Kaggle.

Usage:
    python scripts/evaluation/upload_results_to_dataset.py --username YOUR_KAGGLE_USERNAME
"""

import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Upload evaluation results to Kaggle Dataset')
    parser.add_argument('--username', required=True, help='Your Kaggle username')
    parser.add_argument('--result_dir', default='/kaggle/working/results/experiment02', 
                       help='Results directory to upload')
    parser.add_argument('--dataset_name', default='kl-vq-evaluation-results',
                       help='Dataset slug name')
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    
    if not result_dir.exists():
        print(f"❌ Results directory not found: {result_dir}")
        print("Make sure evaluation has completed first!")
        return 1
    
    print("=" * 70)
    print("CREATING KAGGLE DATASET FOR RESULTS")
    print("=" * 70)
    
    # Create dataset metadata
    dataset_slug = f"{args.username}/{args.dataset_name}"
    
    dataset_metadata = {
        "title": "KL vs VQ Evaluation Results",
        "id": dataset_slug,
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["deep-learning", "autoencoder", "thermal-imaging", "evaluation"],
        "description": f"""# KL vs VQ Autoencoder Evaluation Results

**Experiment**: First-stage autoencoder comparison for LDM Stage 2 selection
**Generated**: {datetime.now().isoformat()}

## Contents

### KL Results (`kl/`)
- `kl_evaluation_results.json` - Complete metrics summary
- `posterior_means.npy` - Posterior mean distributions
- `posterior_logvars.npy` - Posterior log-variance distributions  
- `kl_divergences.npy` - KL divergence per sample
- `latent_distribution_samples.npy` - Latent samples for analysis

### VQ Results (`vq/`)
- `vq_evaluation_results.json` - Complete metrics summary
- `codebook_usage_counts.npy` - Codebook usage histogram
- `perplexity_trajectory.npy` - Perplexity across validation set
- `quantization_error_trajectory.npy` - Quantization error per sample
- `latent_distribution_samples.npy` - Latent samples for analysis

### Comparison (`comparison/`)
- `kl_vs_vq_comparison.json` - Full statistical comparison with recommendation

## Usage

```python
# Load comparison results
import json
from pathlib import Path

data_dir = Path('/kaggle/input/{args.dataset_name}')
with open(data_dir / 'comparison' / 'kl_vs_vq_comparison.json') as f:
    comparison = json.load(f)

print(comparison['summary']['key_findings'])
```
"""
    }
    
    # Save metadata
    metadata_file = result_dir / 'dataset-metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(dataset_metadata, f, indent=2)
    
    print(f"\n✓ Created metadata: {metadata_file}")
    print(f"\nDataset ID: {dataset_slug}")
    print(f"Title: {dataset_metadata['title']}")
    
    print("\n" + "=" * 70)
    print("UPLOADING TO KAGGLE DATASET")
    print("=" * 70)
    
    try:
        # Try to create new dataset
        print(f"\nAttempting to create dataset: {dataset_slug}")
        create_result = subprocess.run(
            ['kaggle', 'datasets', 'create', '-p', str(result_dir), '-r', 'zip'],
            capture_output=True,
            text=True,
            cwd=str(result_dir)
        )
        
        if create_result.returncode == 0:
            print("✅ Dataset created successfully!")
            print(create_result.stdout)
        elif 'already exists' in create_result.stderr.lower():
            # Dataset exists, update it instead
            print("Dataset already exists. Updating version...")
            update_result = subprocess.run(
                ['kaggle', 'datasets', 'version', '-p', str(result_dir), '-m', 
                 f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", '-r', 'zip'],
                capture_output=True,
                text=True,
                cwd=str(result_dir)
            )
            
            if update_result.returncode == 0:
                print("✅ Dataset updated successfully!")
                print(update_result.stdout)
            else:
                print(f"⚠️ Update failed: {update_result.stderr}")
                print_manual_instructions(dataset_slug, result_dir, dataset_metadata)
                return 1
        else:
            print(f"⚠️ Create failed: {create_result.stderr}")
            print_manual_instructions(dataset_slug, result_dir, dataset_metadata)
            return 1
    
    except FileNotFoundError:
        print("⚠️ Kaggle CLI not found")
        print_manual_instructions(dataset_slug, result_dir, dataset_metadata)
        return 1
    
    except Exception as e:
        print(f"⚠️ Error: {e}")
        print_manual_instructions(dataset_slug, result_dir, dataset_metadata)
        return 1
    
    print("\n" + "=" * 70)
    print("✅ UPLOAD COMPLETE")
    print("=" * 70)
    print(f"\n📊 View dataset at: https://www.kaggle.com/datasets/{dataset_slug}")
    print(f"📁 Local results: {result_dir}")
    
    return 0


def print_manual_instructions(dataset_slug, result_dir, metadata):
    """Print manual upload instructions"""
    print("\n📝 MANUAL UPLOAD INSTRUCTIONS:")
    print("=" * 70)
    print("1. Go to: https://www.kaggle.com/datasets")
    print("2. Click 'New Dataset' (or find existing and click 'New Version')")
    print(f"3. Upload all files from: {result_dir}")
    print(f"4. Set title: {metadata['title']}")
    print(f"5. Set slug: {dataset_slug.split('/')[-1]}")
    print("6. Click 'Create' or 'Update Version'")
    print("=" * 70)


if __name__ == '__main__':
    exit(main())
