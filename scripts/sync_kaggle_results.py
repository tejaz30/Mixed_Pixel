#!/usr/bin/env python3
"""
Sync Kaggle Experiment Results to Local Repository

This script downloads results from a completed Kaggle notebook and organizes
them into the appropriate local directory structure.

Usage:
    python scripts/sync_kaggle_results.py --notebook-url <kaggle_notebook_url>
    
    Or for manual download:
    python scripts/sync_kaggle_results.py --manual --kaggle-zip <path_to_downloaded_zip>
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import zipfile


class KaggleResultsSyncer:
    """Handles syncing of Kaggle experiment results to local repository."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def setup_directories(self):
        """Ensure necessary directories exist."""
        dirs = [
            self.repo_root / "checkpoints" / "experiment1",
            self.repo_root / "outputs" / "experiment1",
            self.repo_root / "logs" / "experiment1",
            self.repo_root / "results" / "experiment1",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            
    def check_kaggle_cli(self) -> bool:
        """Check if Kaggle CLI is installed."""
        try:
            result = subprocess.run(
                ["kaggle", "--version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
            
    def download_from_kaggle(self, notebook_url: str) -> Path:
        """
        Download notebook output using Kaggle CLI.
        
        Args:
            notebook_url: URL of the Kaggle notebook (e.g., username/notebook-name)
        
        Returns:
            Path to downloaded zip file
        """
        if not self.check_kaggle_cli():
            print("❌ Kaggle CLI not found!")
            print("\nInstall with: pip install kaggle")
            print("Then configure: https://github.com/Kaggle/kaggle-api#api-credentials")
            sys.exit(1)
            
        # Parse notebook reference
        parts = notebook_url.strip("/").split("/")
        if len(parts) >= 2:
            username = parts[-2]
            notebook = parts[-1]
        else:
            print("❌ Invalid notebook URL format")
            print("Expected: username/notebook-name or full URL")
            sys.exit(1)
            
        print(f"📥 Downloading output from {username}/{notebook}...")
        
        # Download to temp directory
        temp_dir = self.repo_root / "temp_kaggle_download"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            cmd = [
                "kaggle", "kernels", "output",
                f"{username}/{notebook}",
                "-p", str(temp_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Download failed: {result.stderr}")
                sys.exit(1)
                
            print("✓ Download complete")
            return temp_dir
            
        except Exception as e:
            print(f"❌ Error downloading: {e}")
            sys.exit(1)
            
    def extract_manual_zip(self, zip_path: Path) -> Path:
        """Extract manually downloaded zip file."""
        if not zip_path.exists():
            print(f"❌ Zip file not found: {zip_path}")
            sys.exit(1)
            
        print(f"📦 Extracting {zip_path.name}...")
        
        temp_dir = self.repo_root / "temp_kaggle_download"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            print("✓ Extraction complete")
            return temp_dir
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            sys.exit(1)
            
    def organize_results(self, source_dir: Path):
        """Organize downloaded results into repository structure."""
        print("\n📁 Organizing results...")
        
        operations = []
        
        # Map Kaggle output structure to repo structure
        mappings = [
            # Checkpoints
            ("checkpoints/exp1_kl_no_mixed", "checkpoints/experiment1/kl_no_mixed"),
            ("checkpoints/exp1_vq_no_mixed", "checkpoints/experiment1/vq_no_mixed"),
            
            # Outputs (visualizations, etc.)
            ("outputs/exp1_kl_no_mixed", "outputs/experiment1/kl_no_mixed"),
            ("outputs/exp1_vq_no_mixed", "outputs/experiment1/vq_no_mixed"),
            
            # Logs
            ("logs/exp1_kl_no_mixed", "logs/experiment1/kl_no_mixed"),
            ("logs/exp1_vq_no_mixed", "logs/experiment1/vq_no_mixed"),
        ]
        
        for src_rel, dst_rel in mappings:
            src = source_dir / src_rel
            dst = self.repo_root / dst_rel
            
            if src.exists():
                dst.mkdir(parents=True, exist_ok=True)
                
                # Copy files
                for item in src.rglob('*'):
                    if item.is_file():
                        rel_path = item.relative_to(src)
                        dst_file = dst / rel_path
                        dst_file.parent.mkdir(parents=True, exist_ok=True)
                        
                        shutil.copy2(item, dst_file)
                        operations.append(f"  ✓ {dst_rel}/{rel_path}")
                        
        # Copy comparison CSV and visualizations to results directory
        result_files = [
            "kl_vs_vq_comparison.csv",
            "reconstruction_comparison.png",
        ]
        
        results_dir = self.repo_root / "results" / "experiment1"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        for filename in result_files:
            src_file = source_dir / filename
            if src_file.exists():
                # Add timestamp to avoid overwriting
                stem = src_file.stem
                suffix = src_file.suffix
                dst_file = results_dir / f"{stem}_{self.timestamp}{suffix}"
                
                shutil.copy2(src_file, dst_file)
                operations.append(f"  ✓ results/experiment1/{dst_file.name}")
                
        if operations:
            print("\nCopied files:")
            for op in operations[:20]:  # Show first 20
                print(op)
            if len(operations) > 20:
                print(f"  ... and {len(operations) - 20} more files")
        else:
            print("⚠️  No files found to copy. Check source directory structure.")
            
        return len(operations) > 0
        
    def create_experiment_summary(self, source_dir: Path):
        """Create a summary document of the experiment results."""
        print("\n📝 Creating experiment summary...")
        
        summary_path = self.repo_root / "results" / "experiment1" / f"summary_{self.timestamp}.md"
        
        # Try to load comparison CSV
        comparison_data = None
        csv_path = source_dir / "kl_vs_vq_comparison.csv"
        if csv_path.exists():
            try:
                import pandas as pd
                comparison_data = pd.read_csv(csv_path)
            except ImportError:
                print("  (pandas not available, skipping CSV parsing)")
                
        with open(summary_path, 'w') as f:
            f.write(f"# KL vs VQ Experiment Results\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Experiment:** First-Stage Autoencoder Comparison\n")
            f.write(f"**Dataset:** No_Mixed\n\n")
            
            f.write("## Experiment Configuration\n\n")
            f.write("- **Models:** KL-regularized autoencoder vs VQ-based autoencoder\n")
            f.write("- **Latent Channels:** 4\n")
            f.write("- **Compression:** 8x (3 downsampling stages)\n")
            f.write("- **Training Epochs:** 100\n")
            f.write("- **Batch Size:** 4\n")
            f.write("- **Learning Rate:** 1e-3\n\n")
            
            if comparison_data is not None:
                f.write("## Quantitative Results\n\n")
                f.write("| Metric | KL | VQ | Difference (KL - VQ) |\n")
                f.write("|--------|----|----|----------------------|\n")
                
                for _, row in comparison_data.iterrows():
                    f.write(f"| {row['Metric']} | {row['KL']} | {row['VQ']} | {row['Difference (KL - VQ)']} |\n")
                    
                f.write("\n")
                
            f.write("## Files\n\n")
            f.write("### Checkpoints\n")
            f.write("- KL: `checkpoints/experiment1/kl_no_mixed/`\n")
            f.write("- VQ: `checkpoints/experiment1/vq_no_mixed/`\n\n")
            
            f.write("### Outputs\n")
            f.write("- KL: `outputs/experiment1/kl_no_mixed/`\n")
            f.write("- VQ: `outputs/experiment1/vq_no_mixed/`\n\n")
            
            f.write("### Logs\n")
            f.write("- KL: `logs/experiment1/kl_no_mixed/`\n")
            f.write("- VQ: `logs/experiment1/vq_no_mixed/`\n\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. Review quantitative metrics\n")
            f.write("2. Examine reconstruction visualizations\n")
            f.write("3. Analyze boundary preservation (boundary_ratio)\n")
            f.write("4. Update `docs/CURRENT_STATE.md` with findings\n")
            f.write("5. Document decision in `docs/DECISIONS.md`\n")
            f.write("6. Plan next experiment based on results\n")
            
        print(f"✓ Summary created: {summary_path}")
        
    def cleanup(self, temp_dir: Path):
        """Clean up temporary download directory."""
        if temp_dir.exists():
            print("\n🧹 Cleaning up temporary files...")
            shutil.rmtree(temp_dir)
            print("✓ Cleanup complete")
            
    def update_gitignore(self):
        """Ensure large files are in .gitignore."""
        gitignore_path = self.repo_root / ".gitignore"
        
        patterns = [
            "# Large checkpoint files",
            "checkpoints/**/*.pth",
            "*.pth",
            "",
            "# Temporary download directory", 
            "temp_kaggle_download/",
        ]
        
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                current = f.read()
                
            # Check if patterns already exist
            missing = [p for p in patterns if p and p not in current]
            
            if missing:
                print("\n📝 Updating .gitignore...")
                with open(gitignore_path, 'a') as f:
                    f.write("\n")
                    f.write("\n".join(patterns))
                print("✓ .gitignore updated")


def main():
    parser = argparse.ArgumentParser(
        description="Sync Kaggle experiment results to local repository"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--notebook-url",
        help="Kaggle notebook URL or username/notebook-name"
    )
    group.add_argument(
        "--manual",
        action="store_true",
        help="Use manually downloaded zip file"
    )
    
    parser.add_argument(
        "--kaggle-zip",
        type=Path,
        help="Path to manually downloaded Kaggle output zip (use with --manual)"
    )
    
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep temporary download directory"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.manual and not args.kaggle_zip:
        parser.error("--manual requires --kaggle-zip")
        
    # Find repository root
    repo_root = Path(__file__).parent.parent
    if not (repo_root / ".git").exists():
        print("❌ Not in a git repository")
        sys.exit(1)
        
    print("="*60)
    print("Kaggle Results Sync")
    print("="*60)
    print(f"Repository: {repo_root}\n")
    
    syncer = KaggleResultsSyncer(repo_root)
    syncer.setup_directories()
    
    # Download or extract
    if args.manual:
        source_dir = syncer.extract_manual_zip(args.kaggle_zip)
    else:
        source_dir = syncer.download_from_kaggle(args.notebook_url)
        
    # Organize results
    success = syncer.organize_results(source_dir)
    
    if success:
        syncer.create_experiment_summary(source_dir)
        syncer.update_gitignore()
        
        print("\n" + "="*60)
        print("✅ Sync Complete!")
        print("="*60)
        print("\nResults are now in your local repository.")
        print("\nNext steps:")
        print("  1. Review results in results/experiment1/")
        print("  2. Check visualizations in outputs/experiment1/")
        print("  3. Update docs/CURRENT_STATE.md")
        print("  4. Commit relevant files (not .pth checkpoints)")
        print("\nTo commit:")
        print("  git add results/ outputs/ logs/")
        print('  git commit -m "Add KL vs VQ experiment results"')
    else:
        print("\n⚠️  No results found. Check Kaggle output structure.")
        
    # Cleanup
    if not args.no_cleanup:
        syncer.cleanup(source_dir)
        
    print()


if __name__ == "__main__":
    main()
