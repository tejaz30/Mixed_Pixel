#!/usr/bin/env python3
"""
Validate that the repository is ready for Kaggle experiment.

This script checks that all necessary files and configurations are in place
before uploading to Kaggle.

Usage:
    python scripts/validate_kaggle_ready.py
"""

from pathlib import Path
import yaml
import sys


class KaggleReadinessValidator:
    """Validates repository is ready for Kaggle experiment."""
    
    def __init__(self):
        self.repo_root = Path(__file__).parent.parent
        self.errors = []
        self.warnings = []
        self.info = []
        
    def check_file_exists(self, path: Path, description: str, required: bool = True):
        """Check if a file exists."""
        if path.exists():
            self.info.append(f"✓ {description}: {path.relative_to(self.repo_root)}")
            return True
        else:
            msg = f"✗ {description} not found: {path.relative_to(self.repo_root)}"
            if required:
                self.errors.append(msg)
            else:
                self.warnings.append(msg)
            return False
            
    def check_directory_exists(self, path: Path, description: str):
        """Check if a directory exists."""
        if path.exists() and path.is_dir():
            self.info.append(f"✓ {description}: {path.relative_to(self.repo_root)}")
            return True
        else:
            self.warnings.append(f"⚠ {description} not found: {path.relative_to(self.repo_root)}")
            return False
            
    def validate_config_file(self, config_path: Path, model_type: str):
        """Validate a configuration file."""
        if not self.check_file_exists(config_path, f"{model_type} config"):
            return False
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Check essential fields
            required_fields = [
                ('experiment', 'name'),
                ('model', 'type'),
                ('data', 'dataset_root'),
                ('training', 'max_epochs'),
                ('paths', 'checkpoint_dir'),
            ]
            
            for *path, key in required_fields:
                current = config
                for p in path:
                    if p not in current:
                        self.errors.append(f"✗ {model_type} config missing field: {'.'.join(path)}.{key}")
                        return False
                    current = current[p]
                    
                if key not in current:
                    self.errors.append(f"✗ {model_type} config missing field: {'.'.join(path)}.{key}")
                    return False
                    
            self.info.append(f"  ✓ {model_type} config structure valid")
            
            # Check dataset path
            dataset_root = config['data']['dataset_root']
            if dataset_root.startswith('/'):
                self.warnings.append(f"  ⚠ {model_type} config has absolute path: {dataset_root}")
                self.warnings.append(f"    This will need to be updated in Kaggle")
            else:
                self.info.append(f"  ✓ {model_type} config has relative path")
                
            return True
            
        except yaml.YAMLError as e:
            self.errors.append(f"✗ {model_type} config is not valid YAML: {e}")
            return False
        except Exception as e:
            self.errors.append(f"✗ Error reading {model_type} config: {e}")
            return False
            
    def validate_source_modules(self):
        """Check that required source modules exist."""
        print("\nChecking Source Modules...")
        
        required_modules = [
            ("src/models/ldm/first_stage/kl/model.py", "KL autoencoder model"),
            ("src/models/ldm/first_stage/vq/model.py", "VQ autoencoder model"),
            ("src/scripts/training/train_first_stage.py", "First-stage training script"),
            ("src/scripts/evaluation/evaluate_first_stage.py", "Evaluation script"),
        ]
        
        all_found = True
        for path_str, desc in required_modules:
            path = self.repo_root / path_str
            if not self.check_file_exists(path, desc):
                all_found = False
                
        # Check for __init__.py files
        init_files = [
            "src/__init__.py",
            "src/models/__init__.py",
            "src/models/ldm/__init__.py",
            "src/models/ldm/first_stage/__init__.py",
            "src/models/ldm/first_stage/kl/__init__.py",
            "src/models/ldm/first_stage/vq/__init__.py",
        ]
        
        for init_path in init_files:
            path = self.repo_root / init_path
            if not path.exists():
                self.warnings.append(f"⚠ __init__.py missing: {init_path}")
                self.warnings.append(f"  Module imports may fail in Kaggle")
                
        return all_found
        
    def validate_configs(self):
        """Check configuration files."""
        print("\nChecking Configuration Files...")
        
        kl_config = self.repo_root / "configs/experiment1/kl_autoencoder.yaml"
        vq_config = self.repo_root / "configs/experiment1/vq_autoencoder.yaml"
        
        kl_valid = self.validate_config_file(kl_config, "KL")
        vq_valid = self.validate_config_file(vq_config, "VQ")
        
        return kl_valid and vq_valid
        
    def validate_notebook(self):
        """Check Kaggle notebook exists."""
        print("\nChecking Kaggle Notebook...")
        
        notebook_path = self.repo_root / "notebooks/kaggle_kl_vs_vq_experiment.ipynb"
        return self.check_file_exists(notebook_path, "Kaggle experiment notebook")
        
    def validate_git_status(self):
        """Check git status."""
        print("\nChecking Git Status...")
        
        git_dir = self.repo_root / ".git"
        if not git_dir.exists():
            self.errors.append("✗ Not a git repository")
            return False
            
        self.info.append("✓ Git repository found")
        
        # Check if there are uncommitted changes
        import subprocess
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_root,
                capture_output=True,
                text=True
            )
            
            if result.stdout.strip():
                self.warnings.append("⚠ You have uncommitted changes")
                self.warnings.append("  Consider committing before running on Kaggle")
            else:
                self.info.append("✓ Working directory is clean")
                
        except Exception as e:
            self.warnings.append(f"⚠ Could not check git status: {e}")
            
        return True
        
    def check_dependencies(self):
        """Check Python dependencies."""
        print("\nChecking Python Dependencies...")
        
        requirements_file = self.repo_root / "requirements.txt"
        if self.check_file_exists(requirements_file, "Requirements file", required=False):
            try:
                with open(requirements_file, 'r') as f:
                    reqs = f.read()
                    
                essential = ['torch', 'numpy', 'pillow', 'pyyaml']
                missing = [pkg for pkg in essential if pkg not in reqs.lower()]
                
                if missing:
                    self.warnings.append(f"⚠ requirements.txt may be missing: {', '.join(missing)}")
                else:
                    self.info.append("✓ Essential packages listed in requirements.txt")
                    
            except Exception as e:
                self.warnings.append(f"⚠ Could not read requirements.txt: {e}")
        else:
            self.warnings.append("⚠ No requirements.txt found")
            self.warnings.append("  Kaggle will use default packages only")
            
    def print_checklist(self):
        """Print pre-upload checklist."""
        print("\n" + "="*60)
        print("Pre-Upload Checklist")
        print("="*60)
        
        checklist = [
            "[ ] Latest code committed to git",
            "[ ] Code pushed to GitHub",
            "[ ] No_Mixed dataset uploaded to Kaggle",
            "[ ] Kaggle API configured (for auto-download)",
            "[ ] Updated DATASET_NAME in notebook",
            "[ ] Updated Git clone URL in notebook",
        ]
        
        for item in checklist:
            print(item)
            
    def run_validation(self):
        """Run all validation checks."""
        print("="*60)
        print("Kaggle Readiness Validation")
        print("="*60)
        
        # Run checks
        self.validate_configs()
        self.validate_source_modules()
        self.validate_notebook()
        self.validate_git_status()
        self.check_dependencies()
        
        # Print results
        print("\n" + "="*60)
        print("Validation Results")
        print("="*60)
        
        if self.info:
            print(f"\n[PASSED] Checks ({len(self.info)}):")
            for msg in self.info[:10]:  # Show first 10
                print(f"  {msg}")
            if len(self.info) > 10:
                print(f"  ... and {len(self.info) - 10} more")
                
        if self.warnings:
            print(f"\n[WARNING] Warnings ({len(self.warnings)}):")
            for msg in self.warnings:
                print(f"  {msg}")
                
        if self.errors:
            print(f"\n[ERROR] Errors ({len(self.errors)}):")
            for msg in self.errors:
                print(f"  {msg}")
                
        # Summary
        print("\n" + "="*60)
        if self.errors:
            print("[FAIL] VALIDATION FAILED")
            print("="*60)
            print("Please fix errors before uploading to Kaggle.")
            return False
        elif self.warnings:
            print("[PASS] VALIDATION PASSED WITH WARNINGS")
            print("="*60)
            print("You can proceed, but review warnings first.")
            self.print_checklist()
            return True
        else:
            print("[PASS] VALIDATION PASSED")
            print("="*60)
            print("Repository is ready for Kaggle!")
            self.print_checklist()
            return True


def main():
    validator = KaggleReadinessValidator()
    success = validator.run_validation()
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    
    if success:
        print("1. Upload notebook to Kaggle")
        print("2. Enable GPU and add No_Mixed dataset")
        print("3. Update DATASET_NAME and Git URL in notebook")
        print("4. Run experiment")
        print("5. Use sync_kaggle_results.py to download results")
        print("\nSee scripts/KAGGLE_QUICKSTART.md for detailed guide.")
    else:
        print("1. Fix validation errors")
        print("2. Run this script again")
        print("3. Commit fixes to git")
        
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
