# Add this as Cell 2.5 in Kaggle to debug the path issue

import os
from pathlib import Path

print("=" * 70)
print("DATASET PATH DEBUG")
print("=" * 70)

# Check what's in /kaggle/input
print("\n1. Available datasets in /kaggle/input:")
kaggle_input = Path('/kaggle/input')
if kaggle_input.exists():
    for item in kaggle_input.iterdir():
        print(f"   - {item.name}")
        # Check if it's a directory and list contents
        if item.is_dir():
            print(f"     Contents of {item.name}:")
            for subitem in item.iterdir():
                print(f"       - {subitem.name}")
                if subitem.is_dir():
                    # Count images in subdirectories
                    images = list(subitem.rglob('*.png')) + list(subitem.rglob('*.jpg'))
                    if images:
                        print(f"         -> {len(images)} images found")
else:
    print("   ❌ /kaggle/input does not exist (not running on Kaggle?)")

# Check what Cell 2 set
print(f"\n2. KAGGLE_DATASET_PATH variable: {KAGGLE_DATASET_PATH if 'KAGGLE_DATASET_PATH' in dir() else 'NOT SET'}")

# Check what's in the config after Cell 2 updated it
print(f"\n3. Config file after Cell 2 update:")
import yaml
with open(KL_CONFIG, 'r') as f:
    config = yaml.safe_load(f)
print(f"   dataset_root: {config['data']['dataset_root']}")

# Verify the path exists
dataset_path = Path(config['data']['dataset_root'])
print(f"\n4. Does the config path exist? {dataset_path.exists()}")
if dataset_path.exists():
    images = list(dataset_path.glob('*.png')) + list(dataset_path.glob('*.jpg'))
    print(f"   ✓ Found {len(images)} images")
else:
    print(f"   ❌ Path does not exist: {dataset_path}")
    
print("=" * 70)
