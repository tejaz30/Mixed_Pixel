# Kaggle Results Sync Guide

This guide explains how to sync experiment results from Kaggle back to your local repository.

## 📋 Prerequisites

- Python 3.7+
- Kaggle API credentials (for automatic download)
- Your experiment completed on Kaggle

---

## 🚀 Quick Start

### Method 1: Automatic Download (Recommended)

**Setup Kaggle API (First Time Only):**

1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New API Token"
4. Save `kaggle.json` to:
   - Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - Mac/Linux: `~/.kaggle/kaggle.json`

5. Install Kaggle CLI:
   ```bash
   pip install kaggle
   ```

**Run Sync:**

```bash
# Windows
scripts\sync_kaggle_results.bat

# Mac/Linux
python scripts/sync_kaggle_results.py --notebook-url YOUR_USERNAME/your-notebook-name
```

When prompted, enter your notebook URL or just `username/notebook-name`.

---

### Method 2: Manual Download

**Download Results:**

1. Open your completed Kaggle notebook
2. Click the **⋮** menu (top-right)
3. Select **"Download Output"**
4. Save the zip file to your computer

**Run Sync:**

```bash
# Windows
scripts\sync_kaggle_results.bat
# Choose option 2, then enter path to zip

# Mac/Linux
python scripts/sync_kaggle_results.py --manual --kaggle-zip path/to/downloaded.zip
```

---

## 📁 What Gets Synced

The script organizes Kaggle output into your repository:

### From Kaggle → To Local Repository:

```
/kaggle/working/
├── checkpoints/
│   ├── exp1_kl_no_mixed/          → checkpoints/experiment1/kl_no_mixed/
│   └── exp1_vq_no_mixed/          → checkpoints/experiment1/vq_no_mixed/
│
├── outputs/
│   ├── exp1_kl_no_mixed/          → outputs/experiment1/kl_no_mixed/
│   └── exp1_vq_no_mixed/          → outputs/experiment1/vq_no_mixed/
│
├── logs/
│   ├── exp1_kl_no_mixed/          → logs/experiment1/kl_no_mixed/
│   └── exp1_vq_no_mixed/          → logs/experiment1/vq_no_mixed/
│
├── kl_vs_vq_comparison.csv        → results/experiment1/comparison_TIMESTAMP.csv
└── reconstruction_comparison.png   → results/experiment1/reconstruction_TIMESTAMP.png
```

---

## 📊 After Sync

The script creates:

1. **Organized directories** with your experiment results
2. **Timestamped summary** at `results/experiment1/summary_TIMESTAMP.md`
3. **Updated .gitignore** to exclude large checkpoint files

### Next Steps:

1. **Review Results:**
   ```bash
   # View comparison
   cat results/experiment1/comparison_*.csv
   
   # Open visualizations
   start outputs/experiment1/kl_no_mixed/
   start outputs/experiment1/vq_no_mixed/
   ```

2. **Analyze Metrics:**
   - Compare MSE, SSIM, PSNR between KL and VQ
   - Check boundary preservation ratio
   - Examine reconstruction quality visually

3. **Update Documentation:**
   ```bash
   # Update project state
   code docs/CURRENT_STATE.md
   
   # Document decision
   code docs/DECISIONS.md
   ```

4. **Commit Results** (exclude large checkpoints):
   ```bash
   git add results/ outputs/ logs/
   git add docs/CURRENT_STATE.md docs/DECISIONS.md
   git commit -m "Add KL vs VQ experiment results from Kaggle"
   git push
   ```

---

## 🔧 Advanced Options

### Keep Temporary Files

```bash
python scripts/sync_kaggle_results.py --notebook-url user/notebook --no-cleanup
```

### Specify Custom Paths

Edit `sync_kaggle_results.py` to customize the mapping:

```python
mappings = [
    ("source_path", "destination_path"),
    # Add custom mappings here
]
```

---

## ⚠️ Troubleshooting

### "Kaggle CLI not found"

```bash
pip install kaggle
# Then configure kaggle.json as described above
```

### "No files found to copy"

**Check Kaggle output structure:**
- Ensure your notebook saved outputs to `/kaggle/working/`
- Verify checkpoint/output directories were created
- Check that training completed successfully

### "Access denied" errors on Windows

Run Command Prompt or PowerShell as Administrator.

### Large files in Git

The script updates `.gitignore` to exclude `.pth` checkpoint files automatically.

If you accidentally committed large files:
```bash
git rm --cached checkpoints/**/*.pth
git commit -m "Remove large checkpoint files"
```

---

## 📝 Manual Sync (Alternative)

If the script doesn't work, you can manually organize:

1. Download Kaggle output
2. Extract zip file
3. Copy to repository:
   ```
   checkpoints/ → your_repo/checkpoints/experiment1/
   outputs/     → your_repo/outputs/experiment1/
   logs/        → your_repo/logs/experiment1/
   *.csv, *.png → your_repo/results/experiment1/
   ```

---

## 🆘 Support

If you encounter issues:

1. Check that the Kaggle notebook completed successfully
2. Verify output files exist in Kaggle's Output tab
3. Ensure you have write permissions to the repository
4. Check Python and package versions

For Kaggle API issues, see: https://github.com/Kaggle/kaggle-api

---

## 📚 Related Documentation

- [Kaggle Notebook Setup](../notebooks/README.md) _(if exists)_
- [Experiment 1 Design](.kiro/specs/02-kl-vs-vq/design.md)
- [Current Project State](../docs/CURRENT_STATE.md)

