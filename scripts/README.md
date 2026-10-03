# Scripts Directory

Utility scripts for the Mixed-Pixel Detection project.

## 📁 Contents

### Kaggle Experiment Workflow

| File | Purpose | Platform |
|------|---------|----------|
| `validate_kaggle_ready.py` | Check if repo is ready for Kaggle | All |
| `sync_kaggle_results.py` | Download and organize Kaggle results | All |
| `sync_kaggle_results.bat` | Windows wrapper for sync script | Windows |
| `KAGGLE_QUICKSTART.md` | Quick reference guide | All |
| `README_KAGGLE_SYNC.md` | Detailed sync documentation | All |

---

## 🚀 Quick Start: Kaggle Experiment

### 1. Validate Before Upload

```bash
python scripts/validate_kaggle_ready.py
```

This checks:
- ✓ All required source files exist
- ✓ Configuration files are valid
- ✓ Notebook is ready
- ✓ Git status is clean

### 2. Run Experiment on Kaggle

See [`KAGGLE_QUICKSTART.md`](KAGGLE_QUICKSTART.md) for step-by-step instructions.

### 3. Sync Results Back

**Option A: Automatic**
```bash
python scripts/sync_kaggle_results.py --notebook-url YOUR_USERNAME/notebook-name
```

**Option B: Manual**
```bash
python scripts/sync_kaggle_results.py --manual --kaggle-zip path/to/download.zip
```

**Option C: Windows Batch**
```bash
scripts\sync_kaggle_results.bat
```

See [`README_KAGGLE_SYNC.md`](README_KAGGLE_SYNC.md) for detailed instructions.

---

## 📖 Documentation

- **[KAGGLE_QUICKSTART.md](KAGGLE_QUICKSTART.md)** - Quick reference checklist
- **[README_KAGGLE_SYNC.md](README_KAGGLE_SYNC.md)** - Comprehensive sync guide

---

## 🔧 Requirements

### For Validation
- Python 3.7+
- PyYAML (`pip install pyyaml`)

### For Automatic Sync
- Python 3.7+
- Kaggle CLI (`pip install kaggle`)
- Kaggle API credentials (see [setup guide](README_KAGGLE_SYNC.md#setup-kaggle-api))

### For Manual Sync
- Python 3.7+
- Downloaded Kaggle output zip file

---

## 💡 Typical Workflow

```
1. Develop locally
   ├─ Write code in src/
   ├─ Update configs/
   └─ Test with notebooks/environment_check.ipynb

2. Prepare for Kaggle
   ├─ python scripts/validate_kaggle_ready.py
   ├─ git commit & push
   └─ Upload No_Mixed dataset to Kaggle

3. Run on Kaggle
   ├─ Upload notebooks/kaggle_kl_vs_vq_experiment.ipynb
   ├─ Configure GPU and datasets
   └─ Run All (~3-4 hours)

4. Sync results back
   ├─ python scripts/sync_kaggle_results.py
   ├─ Review results/experiment1/
   └─ Update docs/

5. Analyze and decide
   ├─ Compare KL vs VQ metrics
   ├─ Update docs/CURRENT_STATE.md
   ├─ Document decision in docs/DECISIONS.md
   └─ git commit & push results
```

---

## 🎯 What Gets Synced

```
Kaggle Output                       → Local Repository
═══════════════════════════════════════════════════════════
checkpoints/exp1_kl_no_mixed/       → checkpoints/experiment1/kl_no_mixed/
checkpoints/exp1_vq_no_mixed/       → checkpoints/experiment1/vq_no_mixed/
outputs/exp1_kl_no_mixed/           → outputs/experiment1/kl_no_mixed/
outputs/exp1_vq_no_mixed/           → outputs/experiment1/vq_no_mixed/
logs/exp1_kl_no_mixed/              → logs/experiment1/kl_no_mixed/
logs/exp1_vq_no_mixed/              → logs/experiment1/vq_no_mixed/
kl_vs_vq_comparison.csv             → results/experiment1/comparison_TIMESTAMP.csv
reconstruction_comparison.png        → results/experiment1/reconstruction_TIMESTAMP.png
```

Plus generates: `results/experiment1/summary_TIMESTAMP.md`

---

## ⚠️ Important Notes

### Large Files

Checkpoint files (`.pth`) are automatically excluded from git via `.gitignore`.

**Do commit:**
- Comparison CSVs
- Visualizations (PNG)
- Summary markdown files
- Logs (if small)

**Don't commit:**
- Model checkpoints (`.pth` files)
- Large training logs

### File Sizes

Typical experiment results:
- Checkpoints: ~500 MB - 2 GB (excluded from git)
- Outputs: ~10-50 MB (included in git)
- Logs: ~1-10 MB (included in git)

---

## 🆘 Troubleshooting

### Validation Fails

```bash
# Check what's wrong
python scripts/validate_kaggle_ready.py

# Fix issues, then validate again
```

### Sync Issues

See detailed troubleshooting in [`README_KAGGLE_SYNC.md`](README_KAGGLE_SYNC.md#-troubleshooting).

Common fixes:
- Install Kaggle CLI: `pip install kaggle`
- Check dataset name matches
- Verify Kaggle output exists
- Try manual download method

---

## 📚 Additional Resources

- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [Project AGENTS.md](../AGENTS.md)
- [Experiment Specifications](../.kiro/specs/02-kl-vs-vq/)
- [Current State](../docs/CURRENT_STATE.md)

---

## 🔄 Future Scripts

Planned utilities:
- `compare_experiments.py` - Compare multiple experiment runs
- `checkpoint_analyzer.py` - Analyze checkpoint quality
- `dataset_validator.py` - Validate dataset structure
- `config_generator.py` - Generate experiment configs

---

## 🤝 Contributing

When adding new scripts:
1. Add docstring explaining purpose
2. Include usage examples
3. Update this README
4. Add to validation checks if applicable
5. Test on both Windows and Linux if possible
