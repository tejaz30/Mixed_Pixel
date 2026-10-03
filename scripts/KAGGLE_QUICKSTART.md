# Kaggle Experiment - Quick Reference

## 🎯 Step-by-Step Checklist

### 1️⃣ Prepare (Done Once)
- [ ] Push latest code to GitHub
- [ ] Upload No_Mixed dataset to Kaggle
- [ ] Set up Kaggle API credentials (optional, for auto-download)

### 2️⃣ Setup Kaggle Notebook
- [ ] Create new Kaggle notebook
- [ ] Enable GPU (T4 or **T4 x2 for faster training**)
- [ ] Add No_Mixed dataset
- [ ] Upload `notebooks/kaggle_kl_vs_vq_experiment.ipynb`

### 3️⃣ Configure Notebook
- [ ] Update `DATASET_NAME` variable (Cell 2.2) to match your uploaded dataset
- [ ] Update Git clone URL (Cell 1.2) to your repository
- [ ] Verify GPU is enabled

### 4️⃣ Run Experiment
- [ ] Click "Run All" or run cells sequentially
- [ ] Monitor training progress (~3-4 hours on T4)
- [ ] Check for errors in output

### 5️⃣ Download Results

**Option A: Automatic (Kaggle CLI)**
```bash
python scripts/sync_kaggle_results.py --notebook-url YOUR_USERNAME/notebook-name
```

**Option B: Manual**
1. In Kaggle: ⋮ menu → "Download Output"
2. Save zip file
3. Run:
   ```bash
   python scripts/sync_kaggle_results.py --manual --kaggle-zip path/to/file.zip
   ```

**Option C: Windows Batch**
```bash
scripts\sync_kaggle_results.bat
```

### 6️⃣ Review Results
- [ ] Check `results/experiment1/summary_*.md`
- [ ] Review comparison CSV
- [ ] Examine reconstruction visualizations
- [ ] Compare metrics: MSE, SSIM, boundary_ratio

### 7️⃣ Document Findings
- [ ] Update `docs/CURRENT_STATE.md` with results
- [ ] Record decision in `docs/DECISIONS.md`
- [ ] Note any issues in `docs/KNOWN_ISSUES.md`

### 8️⃣ Commit to Repository
```bash
git add results/ outputs/ logs/
git add docs/CURRENT_STATE.md docs/DECISIONS.md
git commit -m "Add KL vs VQ experiment results"
git push
```

---

## 🔑 Key Information

### Dataset Name in Kaggle
You uploaded: `_________________` (fill this in!)

Update in notebook Cell 2.2:
```python
DATASET_NAME = 'your-dataset-name-here'
```

### Repository URL
```
https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

Update in notebook Cell 1.2.

---

## ⏱️ Expected Timeline

| Task | Single GPU (T4) | Dual GPU (T4 x2) |
|------|-----------------|------------------|
| Setup notebook | 5 min | 5 min |
| KL training | 1.5-2 hrs | ~1 hr |
| VQ training | 1.5-2 hrs | ~1 hr |
| Evaluation | 10-15 min | 10-15 min |
| **Total** | **~3-4 hrs** | **~2-2.5 hrs** |

**Note:** Multi-GPU training is automatically enabled when you select "GPU T4 x2" in Kaggle settings!

---

## 🎓 Understanding Results

### Key Metrics to Compare:

1. **MSE (Lower is better)**
   - Pixel-level reconstruction accuracy

2. **SSIM (Higher is better, 0-1)**
   - Structural similarity

3. **Boundary Ratio (Close to 1.0 is neutral)**
   - `boundary_mse / non_boundary_mse`
   - Higher = boundaries harder to reconstruct
   - Lower = boundaries easier than interior

4. **PSNR (Higher is better)**
   - Peak signal-to-noise ratio

### VQ-Specific Metrics:

- **Codebook Utilization** (0-1)
  - What fraction of codebook vectors are used
  - Low (<0.5) = codebook collapse
  
- **Perplexity**
  - Effective codebook size
  - Should be reasonably high

---

## ❓ Quick Troubleshooting

### Notebook Issues

**"Dataset not found"**
- Check `/kaggle/input/` contents (Cell 1.1)
- Update `DATASET_NAME` to match actual name

**"Module not found"**
- Ensure repo was cloned successfully
- Check `src/` directory exists

**"CUDA out of memory"**
- Reduce `batch_size` in configs (try 2 or 3)
- Restart kernel and run again

### Sync Issues

**"Kaggle CLI not found"**
```bash
pip install kaggle
```

**"No files found to copy"**
- Check Kaggle Output tab has files
- Verify training completed successfully
- Try manual download method

**"Permission denied"**
- Ensure you have write access to repo directory
- On Windows, try running as Administrator

---

## 📞 Need Help?

1. Check `scripts/README_KAGGLE_SYNC.md` for detailed guide
2. Review notebook output for error messages
3. Verify GPU was available during training
4. Check that datasets loaded correctly

---

## ✅ Success Indicators

You know it worked when:
- ✓ Training completed 100 epochs for both models
- ✓ Checkpoints saved (best_model.pth exists)
- ✓ Evaluation metrics generated
- ✓ Comparison CSV shows differences between KL and VQ
- ✓ Visualizations show reconstructions
- ✓ Files synced to local repository

---

## 🚀 What's Next?

After this experiment:
1. Analyze which model (KL or VQ) performs better
2. Decide on first-stage architecture for LDM
3. Plan next experiment:
   - If one is clearly better → use for LDM diffusion stage
   - If results are mixed → deeper investigation or ablation
   - If both are poor → revisit architecture or dataset

Document your decision and reasoning!
