# Mixed-Pixel Detection in Agricultural Thermal Imagery

A research project investigating deep-learning architectures for unsupervised mixed-pixel detection in agricultural thermal imagery.

## 🎯 Project Status

**Current Phase:** Experiment 1 - KL vs VQ First-Stage Autoencoder Comparison

**Completed:**
- ✅ Preprocessing pipeline for datasets
- ✅ CAE baseline model and training scripts
- ✅ CAE models trained on four datasets
- ✅ KL and VQ autoencoder implementations
- ✅ Kaggle experiment notebook and sync infrastructure

**In Progress:**
- 🔄 Running KL vs VQ comparison on Kaggle

---

## 🚀 Quick Start

### For Kaggle Experiments

1. **Validate repository:**
   ```bash
   python scripts/validate_kaggle_ready.py
   ```

2. **Run experiment on Kaggle:**
   - Enable GPU T4 x2 for faster training (multi-GPU automatically enabled)
   - See [`scripts/KAGGLE_QUICKSTART.md`](scripts/KAGGLE_QUICKSTART.md)

3. **Sync results:**
   ```bash
   python scripts/sync_kaggle_results.py --notebook-url YOUR_USERNAME/notebook
   ```

### For Local Development

See [`AGENTS.md`](AGENTS.md) for development guidelines and agent instructions.

---

## 📁 Repository Structure

```
├── src/                    # Source code
│   ├── models/            # Model architectures
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation utilities
│   └── preprocessing/     # Dataset preprocessing
├── notebooks/             # Jupyter notebooks
│   └── kaggle_kl_vs_vq_experiment.ipynb
├── configs/               # Experiment configurations
│   └── experiment1/       # KL vs VQ configs
├── scripts/               # Utility scripts
│   ├── validate_kaggle_ready.py
│   ├── sync_kaggle_results.py
│   └── README.md
├── docs/                  # Project documentation
│   ├── CURRENT_STATE.md   # Current project status
│   ├── DECISIONS.md       # Research decisions log
│   └── ...
├── results/               # Experiment results
├── checkpoints/           # Model checkpoints (not in git)
└── outputs/               # Visualizations and outputs
```

---

## 📚 Documentation

- **[AGENTS.md](AGENTS.md)** - Agent instructions and project governance
- **[docs/CURRENT_STATE.md](docs/CURRENT_STATE.md)** - Current project state
- **[docs/DECISIONS.md](docs/DECISIONS.md)** - Research decisions log
- **[scripts/README.md](scripts/README.md)** - Utility scripts overview
- **[.kiro/specs/](/.kiro/specs/)** - Experiment specifications

---

## 🔬 Experiments

### Experiment 1: KL vs VQ First-Stage Autoencoder
- **Status:** In Progress
- **Goal:** Compare KL-regularized and VQ-based autoencoders
- **Dataset:** No_Mixed
- **Documentation:** [.kiro/specs/02-kl-vs-vq/](.kiro/specs/02-kl-vs-vq/)

---

## 🛠️ Setup

### Requirements
- Python 3.8+
- PyTorch
- See `requirements.txt` for full dependencies

### Installation
```bash
pip install -r requirements.txt
```

---

## 🎓 Research Background

This project investigates whether alternative deep-learning architectures can improve upon a conventional Convolutional Autoencoder (CAE) baseline for:
1. Reconstruction quality of thermal imagery
2. Preservation of thermal boundaries
3. Unsupervised mixed-pixel detection

Candidate architectures include:
- Convolutional Autoencoder (CAE) - Baseline
- Latent Diffusion Models (LDM)
- Continuous Feature Convolution (CFC)
- Context Clusters (CoC)

See [.kiro/steering/product.md](.kiro/steering/product.md) for detailed research context.