# Scripts Directory

Organized collection of scripts for training, evaluation, testing, and utilities.

## Structure

### 📁 `training/`
Scripts for training models:
- `train_first_stage.py` - Train KL/VQ autoencoders (Experiment 1)
- `train_cae.py` - Train CAE baseline
- `train_multiple_cae.py` - Train CAE on multiple datasets
- `run_experiment1.py` - Run complete Experiment 1 workflow
- `*.bat` - Windows batch files for convenient training

### 📁 `evaluation/`
Scripts for evaluating trained models:
- `evaluate_first_stage.py` - Evaluate KL/VQ models
- `compare_experiment1.py` - Compare KL vs VQ results
- `run_evaluation.bat` - Batch evaluation runner
- `run_validation.bat` - Validation runner

### 📁 `testing/`
Unit and integration tests:
- `test_cae.py` - CAE model tests
- `test_dataset.py` - Dataset loading tests
- `test_experiment1_models.py` - Experiment 1 model tests
- `test_preprocessing_config.py` - Preprocessing configuration tests

### 📁 `preprocessing/`
Data preprocessing scripts:
- `run_deduplication.bat` - Remove duplicate images
- `run_preprocessing.bat` - Run full preprocessing pipeline

### 📁 `utils/`
Utility scripts:
- `check_training_status.py` - Monitor training progress

## Usage

### Training
```bash
# Train KL autoencoder
python src/scripts/training/train_first_stage.py --config configs/experiment1/kl_autoencoder.yaml

# Train VQ autoencoder
python src/scripts/training/train_first_stage.py --config configs/experiment1/vq_autoencoder.yaml

# Run complete experiment
python src/scripts/training/run_experiment1.py
```

### Evaluation
```bash
# Evaluate model
python src/scripts/evaluation/evaluate_first_stage.py --checkpoint path/to/checkpoint.pth

# Compare KL vs VQ
python src/scripts/evaluation/compare_experiment1.py
```

### Testing
```bash
# Run all tests
pytest src/scripts/testing/

# Run specific test
python src/scripts/testing/test_cae.py
```

### Utilities
```bash
# Check training status
python src/scripts/utils/check_training_status.py
```
