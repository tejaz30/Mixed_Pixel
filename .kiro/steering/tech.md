# Technology Stack

## Development Environment

Primary development environment:

- Kiro IDE
- Git
- GitHub or equivalent Git remote

Kiro is the primary environment for writing, reviewing, organizing, and maintaining project code.

## Compute Environment

GPU- and TPU-intensive experiments are executed using Kaggle Notebooks.

Kaggle is the compute environment, not the primary source-code environment.

Important project logic must remain in the repository rather than existing only inside Kaggle notebooks.

## Programming Language

Primary language:

- Python

## Deep Learning Framework

Primary framework:

- PyTorch

Use PyTorch for model implementation, training, inference, and experiment execution unless a project requirement explicitly requires another framework.

## Current / Expected Libraries

Libraries may include:

- PyTorch
- TorchVision
- NumPy
- SciPy
- Pandas
- OpenCV
- Pillow
- scikit-image
- TensorBoard
- Matplotlib

Additional libraries may be introduced when required by a specific architecture or experiment.

Dependencies must be documented rather than silently added.

## Experiment Execution

The preferred workflow is:

1. Develop code in Kiro.
2. Commit changes to Git.
3. Pull or synchronize the repository in Kaggle.
4. Execute GPU/TPU-heavy experiments in Kaggle.
5. Store experiment outputs in the designated results structure.
6. Record important results and decisions.
7. Commit relevant reproducibility information back to the repository.

## Configuration

Training and experiment configuration should be separated from implementation code wherever practical.

Avoid hard-coding:

- dataset paths
- Kaggle-specific paths
- model hyperparameters
- output directories
- experiment identifiers

## Reproducibility

Experiments should record, where applicable:

- random seed
- dataset version
- train/validation split
- model configuration
- hyperparameters
- optimizer
- learning rate
- batch size
- number of epochs
- hardware environment
- software/dependency versions
- experiment identifier

## Dependency Management

All required dependencies must be explicitly documented.

Do not assume that a package available in the local environment will also be available in Kaggle.

Kaggle-specific dependencies should be clearly identified.

## Hardware Awareness

Code should support CPU execution for development and testing where practical.

Training code must support GPU execution.

TPU-specific implementation should only be introduced when there is a demonstrated benefit and should not unnecessarily complicate the core model implementation.