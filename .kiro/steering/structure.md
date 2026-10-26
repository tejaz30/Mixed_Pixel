# Project Structure

## Repository Structure

The project follows a separation between:

- project rules
- research documentation
- source code
- experiments
- tests
- notebooks
- generated results

Expected high-level structure:

.kiro/
├── steering/
├── specs/
├── agents/
└── hooks/

docs/
├── CURRENT_STATE.md
├── DECISIONS.md
├── CHANGELOG.md
├── TODO.md
└── KNOWN_ISSUES.md

research/
├── experiments/
├── results/
├── figures/
└── reports/

src/
├── data/
├── models/
├── training/
├── evaluation/
└── utils/

tests/

notebooks/

AGENTS.md
README.md

## Source Code

Reusable implementation belongs in `src/`.

Do not place important reusable project logic exclusively inside notebooks.

### `src/data/`

Dataset loading, preprocessing, transforms, data validation, and dataset utilities.

### `src/models/`

Model architectures and model-specific components.

Examples include:

- CAE
- KL autoencoder
- VQ autoencoder
- LDM components
- CFC
- CoC

### `src/training/`

Training loops, optimizers, schedulers, checkpointing, and training utilities.

### `src/evaluation/`

Metrics, evaluation pipelines, reconstruction analysis, boundary analysis, and benchmarking utilities.

### `src/utils/`

Shared utilities that do not belong to a more specific module.

## Research Experiments

`research/experiments/` contains experiment-specific configuration and documentation.

Each major experiment should have a dedicated directory.

Example:

research/experiments/
├── exp01_kl_vs_vq/
├── exp02_compression/
├── exp03_ldm/
├── exp04_cfc/
└── exp05_coc/

Experiments should reference reusable source code rather than duplicating implementation.

## Results

`research/results/` contains experiment outputs.

Results should be associated with an experiment identifier.

Do not overwrite results from previous experiments without explicit justification.

## Figures

`research/figures/` contains figures intended for:

- analysis
- presentations
- reports
- publications

Generated intermediate plots should remain associated with their experiment when possible.

## Notebooks

`notebooks/` contains notebooks primarily used for:

- Kaggle execution
- exploratory analysis
- visualization
- experiment orchestration

Notebooks should not become the only location of core implementation logic.

## Tests

Automated tests belong in `tests/`.

Tests should validate reusable project components and important assumptions.

## Documentation

Project-level state and decisions belong in `docs/`.

Research-specific reasoning should be documented in the relevant experiment or research documentation.

## File Placement Rule

Before creating a new file, determine whether it belongs to:

- source code
- experiment configuration
- experiment results
- documentation
- tests
- notebook/execution layer

Do not create duplicate files in multiple locations without a clear reason.