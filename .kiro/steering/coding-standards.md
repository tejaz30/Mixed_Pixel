# Coding Standards

## General Principles

Code should prioritize:

1. Correctness
2. Reproducibility
3. Readability
4. Modularity
5. Maintainability

Avoid unnecessary abstraction and unnecessary complexity.

## Python

Follow standard Python conventions and use clear, descriptive names.

Prefer:

- `snake_case` for variables and functions
- `PascalCase` for classes
- descriptive names over abbreviations

Avoid unexplained magic numbers.

## Functions

Functions should have a single clear responsibility.

Avoid very large functions that combine:

- data loading
- model creation
- training
- evaluation
- visualization
- result saving

These responsibilities should be separated where practical.

## Classes

Use classes where they improve organization or encapsulation.

PyTorch models should inherit from the appropriate PyTorch base classes.

## Configuration

Do not hard-code experiment-specific hyperparameters inside model implementations.

Prefer explicit configuration objects or configuration files.

## Paths

Do not hard-code local machine paths or Kaggle-specific paths inside reusable source code.

Paths should be supplied through configuration or command-line/environment configuration.

## Logging

Training and evaluation code should provide useful logging.

Important information should include:

- experiment ID
- epoch
- training loss
- validation loss
- evaluation metrics
- checkpoint information

Avoid excessive debug output during normal execution.

## Error Handling

Fail clearly when required inputs are missing or invalid.

Do not silently ignore errors that could affect experiment validity.

## Randomness

Experiments should provide explicit random seeds where reproducibility is required.

Seed relevant libraries and frameworks where practical.

## Checkpoints

Checkpoints should include enough information to reproduce or resume an experiment.

Where appropriate, save:

- model state
- optimizer state
- scheduler state
- epoch
- configuration
- relevant metrics

## Documentation

Public or non-obvious functions should have concise docstrings.

Complex research logic should include comments explaining the reasoning, not merely restating the code.

## Notebooks

Notebooks should primarily orchestrate experiments or perform exploration.

Avoid copying large sections of reusable source code into notebooks.

## Research Code

Do not optimize or refactor research code solely for elegance if doing so makes the experiment harder to reproduce.

When modifying an implementation that affects experimental comparability, document the change.