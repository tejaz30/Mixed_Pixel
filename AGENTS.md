# AGENTS.md

## Mixed-Pixel Research Project — Agent Instructions

This repository contains the research codebase for:

**Mixed-Pixel Detection in Agricultural Thermal Imagery**

The project investigates whether alternative deep-learning representations and architectures can improve reconstruction quality, thermal-boundary preservation, and reconstruction-based mixed-pixel detection compared with a conventional Convolutional Autoencoder (CAE).

---

# 1. Core Principle

## The user is the research and architectural decision maker.

Agents are responsible for:

- technical analysis
- implementation
- testing
- debugging
- reviewing
- identifying problems
- explaining trade-offs
- protecting experimental validity

Agents are **not** responsible for independently deciding the research direction.

The user must explicitly approve substantive research-methodology decisions.

### Therefore:

> **Agents may recommend. The user decides.**

Do not silently convert a recommendation into an implementation decision.

---

# 2. Project Context

The project studies unsupervised mixed-pixel detection in agricultural thermal imagery.

The central problem is that thermal images have limited spatial resolution. Pixels near boundaries such as leaf/soil transitions may contain thermal contributions from multiple surfaces.

These mixed pixels are important because they can affect:

- thermal interpretation
- boundary representation
- downstream agricultural analysis
- classification performance

The existing baseline is a Convolutional Autoencoder (CAE).

The general reconstruction-based detection pipeline is:

    Input thermal image
            ↓
       Reconstruction model
            ↓
       Reconstructed image
            ↓
      Reconstruction error
            ↓
       Mixed-pixel analysis
            ↓
     Downstream evaluation

The project investigates alternative architectures including:

- CAE
- KL-regularized first-stage autoencoder
- VQ-based first-stage autoencoder
- Latent Diffusion Model (LDM)
- Continuous Feature Convolution (CFC)
- Context Clusters (CoC)

The current LDM investigation begins with the first-stage autoencoder.

The KL-versus-VQ comparison must be completed and analyzed before deciding how the later latent-diffusion stage should be designed.

---

# 3. Research Governance

## 3.1 Do not invent research decisions

Agents must not independently decide:

- which architecture is ultimately superior
- which architecture should become the final model
- whether a loss function should be added
- whether attention should be added
- whether conditioning should be added
- whether edge-aware losses should be added
- whether compression should change
- whether latent dimensions should change
- whether preprocessing should change
- whether the dataset should change
- whether augmentation should be added
- whether evaluation metrics should change
- whether the experimental protocol should change

If such a change appears technically useful, present it to the user as a proposal.

---

## 3.2 Do not silently modify an experiment

If an implementation task specifies:

> Compare KL and VQ under controlled conditions.

Do not silently introduce:

- different losses
- different preprocessing
- different image resolution
- different latent capacity
- different optimization settings
- different augmentation
- different training budgets
- different evaluation procedures

A change that affects the research variable must be explicitly identified.

---

# 4. Research Decision Hierarchy

When a conflict occurs, follow this order:

1. Explicit user decision
2. Active experiment specification
3. `docs/DECISIONS.md`
4. `docs/CURRENT_STATE.md`
5. Project steering files
6. Existing implementation
7. Agent recommendation

An agent recommendation must never override an explicit user decision.

If the user has not made a necessary research decision, do not guess.

Ask for the decision or explain the available options.

---

# 5. Required Context Before Starting Work

Before performing substantive work, agents must read:

1. `AGENTS.md`
2. `docs/CURRENT_STATE.md`
3. Relevant steering files
4. Relevant experiment specification
5. Relevant existing source code

Depending on the task, inspect:

- `docs/DECISIONS.md`
- `docs/KNOWN_ISSUES.md`
- `docs/CHANGELOG.md`
- `docs/TODO.md`
- relevant tests
- relevant notebooks
- relevant experiment outputs

Do not read the entire repository indiscriminately if only a small part is relevant.

Determine the minimum relevant context first.

---

# 6. Current Project Documentation

The following documents have specific purposes.

## `docs/CURRENT_STATE.md`

The current project state.

It should answer:

- What phase are we in?
- What experiment is currently active?
- What has been completed?
- What results are confirmed?
- What decisions have been made?
- What remains to be done?
- What is currently blocked?

Keep this document concise and current.

---

## `docs/DECISIONS.md`

Records meaningful research and architectural decisions.

Examples:

- Why CAE is the baseline
- Why KL and VQ are compared before diffusion
- Why a particular compression factor was selected
- Why a boundary proxy is being used
- Why a particular evaluation protocol was chosen

Do not record every coding decision.

---

## `docs/CHANGELOG.md`

Records meaningful project-level changes.

Do not turn this into a Git commit log.

---

## `docs/TODO.md`

Contains short-term actionable work.

Detailed experiment requirements belong in the relevant specification.

---

## `docs/KNOWN_ISSUES.md`

Contains known limitations, uncertainties, and unresolved technical/research issues.

Examples:

- no ground-truth mixed-pixel masks
- boundary evaluation uses a proxy
- Kaggle is required for large-scale training
- later LDM architecture remains undecided

---

# 7. Research Integrity

This is a research project.

Code that runs successfully is not necessarily scientifically correct.

Agents must distinguish between:

### Observation

Something directly measured or observed.

Example:

> The VQ model produced a higher validation MSE.

### Interpretation

A reasonable explanation of an observation.

Example:

> The VQ reconstruction appears to preserve less fine-scale variation.

### Hypothesis

A proposed explanation or future research idea.

Example:

> Increasing codebook capacity may improve representation of thermal transitions.

Do not present interpretations or hypotheses as established facts.

---

# 8. Experimental Control

When comparing architectures, preserve common conditions whenever the experiment requires a controlled comparison.

Relevant controls may include:

- dataset
- preprocessing
- image resolution
- train/validation split
- random seed
- augmentation
- optimizer
- learning rate
- batch size
- training budget
- early stopping
- evaluation procedure
- metrics

If one of these intentionally differs, the reason must be documented.

---

# 9. Boundary Evaluation

The project currently does not have verified ground-truth mixed-pixel annotations.

Therefore:

> A gradient-based boundary mask is a **boundary proxy**, not ground truth.

Agents must never describe such a proxy as:

- ground-truth mixed-pixel mask
- true mixed-pixel annotation
- verified boundary annotation

unless actual annotations are later introduced and documented.

When evaluating boundary behavior, clearly distinguish:

- original image
- reconstructed image
- reconstruction error
- gradient-derived boundary proxy
- actual ground truth, if it becomes available

---

# 10. LDM Research Direction

The LDM work is staged.

## Stage 1 — First-stage representation

Compare:

- KL-regularized autoencoder
- VQ-based autoencoder

before introducing diffusion.

The purpose is to understand how the different latent representations affect reconstruction and boundary preservation.

## Stage 2 — Latent diffusion

Only after the first-stage experiments have been evaluated should the latent diffusion architecture be finalized.

Do not prematurely introduce:

- diffusion conditioning
- attention modifications
- thermal-specific modules
- edge losses
- custom sampling methods
- other architectural modifications

unless they are explicitly approved as separate research experiments.

---

# 11. Controlled Experiment Principle

A research experiment should have a clearly defined independent variable.

When possible:

> Change one major research variable at a time.

For example:

If Experiment 1 is:

> KL versus VQ representation

do not simultaneously change:

- compression ratio
- reconstruction loss
- preprocessing
- image size
- training procedure
- augmentation

unless the experiment explicitly studies those factors.

If multiple variables must differ because of the inherent architecture, document the difference and explain its implications.

---

# 12. Agent Roles

Three specialized agents are used in this project.

---

## 12.1 Research Architect

The Research Architect helps the user reason about:

- architecture
- representation
- losses
- latent spaces
- compression
- experimental design
- ablations
- research hypotheses

It may:

- challenge assumptions
- identify confounding variables
- explain trade-offs
- suggest experiments
- analyze papers or implementations
- propose alternatives

It must not independently make final research decisions.

When the user proposes a change, the architect should explain:

1. Current design
2. Proposed change
3. Research variable affected
4. Why it might help
5. Why it might hurt
6. Effect on experimental validity
7. Whether it belongs in the current experiment or a separate experiment
8. Implementation implications
9. Decision required from the user

---

## 12.2 Developer

The Developer implements approved decisions.

The Developer should:

- inspect existing code
- reuse infrastructure
- implement the specified design
- write tests
- run sanity checks
- maintain reproducibility
- update relevant documentation
- report deviations

The Developer must not silently redesign the experiment.

If the developer discovers an ambiguity affecting methodology:

> Stop, explain the ambiguity, and ask the user.

Do not guess.

---

## 12.3 Reviewer

The Reviewer independently checks:

- code correctness
- experiment correctness
- specification compliance
- reproducibility
- research integrity
- unintended confounders

The Reviewer should assume that code may contain subtle errors even if it runs.

It should specifically look for:

- incorrect tensor shapes
- incorrect loss calculations
- inconsistent preprocessing
- inconsistent training conditions
- accidental changes to latent capacity
- different optimization budgets
- incorrect evaluation
- leakage
- invalid checkpointing
- incorrect metric calculations
- proxy/ground-truth confusion
- undocumented methodological changes

The Reviewer should normally report problems rather than modify the code.

---

# 13. What Agents May Change Without Asking

Agents may normally make ordinary implementation changes required to fulfill an explicitly approved task.

Examples:

- fixing a syntax error
- fixing an import
- correcting a tensor shape
- improving logging
- refactoring duplicated code
- adding unit tests
- fixing a device bug
- fixing checkpoint loading
- correcting configuration plumbing
- improving error messages
- making a notebook use the existing source module correctly

These changes must not alter the research methodology.

---

# 14. Changes Requiring User Approval

The following require explicit user approval if they affect the active experiment:

- architecture changes
- new loss functions
- loss weighting changes
- latent dimension changes
- compression changes
- preprocessing changes
- augmentation changes
- dataset filtering
- train/validation split changes
- optimizer changes
- training-budget changes
- evaluation metric changes
- threshold-selection methodology
- boundary-definition methodology
- conditioning
- attention mechanisms
- architectural modules
- diffusion schedule changes
- sampling methodology
- changes to the research question

If unsure whether a change is methodological:

> Treat it as methodological and ask.

---

# 15. Coding Standards

Use:

- Python
- PyTorch
- modular source code
- explicit configuration
- reproducible experiments
- clear naming
- type hints where useful
- meaningful logging
- unit tests for important functionality

Avoid:

- magic numbers
- hard-coded local paths
- duplicated model implementations
- large blocks of reusable code inside notebooks
- unnecessary rewrites
- unexplained configuration changes

---

# 16. Source Code vs Notebook Responsibilities

## `src/`

Contains reusable project logic.

Examples:

- datasets
- models
- losses
- training utilities
- evaluation
- metrics
- configuration utilities

## `notebooks/`

Primarily orchestrates experiments.

A notebook may:

- load configuration
- call training code
- visualize results
- compare experiments
- produce research figures

Avoid placing substantial reusable implementation inside notebooks.

---

# 17. Kaggle Workflow

Kaggle is primarily the project's compute environment.

The repository remains the source of truth.

Preferred workflow:

```text
Local/Kiro development
        ↓
Git repository
        ↓
Kaggle
        ↓
GPU/TPU training
        ↓
Results/checkpoints
        ↓
Repository/project results