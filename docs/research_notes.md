# Research Status

This document summarizes the current implemented scope and interpretation of the results.

## Implemented Methods

- The codebase has separate runners for microscopy, signal features, tool wear, concrete strength, and multimodal summaries.
- Real online datasets are already connected for tool wear and concrete strength.
- Tool-wear experiments use grouped splits where possible, which is important because random row splits can overestimate performance.
- The microscopy part includes segmentation and active-learning code with public SEM benchmark experiments.
- Figures, reports, configs, and tests are kept in separate folders so the project can grow without becoming only a notebook collection.

## Evaluation Design

- Public benchmark datasets are used where practical.
- Grouped splits are used for tool-wear experiments to reduce leakage.
- Generated signal data is clearly separated from public datasets.
- Metrics are saved as JSON and summarized in Markdown reports.

## Current Interpretation

The strongest current evidence is the real-data manufacturing/property benchmarks and the reproducible project structure. The microscopy experiments are reported as baseline SEM segmentation results rather than state-of-the-art claims.
