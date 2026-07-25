# Methodology

This document describes the experimental method used across the four independent materials-ML projects.

## Why These Methods Are Used

Materials datasets often combine small image sets, high-frequency process signals, tabular properties, and experiment-level grouping. The methodology therefore emphasizes leakage control, reproducible preprocessing, baseline comparison, and clear separation between real benchmark data and development scaffolds.

## How They Are Applied

- Data audit and label QA are run before model training.
- Train/test splits use specimen, tool, batch, or experiment groups where metadata allows it.
- Classification, regression, and segmentation tasks are evaluated with task-specific metrics.
- Synthetic data is treated as controlled augmentation or pipeline scaffolding, not as real benchmark evidence.
- Dataset provenance, license notes, sample counts, label definitions, split strategy, preprocessing, and limitations are recorded in dataset cards.
- Figures and reports are generated from scripts rather than hand-written result summaries.

## Method Blocks

| Method block | Purpose | Output |
|---|---|
| Data ingestion | load public datasets or generated development data | cleaned tables, manifests, sample previews |
| Preprocessing | standardize images, signals, and tabular variables | normalized arrays and feature matrices |
| Feature extraction | convert raw signals and metadata into ML inputs | RMS, spectral, energy, force, and property descriptors |
| Model training | fit baseline ML and segmentation models | saved metrics and qualitative outputs |
| Evaluation | report held-out performance using suitable metrics | Markdown reports, JSON metrics, figures |

## Current Results

| Benchmark | Task | Result |
|---|---|---|
| Vicomtech tool wear | held-out tool-ID regression | R2 `0.8680` |
| Vicomtech tool wear | held-out tool-ID wear-stage classification | macro F1 `0.6472` |
| Katulu Uniwear | held-out experiment wear-stage classification | macro F1 `0.5205` |
| Concrete strength | tabular property regression | R2 `0.8990` |
| NASA EBC SEM suite | segmentation smoke test | foreground IoU `0.1174` |
