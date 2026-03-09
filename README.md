# Microscopy CV Research

Research-structured computer vision project for microscopic images with three connected tracks:

1. Supervised learning with pretrained encoders for microscopy-style images
2. Synthetic data generation for balancing, augmentation, and stress testing
3. Hybrid learning that combines classification, regression, and synthetic mixing

## Framework

```mermaid
flowchart LR
    A[Microscopy Images] --> B[Data Audit and Group Split]
    B --> C[Supervised Encoder Training]
    B --> D[Synthetic Image Generation]
    C --> E[Classification Output]
    C --> F[Regression Output]
    D --> G[Synthetic Dataset]
    G --> H[Hybrid Multitask Training]
    B --> H
    H --> I[Joint Structure Property Prediction]
    A --> K[Real Benchmark Ingestion]
    K --> L[SEM Segmentation Showcase]
    K --> M[TEM Target Registry]
    K --> N[EBSD Target Registry]
    L --> J[Reports and Checkpoints]
    E --> J
    F --> J
    I --> J
```

## Quick Showcase

See the full portfolio-style page here: [SHOWCASE.md](SHOWCASE.md)

See the real benchmark target map here: [reports/real_benchmark_portfolio.md](reports/real_benchmark_portfolio.md)

See the integrated NASA SEM benchmark showcase here: [reports/real_sem_benchmark_showcase.md](reports/real_sem_benchmark_showcase.md)

![Benchmark overview](reports/figures/benchmark_overview.png)

![NASA SEM benchmark](reports/figures/real_sem_benchmark.png)

## What This Project Does

- builds a 1,440-image microscopy-style starter dataset with specimen-level grouping
- trains supervised classification and regression baselines
- generates synthetic microscopy-like images for augmentation studies
- runs a hybrid multitask model with classification plus regression outputs
- supports public-dataset showcase collection from external microscopy sources
- integrates a real NASA SEM segmentation benchmark showcase with actual images and masks

## Current Reality

The repo now has one real microscopy benchmark integrated directly into the portfolio:
- NASA MicroNet SEM segmentation benchmark data

What is still missing for a fuller research-grade benchmark suite:
- real TEM or EMPIAR-derived encoder transfer evaluation
- real EBSD indexing, phase, or orientation prediction with sample patterns and metrics
- task-specific model evaluation on those additional modalities

## Runnable Entry Points

```powershell
C:/Users/ghans/AppData/Local/Programs/Python/Python312/python.exe scripts/create_sample_dataset.py
C:/Users/ghans/AppData/Local/Programs/Python/Python312/python.exe scripts/run_supervised.py --config configs/supervised_classification.json
C:/Users/ghans/AppData/Local/Programs/Python/Python312/python.exe scripts/run_supervised.py --config configs/supervised_regression.json
C:/Users/ghans/AppData/Local/Programs/Python/Python312/python.exe scripts/run_synthetic.py
C:/Users/ghans/AppData/Local/Programs/Python/Python312/python.exe scripts/run_hybrid.py
C:/Users/ghans/AppData/Local/Programs/Python/Python312/python.exe scripts/build_public_showcase.py
C:/Users/ghans/AppData/Local/Programs/Python/Python312/python.exe scripts/build_benchmark_portfolio.py
C:/Users/ghans/AppData/Local/Programs/Python/Python312/python.exe scripts/build_real_sem_showcase.py
```

## Current Benchmark Results

| Track | Output | Result |
|---|---|---|
| Supervised classification | accuracy | 1.0000 |
| Supervised classification | macro F1 | 1.0000 |
| Supervised regression | MAE | 0.0414 |
| Supervised regression | RMSE | 0.0507 |
| Supervised regression | R2 | 0.9678 |
| Hybrid multitask | classification accuracy | 1.0000 |
| Hybrid multitask | classification macro F1 | 1.0000 |
| Hybrid multitask | regression MAE | 0.0328 |
| Hybrid multitask | regression RMSE | 0.0395 |
| Hybrid multitask | regression R2 | 0.9804 |
| Synthetic generation | generated images | 120 |

## Real Benchmark Evidence

### NASA SEM Segmentation Showcase

- datasets integrated: `EBC1`, `EBC2`, `EBC3`, `Super1`, `Super2`, `Super3`, `Super4`
- task type: semantic segmentation
- evidence generated: real test-image inputs, masks, overlays, and split-count reports

![NASA SEM benchmark](reports/figures/real_sem_benchmark.png)

## Example Images

### Starter Dataset Samples

| Grain | Pore | Crack |
|---|---|---|
| ![grain](data/raw/images/specimen_000_00.png) | ![pore](data/raw/images/specimen_060_00.png) | ![crack](data/raw/images/specimen_120_00.png) |

### Synthetic Generation Samples

| Synthetic Grain | Synthetic Pore | Synthetic Crack |
|---|---|---|
| ![syn-grain](data/interim/synthetic_images/synthetic_grain_000.png) | ![syn-pore](data/interim/synthetic_images/synthetic_pore_000.png) | ![syn-crack](data/interim/synthetic_images/synthetic_crack_000.png) |

## Public Microscopy Showcase

The project can also download official public microscopy datasets, extract readable preview images, and validate that the project loader can work on them.

Current local showcase sources:
- BBBC028 Human HT29 Colon-Cancer Cells
- BBBC033 Human Motor Neurons
- BBBC053 Image-Based Profiling MitoCheck

## Real Benchmark Targets

| Modality | Example task | Current status |
|---|---|---|
| SEM | materials segmentation | integrated in portfolio from NASA benchmark data |
| TEM | encoder transfer or retrieval | target registry only |
| EBSD | phase or orientation prediction | target registry only |

The benchmark registry for these targets is tracked in [configs/real_benchmark_targets.json](configs/real_benchmark_targets.json), the generated target summary is in [reports/real_benchmark_portfolio.md](reports/real_benchmark_portfolio.md), and the integrated SEM showcase is in [reports/real_sem_benchmark_showcase.md](reports/real_sem_benchmark_showcase.md).

## Important Note

The synthetic starter results are still framework-level results, not publishable microscopy evidence. The SEM section is now real benchmark evidence from NASA's public microscopy benchmark data, while TEM and EBSD are still planned benchmark integrations.
