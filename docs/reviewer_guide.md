# Reviewer Guide

This page explains how to read **MicroForge AI** from three common reviewer viewpoints: professor, technical hiring manager, and HR/recruiter.

## If You Are A Professor Or Research Supervisor

Look first for scientific discipline, not only high numbers.

| Review Question | Where This Project Answers It |
|---|---|
| Is the problem scientifically relevant? | The project connects microstructure images, manufacturing signals, tool wear, and property prediction. |
| Are the datasets real or clearly labeled? | [Dataset cards](datasets.md) separate implemented real datasets from optional large targets and simulated scaffolding. |
| Is leakage controlled? | Tool-wear benchmarks use held-out tool IDs or experiment tags instead of only random row splits. |
| Are limitations stated honestly? | SEM segmentation is labeled as a smoke-test baseline, not a finished state-of-the-art claim. |
| Is the next research path clear? | [Methodology notes](methodology.md), [structure-property roadmap](structure_property_roadmap.md), and project READMEs list upgrade paths. |

### Professor-Level Strengths

- The repository is interdisciplinary: microscopy CV, sensor/signal analysis, process monitoring, and materials informatics are connected in one workflow.
- The benchmark design values reproducibility: scripts, configs, reports, figures, tests, and dataset cards are separated.
- The validation approach is careful for manufacturing data, especially grouped splits for tool/experiment separation.
- The project avoids pretending synthetic data is equivalent to real experimental data.

### Professor-Level Gaps To Improve Next

- Add stronger SEM/TEM/EBSD datasets with expert labels and report cross-dataset generalization.
- Add multiple random seeds, confidence intervals, and ablation tables for model choices.
- Add pretrained microscopy encoders such as MicroNet/Swin and compare against random initialization.
- Add publication-style error analysis with failure examples and class-wise segmentation metrics.
- Add a short model card for each trained model and a data-license/provenance table.

## If You Are A Technical Hiring Manager

Look for whether the candidate can build an end-to-end ML system, not only notebooks.

| Capability | Evidence |
|---|---|
| Data ingestion | Online GitHub datasets are downloaded and parsed by scripts. |
| Cleaning/preprocessing | Numeric validation, schema handling, signal features, and dataset-specific runners. |
| Modeling | RandomForest baselines, UNet segmentation baseline, active-learning workflow hooks. |
| Evaluation | R2, macro F1, segmentation IoU, generated reports, and saved figures. |
| Software quality | Config-driven scripts, reusable `src/` package, smoke tests, and separated docs. |

## If You Are HR Or A Recruiter

Start with the main [README](../README.md), then open one role-relevant folder.

| Role Type | Best Folder |
|---|---|
| Computer vision / microscopy AI | [projects/01_microscopy_cv](../projects/01_microscopy_cv) |
| Manufacturing ML / predictive maintenance | [projects/03_tool_wear_benchmarks](../projects/03_tool_wear_benchmarks) |
| Materials informatics | [projects/04_materials_property_ml](../projects/04_materials_property_ml) |
| General ML engineering | [projects/05_multimodal_platform](../projects/05_multimodal_platform) |

## Best Current Positioning

MicroForge AI should be presented as a **research-oriented ML portfolio**, not as a completed scientific paper yet. Its strongest current value is the clean architecture, real-data benchmark foundation, and clear path toward publishable microscopy/materials-AI research.
