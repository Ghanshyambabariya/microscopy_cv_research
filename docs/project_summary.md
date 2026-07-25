# Project Summary

MicroForge AI is a materials ML research workspace that connects four areas I want to keep developing together:

- microscopy image analysis
- high-frequency process signals
- machining/tool-wear prediction
- materials-property regression

The repository is split into small projects so that each part can be read and tested independently. The shared code lives in `src/`, while scripts, configs, reports, and copied result figures are kept separate.

## Current Focus

| Area | Current Work |
|---|---|
| Microscopy CV | SEM segmentation baseline, active-learning loop, synthetic data scaffolding |
| Process signals | 20 kHz force/torque signal simulation and feature extraction |
| Tool wear | real Vicomtech and Uniwear benchmarks with grouped validation |
| Property ML | concrete compressive-strength regression from tabular variables |
| Automation | config-driven benchmark scripts, generated reports, and tests |

## Short Description

Built a materials ML research workspace combining SEM image segmentation, active learning, process-signal feature extraction, real tool-wear benchmarks, and materials-property regression with reproducible Python scripts and generated reports.

## Current Scope

The current SEM work is reported as a baseline measurement rather than a final research result. The implemented comparison uses lightweight segmentation models and public benchmark data where available.
