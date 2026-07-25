# Project Summary

This repository contains four separate materials-ML projects:

- microscopy image analysis
- high-frequency process signals
- machining/tool-wear prediction
- materials-property regression

The repository is split so each project can be read and tested independently. Shared implementation code lives in `src/`, while scripts, configs, reports, and copied result figures are kept separate.

## Current Focus

| Area | Current Work |
|---|---|
| Microscopy CV | SEM segmentation baseline, active-learning loop, synthetic data scaffolding |
| Process signals | 20 kHz force/torque signal simulation and feature extraction |
| Tool wear | real Vicomtech and Uniwear benchmarks with grouped validation |
| Property ML | concrete compressive-strength regression from tabular variables |

## Short Description

Four separate materials-ML projects covering SEM image segmentation, high-frequency process-signal feature extraction, real tool-wear benchmarks, and tabular materials-property regression.

## Current Scope

The current SEM work is reported as a baseline measurement rather than a final research result. The implemented comparison uses lightweight segmentation models and public benchmark data where available.
