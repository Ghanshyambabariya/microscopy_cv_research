# Microscopy Benchmark Scope

## Current State

The project includes:
- a clean training pipeline
- synthetic image generation
- grouped train/validation/test splits
- generated result figures

The current microscopy results are reported as baseline SEM segmentation experiments, not as a complete SEM/TEM/EBSD benchmark suite.

## Benchmark Scope

### 1. Starter Benchmark

The classification and regression starter metrics come from internally generated microscopy-style images. This verifies the pipeline but is separated from public-data benchmark evidence.

### 2. Public SEM Benchmark

The repository includes SEM segmentation experiments using NASA MicroNet benchmark-style data and an active-learning loop.

### 3. Task Coverage

The implemented microscopy task coverage includes:
- segmentation
- synthetic classification/regression scaffolds
- active-learning sample selection

## Reported Evidence

The repository reports test images, predicted masks, metric tables, and leaderboard entries for the current SEM segmentation baseline.

## Encoder Benchmark Design

Encoder comparison is organized around:
- a microscopy-specialized encoder such as NASA MicroNet
- a large EM or biomedical encoder
- a strong generic vision backbone

## Modality References

- SEM segmentation on materials microstructure
- TEM representation learning or classification
- EBSD indexing, orientation, or phase prediction
