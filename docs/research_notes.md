# Research Notes

These notes summarize how I am thinking about the project and what still needs improvement.

## What Is Working

- The codebase has separate runners for microscopy, signal features, tool wear, concrete strength, and multimodal summaries.
- Real online datasets are already connected for tool wear and concrete strength.
- Tool-wear experiments use grouped splits where possible, which is important because random row splits can overestimate performance.
- The microscopy part includes segmentation and active-learning code, but it still needs stronger real SEM/TEM/EBSD datasets.
- Figures, reports, configs, and tests are kept in separate folders so the project can grow without becoming only a notebook collection.

## What Needs More Work

- Add larger and better-labeled microscopy datasets.
- Run repeated seeds and report mean plus standard deviation.
- Add model ablations for encoder choice, image size, augmentation, and loss function.
- Add more failure-case figures, especially for segmentation.
- Track dataset version, split seed, model name, and metrics for every run.

## Current Interpretation

The strongest part of the repository right now is the overall system structure and the real-data manufacturing/property benchmarks. The microscopy part is useful as a framework and baseline, but it should not yet be described as state of the art.
