# Encoder Benchmark Design

This document records encoder families used as references for microscopy representation learning.

## Why Encoder Choice Matters

Microscopy images differ from natural images in texture, scale, noise, contrast, and annotation density. Comparing microscopy-specific, biomedical, and generic encoders helps separate the effect of domain pretraining from the effect of model size.

## Encoder Families

### Microscopy-specialized

- NASA pretrained microscopy models (MicroNet encoders)
  - repo: https://github.com/nasa/pretrained-microscopy-models
  - good when your images resemble microscopy texture statistics more than natural images

- Cytoself
  - repo: https://github.com/royerlab/cytoself
  - useful when localization patterns or cell morphology embeddings are central

### Histopathology / biomedical foundation models

- UNI
  - repo: https://github.com/KatherLab/uni
  - strong pathology encoder for transfer and linear probing

- TITAN
  - repo: https://github.com/mahmoodlab/TITAN
  - useful for multimodal experiments with reports or text metadata

### Strong generic backbones to benchmark anyway

- DINOv2
- ConvNeXtV2
- ViT / DeiT family

## How They Are Used

The benchmark design compares three encoder groups: microscopy-specialized encoders, biomedical foundation encoders, and generic self-supervised vision encoders. Each encoder group is evaluated with the same dataset split, preprocessing path, downstream head, and metric table.
