# Encoder Recommendations

## Strong Starting Points

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
  - useful if later your project becomes multimodal with reports or text metadata

### Strong generic backbones to benchmark anyway

- DINOv2
- ConvNeXtV2
- ViT / DeiT family

## Practical Recommendation

Benchmark three encoder families:
- microscopy-specialized encoder
- pathology/biomedical foundation encoder
- strong generic self-supervised encoder

This makes your paper stronger because you can show whether domain pretraining really matters for your microscopic image type.
