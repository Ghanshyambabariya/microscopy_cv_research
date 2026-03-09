# NASA EBC SEM Segmentation Experiment

This report records a real segmentation baseline trained on NASA's public EBC benchmark segmentation data from the `pretrained-microscopy-models` repository.

## Setup

- Modality: SEM-style materials microscopy
- Task: semantic segmentation
- Datasets: `EBC1`, `EBC2`, `EBC3`
- Model: `UNetSmall`
- Training device: CPU
- Image size: `256 x 256`
- Epochs: `10`
- Batch size: `4`
- Loss: weighted cross-entropy

## Dataset Sizes

- Train samples: `37`
- Validation samples: `9`
- Test samples: `12`

## Test Metrics

- Pixel accuracy: `0.9480`
- Mean IoU (foreground classes only): `0.4334`
- Mean Dice (foreground classes only): `0.5293`
- Class 1 IoU: `0.7578`
- Class 1 Dice: `0.8622`
- Class 2 IoU: `0.1089`
- Class 2 Dice: `0.1964`

## Interpretation

- This is a real microscopy experiment, not a synthetic starter result.
- The model segments the dominant foreground phase well.
- The rare crack-like class is still difficult, but the weighted training run now predicts it instead of collapsing to zero.
- This is a valid baseline for the portfolio, not the final best model.

## Prediction Examples

![NASA EBC predictions](reports/figures/sem_ebc_predictions.png)
