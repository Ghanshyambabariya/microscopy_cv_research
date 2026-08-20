# NASA EBC SEM Architecture Comparison

Both models were trained on the same public NASA EBC SEM split with 37 training images, 9 validation images, and 12 held-out test images.

| Model | Pixel Accuracy | Foreground IoU | Dice/F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| U-Net small | 0.9446 | 0.4388 | 0.5411 | 0.5053 | 0.7485 |
| FCN small | 0.8751 | 0.3168 | 0.4362 | 0.3703 | 0.6857 |

The U-Net baseline is stronger on the current split. The foreground class remains the limiting case, so the qualitative panels and saliency maps should be read together with the aggregate metrics.

Figures:

- U-Net panel: `reports/figures/sem_ebc_predictions.png`
- FCN panel: `reports/figures/sem_ebc_fcn_predictions.png`
