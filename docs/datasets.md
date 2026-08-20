# Dataset Provenance

This file records the public data sources used or documented in the project. Dataset files are loaded from their original public locations or from local generated artifacts; large third-party archives are not redistributed directly in this repository.

## NASA EBC SEM Segmentation

| Field | Value |
|---|---|
| Name | NASA EBC SEM benchmark segmentation data |
| Original authors / institution | NASA materials microscopy research group |
| Source URL / DOI | https://github.com/nasa/pretrained-microscopy-models |
| License | Follow the source repository license and dataset terms before reuse |
| Modality | SEM microscopy images with segmentation masks |
| Target | Pixel-level semantic segmentation |
| Samples used | Current EBC run: 37 train, 9 validation, 12 test images from `EBC1`, `EBC2`, and `EBC3` |
| Split methodology | Dataset-level files are split into train/validation/test partitions by the local training pipeline with a fixed seed |
| Preprocessing | RGB conversion, resizing to 256 x 256, tensor normalization, mask remapping, class weighting for imbalance |
| Limitations | Small held-out test set and strong class imbalance; minority foreground performance is lower than background performance |
| Redistribution | Source data should be obtained from the NASA repository rather than copied into this repository |

## Vicomtech Machine-Tool Wear

| Field | Value |
|---|---|
| Name | Vicomtech dataset-machine-tool-wear |
| Original authors / institution | Vicomtech |
| Source URL / DOI | https://github.com/Vicomtech/dataset-machine-tool-wear |
| License | Not restated here; verify directly at the source repository |
| Modality | Tabular machining/process features |
| Target | Tool-wear regression and wear-stage classification |
| Samples used | Loaded by `scripts/run_external_tool_wear.py` from the public repository |
| Split methodology | Grouped held-out validation by tool ID |
| Preprocessing | Numeric cleaning, target preparation, grouped train/test split, regression/classification label preparation |
| Limitations | Generalization is evaluated across held-out tools, but source-specific acquisition conditions still matter |
| Redistribution | Use the upstream repository as the source of record |

## Katulu Uniwear

| Field | Value |
|---|---|
| Name | Katulu Uniwear Dataset |
| Original authors / institution | Katulu |
| Source URL / DOI | https://github.com/katulu-io/uniwear-dataset |
| License | Not restated here; verify directly at the source repository |
| Modality | Force/vibration time-series measurements |
| Target | Tool-wear regression and wear-stage classification |
| Samples used | Loaded by `scripts/run_external_uniwear.py` from the public repository |
| Split methodology | Grouped held-out validation by experiment tag |
| Preprocessing | Signal table cleaning, numeric feature extraction, grouped split, model-ready feature matrix generation |
| Limitations | The current result is a baseline; domain shifts between experiments can reduce regression performance |
| Redistribution | Use the upstream repository as the source of record |

## Concrete Compressive Strength

| Field | Value |
|---|---|
| Name | Concrete Compressive Strength |
| Original authors / institution | UCI Machine Learning Repository dataset; CSV mirror maintained in the `stedy` repository |
| Source URL / DOI | https://github.com/stedy/Machine-Learning-with-R-datasets |
| License | Not restated here; verify the original UCI dataset and mirror terms before redistribution |
| Modality | Tabular materials composition/process variables |
| Target | Compressive strength regression |
| Samples used | 1,005 cleaned rows after duplicate removal |
| Split methodology | Fixed-seed random train/test split: 753 train, 252 test; five-fold cross-validation on the training partition |
| Preprocessing | Duplicate removal, numeric validation, feature/target separation |
| Limitations | Random splitting can overestimate performance when near-duplicate mix designs are present; grouped or time-aware splits would be stricter if metadata were available |
| Redistribution | The runner downloads the public CSV mirror; source terms should be checked before republishing the dataset |

## CoMMonS Microscopy Dataset

| Field | Value |
|---|---|
| Name | CoMMonS microscopic material surface dataset |
| Original authors / institution | Georgia Tech research group associated with the CoMMonS repository |
| Source URL / DOI | https://github.com/olivesgatech/CoMMonS |
| License | Not restated here; verify directly at the source repository |
| Modality | Microscopy/material surface images |
| Target | Material surface property or class labels depending on the selected benchmark subset |
| Samples used | Documented as a large optional target, not part of the default lightweight run |
| Split methodology | To be defined for a dedicated image benchmark run |
| Preprocessing | Planned image validation, resizing, channel normalization, and metadata cleaning |
| Limitations | The sampled archive is large, so it is not bundled with this repository |
| Redistribution | Use the upstream repository as the source of record |

## Synthetic High-Frequency Process Signals

| Field | Value |
|---|---|
| Name | Synthetic 20 kHz process-signal dataset |
| Original authors / institution | Generated locally by this repository |
| Source URL / DOI | Not applicable |
| License | Repository license applies to generated code; generated data are development artifacts |
| Modality | Force and moment channels: `Fx`, `Fy`, `Fz`, `Mz` |
| Target | Process-state classification and synthetic property regression |
| Samples used | Configured by `configs/materials_signal.json` |
| Split methodology | Fixed-seed train/test split on extracted windows |
| Preprocessing | Windowing, time-domain features, spectral features, resultant-force features, and energy descriptors |
| Limitations | Synthetic signals reproduce workflow structure only; results are not evidence of performance on industrial measurements |
| Redistribution | Safe to regenerate locally; no proprietary data are included |
