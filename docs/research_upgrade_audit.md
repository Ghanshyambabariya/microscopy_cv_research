# Research Upgrade Audit

This audit records the current research alignment for each independent project after the documentation split.

## 01 Microscopy CV

Why used: microscopy segmentation is a core materials-characterization task because pixel-level masks support phase, defect, porosity, and microstructure quantification.

How used: the project loads public NASA EBC SEM image/mask data, normalizes images, trains U-Net and FCN baselines, evaluates pixel accuracy, foreground IoU, Dice, precision, and recall, and saves prediction panels with attribution.

Current result: NASA EBC SEM trained U-Net foreground IoU `0.4388`; FCN foreground IoU `0.3168`; Automatic-SEM foreground IoU `0.4963` in the compact SEM suite table.

Research upgrade applied: the project page now states the NASA public-data source, compares two segmentation architectures, reports precision/recall with IoU and Dice, and includes input-gradient saliency in the qualitative panel.

## 02 Process-Signal ML

Why used: high-frequency force and moment data are too dense for direct tabular ML, so physics-aware window features are needed before classification or regression.

How used: the project windows 20 kHz `Fx`, `Fy`, `Fz`, and `Mz` signals and extracts RMS, peak, crest factor, spectral centroid, bandpower, resultant force, and energy features.

Current result: process-state classification accuracy `1.0000`; property regression R2 `0.9998` on the generated force/moment benchmark.

Research upgrade applied: the project page now states clearly that the signal data is generated development data and that the scientific output is the feature-extraction and evaluation pipeline.

## 03 Tool-Wear Prediction Benchmark

Why used: tool wear is a practical condition-monitoring task where random-row splits can overestimate performance. Grouped validation better tests generalization to unseen tools or experiments.

How used: the project downloads public machining datasets, cleans numeric sensor columns, derives regression/classification targets, splits by tool ID or experiment tag, and trains Random Forest baselines.

Current result: Vicomtech flank-wear regression R2 `0.8680`; Vicomtech wear-stage macro F1 `0.6472`; Katulu Uniwear macro F1 `0.5205`.

Research upgrade applied: the project page now emphasizes grouped validation, leakage control, and separate dataset reports rather than broad predictive-maintenance wording.

## 04 Materials Property ML

Why used: tabular property prediction is a standard materials-informatics baseline before larger deep-learning models are justified.

How used: the project downloads a public concrete-strength dataset, cleans composition/process variables, compares Ridge, Random Forest, and XGBoost regressors, reports MAE, RMSE, R2, and five-fold cross-validation, and uses SHAP for explainability.

Current result: XGBoost concrete compressive-strength regression R2 `0.9173`; Random Forest R2 `0.8990`.

Research upgrade applied: the project page now includes a model-comparison table, SHAP global ranking, SHAP beeswarm, and individual prediction explanation.

## Repository-Level Checks

- Public project index lists four independent projects only.
- Old shared-layer wording was removed from the visible project list.
- Guidance, CV, role, application, reviewer, and recommendation phrasing was scanned out of public Markdown.
- Markdown links and visual assets are covered by tests.
- Project figures use a shared visualization style with annotated metrics, residual-aware regression plots, normalized confusion matrices, and direct segmentation error-rate labels.
