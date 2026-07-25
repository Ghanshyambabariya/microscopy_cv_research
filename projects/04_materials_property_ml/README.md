# 04. Materials Property ML

[Back to project index](../README.md) | [Back to main README](../../README.md)

Independent project for material-property regression from tabular composition and process variables.

## Why Used

Many materials problems are tabular before they become deep-learning problems. A clean baseline with sensible preprocessing, feature importance, and a clear metric is often the first useful step.

## How Used

- downloaded a public tabular property dataset
- cleaned numeric features and target values
- split the data into train/test sets
- trained a Random Forest regression baseline
- reported MAE, RMSE, R2, and feature importance
- saved a result plot and Markdown report

## Final Results

| Dataset | Task | Model | Result |
|---|---|---|---|
| concrete compressive strength | property regression | Random Forest | R2 `0.8990` |

Top drivers in the current model include curing age, cement, water, superplasticizer, and slag.

## Research Alignment

The implementation follows materials-informatics baseline practice by using explicit preprocessing, interpretable feature importance, and regression metrics before introducing larger models.

## Results

![Concrete strength](results/external_concrete_strength.png)

Report: [results/external_concrete_strength_report.md](results/external_concrete_strength_report.md)

## Main Files

- `scripts/run_external_concrete.py`
- `configs/external_concrete_strength.json`
- `src/microscopy_cv_research/training/external_concrete.py`
- `docs/structure_property_modelling.md`

## Run

```powershell
python scripts/run_external_concrete.py --config configs/external_concrete_strength.json
```
