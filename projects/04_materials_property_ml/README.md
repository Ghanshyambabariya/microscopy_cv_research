# Project 04: Materials Property ML

**Focus:** materials-informatics regression from composition/process variables to properties.

This project adds a compact real tabular materials dataset: concrete mix composition and curing age to compressive strength.

## Why This Project Is Unique

- Shows materials-property prediction beyond images and process signals.
- Uses a clean real dataset that runs quickly and gives interpretable feature importances.
- Demonstrates composition/process/property regression, a core materials-informatics workflow.

## Main Evidence

| Dataset | Task | Model | Result |
|---|---|---|---|
| Concrete compressive strength | property regression | RandomForest | R2 `0.8990` |

Top drivers from the trained model include curing age, cement, water, superplasticizer, and slag.

## Results

![Concrete strength](results/external_concrete_strength.png)

See: [results/external_concrete_strength_report.md](results/external_concrete_strength_report.md)

## Run

```powershell
python scripts/run_external_concrete.py --config configs/external_concrete_strength.json
```

## Next Upgrade

Add composition-based alloy, battery, polymer, or heat-treatment property datasets.

