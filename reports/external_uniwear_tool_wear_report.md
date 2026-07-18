# External Uniwear Tool-Wear Benchmark

- source: [Katulu Uniwear Dataset](https://github.com/katulu-io/uniwear-dataset)
- cleaned rows: `39903`
- experiment groups: `12`
- window features: `1229`
- group split: train `839`, test `390`

## Results

- tool-wear regression MAE: `0.0383`
- tool-wear regression RMSE: `0.0517`
- tool-wear regression R2: `0.2397`
- wear-stage accuracy: `0.5667`
- wear-stage macro F1: `0.5205`

![External Uniwear benchmark](figures/external_uniwear_tool_wear.png)

## Why This Matters

This benchmark adds a second real online materials-process dataset with vibration and force signals. It tests whether the platform can ingest a different schema, window the time series, extract features, split by experiment, train models, and return quantitative results.
