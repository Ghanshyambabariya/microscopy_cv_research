# Methodology Notes

These notes define the standard MicroForge AI should follow as it grows into publishable materials-AI research.

## Recommended Experimental Order

1. Data audit and label QA
2. Supervised classification benchmark
3. Supervised regression benchmark for structure-property prediction
4. Synthetic generation quality study
5. Synthetic-to-real downstream utility study
6. Hybrid multitask or consistency-based learning

## Research Standards To Preserve

- Split by specimen, patient, wafer, batch, or acquisition session
- Report mean and standard deviation across multiple seeds
- Separate model selection, ablation, and final blind test evaluation
- Record acquisition metadata: microscope, magnification, stain, illumination, batch
- Treat synthetic data as a controlled intervention, not as a free accuracy booster
- Report dataset provenance, license, number of samples, label source, and known bias
- Include failure cases, not only best examples
- Compare simple baselines against deep models before claiming improvement
- Keep train, validation, and test transforms explicit and version controlled

## Professor Review Checklist

| Standard | Expected Evidence |
|---|---|
| Dataset validity | source link, label description, sample count, license note |
| Leakage control | split by physical unit, tool ID, experiment, batch, or acquisition session |
| Reproducibility | config file, script command, random seed, saved report |
| Baseline strength | simple model, stronger model, and ablation comparison |
| Scientific interpretation | error analysis tied back to microstructure/process/property meaning |

## Minimum Paper-Style Sections

- Problem statement
- Data acquisition and labeling
- Preprocessing and split strategy
- Encoder comparison
- Synthetic generation method
- Hybrid learning method
- Downstream evaluation
- Error analysis and failure cases
