# Methodology Notes

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

## Minimum Paper-Style Sections

- Problem statement
- Data acquisition and labeling
- Preprocessing and split strategy
- Encoder comparison
- Synthetic generation method
- Hybrid learning method
- Downstream evaluation
- Error analysis and failure cases
