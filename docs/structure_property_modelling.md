# Structure-Property Modelling

## Scientific Task

Microscopic images often support two broad scientific tasks:

1. Classification
- defect category
- microstructure family
- phase class
- treatment condition
- crack / pore / inclusion presence

2. Regression
- grain size
- porosity fraction
- hardness
- roughness proxy
- strength / conductivity / toughness estimate

## Methodology

Structure-property modelling is implemented as a progression from image-only classification to property regression and multimodal prediction. The workflow separates visual descriptors, process metadata, and tabular material variables so each contribution can be measured.

## Reference Projects

### For microscopy generation
- CELL-Diff: https://github.com/BoHuangLab/CELL-Diff
- guided-I2I on JUMP microscopy imagery: https://github.com/crosszamirski/guided-I2I
- DISPR for diffusion-based microscopy augmentation ideas: https://github.com/marrlab/DISPR

### For structure/property prediction framing
- CGCNN: https://github.com/txie-93/cgcnn
- OGCNN: https://github.com/RishikeshMagar/OGCNN

CGCNN and OGCNN are not microscopy-image projects. They are included as references for property-prediction framing, regression metrics, and model evaluation discipline.
