# Structure-Property Roadmap

## Where This Project Can Go

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

## Better Scientific Formulation

If your true goal is property prediction, do not stop at classification.

Recommended progression:
- image-only classification benchmark
- image-only regression benchmark
- multitask model predicting both class and property
- metadata-aware model that uses process settings together with images

## Online Projects Worth Borrowing Ideas From

### For microscopy generation
- CELL-Diff: https://github.com/BoHuangLab/CELL-Diff
- guided-I2I on JUMP microscopy imagery: https://github.com/crosszamirski/guided-I2I
- DISPR for diffusion-based microscopy augmentation ideas: https://github.com/marrlab/DISPR

### For structure/property prediction framing
- CGCNN: https://github.com/txie-93/cgcnn
- OGCNN: https://github.com/RishikeshMagar/OGCNN

These last two are not microscopy-image projects, but they are excellent references for how to formulate property prediction as a serious regression or classification research problem with strong evaluation discipline.

## My Suggestion

Use your microscopic image encoder as the visual branch and later add process metadata or structure descriptors as a second branch. That usually gives a more believable structure-property model than images alone.
