# Real Microscopy Benchmark Map

This report separates implemented microscopy benchmark evidence from documented public benchmark references.

## Encoder References

| Encoder or Corpus | Role | Why it matters | Source |
|---|---|---|---|
| NASA MicroNet | microscopy-pretrained encoder family | Microscopy-specific transfer learning baseline with benchmark segmentation datasets and pretrained encoders. | https://github.com/nasa/pretrained-microscopy-models |
| CEM500K | cellular electron microscopy pretraining corpus | Large-scale unlabeled TEM-style micrograph corpus for self-supervised or transfer learning. | https://empiar.ipr.pdbj.org/ko/entry/10592/ |

## Real Benchmark Targets

| Modality | Task | Dataset | Status | Methodological purpose | Source |
|---|---|---|---|---|---|
| SEM | semantic segmentation | NASA benchmark segmentation data | not yet integrated | Transfer learning from microscopy encoders on materials microstructure segmentation. | https://github.com/nasa/pretrained-microscopy-models |
| SEM | indentation-mark segmentation | Indentation mark segmentation data | not yet integrated | Defect or feature segmentation on real SEM images tied to materials characterization. | https://zenodo.org/record/7639189 |
| TEM | representation learning and transfer | CEM500K | not yet integrated | Comparison of generic, biomedical, and microscopy-specific encoders on real TEM-like imagery. | https://empiar.ipr.pdbj.org/ko/entry/10592/ |
| TEM | public archive showcase | EMPIAR | partially represented by external showcase only | Open electron microscopy data ingestion reference. | https://www.ebi.ac.uk/empiar/policies/ |
| EBSD | orientation or phase prediction | Northwestern simulated EBSD dataset | not yet integrated | Connection between microscopy CV and structure-property or crystallographic prediction tasks. | https://www.scholars.northwestern.edu/en/datasets/simulated-ebsd-dataset |
| EBSD | pattern indexing reference | EBSD indexing CNN | not yet integrated | EBSD-oriented learning task reference beyond texture-class labels. | https://github.com/NU-CUCIS/EBSD-indexing |

## Interpretation

- Current integrated evidence is SEM segmentation with generated masks, metric tables, and qualitative figures.
- TEM and EBSD entries are documented as public benchmark references, not reported results.
- The table separates implemented status from reference status for each modality.
