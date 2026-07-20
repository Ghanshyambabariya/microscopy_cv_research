# Real Microscopy Benchmark Notes

This report separates the current synthetic starter benchmark from the real microscopy targets the project should support next.

## Encoder References

| Encoder or Corpus | Role | Why it matters | Source |
|---|---|---|---|
| NASA MicroNet | microscopy-pretrained encoder family | Microscopy-specific transfer learning baseline with benchmark segmentation datasets and pretrained encoders. | https://github.com/nasa/pretrained-microscopy-models |
| CEM500K | cellular electron microscopy pretraining corpus | Large-scale unlabeled TEM-style micrograph corpus for self-supervised or transfer learning. | https://empiar.ipr.pdbj.org/ko/entry/10592/ |

## Real Benchmark Targets

| Modality | Task | Dataset | Status | Project goal | Source |
|---|---|---|---|---|---|
| SEM | semantic segmentation | NASA benchmark segmentation data | not yet integrated | Show transfer learning from microscopy encoders on materials microstructure segmentation. | https://github.com/nasa/pretrained-microscopy-models |
| SEM | indentation-mark segmentation | Indentation mark segmentation data | not yet integrated | Demonstrate defect or feature segmentation on real SEM images tied to materials characterization. | https://zenodo.org/record/7639189 |
| TEM | representation learning and transfer | CEM500K | not yet integrated | Compare generic, biomedical, and microscopy-specific encoders on real TEM-like imagery. | https://empiar.ipr.pdbj.org/ko/entry/10592/ |
| TEM | public archive showcase | EMPIAR | partially represented by external showcase only | Show that the framework is ready to ingest open electron microscopy data, not only synthetic images. | https://www.ebi.ac.uk/empiar/policies/ |
| EBSD | orientation or phase prediction | Northwestern simulated EBSD dataset | not yet integrated | Connect microscopy CV with structure-property or crystallographic prediction tasks. | https://www.scholars.northwestern.edu/en/datasets/simulated-ebsd-dataset |
| EBSD | pattern indexing reference | EBSD indexing CNN | not yet integrated | Benchmark the repo against a real EBSD-oriented learning task instead of only texture-class toy labels. | https://github.com/NU-CUCIS/EBSD-indexing |

## Interpretation

- The current repo is a valid framework demonstration, but it is not yet comparable to NASA-style microscopy transfer-learning benchmarks.
- The next scientific step is to ingest at least one real SEM task, one TEM task, and one EBSD task and report task-appropriate metrics.
- Real evidence should include actual test images, predicted outputs, metric tables, and failure-case visualizations.
