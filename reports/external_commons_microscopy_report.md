# External CoMMonS Microscopy Target

- source: [CoMMonS microscopic material surface dataset](https://github.com/olivesgatech/CoMMonS)
- sampled archive size: about `1.1 GB`
- status: `large_dataset_not_downloaded`

## Current Handling

CoMMonS is configured as an optional large microscopy-material benchmark. The sampled archive is about 1.1 GB, so the default run documents the target without downloading or committing the archive.

## Runnable Command

`python scripts/run_external_commons.py --config configs/external_commons_microscopy.json --allow-large-download`
