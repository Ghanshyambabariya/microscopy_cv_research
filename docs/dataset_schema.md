# Dataset Schema

Minimum table columns for `data/processed/labels.csv`:

- `image_path`
- `target_class`
- `property_value`
- `specimen_id`
- `split_group`
- `magnification`
- `stain`
- `batch_id`
- `acquisition_date`

## Notes

- `specimen_id` or `split_group` drives leakage-safe splits.
- `property_value` can be omitted if you are not running regression yet.
- Additional metadata is preserved even if unused in the first benchmark run.
