# Build Stats Count Discrepancy Note

The `expected_images` and `images_written` fields in `magformer_datasets/20260318_1K_1566/build_stats.json` describe two different stages of the dataset build.

## Fields checked

- `build_stats.tasks_total = 2264`
- `build_stats.expected_images = 4528`
- `build_stats.images_written = 1566`
- `build_stats.views_skipped_mask_parity = 2962`
- `dataset_info.build_dataset.num_views = 2`

## Arithmetic

- `tasks_total × num_views = expected_images`
- `2264 × 2 = 4528`

This means `expected_images` is the pre-filter render count implied by the number of build tasks and the configured two views per task.

## Why `images_written` is smaller

`images_written = 1566` is the retained output count after build-time filtering, not the raw expected render count.

The strongest direct evidence is that:

- `expected_images - images_written = 4528 - 1566 = 2962`
- `build_stats.views_skipped_mask_parity = 2962`

The JSON therefore supports this interpretation:

- `expected_images` counts all planned views before filtering.
- `images_written` counts only the views that were actually written after filtering.
- In this dataset build, the reduction is fully explained by `views_skipped_mask_parity`.

## Conclusion

The `4528` vs `1566` gap is expected behavior for this dataset build, not an unexplained counting bug. The two fields are measuring different stages of the pipeline.
