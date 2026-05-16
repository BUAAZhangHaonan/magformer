# VC-SUDA R61 target-scale visualization - 2026-05-16

## Conclusion

R61 supports the R60 diagnosis.

On the same 200 `target_unlabeled` images, Teacher8499 and R59 predict boxes that are far too large for the target GT scale, while R46 sits on the GT scale and has much higher GT best-IoU.

## Inputs

No training and no new eval were run. R61 reused the existing COCO result JSONs:

- target annotations: `magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json`
- target images: `magformer_datasets/pseudo_real_512/images/train`
- Teacher predictions: `output/experiments/teacher_pseudoreal_target_unlabeled200_1024_backmap_recheck_20260515/coco_instances_results.json`
- R59 predictions: `output/diagnostics/r59_retention_min_ckpt0250_target_unlabeled200_20260516/coco_instances_results.json`
- R46 predictions: `output/diagnostics/r46_pseudoreal_source_true_resume_ckpt0750_target_unlabeled200_1024_backmap_20260516/coco_instances_results.json`

## Output

The script is `tools/visualize_target_scale_predictions.py`.

It wrote:

- contact sheet: `output/diagnostics/r61_target_scale_visualization_20260516/contact_sheet.png`
- selected-image stats JSON: `output/diagnostics/r61_target_scale_visualization_20260516/selected_stats.json`
- selected-image stats CSV: `output/diagnostics/r61_target_scale_visualization_20260516/selected_stats.csv`
- all-image stats JSON/CSV: `output/diagnostics/r61_target_scale_visualization_20260516/all_image_stats.json` and `.csv`
- summary: `output/diagnostics/r61_target_scale_visualization_20260516/summary.json`

The contact sheet has four columns per image: `GT / Teacher / R59 / R46`.

## Image Selection

Selection is deterministic and explainable.

The script sorts images by this preference:

1. R46 GT best-IoU p50 is at least `0.5`.
2. Teacher and R59 predicted bbox area p50 are both at least `2x` GT bbox area p50.
3. Higher R46 IoU wins, then stronger Teacher/R59 oversize wins.

The selected image ids were:

`197, 113, 150, 212, 127, 157, 116, 182, 193, 201`.

## Scale Statistics

All-image median over the 200 target images:

| Source | bbox area p50 over images | GT best-IoU p50 over images | pred count p50 |
|---|---:|---:|---:|
| GT | `779.75` | n/a | n/a |
| Teacher8499 | `6657.0` | `0.1407655469` | `56.0` |
| R59 | `4021.0` | `0.2508430642` | `124.5` |
| R46 | `777.5` | `0.7619912591` | `63.0` |

Ratio to GT bbox area p50:

| Source | ratio |
|---|---:|
| Teacher8499 | `8.54x` |
| R59 | `5.16x` |
| R46 | `1.00x` |

Selected-sheet examples show the same pattern. For image `197`, GT bbox area p50 is `910`; Teacher is `6223`, R59 is `4312`, and R46 is `899`. R46 best-IoU p50 is `0.9354`, while Teacher and R59 are `0.1676` and `0.2576`.

## Readout

This is a scale failure, not an empty-export failure.

Teacher and R59 both produce predictions, but their median boxes are several times larger than target GT. R46 produces a similar number of predictions to Teacher, but its median bbox area is aligned with GT and its best-IoU p50 is much higher. That matches R60: Teacher/R59 are target-scale mismatched, while R46 is target-scale aligned.

## Validation

- `python -m pytest -q tests/test_visualize_target_scale_predictions.py`: passed, `4 passed`.
- `python tools/visualize_target_scale_predictions.py`: passed and generated the output files above.
- The script discovers prediction JSONs from existing eval directories and does not invoke training or model eval.
