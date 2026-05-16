# VC-SUDA R52 target_unlabeled sampling setup, 2026-05-16

R52 is a prediction-only sampling mechanism check. It does not train 250 iter and does not run model eval.

## Goal

R46 is the current best Stage C line on `target_unlabeled200`, with segm AP/AP75 `0.323252/0.287485`. R50/R51 source-scale changes did not improve the gate, so R52 changes only target_unlabeled sampling during Stage C training setup.

## Implementation

- Added `tools/build_target_unlabeled_sampling_stats.py`.
- Added `vc_suda.target_unlabeled_sampling` config with `enabled` and `stats_path`.
- Added `SemiSupervisedDataset.target_unlabeled_index_sequence`.
- Kept source indexing as `idx % len(source)`.
- Kept target unlabeled collate image-only; labels, masks, boxes, and annotations are not emitted for target weak/strong views.

The stats builder reads target COCO `images` only. It ignores target `annotations` and rejects GT-like fields in prediction/stats inputs such as `gt_count`, `gt_density_bucket`, and `annotations`.

## Stats

Command:

```bash
python tools/build_target_unlabeled_sampling_stats.py \
  --target-ann magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json \
  --r46-pred output/diagnostics/r46_pseudoreal_source_true_resume_ckpt0750_target_unlabeled200_1024_backmap_20260516/coco_instances_results.json \
  --r12-pred output/diagnostics/r33_r12_ckpt499_sanity_unlabeled200_20260516/coco_instances_results.json \
  --output output/diagnostics/r52_target_unlabeled_sampling_20260516/target_sampling_stats.json
```

Summary:

| Bucket | Images | Repeat |
|---|---:|---:|
| normal | 158 | 1 |
| dense | 30 | 2 |
| dense_tiny | 12 | 3 |

Sequence length: `254`.

## Config

New config:

`configs/vc_suda_stage_c_r52_pseudoreal_source_target_sampling_true_resume_1024_teacher8499.yaml`

It is based on R46. The only training semantic change is:

```yaml
vc_suda:
  target_unlabeled_sampling:
    enabled: true
    stats_path: output/diagnostics/r52_target_unlabeled_sampling_20260516/target_sampling_stats.json
```

Name, output directory, and logger run name were changed to keep artifacts separate. Source remains pseudo_real source, and target splits, resume, LR, losses, runtime, and max_iter stay aligned with R46.

## Smoke

Dataset instantiation with the R52 config reported:

- `source=1008`
- `target_labeled=25`
- `target_unlabeled=200`
- `target_unlabeled_sequence=254`
- repeat histogram `{1: 158, 2: 30, 3: 12}`
- first source indices `[0, 1, 2, 3, 4, 5]`
- first target ids `[80, 10, 85, 13, 82, 28, 217, 131, 62, 62]`

No 250-iter training was started.
