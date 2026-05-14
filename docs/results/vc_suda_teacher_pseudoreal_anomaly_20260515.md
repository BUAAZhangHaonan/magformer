# VC-SUDA Teacher Pseudo-Real Anomaly - 2026-05-15

## Scope

This note records the Teacher direct eval anomaly on the pseudo_real target validation split. It also records why this result must not replace StageB or StageC/R3 target-domain evaluation.

## Fixed Points

- Host: WS-4029GP-TRT via `ssh 4029`.
- Repository: `/home/hdd3/zhanghaonan/magformer`.
- Teacher checkpoint: `output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth`.
- Teacher eval config for the pseudo_real rerun: `configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml`.
- Pseudo_real rerun protocol: explicit dataset root, annotation file, and split.

## Known-Good Source-Domain Check

The same Teacher checkpoint is normal on the original 1.5K first50 1024 backmap check.

| Eval target | bbox AP | segm AP |
| --- | ---: | ---: |
| original 1.5K first50, 1024 backmap | `0.6519639237` | `0.6253031403` |

This rules out a broken checkpoint as the primary explanation for the pseudo_real result.

## Pseudo-Real Direct Eval Result

The Teacher remains near zero on pseudo_real val28 after rerunning with explicit dataset root, annotation file, and split.

| Eval target | bbox AP | segm AP |
| --- | ---: | ---: |
| pseudo_real val28 | `0.000001615` | `0.000003715` |

The rerun used `configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml` and explicit dataset inputs, so this is not explained by a silent default split switch.

## JSON Sanity

The prediction JSON is structurally valid.

- Image IDs: `28` unique IDs.
- Predictions: `1509`.
- Mask encoding: RLE, `512x512`.
- Category: `category_id = 1`.
- Bboxes: inside image bounds.
- Score median: `0.6433`.

This points away from empty output, category mismatch, corrupt RLE, or out-of-bounds boxes.

## Shape Diagnosis

The dominant mismatch is object scale and annotation convention, not a global shift.

| Quantity | Value |
| --- | ---: |
| pseudo_real GT bbox median size | about `27x28` |
| Teacher predicted bbox median size | about `70x87` |
| bbox max IoU median | `0.157` |
| predictions with bbox IoU >= 0.5 | `1 / 1509` |
| mask max IoU median | `0.161` |
| predictions with mask IoU >= 0.5 | `2 / 1509` |

Centers do not show an obvious global translation. The failure is better explained by target-domain scale or annotation convention mismatch: the Teacher often predicts plausible target-region responses, but the boxes and masks are much larger than the pseudo_real GT objects.

## Relation To StageB And R3

The roughly 30 AP target_unlabeled200 results are not Teacher direct results.

| Model | Checkpoint | target_unlabeled200 segm AP |
| --- | --- | ---: |
| StageB | `output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth` | `0.2980` |
| StageC/R3 | `output/vc_suda/stage_c_r3_a10_1024_teacher8499/checkpoint_iter_0000999.pth` | `0.3068` |

Those numbers come from the same backmap protocol on target_unlabeled200, after adaptation. They should stay separate from Teacher direct pseudo_real val28 evaluation.

## Conclusion

Teacher direct pseudo_real val28 AP is near zero because the target split has a scale or annotation convention mismatch with the Teacher predictions. It is not a sign that the Teacher checkpoint is broken, because the same checkpoint is strong on original 1.5K first50 1024 backmap.

Use this result only as an anomaly diagnosis. Do not use it to replace StageB or StageC/R3 target-domain evaluation. Do not use older incompatible low-AP readings as evidence for the Teacher baseline.

The next useful diagnostic is a target-labeled overfit run. That should check whether the supervised target chain can learn the target annotation convention when labels are available.

## Evidence Boundary

This note intentionally relies only on the explicit rerun, JSON sanity checks, shape diagnosis, and the target_unlabeled200 StageB/R3 backmap results listed above. Older incompatible readings and superseded stage comparisons are not used here.
