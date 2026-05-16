# VC-SUDA R63 StageB Source/Target Trajectory - 2026-05-16

## Conclusion

StageB does not have a checkpoint that satisfies both gates. None of the evaluated StageB checkpoints reaches original first50 source segm AP `>=0.55` and target_unlabeled200 segm AP `>=0.30` at the same time.

The source gate is already broken at the earliest available StageB checkpoint. Teacher8499 has original first50 source segm AP `0.625293` and target area ratio `8.537x` on target_unlabeled200. By StageB `ckpt0999`, target area ratio has moved to `1.570x`, but source segm AP is only `0.470519`. `ckpt1499` briefly recovers source to `0.539706`, still below `0.55`, while target segm AP falls to `0.229988`. Final `ckpt8999` has the best target AP among these StageB points, `0.297990`, but source is only `0.450872`.

This means the next step should implement a real multi-source/source-retention objective instead of more single-source StageB configuration trials.

## Checkpoints

StageB checkpoint files found:

| Directory | Available checkpoint files |
| --- | --- |
| `output/experiments/vc_suda_stage_b_1024_teacher8499_g4_7_retry_20260513_2207/` | `checkpoint_iter_0000999.pth`, `checkpoint_iter_0001499.pth` |
| `output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/` | `checkpoint_iter_0008999.pth`, `checkpoint_iter_0009000.pth`, `model_best.pth` |

No StageB `0000249`, `0000499`, or `0001999` checkpoint was present in these StageB output directories.

## Protocol

Source sanity used the fixed original first50 wrapper:

- Entry point: `tools/evaluate_teacher_first50_1024_backmap.py`
- Checker: `tools/check_eval_protocol.py original_first50_teacher --allow-nondefault-weights`
- Base config: `configs/finetune_1k_full_1024.yaml`
- Dataset: `magformer_datasets/20260318_1K_1566/annotations/instances_all.json`
- Split: `all`
- Max images: `50`
- Image size: `1024`
- Score/mask threshold: `0.05/0.5`
- Inference topk/maxDets: `100/100`
- Device: GPU4, PyTorch MSDA

Target eval used the established target_unlabeled200 protocol:

- Checker: `tools/check_eval_protocol.py pseudo_real_target_unlabeled200`
- Eval entry point: `tools/evaluate_1024_backmap.py --force-pytorch-msda`
- Base config: `configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml`
- Dataset: `magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json`
- Split: `train`
- Images: `200`
- Image size: `1024`
- Score/mask threshold: `0.05/0.5`
- Inference topk/maxDets: `200/200`
- Device: GPU5 for new target evals, PyTorch MSDA

## Eval Outputs

| Checkpoint | Source eval output | Target eval output |
| --- | --- | --- |
| Teacher8499 reference | `output/diagnostics/r44_teacher_first50_protocol_fix_20260516/` | `output/experiments/teacher_pseudoreal_target_unlabeled200_1024_backmap_recheck_20260515/` |
| StageB `ckpt0999` | `output/diagnostics/r63_stageb_ckpt0999_original_first50_20260516/` | `output/diagnostics/r63_stageb_ckpt0999_target_unlabeled200_20260516/` |
| StageB `ckpt1499` | `output/diagnostics/r63_stageb_ckpt1499_original_first50_20260516/` | `output/diagnostics/r63_stageb_ckpt1499_target_unlabeled200_20260516/` |
| StageB `ckpt8999` | `output/diagnostics/r63_stageb_ckpt8999_original_first50_20260516/` | `output/experiments/diagnostic_stageb_vs_r3a10_target_unlabeled_20260515/stage_b/` |

For reused final StageB target metrics, the R63 static target protocol summary is saved at `output/diagnostics/r63_stageb_ckpt8999_target_unlabeled200_protocol_20260516/`.

## Metrics

| Point | Source bbox AP | Source segm AP | Target bbox AP | Target segm AP | Target pred bbox area p50 | Target GT bbox area p50 | Area ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Teacher8499 reference | `0.651953` | `0.625293` | `0.000042` | `0.000026` | `6657.00` | `779.75` | `8.537x` |
| StageB `ckpt0999` | `0.513042` | `0.470519` | `0.319421` | `0.243529` | `1224.00` | `779.75` | `1.570x` |
| StageB `ckpt1499` | `0.579031` | `0.539706` | `0.307304` | `0.229988` | `1223.25` | `779.75` | `1.569x` |
| StageB `ckpt8999` | `0.522730` | `0.450872` | `0.378422` | `0.297990` | `931.50` | `779.75` | `1.195x` |

Area ratio is `median image-level prediction bbox-area p50 / median image-level GT bbox-area p50` over target_unlabeled200. This matches the R61 scale-readout style and reuses existing prediction JSONs where possible.

## Readout

- No crossing exists in the available StageB trajectory. `ckpt8999` is closest on target AP, but misses both gates: source `0.450872 < 0.55`, target `0.297990 < 0.30`.
- Source first50 segm AP dropped below `0.55` before or by the earliest available StageB checkpoint, `ckpt0999`. The repo does not retain an earlier StageB checkpoint to localize the exact iteration between Teacher8499 and `ckpt0999`.
- The target scale adaptation happens without source retention. Area ratio moves from Teacher `8.537x` to StageB `1.570x` at `ckpt0999` and `1.195x` at `ckpt8999`, while source AP remains below the source gate throughout.
- Continuing single-source StageB configuration trials is not the right next step. The evidence points to adding explicit multi-source/source-retention, then rechecking whether target scale adaptation and source sanity can coexist.

## Validation

- Source protocol checks passed for `ckpt0999`, `ckpt1499`, and `ckpt8999`; summaries are in each source output directory.
- Target protocol checks passed for `ckpt0999`, `ckpt1499`, and reused `ckpt8999`; summaries are in each target output/protocol directory.
- New evals were eval-only. No training was started and no model code was changed.
