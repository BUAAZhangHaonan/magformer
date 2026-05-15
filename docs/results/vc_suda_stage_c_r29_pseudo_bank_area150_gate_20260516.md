# VC-SUDA Stage C R29 Pseudo Bank Area150 Gate - 2026-05-16

Conclusion: R29 does not pass the offline pseudo-bank training gate. Lowering `area_min` from `300` to `150` keeps global precision and area ratio within bounds, but dense `90-100` recall is still too low. Do not create a training config from this bank.

## Scope

- No training was started.
- No training config was created.
- Output artifacts were written under `output/diagnostics/r29_pseudo_bank_area150_gate_20260516/` and must stay uncommitted.
- R28 does not continue. Its formal external segm AP `0.246710` is far below the R15/R12 `ckpt499` topk200 reference `0.320048`, so the 512 short path remains a plumbing check only.
- The next training step is allowed only if the R29 pseudo-bank gate passes. R29 failed, so the next action is more bank diagnosis, not training.

## Fixed Inputs

R23 was checked read-only first. The fixed prediction source for this gate is the R15/R12 `ckpt499` topk200/maxDets200 output:

- Predictions: `output/diagnostics/r15_topk200_maxdets200_20260515/coco_instances_results.json`.
- Source COCO: `magformer_datasets/20260318_1K_32254/annotations/instances_train.json`.
- Target COCO: `magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json`.
- Hidden GT for quality readout: `magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json`.

R23 simple high-precision used `score_min=0.94`, `fill_min=0.55`, and `area_min=300`. It had no per-density topk or side constraints. R29 changes only `area_min: 300 -> 150`.

## Command

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/build_pseudo_bank.py \
  --predictions output/diagnostics/r15_topk200_maxdets200_20260515/coco_instances_results.json \
  --source-coco magformer_datasets/20260318_1K_32254/annotations/instances_train.json \
  --target-coco magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json \
  --hidden-gt magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json \
  --output-json output/diagnostics/r29_pseudo_bank_area150_gate_20260516/area150_pseudo_bank.json \
  --metrics-json output/diagnostics/r29_pseudo_bank_area150_gate_20260516/area150_metrics.json \
  --metrics-md output/diagnostics/r29_pseudo_bank_area150_gate_20260516/area150_metrics.md \
  --name r29_area150_gate \
  --score-min 0.94 --fill-min 0.55 --area-min 150
```

## Results

| scope | kept | kept/img | P50 | R50 | P75 | R75 |
|---|---:|---:|---:|---:|---:|---:|
| global | 6,097 | 30.485 | 0.936854 | 0.486128 | 0.604232 | 0.313532 |
| dense `90-100` | 1,485 | - | 0.884848 | 0.282398 | 0.440404 | 0.140554 |

Dense `90-100` bucket details:

| images | gt | raw_pred | kept | P50 | R50 | P75 | R75 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 47 | 4,653 | 4,981 | 1,485 | 0.884848 | 0.282398 | 0.440404 | 0.140554 |

Matched TP pred/GT mask area ratio at IoU 0.50:

| count | mean | p50 | p75 | p90 |
|---:|---:|---:|---:|---:|
| 5,712 | 1.141534 | 1.115596 | 1.220701 | 1.356843 |

## Gate

| criterion | required | observed | pass |
|---|---:|---:|---|
| global P50 | >= 0.92 | 0.936854 | yes |
| global P75 | >= 0.58 | 0.604232 | yes |
| dense `90-100` R50 | >= 0.35 | 0.282398 | no |
| dense `90-100` R75 | >= 0.18 | 0.140554 | no |
| dense `90-100` P50 | >= 0.85 | 0.884848 | yes |
| matched TP area ratio p90 | <= 1.40 | 1.356843 | yes |

Gate decision: fail.

## Read

Lowering `area_min` recovers only `45` kept masks versus R23 simple high precision, from `6,052` to `6,097`. Dense kept masks rise only from `1,464` to `1,485`, and dense R50 moves from `0.279390` to `0.282398`. This does not solve the dense recall blocker.

R28 should not be continued because it is much worse than the R15/R12 reference on the formal external target eval. R29 also does not unlock training because the bank fails the dense recall gate. No training config should be created from this result.
