# VC-SUDA Stage C R15 TopK/MaxDets Recovery - 2026-05-15

Conclusion: raising eval-only inference topk and COCO maxDets from 100 to 200 removes the dense truncation, but it does not materially improve R12 `ckpt499` segmentation AP.

## Scope

- No training was started.
- No checkpoint sweep was run.
- The only eval used R12 `checkpoint_iter_0000499.pth` on full `target_unlabeled200`.
- Output artifacts were written under `output/diagnostics/r15_topk200_maxdets200_20260515/` and should stay uncommitted.

## Code Path

The default candidate budget remains 100.

R15 adds explicit eval-time controls:

- `runtime.eval_inference_topk`, default `100`
- `runtime.eval_max_dets`, default `100`
- `tools/evaluate.py --inference-topk/--max-dets`
- `tools/evaluate_1024_backmap.py --inference-topk/--max-dets`
- `tools/export_results.py --inference-topk/--max-dets`

For `max_dets=200`, `COCOEvaluator` sets `COCOeval.params.maxDets=[1,10,200]`. Metric extraction reads AP directly from the COCO precision tensor at the configured final maxDets, because pycocotools' `stats[0]` summary slot is hardcoded to `maxDets=100`.

## Diagnostic Command

```bash
CUDA_VISIBLE_DEVICES=0 /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/evaluate_1024_backmap.py \
  --base-config output/experiments/vc_suda_stage_c_r12_32ksource_iter0500_target_unlabeled200_1024_backmap_20260515_1606/eval_base_r12_ckpt499.yaml \
  --dataset-root magformer_datasets/pseudo_real_512 \
  --ann annotations/instances_target_unlabeled.json \
  --split train \
  --weights output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth \
  --output-dir output/diagnostics/r15_topk200_maxdets200_20260515 \
  --image-size 1024 \
  --batch-size 4 \
  --num-workers 4 \
  --score-threshold 0.05 \
  --mask-threshold 0.5 \
  --iou-types bbox,segm \
  --inference-topk 200 \
  --max-dets 200 \
  --dump-inference-stats output/diagnostics/r15_topk200_maxdets200_20260515/inference_stats.json \
  --force-pytorch-msda
```

The run evaluated all 200 target-unlabeled images.

## Metrics

| setting | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | exported |
|---|---:|---:|---:|---:|---:|---:|---:|
| R14 topk100/maxDets100 | 0.393266 | 0.734464 | 0.379222 | 0.320046 | 0.648343 | 0.284144 | 13,122 |
| R15 topk200/maxDets200 | 0.394476 | 0.741115 | 0.379187 | 0.320048 | 0.648363 | 0.284144 | 13,806 |
| Delta | +0.001210 | +0.006651 | -0.000036 | +0.000002 | +0.000020 | +0.000000 | +684 |

## Truncation Readout

| setting | pre_topk sum | topk_limit | post_topk sum | topk truncated | exported max | images exported at cap |
|---|---:|---:|---:|---:|---:|---:|
| R14 topk100/maxDets100 | 40,000 | 100 | 20,000 | 200 / 200 | 100 | 10 |
| R15 topk200/maxDets200 | 40,000 | 200 | 40,000 | 0 / 200 | 127 | 0 |

R15 proves the hard topk cap was removed. It also proves this cap was not the main reason segm AP was stuck near `0.3200`.

Dense bucket `90-100` changed from `4,343` exported predictions for `4,653` GT objects to `4,981` exported predictions. That extra candidate budget did not move segm AP.

## Interpretation

The dense truncation issue is resolved for eval-only topk/maxDets=200.

The next bottleneck is not candidate count after inference topk. The useful signal is that bbox AP50 rises by `+0.006651`, while bbox AP75 and segm AP stay flat. This points more toward localization/mask quality than export truncation.

## Validation

Focused tests:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python -m pytest \
  tests/test_magformer_raw_inference.py::test_inference_raw_collects_topk_candidate_stats \
  tests/test_magformer_raw_inference.py::test_inference_raw_accepts_explicit_topk_limit_for_dense_eval \
  tests/test_eval_runtime_contract.py::test_cocoevaluator_forwards_custom_max_dets_to_cocoeval \
  tests/test_eval_runtime_contract.py::test_cocoevaluator_extracts_ap_for_custom_final_max_dets \
  tests/test_coco_export_parity.py::test_evaluate_1024_defaults_to_100_and_accepts_dense_200_flags -q
```

The custom maxDets AP extraction test was verified red first, then green after the evaluator fix.
