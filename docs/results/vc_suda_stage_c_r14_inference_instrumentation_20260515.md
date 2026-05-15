# VC-SUDA Stage C R14 Inference Instrumentation - 2026-05-15

Conclusion: R12 ckpt499 is hitting the inference topk candidate cap. This is a real eval-time blocking risk, not only an exported JSON artifact.

## Scope

- No training was started.
- No checkpoint sweep was run.
- The only diagnostic eval used R12 `checkpoint_iter_0000499.pth` on full `target_unlabeled200`.
- Output artifacts were written under `output/diagnostics/r14_inference_stats_20260515/` and should stay uncommitted.

## Code Change

Instrumentation is opt-in through `--dump-inference-stats`.

It records per image:

- `pre_topk_candidate_count`
- `topk_limit`
- `post_topk_count`
- `post_score_count`
- `post_mask_nonempty_count`
- `exported_count`
- `gt_count`
- `gt_density_bucket`

The default eval path does not write stats.

## Diagnostic Command

```bash
CUDA_VISIBLE_DEVICES=0 /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/evaluate_1024_backmap.py \
  --base-config output/experiments/vc_suda_stage_c_r12_32ksource_iter0500_target_unlabeled200_1024_backmap_20260515_1606/eval_base_r12_ckpt499.yaml \
  --dataset-root magformer_datasets/pseudo_real_512 \
  --ann annotations/instances_target_unlabeled.json \
  --split train \
  --weights output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth \
  --output-dir output/diagnostics/r14_inference_stats_20260515 \
  --image-size 1024 \
  --batch-size 4 \
  --num-workers 4 \
  --score-threshold 0.05 \
  --mask-threshold 0.5 \
  --iou-types bbox,segm \
  --dump-inference-stats output/diagnostics/r14_inference_stats_20260515/inference_stats.json \
  --force-pytorch-msda
```

The run evaluated all 200 target-unlabeled images. Metrics matched the prior ckpt499 result: bbox AP `0.3932659516`, segm AP `0.3200463988`, and `13,122` exported predictions.

## Stats Readout

Per-image topk:

| field | min | p50 | p90 | max | sum |
|---|---:|---:|---:|---:|---:|
| pre_topk_candidate_count | 200 | 200 | 200 | 200 | 40,000 |
| topk_limit | 100 | 100 | 100 | 100 | 20,000 |
| post_topk_count | 100 | 100 | 100 | 100 | 20,000 |
| post_score_count | 25 | 62.5 | 97 | 100 | 13,122 |
| post_mask_nonempty_count | 25 | 62.5 | 97 | 100 | 13,122 |
| exported_count | 25 | 62.5 | 97 | 100 | 13,122 |

Summary:

- `topk_truncated_images`: `200 / 200`.
- `score_filtered_images`: `190 / 200`.
- `mask_empty_filtered_images`: `0 / 200`.
- Images with `exported_count == 100`: `14, 27, 31, 64, 99, 183, 192, 219, 221, 223`.

GT density buckets:

| bucket | images | gt | pre_topk | post_topk | post_score | post_mask_nonempty | exported | topk_truncated_images |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-30 | 22 | 550 | 4,400 | 2,200 | 753 | 753 | 753 | 22 |
| 31-60 | 131 | 6,547 | 26,200 | 13,100 | 8,026 | 8,026 | 8,026 | 131 |
| 90-100 | 47 | 4,653 | 9,400 | 4,700 | 4,343 | 4,343 | 4,343 | 47 |

## Interpretation

The current inference path creates `200` class-query candidates per image and immediately keeps only `100`. Since every image has `pre_topk_candidate_count=200` and `post_topk_count=100`, the topk cap is active everywhere.

The strongest blocking evidence is the dense `90-100` bucket. It exports `4,343` predictions for `4,653` GT objects after topk, and 10 dense images export exactly `100` predictions. On those images, score filtering and non-empty mask filtering do not reduce the count after topk, so the cap directly decides the exported candidate budget.

Mask non-empty filtering is not the culprit in this run: `post_mask_nonempty_count` equals `post_score_count` for every image.

## Recommendation

Run one eval-only experiment that raises inference topk and COCO `maxDets` together, then compare against this R14 baseline. Do not train before that check.

The next change should be a controlled eval-time code path or CLI/runtime knob for candidate budget, for example `inference_topk=200` and `max_dets=200`. It should keep the default at `100` until the eval-only result proves the change helps.
