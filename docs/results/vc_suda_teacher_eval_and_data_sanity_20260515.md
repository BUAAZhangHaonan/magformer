# VC-SUDA Teacher Eval and Data Sanity - 2026-05-15

## Scope

This note records the 2026-05-15 process check, Teacher first50 1024 backmap recheck, and data validity checks for the VC-SUDA milestone.

## Process Check

- Host: WS-4029GP-TRT via `ssh 4029`.
- Repository: `/home/hdd3/zhanghaonan/magformer`.
- HEAD at check time: `059ade1`.
- GPU 4-7 were idle.
- No `magformer` train or eval process was found.
- `tmux` sessions existed for `eval_1k`, `eval_32k`, `vc_suda_stage_c_20260514_0734`, and `vc_suda_stage_c_r3_c05_20260514`.
- The panes were stopped at `bash`.
- The working tree only showed untracked output at that time: `?? output/upper_bound/`.

## Teacher First50 Recheck

Fresh 1024 backmap recheck output:

- `output/experiments/teacher_first50_1024_backmap_recheck_20260515/`

Metrics:

| Metric | AP |
| --- | ---: |
| bbox | `0.6519639237` |
| segm | `0.6253031403` |

Command protocol:

```bash
python tools/evaluate_1024_backmap.py \
  --dataset-root /home/hdd3/zhanghaonan/magformer/magformer_datasets/20260318_1K_1566 \
  --ann annotations/instances_all.json \
  --split all \
  --weights output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth \
  --image-size 1024 \
  --batch-size 4 \
  --max-images 50 \
  --iou-types bbox,segm
```

Important protocol point:

- The original 1.5K dataset root, annotation file, and split must be passed explicitly.
- Dataset root: `/home/hdd3/zhanghaonan/magformer/magformer_datasets/20260318_1K_1566`.
- Annotation: `annotations/instances_all.json`.
- Split: `all`.
- Weights: `output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth`.
- Image size: `1024`.
- Batch size: `4`.
- Max images: `50`.
- IoU types: `bbox,segm`.

Prediction JSON recheck:

- Predictions: `3421`.
- Unique image IDs: `50`.
- Bboxes are COCO `xywh`.
- Segmentations are RLE with size `1024x1024`.
- `category_id` is `1`.

## Root Cause Readout

The `1.9 AP` reading is not the real Teacher first50 result. It is more consistent with a temporary script path, a default pseudo-real path, or a wrong evaluation protocol.

For original 1.5K evaluation, always pass `dataset-root`, `ann`, and `split` explicitly. Otherwise the eval may silently use the pseudo-real default path or another mismatched target.

## Data Validity

The following datasets were checked:

- original first50
- pseudo_real val
- target_labeled
- target_unlabeled
- target_unlabeled_dev40

No checked split had empty images, missing target categories, bad boxes, bad masks, or missing image files.

Contact sheets were written under:

- `output/diagnostics/data_sanity_20260515/`

The contact sheets are diagnostic images and are not intended for commit.

## Data Statistics

| Split | Images | Annotations | Valid inst min | Valid inst p50 | Valid inst mean | Valid inst max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original first50 | `50` | `3315` | `25` | `50` | `66.30` | `100` |
| pseudo_real val | `28` | `1892` | `50` | `50` | `67.57` | `100` |
| target_labeled | `25` | `1697` | `50` | `50` | `67.88` | `100` |
| target_unlabeled | `200` | `11750` | `25` | `50` | `58.75` | `100` |
| target_unlabeled_dev40 | `40` | `2367` | `25` | `50` | `59.18` | `100` |

## Milestone Conclusion

The fresh Teacher first50 1024 backmap result is high and stable: bbox AP `0.6519639237`, segm AP `0.6253031403`. The earlier `1.9 AP` reading should not be used as the Teacher first50 baseline.

The checked data splits are structurally valid. The next reports should keep evaluation protocol labels explicit, especially whether the run uses original 1.5K data or pseudo-real target data.

Follow-up anomaly note: [VC-SUDA Teacher Pseudo-Real Anomaly - 2026-05-15](vc_suda_teacher_pseudoreal_anomaly_20260515.md) records the explicit pseudo_real val28 rerun, JSON sanity, shape diagnosis, and why Teacher direct pseudo_real AP must stay separate from StageB/R3 target-domain evaluation.
