# VC-SUDA R83 TTA COCO pseudo bank, 2026-05-17

R83 materializes the R82 R80 4-view TTA candidate bank into a COCO annotation JSON with segmentation RLE. This is a bank build and validation task only. No R83 training was launched.

## Scope

- Host: `4029`.
- Project: `/home/hdd3/zhanghaonan/magformer`.
- Conda env: `magformer`.
- GPU: `CUDA_VISIBLE_DEVICES=4`, tool device `cuda:0`.
- Config: `configs/vc_suda_stage_c_r80_unsup_schedule_1024.yaml`.
- Weights: `output/vc_suda/stage_c_r80_unsup_schedule_1024/checkpoint_iter_0000750.pth`.
- Target split: `magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json`, train split, 200 images.
- Views: scale `1.0/1.25` x `noflip/hflip`.
- Union: same class, mask IoU `>=0.55` or bbox IoU `>=0.75`; representative mask is the highest-quality candidate.
- Training filter: score `>=0.90`, mask area `>=20`, fill ratio `>=0.1`.

## Artifacts

- COCO bank: `output/diagnostics/r83_tta_coco_bank_20260517/instances_tta_pseudo_score090.json`.
- Summary JSON: `output/diagnostics/r83_tta_coco_bank_20260517/r83_tta_coco_bank_summary.json`.
- Bucket JSON: `output/diagnostics/r83_tta_coco_bank_20260517/r83_tta_coco_bank_buckets.json`.
- Runtime config: `output/diagnostics/r83_tta_coco_bank_20260517/r83_tta_coco_bank_runtime.yaml`.
- Full log: `output/logs/r83_tta_coco_bank_full200_20260517.log`.
- Validate-only log: `output/logs/r83_tta_coco_bank_validate_only_20260517.log`.
- Smoke log: `output/logs/r83_tta_coco_bank_smoke5_20260517.log`.

## Bank Statistics

| item | value |
|---|---:|
| images | 200 |
| annotations | 13,386 |
| empty images | 0 |
| categories | `[1]` |
| raw 4-view candidates | 72,119 |
| union candidates | 27,409 |
| dropped by score `<0.90` | 14,023 |
| dropped by mask area `<20` | 0 |
| dropped by fill ratio `<0.1` | 0 |
| R82 candidate JSON same-filter expected annotations | 13,386 |
| R82 count delta | 0 |
| duplicate same-class pairs mask IoU `>=0.95` | 0 |
| duplicate same-class pairs bbox IoU `>=0.95` | 0 |

| distribution | mean | p10 | p50 | p90 |
|---|---:|---:|---:|---:|
| score | 0.9533 | 0.9202 | 0.9594 | 0.9751 |
| mask area | 456.0 | 207.0 | 470.0 | 681.5 |
| bbox area | 692.8 | 380.0 | 700.0 | 986.0 |
| fill ratio | 0.6523 | 0.4828 | 0.6703 | 0.7913 |

## Coverage

These are bbox-IoU proxy coverage numbers against target_unlabeled GT, matching the R82 diagnostic convention. The candidate rows are the full R82 union bank. The kept rows are the R83 training-filtered bank.

| bucket | images | GT/proxy | candidates | kept | keep-rate | candidate cov@50/@75 | kept cov@50/@75 |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 200 | 11,750 | 27,409 | 13,386 | 0.488 | 0.837 / 0.478 | 0.800 / 0.465 |
| tiny_area_le_256 | 185 | 2,934 | 2,085 | 1,388 | 0.666 | 0.540 / 0.118 | 0.439 / 0.093 |
| bottom20_area | 181 | 2,355 | 1,512 | 967 | 0.640 | 0.483 / 0.090 | 0.377 / 0.067 |

The candidate coverage is identical to R82 because raw/union counts match the R82 full200 log exactly: `72,119` raw and `27,409` union. The kept coverage is lower than R82 score `>=0.1` kept coverage because R83 intentionally uses score `>=0.90` for training precision.

## Validation

Passed checks:

- `pycocotools.COCO` loads the bank.
- `CocoRgbdDataset` loads the bank as train annotations.
- 200 images are present.
- 0 empty images.
- Every annotation has segmentation RLE.
- Every RLE decodes to the image size.
- RLE area equals annotation `area`.
- RLE bbox equals annotation `bbox`.
- All bboxes are inside image bounds.
- Category IDs are legal.
- Image and depth paths exist.
- JSON numeric values are finite.
- Smoke/full/validate logs have no `Traceback`, `OOM`, `RuntimeError`, or non-finite errors.

Commands run:

```bash
CUDA_VISIBLE_DEVICES=4 /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/build_r83_tta_coco_bank.py \
  --output-dir output/diagnostics/r83_tta_coco_bank_smoke5_20260517 \
  --output-json output/diagnostics/r83_tta_coco_bank_smoke5_20260517/instances_tta_pseudo_score090_smoke5.json \
  --max-images 5 --report-interval 1

CUDA_VISIBLE_DEVICES=4 /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/build_r83_tta_coco_bank.py \
  --output-dir output/diagnostics/r83_tta_coco_bank_20260517 \
  --output-json output/diagnostics/r83_tta_coco_bank_20260517/instances_tta_pseudo_score090.json \
  --max-images 200 --report-interval 20

CUDA_VISIBLE_DEVICES=4 /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/build_r83_tta_coco_bank.py \
  --validate-only \
  --output-dir output/diagnostics/r83_tta_coco_bank_20260517 \
  --output-json output/diagnostics/r83_tta_coco_bank_20260517/instances_tta_pseudo_score090.json \
  --max-images 200
```

## Conclusion

The R83 offline pseudo bank is ready for the next training step. It is faithful to the R82 R80 4-view TTA candidate bank, includes real segmentation RLE from model masks, and passes COCO/RGB-D loader validation. R83 training should use the generated bank path above and keep the score `>=0.90`, mask area `>=20`, fill ratio `>=0.1` policy unless a later ablation changes the precision/recall target.
