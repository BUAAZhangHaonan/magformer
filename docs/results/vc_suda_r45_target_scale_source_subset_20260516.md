# VC-SUDA R45 Target-Scale Source Subset Diagnosis, 2026-05-16

R45 fails the prior gate. The 32K source annotation cannot provide an image-level source subset that is close to the pseudo_real target scale under the current COCO annotations.

No training, model eval, or GPU job was run.

## Inputs and Outputs

Tool:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/build_target_scale_source_subset.py \
  --source-ann magformer_datasets/20260318_1K_32254/annotations/instances_train.json \
  --target-ann magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json \
  --output-ann magformer_datasets/20260318_1K_32254/annotations/instances_train_target_scale_subset_r45.json \
  --target-images 200 \
  --seed 45 \
  --summary-json output/diagnostics/r45_target_scale_source_subset_20260516/summary.json
```

Independent protocol comparison:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/analyze_vc_suda_dataset_protocol.py \
  --ann target_unlabeled200=magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json \
  --ann ordinary_32254_train=magformer_datasets/20260318_1K_32254/annotations/instances_train.json \
  --ann r45_target_scale_subset=magformer_datasets/20260318_1K_32254/annotations/instances_train_target_scale_subset_r45.json \
  --output-json output/diagnostics/r45_target_scale_source_subset_20260516/protocol_summary.json \
  --output-md output/diagnostics/r45_target_scale_source_subset_20260516/protocol_summary.md
```

Generated files:

- `magformer_datasets/20260318_1K_32254/annotations/instances_train_target_scale_subset_r45.json`
- `output/diagnostics/r45_target_scale_source_subset_20260516/summary.json`
- `output/diagnostics/r45_target_scale_source_subset_20260516/protocol_summary.json`
- `output/diagnostics/r45_target_scale_source_subset_20260516/protocol_summary.md`

## Result

| dataset | images | instances | inst/img p50 | inst/img p90 | mask area p50 | area<=256 |
|---|---:|---:|---:|---:|---:|---:|
| target_unlabeled200 | 200 | 11,750 | 50 | 100 | 452 | 0.249702 |
| ordinary_32254_train | 25,654 | 1,398,531 | 50 | 99 | 7,572 | 0.011850 |
| R45 subset | 200 | 10,512 | 50 | 50 | 4,756 | 0.013033 |

Density buckets from the independent protocol summary:

| dataset | density 25-30 | density 46-60 | density >90 |
|---|---:|---:|---:|
| target_unlabeled200 | 22 | 131 | 47 |
| ordinary_32254_train | 4,145 | 15,258 | 5,192 |
| R45 subset | 0 | 187 | 12 |

## Gate

The prior gate requires mask area p50 inside `0.75x-1.5x` of target p50 `452`, so the allowed range is `339-678`.

R45 subset mask area p50 is `4,756`, or `10.52x` target. It fails the p50 gate.

R45 subset `area<=256` is `0.013033`. This is only slightly above the original 32K source `0.011850` and remains far below target `0.249702`. It fails the small-area gate.

## Interpretation

This is not a train-now result. The best image-level 200-image subset selected by the fixed scoring rule improves source p50 from `7,572` to `4,756`, but it is still far outside the target-scale band.

A read-only source candidate audit gives the same answer: across all 25,654 source images, the minimum image-level median mask area is `2,779`; there are `0` source images with image median `<=678` or `<=452`. The maximum image-level `area<=256` ratio is `0.155556`, and `0` source images reach `0.20`.

So R45 stops here. Do not enter training from this subset.
