# VC-SUDA Stage C R37 Balanced +25 True-Resume Plan - 2026-05-16

## Conclusion

R37 is the realistic-balanced `+25` split for the next Stage C true-resume gate. It does not reuse the R31 promoted set, because R36 showed that R31 promoted 25 is biased toward dense and harder images.

No training is started by this plan.

## R36 Readout

R36 found that R31 promoted 25 is too dense for a one-shot supervised expansion:

- Promoted 25 instance-count p50 was `99`, while the remaining target_unlabeled 175 p50 was `50`.
- Promoted 25 teacher segm AP was `0.231670`, while the remaining 175 was `0.336575`.
- R35 original-split true resume stayed near R12 (`0.320048 -> 0.319300`), while R34 `+25` true resume dropped to `0.301879`.

## Actual target_unlabeled200 Density

The current `instances_target_unlabeled.json` has exactly 200 images and 11,750 annotations. Instance counts are not continuous; they form three real density groups.

| instance count | images |
|---:|---:|
| 25 | 22 |
| 49 | 3 |
| 50 | 128 |
| 92 | 1 |
| 94 | 1 |
| 95 | 1 |
| 97 | 2 |
| 98 | 5 |
| 99 | 12 |
| 100 | 25 |

Bucket summary:

| bucket | available images | selected images |
|---|---:|---:|
| 25-30 | 22 | 3 |
| 46-60 | 131 | 16 |
| >90 | 47 | 6 |

There are no images outside `25-30`, `46-60`, and `>90`.

## R37 Selection

R37 follows the observed 200-image mix, about `22/131/47`, and selects `3/16/6` by bucket. Each bucket is sorted by original `image_id` ascending before taking the first images.

Selected original target_unlabeled image ids:

```text
35, 46, 49, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18, 19, 21, 14, 15, 16, 20, 27, 31
```

| bucket | original image_id | instances | file_name |
|---|---:|---:|---|
| 25-30 | 35 | 25 | `74437625201002_25_scene_000003_000571_v0.png` |
| 25-30 | 46 | 25 | `74437625201002_25_scene_000006_001238_v1.png` |
| 25-30 | 49 | 25 | `679303124022_25_scene_000004_000693_v1.png` |
| 46-60 | 1 | 50 | `15414194BA211_50_scene_000027_000201_v0.png` |
| 46-60 | 2 | 50 | `RB44145_50_scene_000026_000908_v0.png` |
| 46-60 | 4 | 50 | `76030110x_50_scene_000020_000512_v0.png` |
| 46-60 | 5 | 50 | `7449150079_50_scene_000028_000948_v0.png` |
| 46-60 | 6 | 50 | `RC-MASTER-SMD_C6032-28_T_50_scene_000029_001068_v1.png` |
| 46-60 | 7 | 50 | `687106149022_50_scene_000004_000376_v0.png` |
| 46-60 | 8 | 50 | `76030110x_50_scene_000029_001148_v0.png` |
| 46-60 | 9 | 50 | `658410821024_50_scene_000014_000288_v1.png` |
| 46-60 | 10 | 50 | `615002138421_50_scene_000007_000919_v0.png` |
| 46-60 | 11 | 50 | `686106148922_50_scene_000010_001088_v1.png` |
| 46-60 | 12 | 50 | `687106149022_50_scene_000014_000393_v0.png` |
| 46-60 | 13 | 50 | `679303124022_50_scene_000017_000510_v0.png` |
| 46-60 | 17 | 50 | `6_50_scene_000011_000125_v1.png` |
| 46-60 | 18 | 50 | `15414194BA211_50_scene_000027_000202_v1.png` |
| 46-60 | 19 | 50 | `6_50_scene_000006_000111_v0.png` |
| 46-60 | 21 | 50 | `615008138321_50_scene_000008_000842_v0.png` |
| >90 | 14 | 100 | `7448262510_XXL_100_scene_000000_000224_v0.png` |
| >90 | 15 | 99 | `658410821024_100_scene_000004_000695_v0.png` |
| >90 | 16 | 100 | `490107670612_100_scene_000004_000825_v1.png` |
| >90 | 20 | 99 | `SOIC127P1030X265-18N_100_scene_000011_000957_v0.png` |
| >90 | 27 | 100 | `618025231421_100_scene_000006_000390_v1.png` |
| >90 | 31 | 100 | `A-DF15A_KG-T2S_1_100_scene_000003_000969_v1.png` |

## Generated Split

- `magformer_datasets/pseudo_real_512/annotations/instances_target_labeled_r37_balanced_plus25.json`
- `magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled_r37_balanced_minus25.json`
- New labeled size: `50` images.
- New unlabeled size: `175` images.
- Image ids are remapped with the same namespace pattern used by R31: original labeled `1000000+id`, selected target_unlabeled `1100000+id`, remaining target_unlabeled `2000000+id`.

Validation checks passed during generation:

- labeled/unlabeled/val `image_id` sets are disjoint.
- labeled/unlabeled/val `file_name` sets are disjoint.
- every annotation `image_id` exists in its image set.
- no split contains empty images.
- annotation `bbox`, `area`, and `category_id` fields are valid.

## Config

Config: `configs/vc_suda_stage_c_r37_balanced_plus25_true_resume_1024_teacher8499.yaml`

It is copied from R34 and changes only run identity/output/log paths plus R37 annotation paths. It keeps the R34 runtime and training policy unchanged:

- `runtime.resume`: `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth`
- `solver.max_iter`: `750`
- `data.image_size`: `1024`
- source root: `magformer_datasets/20260318_1K_32254`
- loss, LR, pseudo-label, augmentation, and 32K source settings unchanged from R34.

## Gate

Run only after preflight passes.

Formal gate metric: iter750 external `target_unlabeled200` 1024 backmap eval with bbox + segm, topk/maxDets 200.

Decision rule:

- If segm AP `>= 0.320048`, R37 may continue.
- If segm AP is `0.319162-0.320048`, record the result but do not extend the run.
- If segm AP `< 0.319162`, stop.

If R37 fails, stop the label-expansion route instead of trying a larger label expansion.
