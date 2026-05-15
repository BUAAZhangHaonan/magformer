# VC-SUDA Stage C R31 Target Labeled +25 Plan - 2026-05-16

## Conclusion

R31 tests one variable: move 25 deterministic target_unlabeled images into target_labeled. The 1024 Stage C setup, 32K source branch, LR, losses, pseudo threshold, augmentation, depth normalization, and R12 `ckpt499` warm start stay unchanged.

## Single Variable

- Base config: `configs/vc_suda_stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499.yaml`.
- R31 config: `configs/vc_suda_stage_c_r31_target_labeled_plus25_1024_teacher8499.yaml`.
- Output: `output/vc_suda/stage_c_r31_target_labeled_plus25_1024_teacher8499`.
- Warm start: `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth`.
- New target labeled ann: `annotations/instances_target_labeled_r31_plus25.json`.
- New target unlabeled ann: `annotations/instances_target_unlabeled_r31_minus25.json`.
- Training gate length: `solver.max_iter: 250`.
- Quick eval/checkpoint period: `250`.
- Checkpoint retention: `runtime.checkpoint_max_keep: null`.
- Built-in best saving: `runtime.eval_saves_best: false`.

## Split Construction

Selection is deterministic from `instances_target_unlabeled.json`. Each bucket is sorted by original `image_id` ascending, and no fallback is used.

Bucket availability and use:

| Bucket | Available | Selected |
|---|---:|---:|
| 90-100 instances | 47 | 15 |
| 31-60 instances | 131 | 8 |
| 25-30 instances | 22 | 2 |

Selected images:

| Bucket | Original image_id | Instances | file_name |
|---|---:|---:|---|
| 90-100 | 14 | 100 | `7448262510_XXL_100_scene_000000_000224_v0.png` |
| 90-100 | 15 | 99 | `658410821024_100_scene_000004_000695_v0.png` |
| 90-100 | 16 | 100 | `490107670612_100_scene_000004_000825_v1.png` |
| 90-100 | 20 | 99 | `SOIC127P1030X265-18N_100_scene_000011_000957_v0.png` |
| 90-100 | 27 | 100 | `618025231421_100_scene_000006_000390_v1.png` |
| 90-100 | 31 | 100 | `A-DF15A_KG-T2S_1_100_scene_000003_000969_v1.png` |
| 90-100 | 36 | 97 | `679303124022_100_scene_000003_000488_v0.png` |
| 90-100 | 39 | 100 | `615002138421_100_scene_000006_000677_v1.png` |
| 90-100 | 41 | 100 | `76030110x_100_scene_000004_000139_v0.png` |
| 90-100 | 48 | 100 | `68710814522_100_scene_000007_000044_v1.png` |
| 90-100 | 62 | 97 | `749014013_100_scene_000005_000867_v0.png` |
| 90-100 | 64 | 100 | `618025231421_100_scene_000005_001166_v1.png` |
| 90-100 | 70 | 100 | `7448262510_XXL_100_scene_000004_000142_v1.png` |
| 90-100 | 77 | 99 | `615008160321_100_scene_000000_000630_v0.png` |
| 90-100 | 83 | 100 | `490107670612_100_scene_000008_000002_v0.png` |
| 31-60 | 1 | 50 | `15414194BA211_50_scene_000027_000201_v0.png` |
| 31-60 | 2 | 50 | `RB44145_50_scene_000026_000908_v0.png` |
| 31-60 | 4 | 50 | `76030110x_50_scene_000020_000512_v0.png` |
| 31-60 | 5 | 50 | `7449150079_50_scene_000028_000948_v0.png` |
| 31-60 | 6 | 50 | `RC-MASTER-SMD_C6032-28_T_50_scene_000029_001068_v1.png` |
| 31-60 | 7 | 50 | `687106149022_50_scene_000004_000376_v0.png` |
| 31-60 | 8 | 50 | `76030110x_50_scene_000029_001148_v0.png` |
| 31-60 | 9 | 50 | `658410821024_50_scene_000014_000288_v1.png` |
| 25-30 | 35 | 25 | `74437625201002_25_scene_000003_000571_v0.png` |
| 25-30 | 46 | 25 | `74437625201002_25_scene_000006_001238_v1.png` |

Output split counts:

| Split | Images | Annotations |
|---|---:|---:|
| target_labeled R31 +25 | 50 | 3638 |
| target_unlabeled R31 -25 | 175 | 9809 |

The original pseudo_real split files reuse `image_id` values across splits, and promoted ids also collide with original target_labeled ids. R31 uses deterministic split-specific id offsets in the new files so labeled, unlabeled, and val have no `image_id` or `file_name` overlap. Annotation `image_id` values were updated to match the new image ids, and annotation ids use the same split-specific offset style.

## Validation Contract

The split generation gate checks:

- New labeled has exactly `50` images and no empty image.
- New unlabeled has exactly `175` images and no empty image.
- Labeled, unlabeled, and val have no `image_id` or `file_name` overlap.
- Every annotation `image_id` exists in its file.
- Bbox width/height and area are positive.
- Category ids are in the shared category set.

## Preflight

Run before training:

```bash
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate magformer
python tools/verify_vc_suda_stage.py \
  --config configs/vc_suda_stage_c_r31_target_labeled_plus25_1024_teacher8499.yaml \
  --stage C \
  --emit-data-evidence \
  --evidence-max-samples 1
```

This preflight is read-only for model state and does not start training.

Preflight result on 2026-05-16:

- `PASS config=configs/vc_suda_stage_c_r31_target_labeled_plus25_1024_teacher8499.yaml`.
- Checks included `checkpoint_semantics`, `source_target_evidence`, `unlabeled_batch_strip`, and `depth_nonconstant`.
- Evidence counts: source `25654`, target_labeled `50`, target_unlabeled `175`, val `28`.
- Warm start role: `r12_ckpt499_continuation`.

## Gate

Use the formal external `target_unlabeled175` 1024-backmap evaluation for the decision after the 250-step checkpoint. The gate should use bbox+segm and the same topk200/maxDets200 protocol used for R15/R12.

Success gate:

- Continue only if formal target_unlabeled segm AP is at least the current best R15/R12 line, `0.320048`.

Loose no-drop line:

- `0.319162` is the R8B `ckpt999` no-drop reference.
- A result between `0.319162` and `0.320048` is not a success. Record it as non-collapsed, but do not extend R31 based only on that.

Failure gate:

- Stop if formal target_unlabeled segm AP is below `0.319162`.
- Stop if bbox+segm evaluation cannot complete under the standard 1024 backmap path.

## Gate Result

R31 completed the planned first gate on 2026-05-16. The training run used a clean environment with `LD_LIBRARY_PATH` and `CUDA_HOME` unset, CUDA MSDA training via `MAGFORMER_MS_DEFORM_ATTN_BACKEND=cuda`, physical GPUs `4,5,6,7` exposed as local `--gpus 0,1,2,3`, and finished `250/250` iterations. The first gate checkpoint exists at `output/vc_suda/stage_c_r31_target_labeled_plus25_1024_teacher8499/checkpoint_iter_0000249.pth`.

Internal val28 bbox smoke result at the end of training:

| Eval | AP | AP50 | AP75 | Note |
|---|---:|---:|---:|---|
| val28 bbox | 0.0934 | 0.3390 | 0.0261 | Smoke only; do not use for model-selection conclusions. |

Formal external `target_unlabeled200` 1024 backmap bbox+segm eval used `ckpt249`, `--inference-topk 200`, and `--max-dets 200`.

| Type | AP | AP50 | AP75 |
|---|---:|---:|---:|
| bbox | 0.3710117 | 0.7082011 | 0.3447274 |
| segm | 0.2946807 | 0.6293110 | 0.2364584 |

Prediction count was `15254`. The inference stats used `topk_limit=200`, had `topk_truncated_images=0/200`, and the COCO eval used maxDets `200`.

Gate decision: fail. R31 segm AP `0.2946807` is below the no-drop reference `0.319162`, so R31 stops at `ckpt249`; do not continue to `ckpt499`.

## Retrospective

Adding a small amount of hidden target GT (`+25`) did not solve the current bottleneck. It reduced the R12/R15 external target_unlabeled200 segm AP line from `0.320048` to `0.294681`, so the issue is not only the count of labeled target images.

The likely failure mode is that short fine-tuning with the EMA teacher and pseudo branch damages the already useful teacher behavior. The next step should be to diagnose R31 versus R12 prediction differences, or run a teacher-frozen / supervised-only upper-bound check. Continuing to expand target labels before that diagnosis is not the right next move.
