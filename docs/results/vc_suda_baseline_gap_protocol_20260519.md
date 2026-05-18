# VC-SUDA Baseline Gap Protocol - 2026-05-19

## Research Question

The current question is not whether VC-SUDA reaches Teacher-10AP alone. The useful comparison is:

1. How large is the source-to-pseudo-real sim2real gap under the same 32K synthetic setup?
2. Does VC-SUDA improve over fair labeled-only baselines on the same pseudo-real target splits?

Teacher-10AP can stay as a historical sanity target, but it should not be the only target-domain success criterion.

## Fair Protocol

Use a fixed source and target protocol:

- Source setup: train on the 32K synthetic source cache at 1024 input resolution.
- Pseudo-real target setup: use the deterministic target150 train split, val28 anchor split, and remaining75 non-leakage split.
- MagFormer target evaluation: run 1024 backmap eval with `topk=200` and `maxDets=200` for val28 and remaining75.
- Official RGB Mask2Former target evaluation: run the official wrapper with fixed test input `INPUT.MIN_SIZE_TEST 1024`, `INPUT.MAX_SIZE_TEST 1024`, `MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100`, `TEST.DETECTIONS_PER_IMAGE 100`, then replay `coco_instances_results.json` against the original source-RLE annotations with `tools/replay_official_coco_results.py`.
- Report source-val AP and target AP separately, then compute the practical sim2real gap from source-only target eval and labeled-only target eval.

Leakage risks:

- `full200` includes promoted target training images after target150, so it is only a continuity reference.
- `train200` is an oracle fit check, not a fair generalization number.
- Pseudo-bank tuning on remaining75 can leak validation pressure into the unsupervised target pool. Treat it as diagnosis unless the bank rule is frozen before the final comparison.

## Current Facts

| Run | Role | Split / protocol | segm AP | Status |
| --- | --- | --- | ---: | --- |
| R98 | MagFormer source-only | 32K source val subset, first 300 val images | `0.810410` | Strong source-domain learning signal. |
| R98 | MagFormer source-only | pseudo-real target_unlabeled200, 1024 backmap | `0.0000009269` | Near-zero target transfer under the documented source-only check. |
| R114 | MagFormer labeled-only target150 | val28, 1024 backmap | `0.321831` | Current fair labeled-only anchor. |
| R114 | MagFormer labeled-only target150 | remaining75, 1024 backmap | `0.392173` | Current non-leakage labeled-only target score. |
| R134 | Official RGB Mask2Former source-only | source val, 32K source cache | `0.602268` | Best source checkpoint is `model_0004499.pth`. |
| R134 | Official RGB Mask2Former source-only | val28 / remaining75, fixed1024 source-RLE replay | `0.00000103 / 0.00000040` | Source-only RGB transfer to pseudo-real is effectively zero. |
| R136 | Official RGB Mask2Former labeled-only target150 | val28 / remaining75, fixed1024 built-in maxDets100 | `0.226573 / 0.291094` | Fairer fixed-size diagnostic row; below R114 by `0.095258 / 0.101079` segm AP. |
| R136 | Official RGB Mask2Former labeled-only target150 | val28 / remaining75, fixed1024 source-RLE replay maxDets100/200 | `0.226913 / 0.290370` | Replay row against original source-RLE annotations; maxDets100 and 200 match because predictions are capped at 100 per image. |
| R137 | Official RGB Mask2Former labeled-only target150, continued to 2000 iters | val28 / remaining75, fixed1024 built-in maxDets100 | `0.249466 / 0.322779` | Fixed-size eval-only row; improves R136 by `0.022893 / 0.031685`, but remains below R114 by `0.072365 / 0.069394`. |
| R137 | Official RGB Mask2Former labeled-only target150, continued to 2000 iters | val28 / remaining75, fixed1024 source-RLE replay maxDets100/200 | `0.253215 / 0.324278` | Replay row against original source-RLE annotations; improves R136 replay by `0.026303 / 0.033907`, but remains below R114 by `0.068615 / 0.067895`. |
| R118 | VC-SUDA from R114 | val28 / remaining75 / full200 at iter0099 | `0.321894 / 0.392568 / 0.520202` | Approximately neutral and failed its gates. |
| R122 | VC-SUDA depth-boundary from R114 | val28 / remaining75 at iter0099 | `0.322904 / 0.392410` | AP is approximately neutral; bucket go/no-go failed. |

Interpretation: R114 already gives a strong labeled-only target baseline. The early VC-SUDA variants have not shown a clear target-domain gain over that baseline. R134 now shows that official RGB Mask2Former source-only also has an almost complete sim2real collapse on pseudo-real target splits before target finetuning.

## R134 Source-Only Official RGB Target Eval

Checkpoint:

```text
output/baseline/r134_official_m2f_32k_cache_continue_3000_5000_g67/model_0004499.pth
```

Output dirs:

```text
output/diagnostics/r134_official_m2f_sourceonly_val28_fixed1024_eval
output/diagnostics/r134_official_m2f_sourceonly_remaining75_fixed1024_eval
```

Both eval commands used `CUDA_VISIBLE_DEVICES=6,7`, `--eval-only`, the official wrapper `baselines/run_official_mask2former_ecc.py`, fixed1024 test input, 100 object queries, and 100 detections per image.

| Split | Built-in bbox AP | Built-in segm AP | Replay bbox AP | Replay segm AP | Replay kept / input |
| --- | ---: | ---: | ---: | ---: | ---: |
| val28 | `0.0000` | `0.0001` | `0.000000` | `0.000103` | `1692 / 2800` |
| remaining75 | `0.0000` | `0.0000` | `0.000000` | `0.000040` | `4409 / 7500` |

Replay metric files:

```text
output/diagnostics/r134_official_m2f_sourceonly_val28_fixed1024_eval/source_rle_replay_metrics.cocoeval.json
output/diagnostics/r134_official_m2f_sourceonly_remaining75_fixed1024_eval/source_rle_replay_metrics.cocoeval.json
```

The replay `maxDets100` and `maxDets200` values match because the official prediction JSON contains at most 100 detections per image.

## R137 Target150 Official RGB Fixed1024 Eval

Checkpoint:

```text
output/baseline/r137_official_m2f_r136iter1000_target150_continue_1000_2000_g67/model_final.pth
```

Output dirs:

```text
output/diagnostics/r137_official_m2f_val28_fixed1024_eval
output/diagnostics/r137_official_m2f_remaining75_fixed1024_eval
```

Both eval commands used `CUDA_VISIBLE_DEVICES=6,7`, `--eval-only`, the official wrapper `baselines/run_official_mask2former_ecc.py`, fixed1024 test input, 100 object queries, and 100 detections per image. The logs show `Start inference` and no `Starting training` line.

| Split | Built-in bbox AP | Built-in segm AP | Replay bbox AP | Replay segm AP | Replay kept / input |
| --- | ---: | ---: | ---: | ---: | ---: |
| val28 | `27.1572` | `24.9466` | `27.157245` | `25.321549` | `2407 / 2800` |
| remaining75 | `34.2840` | `32.2779` | `34.284000` | `32.427769` | `6511 / 7500` |

Replay metric files:

```text
output/diagnostics/r137_official_m2f_val28_fixed1024_eval/source_rle_replay_metrics.cocoeval.json
output/diagnostics/r137_official_m2f_remaining75_fixed1024_eval/source_rle_replay_metrics.cocoeval.json
```

The R137 replay row improves R136 by `2.630275` AP on val28 and `3.390738` AP on remaining75. It still trails R114 by `6.861504` AP on val28 and `6.789497` AP on remaining75.

## Gap And Recovery

All values in this table are segm AP points. Recovery means `target150 AP - source-only target AP`. Recovery percent means `recovery / (R134 source-val AP - source-only target AP)`.

R134 source-val segm AP at iter4499 is `60.226842`.

| Split | R134 source-only target AP | Source-to-target gap | R136 replay AP | R136 recovered | R137 replay AP | R137 recovered | R114 target150 AP | R114 recovered | R114 - R137 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val28 | `0.000103` | `60.226739` | `22.691274` | `37.68%` | `25.321549` | `42.04%` | `32.183053` | `53.44%` | `6.861504` |
| remaining75 | `0.000040` | `60.226802` | `29.037031` | `48.21%` | `32.427769` | `53.84%` | `39.217266` | `65.12%` | `6.789497` |

R137 recovers more of the source-only collapse than R136 after target150 finetuning, but it still trails MagFormer R114 by about `6.86` AP on val28 and `6.79` AP on remaining75 under the documented target protocols.

## Non-MagFormer Baseline Status

R137 official RGB Mask2Former now has fixed1024 eval-only and source-RLE replay rows for val28 and remaining75. R134 official RGB Mask2Former source-only also has matching fixed1024 target rows, so the labeled-only recovery can be measured from an explicit source-only target anchor instead of inferred from source-val AP alone.

Historical R97 facts before the R134-R136 chain:

- It used official RGB-only Mask2Former on the same 32K source cache.
- It ran only 1500 iterations.
- Its source val segm AP reached `36.2643` at iter1500.
- Its bbox AP was unusable in that export because all serialized prediction boxes were zero, while segmentation AP was non-zero.

Current caveat:

- Do not call the official RGB target rows true topk200 results: replay `maxDets=200` equals `maxDets=100` because the official prediction JSON contains at most 100 detections per image.

## Next Execution Matrix

| Step | Run target | Purpose | Status |
| ---: | --- | --- | --- |
| 1 | Continue R97 source on 32K / 1024 | Get a fairer official RGB Mask2Former source baseline. | Done by R134; best source segm AP is `60.226842` at iter4499. |
| 2 | R134 source-only fixed1024 target eval | Measure source-only RGB pseudo-real AP before target finetuning. | Done; source-only target AP is effectively zero on val28 and remaining75. |
| 3 | R134 iter4499 labeled-only target150 finetune | Match the MagFormer R114 target-supervised setup. | Done by R136. |
| 4 | Evaluate R136/R137 on val28 and remaining75 | Compare non-MagFormer labeled-only against R114 and VC-SUDA. | Done; R137 narrows the R136 gap but remains below R114 on both target splits. |

Decision rule: VC-SUDA should be claimed as useful only if it beats the frozen labeled-only target baseline on val28 and remaining75 without relying on full200/train200 or post-hoc pseudo-bank tuning.
