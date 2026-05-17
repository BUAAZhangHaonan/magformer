# R101 MagFormer Cleanfit R99 Solver Ablation

Date: 2026-05-18
Branch: feature/vc-suda-sim2real
Config: configs/baseline_supervised_r101_magformer_cleanfit_r99solver_1024.yaml

## Goal

Separate whether the R100 gain comes from optimizer/LR/schedule or from disabling augmentation/noise.

## Setup

R101 keeps the R100 clean/no-aug data setup and R98 initialization, but switches the solver back to the R99 weak solver:

- base_lr: 1.0e-6
- warmup_factor: 0.001
- warmup_iters: 50
- weight_decay: 0.07
- lr_scheduler: cosine

## Results

Training completed at 500 iter in tmux session `r101_magformer_cleanfit_r99solver`.

- Config commit: `7f4b7ae129f15d496b57b5d7bfda433d9164e1e8`
- Output dir: `output/baseline/r101_magformer_cleanfit_r99solver_1024`
- Final checkpoint: `output/baseline/r101_magformer_cleanfit_r99solver_1024/checkpoint_iter_0000500.pth`
- External train25 eval dir: `output/diagnostics/r101_magformer_cleanfit_r99solver_iter0500_train_labeled25_1024_backmap_topk200_20260518`

External 1024 backmap protocol matches R99/R100: bbox+segm, input `1024`, score threshold `0.05`, mask threshold `0.5`, inference topk `200`, COCO maxDets `200`, `MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch`, `--force-pytorch-msda`, strict load `774/774` keys.

| eval | images | GT anns | predictions | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | topk truncated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train25 | 25 | 1697 | 2887 | `0.175436` | `0.505351` | `0.075867` | `0.117507` | `0.374081` | `0.025809` | `0/25` |

For comparison:

| run | train25 segm AP | train25 segm AP75 |
| --- | ---: | ---: |
| R99 | `0.082423` | `0.017616` |
| R101 | `0.117507` | `0.025809` |
| R100 | `0.481398` | `0.523051` |

## Gate

Evaluate train25 first. If train25 segm AP75 < 0.10, stop and do not evaluate val28 or target_unlabeled200.

Decision: stop. R101 train25 segm AP75 is `0.025809`, below `0.10`. Val28 and target_unlabeled200 were not evaluated.

## Conclusion

R101 is much closer to R99 than to R100. Keeping the clean/no-aug data setup while reverting only the solver to R99 leaves train25 AP75 near the failed R99 line, not the recovered R100 line.

This supports the claim that the R100 jump mainly came from the stronger solver setup, not just from disabling flip/photo aug/depth noise.
