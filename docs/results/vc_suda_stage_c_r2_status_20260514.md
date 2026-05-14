# VC-SUDA Stage C-R2 status - 2026-05-14

## Current status

Stage C-R2 has been stopped after iter2047. The tmux session `vc_suda_stage_c_r2_20260514` was stopped gracefully with Ctrl-C. It was not stopped with `kill -9`.

R2 iter1000 is the current candidate checkpoint, but it is not the final target. The full eval segm AP is 24.36, which is still far from the 61+ AP target.

Iter2000 regressed in bbox-only quick eval, so continuing this run was stopped to save compute.

## Stop record

- Remote: `4029:/home/hdd3/zhanghaonan/magformer`
- tmux session: `vc_suda_stage_c_r2_20260514`
- Final max training iter reached: 2047
- Stop method: graceful Ctrl-C, no `kill -9`
- GPU status: GPUs 4-7 released
- Process status: no remaining R2 `torchrun` or `tools/train.py` process
- Output dir: `output/vc_suda/stage_c_r2_1024_teacher8499`

## Retained checkpoints

- `output/vc_suda/stage_c_r2_1024_teacher8499/checkpoint_iter_0000999.pth`
- `output/vc_suda/stage_c_r2_1024_teacher8499/checkpoint_iter_0001999.pth`
- `output/vc_suda/stage_c_r2_1024_teacher8499/model_best.pth`

## Loaded weights and schedule

- Warm-start checkpoint: `output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth`
- Warm-start result: missing keys = 0, unexpected keys = 0
- `max_iter=4000`
- `eval_period=1000`
- `ckpt_period=1000`
- Fast eval mode: bbox-only
- `eval_max_images=28`
- `eval_batch_size=4`

## Stage C-R2 bbox-only quick eval

The iter1000 eval row is recorded at iter999 in `metrics_log.csv`, which corresponds to the iter1000 scheduled eval. The iter2000 eval row is recorded at iter1999.

| eval | bbox AP | bbox AP50 | bbox AP75 | segm metrics |
| --- | ---: | ---: | ---: | --- |
| Stage C-R2 iter1000 bbox-only quick eval | 0.1713 | 0.5041 | 0.0806 | not run |
| Stage C-R2 iter2000 bbox-only quick eval | 0.1522 | 0.4801 | 0.0637 | not run |

Conclusion: iter2000 regressed against iter1000, so stopping the run was the right compute-saving choice.

## Stage C-R2 iter1000 full eval

Full eval used `tools/evaluate_1024_backmap.py --force-pytorch-msda` on pseudo_real val 28 with `checkpoint_iter_0000999.pth`.

| eval | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage C-R2 iter1000 full eval | 0.3192 | 0.6763 | 0.2635 | 0.2436 | 0.5716 | 0.1618 |

Conclusion: iter1000 is the current R2 candidate. It improves strongly under the 1024 backmap full eval, but it is still not the final target because segm AP is 24.36 and remains far from 61+.

## Stage B comparison note

The old Stage B full eval result was:

| eval | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage B old full eval entry | 0.1637 | 0.4824 | 0.0727 | 0.1128 | 0.3810 | 0.0350 |

This Stage B number came from the old eval entry. It should not be treated as a same-protocol comparison against Stage C-R2 iter1000 full eval. Stage B must be re-evaluated through the same `tools/evaluate_1024_backmap.py --force-pytorch-msda` 1024 backmap entry before making the final R2 comparison.

## Next required steps

1. Re-evaluate Stage B through the same 1024 backmap full eval entry to avoid comparing different eval protocols.
2. Use that same-protocol Stage B result as the baseline for judging R2.
3. Run R3 after the same-protocol Stage B comparison is available.
