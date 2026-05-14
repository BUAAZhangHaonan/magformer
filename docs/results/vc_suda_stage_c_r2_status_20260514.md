# VC-SUDA Stage C-R2 status - 2026-05-14

## Current status

Stage C-R2 has been stopped after iter2047. The tmux session `vc_suda_stage_c_r2_20260514` was stopped gracefully with Ctrl-C. It was not stopped with `kill -9`.

R2 iter1000 is only the internal R2 stop-loss checkpoint. Under the same 1024 backmap full eval protocol, it is lower than Stage B and should not be recommended as a model that improves on Stage B.

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

Conclusion: iter1000 is not better than Stage B under the same eval protocol. It is only the R2 internal stop-loss point.

## Corrected same-protocol Stage B comparison

Stage B has now been evaluated through the same 1024 backmap full eval entry. The output directory is `output/experiments/vc_suda_stage_b_1024_backmap_full_eval_20260514`.

| eval | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage B same-protocol 1024 backmap full eval | 0.332271 | 0.693367 | 0.287965 | 0.252354 | 0.583612 | 0.175047 |
| Stage C-R2 iter1000 same-protocol full eval | 0.3192 | 0.6763 | 0.2635 | 0.2436 | 0.5716 | 0.1618 |

Corrected conclusion: Stage C-R2 iter1000 is lower than Stage B on bbox AP/AP50/AP75 and segm AP/AP50/AP75. The earlier comparison against the old Stage B eval entry used different eval protocols and should not be used to claim that R2 is better than Stage B.

R2 should therefore not be recommended as a Stage B improvement. The only retained value from R2 is that iter1000 is the internal stop-loss point, because iter2000 regressed in quick bbox eval.

## R3 direction

Run a threshold single-variable path first. The recommended next run is R3-A with pseudo threshold `0.15` and all other settings unchanged. Before starting R3-A, run the pseudo threshold sweep gate.

## Next required steps

1. Do not continue R2 training.
2. Do not recommend Stage C-R2 iter1000 as better than Stage B.
3. Run the pseudo threshold sweep gate before starting R3-A.
4. Start R3-A with threshold `0.15` only if the gate passes.
