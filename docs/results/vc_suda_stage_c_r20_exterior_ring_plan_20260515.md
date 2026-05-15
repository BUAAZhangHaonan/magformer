# VC-SUDA Stage C R20 Exterior Ring Loss Plan - 2026-05-15

## Conclusion

R20 is stopped after the external `ckpt249` eval. The matched pseudo-positive
exterior ring loss did not clear the first-checkpoint hard line on
target_unlabeled200, so `ckpt499` was not evaluated.

## Design

- Config: `configs/vc_suda_stage_c_r20_exterior_ring_r12_ckpt499_1024_teacher8499.yaml`.
- Init checkpoint: `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth`.
- Radius: `2`.
- Weight: `0.05`.
- Scope: only matched pseudo-positive masks after Hungarian matching.
- Loss: mean predicted foreground probability on the binary exterior ring.
- Default behavior: disabled by schema default.

The weight uses the R19 dry-run signal directly. The matched ring probability
mean was `0.111685`, so a raw `0.05` ring weight gives about `0.0056` main-head
loss before the existing Stage C `unsupervised_weight: 0.02`. That is small
beside pseudo mask/dice weights `5.0`, but large enough to show up in dry-run
loss diagnostics. This keeps R20 cautious without making the new term vanish.

## Guardrails

- No source, LR, pseudo threshold, mask/dice, top-k, or data changes.
- No unmatched-query background pressure.
- `checkpoint_max_keep: null`.
- No sweep.
- GPU launch only on `4,5,6,7`.

## Validation Plan

1. Unit tests:
   - default-off does not change pseudo loss;
   - enabled ring loss is positive;
   - changing only the target interior does not change ring loss;
   - zero-ring cases return zero loss and record counts.
2. Preflight:
   - `tools/verify_vc_suda_stage.py --config configs/vc_suda_stage_c_r20_exterior_ring_r12_ckpt499_1024_teacher8499.yaml --require-stage C`.
3. Dry-run:
   - run R19 signal diagnostics on R20 config and R12 `ckpt499`;
   - confirm `pseudo_exterior_ring_loss`, matched count, ring pixel count, and exterior ring probability are logged.
4. First checkpoint external eval:
   - evaluate `checkpoint_iter_0000249.pth`;
   - dataset `target_unlabeled200`;
   - 1024 backmap;
   - `--inference-topk 200`;
   - `--max-dets 200`.

## First Checkpoint Stop Rule

Hard stop R20 at `ckpt249` if either:

- segm AP `< 0.3195`;
- AP75 `< 0.284144`.

Even if AP is flat, stop if low-IoU false positives do not drop by at least 2%.

## External Eval - ckpt249

Completed on `2026-05-15 20:52 CST` from remote host `4029` at commit
`459fc49bf47cdc08412651856a82dc4186965e65`. GPU4-7 were checked before launch;
only Xorg was attached to GPU4-7, and no magformer process was running.

Artifacts, kept out of git:

- Output dir: `output/experiments/vc_suda_stage_c_r20_exterior_ring_iter0250_target_unlabeled200_1024_backmap_topk200_20260515_204935`
- Command: `output/experiments/vc_suda_stage_c_r20_exterior_ring_iter0250_target_unlabeled200_1024_backmap_topk200_20260515_204935/command.sh`
- Log: `output/experiments/vc_suda_stage_c_r20_exterior_ring_iter0250_target_unlabeled200_1024_backmap_topk200_20260515_204935/eval.log`
- Metrics: `output/experiments/vc_suda_stage_c_r20_exterior_ring_iter0250_target_unlabeled200_1024_backmap_topk200_20260515_204935/metrics.cocoeval.json`
- Inference stats: `output/experiments/vc_suda_stage_c_r20_exterior_ring_iter0250_target_unlabeled200_1024_backmap_topk200_20260515_204935/inference_stats.json`

Protocol: `tools/evaluate_1024_backmap.py`, target_unlabeled200,
`annotations/instances_target_unlabeled.json`, `split=train`, 1024 input,
bbox+segm, score threshold `0.05`, mask threshold `0.5`, forced PyTorch MSDA,
`--inference-topk 200`, and `--max-dets 200`. Strict weight load matched
`774/774` keys with `0` missing, `0` unexpected, and `0` shape mismatches.

| checkpoint | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | exported | topk truncated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ckpt249` | 0.390561 | 0.731578 | 0.376013 | 0.316952 | 0.648726 | 0.280995 | 13,652 | 0 / 200 |

Decision:

- R20 `ckpt249` segm AP `0.316952` is below the hard line `0.3195` by `-0.002548`.
- R20 `ckpt249` segm AP75 `0.280995` is below the hard line `0.284144` by `-0.003149`.
- It is below current best R15/R12 topk200 segm AP `0.320048` by `-0.003096`, so `ckpt499` was not evaluated.
