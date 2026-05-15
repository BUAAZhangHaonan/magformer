# VC-SUDA Stage C R18 Pseudo Unmatched Negative Plan - 2026-05-15

## Conclusion

Run one short R18 continuation from R12 `ckpt499` with only one new variable:
target-unlabeled pseudo-branch high-score unmatched query background CE.

## Design

- Config: `configs/vc_suda_stage_c_r18_pseudo_unmatched_neg_r12_ckpt499_1024_teacher8499.yaml`.
- Init checkpoint: `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth`.
- Threshold: `pseudo_unmatched_negative_score_thresh: 0.9`.
- Weight: `pseudo_unmatched_negative_weight: 0.05`.
- Scope: target-unlabeled pseudo branch only, after Hungarian pseudo-positive matching.
- Default behavior: disabled by schema default.

The weight is deliberately below the existing pseudo CE scale. `pseudo_loss_ce`
uses weight `2.0`; R18 adds `0.05`, or 2.5% of that CE scale, and it is still
multiplied by the existing Stage C `unsupervised_weight: 0.02`. The threshold
starts at `0.9` because R17 found many unmatched queries above that score while
their max IoU to kept pseudo masks stayed low.

## Guardrails

- No source, LR, pseudo threshold, mask/dice, top-k, or data changes.
- `checkpoint_max_keep: null`.
- No sweep.
- GPU launch only on `4,5,6,7`.

## Validation Plan

1. Unit tests:
   - default-off loss keys are unchanged;
   - enabled loss penalizes only high-score unmatched foreground queries;
   - low-score unmatched and matched queries are not penalized;
   - config fields have strict types.
2. Preflight:
   - `tools/verify_vc_suda_stage.py --config configs/vc_suda_stage_c_r18_pseudo_unmatched_neg_r12_ckpt499_1024_teacher8499.yaml --require-stage C`.
3. Dry-run:
   - run R17 signal diagnostics on R18 config and R12 `ckpt499`;
   - confirm `pseudo_unmatched_high_score_count` and `pseudo_unmatched_negative_loss` appear in train loss diagnostics.
4. First checkpoint external eval:
   - evaluate first numbered checkpoint with external 1024 backmap on `target_unlabeled200`, bbox+segm.
   - hard stop if segm AP is below R7/R8B stopline; current R8B ckpt999 reference is `0.319162`, and current best R15/R12 topk200 is `0.320048`.

## First Checkpoint Eval Result

External 1024 backmap eval for R18 completed on `2026-05-15 19:16 CST`. The evaluated checkpoint was `output/vc_suda/stage_c_r18_pseudo_unmatched_neg_r12_ckpt499_1024_teacher8499/checkpoint_iter_0000249.pth`.

Eval artifacts:

- Output dir: `output/experiments/vc_suda_stage_c_r18_pun_iter0250_target_unlabeled200_1024_backmap_topk200_20260515_1913`
- Command: `output/experiments/vc_suda_stage_c_r18_pun_iter0250_target_unlabeled200_1024_backmap_topk200_20260515_1913/command.sh`
- Log: `output/experiments/vc_suda_stage_c_r18_pun_iter0250_target_unlabeled200_1024_backmap_topk200_20260515_1913/eval.log`
- Metrics: `output/experiments/vc_suda_stage_c_r18_pun_iter0250_target_unlabeled200_1024_backmap_topk200_20260515_1913/metrics.cocoeval.json`
- Inference stats: `output/experiments/vc_suda_stage_c_r18_pun_iter0250_target_unlabeled200_1024_backmap_topk200_20260515_1913/inference_stats.json`

Protocol: `tools/evaluate_1024_backmap.py`, target_unlabeled200, `annotations/instances_target_unlabeled.json`, `split=train`, 1024 input, bbox+segm, score threshold `0.05`, mask threshold `0.5`, forced PyTorch MSDA path, `--inference-topk 200`, and `--max-dets 200`. Strict weight load matched `774/774` keys with `0` missing, `0` unexpected, and `0` shape mismatches. A first attempt using the R18 training resolved config failed before model load because the eval override made `vc_suda.target_unlabeled_ann` match `data.val_ann`; the completed eval used the existing R15/R12 eval base and the R18 checkpoint weights.

| Checkpoint | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | Predictions | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ckpt249 | 0.393006 | 0.733227 | 0.380170 | 0.318701 | 0.648298 | 0.282244 | 13760 | Stopped: below current best and below 0.319 |

Readout:

- R18 `ckpt249` clears the R7 stopline `0.3171` by `+0.001601`, so it is not a catastrophic failure.
- It is below current best R15/R12 topk200 `0.320048` by `-0.001347`, so R18 did not improve the target_unlabeled200 gate.
- It is below `0.319`; per the first-checkpoint monitor rule, R18 was stopped after this eval instead of continuing to `ckpt499`.
- Runtime check before stopping found active training on GPU4-7, finite logged losses through iter `340`, no current `train_launch.log` Traceback/OOM/non-finite hits, and CPU/RAM below the hard-stop line. After `Ctrl-C`, the tmux pane exited, no R18 train processes remained, and GPU4-7 returned to idle.
