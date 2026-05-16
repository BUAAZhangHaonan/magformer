# VC-SUDA Stage C R32 Plus25 Unsupervised Weight 0 Plan - 2026-05-16

## Conclusion

R32 completed the planned 250-iter single-variable check, and the formal external gate failed. Stop R32 at `ckpt249`; do not continue to `ckpt499`.

The key result is that disabling the pseudo unlabeled loss recovered R31 only slightly, from segm AP `0.2947` to `0.2997`, and still stayed below the R12/R15 reference `0.320048`. This means the R31 drop is not explained only by pseudo loss. The `+25` short continuation itself, or the EMA/optimizer/LR schedule/warm-start path, may be damaging the existing teacher.

## R31 Regression Evidence

R31 degraded against the R15/R12 topk200 reference:

| Signal | R31 | R15/R12 reference | Read |
|---|---:|---:|---|
| segm AP | 0.2946807 | 0.320048 | worse |
| predictions | 15,254 | 13,806 | more predictions, lower AP |
| score p50 | 0.9037 | 0.9354 | lower confidence |
| mask area ratio mean | 0.00238 | 0.00177 | masks grew on average |

Both promoted and non-promoted target groups degraded, so the problem is not isolated to the 25 images moved into the labeled split.

## R32 Purpose

R32 asks one question: did the R31 drop come from the unsupervised pseudo branch rather than from the +25 split itself?

Only one variable changes from R31:

- `vc_suda.unsupervised_weight: 0.02 -> 0.0`

Everything else stays fixed:

- R31 split: `annotations/instances_target_labeled_r31_plus25.json` and `annotations/instances_target_unlabeled_r31_minus25.json`.
- 1024 input path and 32K source branch.
- R12 `ckpt499` warm start: `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth`.
- LR, pseudo threshold, pseudo `max_instances`, supervised loss weights, augmentation, depth settings, `max_iter=250`, checkpoint period, eval period, and runtime settings.

## Config

- Config: `configs/vc_suda_stage_c_r32_plus25_unsup0_1024_teacher8499.yaml`.
- Output: `output/vc_suda/stage_c_r32_plus25_unsup0_1024_teacher8499`.
- Log dir: `output/vc_suda/stage_c_r32_plus25_unsup0_1024_teacher8499/logs`.
- Run name: `vc_suda_stage_c_r32_plus25_unsup0_1024_teacher8499`.

## Preflight

Required command before launch:

```bash
python tools/verify_vc_suda_stage.py \
  --config configs/vc_suda_stage_c_r32_plus25_unsup0_1024_teacher8499.yaml \
  --stage C \
  --emit-data-evidence \
  --evidence-max-samples 1
```

Expected checks:

- Stage C config resolves.
- Source is the 32K source branch.
- Target labeled and target unlabeled are the R31 +25/-25 split.
- Warm start is R12 `ckpt499`.
- `vc_suda.unsupervised_weight` is `0.0`.
- Unlabeled labels are stripped and depth is non-constant.

## Gate

Use the formal external `target_unlabeled200` 1024 backmap bbox+segm protocol. The built-in val28 eval is only a smoke signal.

Decision rules:

- Continue only if formal `target_unlabeled200` segm AP is at least `0.320048`.
- If segm AP is `0.319162-0.320048`, record it as non-collapsed, but do not extend R32.
- If segm AP is below `0.319162`, stop R32.

The gate must report bbox and segm metrics, prediction count, and topk/maxDets truncation state.

## R32 Gate Result

Run evidence:

- Training completed `250/250` in `output/vc_suda/stage_c_r32_plus25_unsup0_1024_teacher8499/train_launch_cuda_msda_cleanenv_retry.log`.
- `checkpoint_iter_0000249.pth` exists in `output/vc_suda/stage_c_r32_plus25_unsup0_1024_teacher8499/`.
- Formal external eval completed with `EXIT_CODE=0` in `output/diagnostics/r32_plus25_unsup0_ckpt249_unlabeled200_full_retry_20260516/eval.log`.

External `target_unlabeled200` 1024 backmap result at `ckpt249`:

| Metric | AP | AP50 | AP75 |
|---|---:|---:|---:|
| bbox | 0.3775 | 0.7082 | 0.3637 |
| segm | 0.2997 | 0.6298 | 0.2479 |

Eval settings and counts:

- `topk/maxDets=200`.
- Prediction count: `15,492`.
- Topk truncation: `0/200` images truncated.

Gate decision:

- Gate fail: segm AP `0.2997 < 0.319162`.
- Stop R32 here. Do not continue to `ckpt499`.

## R32 Review

R32 is a small recovery from R31, but it does not recover the R12/R15 line. R31 reached segm AP `0.2947`; R32 reached `0.2997`; R12/R15 remains `0.320048`.

This makes the R31 drop unlikely to be only the pseudo unlabeled loss. The `+25` short continuation itself, or the EMA, optimizer, LR schedule, or warm-start method, may be damaging the existing teacher.

Next, do not expand the label count again and do not tune the pseudo threshold. First run a no-train checkpoint/eval sanity check, or a Stage-B / supervised-only ablation, so the effect of fine-tuning itself can be isolated.
