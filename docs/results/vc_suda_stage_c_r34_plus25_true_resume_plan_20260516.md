# VC-SUDA Stage C R34 Plus25 True Resume Plan - 2026-05-16

## Conclusion

R34 tests one variable: replace the R31/R32 model-only warm-start from R12 `ckpt499` with a true resume from the same checkpoint. The R31 split, 1024 path, 32K source, LR, losses, pseudo threshold, `unsupervised_weight=0.02`, `max_instances`, and augmentation stay fixed.

## Diagnosis

R31 and R32 both started from R12 `checkpoint_iter_0000499.pth` through `model.finetune_weights`. That loads model parameters only. It resets optimizer state, LR scheduler state, AMP scaler, VC-SUDA EMA teacher state, and trainer `current_iter`.

That reset is a strong confound. R31 fell to segm AP `0.2946807`. R32 set only `unsupervised_weight=0` and recovered only to `0.2997`. The drop is therefore not explained only by the pseudo unlabeled loss.

## R33 Baseline

R33 did not train. It re-evaluated R12 `checkpoint_iter_0000499.pth` with the current code and current data on external `target_unlabeled200` 1024 backmap topk200/maxDets200.

R33 reproduced the R15/R12 reference:

| Metric | Value |
|---|---:|
| bbox AP/AP50/AP75 | `0.394476/0.741115/0.379187` |
| segm AP/AP50/AP75 | `0.320048/0.648363/0.284144` |
| predictions | `13,806` |
| topk-truncated images | `0/200` |

This means the R31/R32 regression is not from benchmark drift.

## R34 Config

- Config: `configs/vc_suda_stage_c_r34_plus25_true_resume_1024_teacher8499.yaml`.
- Base: `configs/vc_suda_stage_c_r31_target_labeled_plus25_1024_teacher8499.yaml`.
- Output: `output/vc_suda/stage_c_r34_plus25_true_resume_1024_teacher8499`.
- True resume: `runtime.resume: output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth`.
- Model-only warm-start disabled: `model.finetune_weights: null`.
- `solver.max_iter: 750`, because the resume checkpoint carries `iter=499` and the gate needs the next 250-iter checkpoint/final state around `ckpt749/750`.
- `runtime.eval_period: 250` and `runtime.checkpoint_period: 250` stay from R31, so the run can produce the gate checkpoint around `checkpoint_iter_0000749.pth`.

## Fixed Variables

R34 keeps these from R31:

- Target split: `instances_target_labeled_r31_plus25.json` and `instances_target_unlabeled_r31_minus25.json`.
- Resolution: `data.image_size: 1024`.
- Source root: `magformer_datasets/20260318_1K_32254`.
- LR: `solver.base_lr: 1e-5`.
- Loss weights: mask/dice/class and contrastive settings unchanged.
- Pseudo threshold: `0.10` start/end/quality threshold.
- `vc_suda.unsupervised_weight: 0.02`.
- `vc_suda.pseudo_label.max_instances: 100`.
- Augmentation and depth settings unchanged.

## Preflight

Run only preflight before launch:

```bash
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate magformer
python tools/verify_vc_suda_stage.py   --config configs/vc_suda_stage_c_r34_plus25_true_resume_1024_teacher8499.yaml   --stage C   --emit-data-evidence   --evidence-max-samples 2
```

This is read-only. It must not start training.

## Gate

After the 250-iter true-resume continuation, evaluate external `target_unlabeled200` at 1024 backmap with bbox+segm and topk200/maxDets200.

Decision:

- `segm AP >= 0.320048`: continue R34.
- `0.319162 <= segm AP < 0.320048`: record as non-collapsed, but do not extend the run.
- `segm AP < 0.319162`: stop R34.

The gate metric is external `target_unlabeled200` 1024 backmap segm AP. The built-in 28-image eval is only a smoke signal.

## Result

R34 true-resumed from R12 `checkpoint_iter_0000499.pth`, started at resume iter `499`, ran to `750/750`, and exited cleanly with `EXIT_CODE=0`. Both `checkpoint_iter_0000749.pth` and `checkpoint_iter_0000750.pth` are present in `output/vc_suda/stage_c_r34_plus25_true_resume_1024_teacher8499`.

External `target_unlabeled200` 1024 backmap eval used topk/maxDets=`200`.

| Metric | Value |
|---|---:|
| prediction count | `14679` |
| bbox AP/AP50/AP75 | `0.379715/0.716758/0.358188` |
| segm AP/AP50/AP75 | `0.301879/0.631840/0.248796` |

## Gate Decision

R34 fails the gate because external `target_unlabeled200` 1024 backmap segm AP is `0.301879 < 0.319162`.

Stop R34. Do not extend the run.

## Review

True resume is slightly better than R31/R32, but it is still far below the R12/R15 `0.320048` line. The drop is therefore not only from the model-only warm-start reset. The `+25` continuation itself still damages target mask quality.

Do not keep expanding `+25` training. The next step should be a no-train promoted/non-promoted eval protocol check, or a Stage-B / supervised-only upper-bound / data-mixing isolation check.
