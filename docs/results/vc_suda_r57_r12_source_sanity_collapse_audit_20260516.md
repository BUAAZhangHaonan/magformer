# VC-SUDA R57 R12 source sanity collapse audit - 2026-05-16

## Conclusion

R12 source sanity collapse is real.

It is not explained by an eval protocol mistake and not by checkpoint key mismatch. Under the same corrected original 1.5K first50 protocol, Teacher8499 reaches source segm AP `0.6252930576`, while R12 `ckpt499` reaches only `0.2361521319`. R46 `ckpt750`, after changing the source anchor back to pseudo-real source and true-resuming from R12, recovers to `0.4009417296`, but it still remains far below Teacher.

The strongest root-cause candidates are training state and data mix, not pseudo loss overwhelming source loss.

## Evidence Files

Primary metric files:

- R12 original first50: `output/diagnostics/r56_source_sanity_r12_ckpt0499_original_first50_20260516/metrics.cocoeval.json`.
- Teacher original first50: `output/diagnostics/r44_teacher_first50_protocol_fix_20260516/metrics.cocoeval.json`.
- R46 original first50: `output/diagnostics/r55_r46_ckpt0750_original_first50_1024_backmap_20260516/metrics.cocoeval.json`.

R12 config and logs:

- Resolved config: `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/config_resolved.yaml`.
- Launch log: `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/train_launch_restart_20260515.log`.
- Scalar log: `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/metrics_log.jsonl`.
- R12 config: `configs/vc_suda_stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499.yaml`.
- R46 config: `configs/vc_suda_stage_c_r46_pseudoreal_source_true_resume_1024_teacher8499.yaml`.

Training-code evidence:

- `magformer/engine/vc_suda_trainer.py`.
- `magformer/data/semi_supervised_dataset.py`.
- `magformer/models/magformer/vc_suda_criterion.py`.
- EMA wrapper construction path: `tools/train.py`, `magformer/models/common/ema_teacher.py`.

## Metric Facts

| Check | Path | bbox AP/AP50/AP75 | segm AP/AP50/AP75 |
|---|---|---:|---:|
| Teacher8499 original first50 | `output/diagnostics/r44_teacher_first50_protocol_fix_20260516/metrics.cocoeval.json` | `0.6519530084/0.8531437401/0.7175465372` | `0.6252930576/0.8679527609/0.7243386328` |
| R12 `ckpt499` original first50 | `output/diagnostics/r56_source_sanity_r12_ckpt0499_original_first50_20260516/metrics.cocoeval.json` | `0.2921127104/0.6266512287/0.2366207989` | `0.2361521319/0.5620645370/0.1582073515` |
| R46 `ckpt750` original first50 | `output/diagnostics/r55_r46_ckpt0750_original_first50_1024_backmap_20260516/metrics.cocoeval.json` | `0.4657849322/0.7669137408/0.4908037106` | `0.4009417296/0.7265082520/0.3993570474` |

The R12 drop from Teacher is `-0.3891409258` segm AP. That is too large to treat as eval noise. R56 also records protocol checker passes and strict weight loads for the source-sanity evals, so this is not a protocol/key-loading artifact.

## Config Findings

R12 is not a true resume from Teacher8499.

In `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/config_resolved.yaml` and `configs/vc_suda_stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499.yaml`:

- `model.finetune_weights: output/vc_suda/stage_c_r8b_lsj10_low_lr_continue_1024_teacher8499/checkpoint_iter_0000999.pth`.
- `runtime.resume: null`.
- `vc_suda.ema_teacher.enabled: true`, `ema_momentum: 0.999`, `warmup_steps: 500`.
- `vc_suda.source_root: magformer_datasets/20260318_1K_32254`.
- `vc_suda.source_ann: annotations/instances_train.json`.
- `vc_suda.unsupervised_weight: 0.02`.
- `vc_suda.target_labeled_weight: 1.0`.
- `data.image_size: 1024`, `min_scale: 1.0`, `max_scale: 1.0`.
- `data.depth.norm: minmax`, `data.depth.per_sample_norm: true`.
- `solver.base_lr: 1.0e-05`.

The R12 launch log confirms model-only warm-start behavior:

- `[Train] Applying warm-start from model.finetune_weights: ...checkpoint_iter_0000999.pth`.
- `[Train] Warm-start loaded model weights ...`.
- `[Train] Warm-start missing keys: 0, unexpected keys: 0`.
- `[Train] Building optimizer...` and `[Train] Building LR scheduler...` after the warm-start.
- `[VCSUDA] EMA teacher enabled, momentum=0.999, warmup_steps=500`.

So R12 starts from R8B `checkpoint_iter_0000999.pth` model weights only. It does not restore optimizer, scheduler, scaler, current iter, or VC-SUDA EMA teacher state from Teacher8499.

## Loss Scale Findings

R12 source loss is active.

The R12 scalar log `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/metrics_log.jsonl` contains `54` train records from iter `0` to `980`. Mean logged values are:

| Signal | Mean | Min | Max | Last logged |
|---|---:|---:|---:|---:|
| `train/source_total_loss` | `25.790761` | `2.273142` | `116.839775` | `5.670127` |
| `train/tl_total_loss_weighted` | `20.725614` | `7.311731` | `45.844807` | `9.420628` |
| `train/pseudo_total` | `1.542300` | `0.587402` | `2.673828` | `1.041016` |
| `train/pseudo_total * 0.02` | `0.030846` | `0.011748` | `0.053477` | `0.020820` |

The pseudo branch is real, but after `unsupervised_weight=0.02` it is much smaller than source and target_labeled losses. It is not the main force that covers source loss in R12.

## Code Findings

`magformer/data/semi_supervised_dataset.py` shows the data mix is simultaneous per step:

- `__getitem__` always returns one `source` sample.
- If present, it also returns one `target_labeled` sample using `idx % len(self.target_labeled)`.
- If present, it also returns weak/strong target_unlabeled views using `idx % len(self.target_unlabeled_index_sequence)`.
- `__len__` is the max of source length and target_unlabeled sequence length.

The R12 launch log reports `source=25654`, `target_labeled=25`, `target_unlabeled=200`. Because the 25 target_labeled images are indexed by modulo, they are repeated at very high frequency across the long source-driven training length.

`magformer/engine/vc_suda_trainer.py` shows the loss composition:

- Source supervised loss is computed first and stored as `source_total_loss`.
- Target_labeled loss is added with `tl_total_loss_weighted = tl_total_loss_raw * target_labeled_weight`.
- Pseudo loss is added only as `unsup_weight * pseudo_total`.
- The optimizer then backprops the combined loss.

`magformer/models/magformer/vc_suda_criterion.py` shows pseudo loss uses Hungarian matching to teacher pseudo labels and returns `pseudo_total` from weighted CE, mask, dice, and optional pseudo terms. The trainer applies the outer `unsupervised_weight` after that.

The EMA teacher is not a fixed Teacher8499 model. `tools/train.py` constructs `EMATeacherWrapper(model, ...)` for Stage C, and `magformer/models/common/ema_teacher.py` deep-copies the current student as teacher. `magformer/engine/vc_suda_trainer.py` only restores an EMA state when `runtime.resume` is used and the checkpoint contains `ema_teacher_state_dict`. R12 has `runtime.resume: null`, so its teacher starts as a copy of the warm-started R8B student.

## Root-Cause Candidates

1. R12 is model-only warm-started from R8B `checkpoint_iter_0000999.pth`, not true-resumed from Teacher8499. Optimizer, scheduler, scaler, trainer iteration, and EMA teacher state are reset. The pseudo-label teacher is the current student deepcopy, not a fixed Teacher8499 source teacher.

2. R12 changes the source anchor to `magformer_datasets/20260318_1K_32254` plus `annotations/instances_train.json`, while the sanity eval is original 1.5K first50. R46 changes source back to `magformer_datasets/pseudo_real_512` plus `annotations/instances_source.json`, and source sanity rises to `0.4009417296`.

3. R12 trains source, 25 target_labeled images, and target_unlabeled in every step. The 25 target_labeled images are repeated by modulo with `target_labeled_weight=1.0`, so they can strongly pull the model away from the Teacher source distribution.

## Lower-Priority Noise

Depth norm, LR, image size, and pseudo loss do not look like single main causes.

- Depth norm differs across historical paths, but R12/R46 use `norm: minmax` and `per_sample_norm: true`.
- LR is `1.0e-05`, lower than the Teacher training LR, so LR alone is not the obvious source of a fast source collapse.
- R12 and R46 both use `image_size: 1024` with fixed scale `1.0/1.0`.
- Pseudo loss is small after the `0.02` outer weight.

## Minimal Next Step

Run eval only. Do not train.

Evaluate two existing checkpoints on the same original 1.5K first50 contract used by R44/R56:

1. R8B `checkpoint_iter_0000999.pth` on original first50.
2. R12 `checkpoint_iter_0000249.pth` on original first50.

R12 `ckpt499` is already evaluated at `0.2361521319`. These two extra points locate the collapse boundary:

- If R8B `ckpt999` is already low, the source sanity was lost before R12.
- If R8B is healthy but R12 `ckpt249` is low, the collapse occurs within the first 249 R12 iterations.
- If R12 `ckpt249` is still healthy but `ckpt499` is low, the collapse happens in the 249 to 499 interval.

## Validation

This audit did not run training or model eval. It reads existing configs, logs, scalar logs, metrics, and source code only.
