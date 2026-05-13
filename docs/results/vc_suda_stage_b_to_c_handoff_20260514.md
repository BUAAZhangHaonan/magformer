# VC-SUDA Stage B to C Handoff - 2026-05-14

## Scope

- Host: WS-4029GP-TRT via `ssh 4029`.
- Repository: `/home/hdd3/zhanghaonan/magformer`.
- Branch: `feature/vc-suda-sim2real`.
- Stage B run dir: `output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821`.
- Stage B final checkpoint gate: `output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth`.
- Stage C config: `configs/vc_suda_stage_c_1024_teacher8499.yaml`.

This is a runbook only. Do not run these commands until Stage B has stopped and `checkpoint_iter_0008999.pth` exists. Do not start Stage C training from this handoff.

## Hard Rules

- Wait for Stage B to release physical GPUs 4-7 before running any GPU command below.
- Do not stop, restart, resume, or attach a new training job from this runbook.
- Do not use `runtime.resume` for Stage C. Stage C must use `model.finetune_weights` for model-only warm-start from the completed Stage B checkpoint.
- Do not pass the Stage B checkpoint as both resume state and finetune weights. Resume is optimizer/trainer state; finetune is model weights only.
- Keep the existing configs and tools. The post-eval uses `configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml`; Stage C verification and pseudo-label diagnostics use `configs/vc_suda_stage_c_1024_teacher8499.yaml`.

## Common Setup

```bash
ssh 4029
cd /home/hdd3/zhanghaonan/magformer
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate magformer
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

Confirm the checkpoint exists before any gate:

```bash
test -s output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth
```

Check GPU 4-7 are no longer occupied by Stage B before GPU use:

```bash
nvidia-smi -i 4,5,6,7
```

Pass condition: GPUs 4-7 are free enough for eval/diagnostics and no Stage B train or torchrun process is still using them.
Fail condition: any Stage B train process is still present, or GPU memory/utilization still shows the Stage B job. Wait; do not stop it from this runbook.

## 1. Stage B Segm Post-Eval

Run only after the final checkpoint exists and Stage B has released GPUs 4-7. Use a single free GPU from 4-7. Under the `CUDA_VISIBLE_DEVICES=4` mask, the eval process sees that GPU as logical `cuda:0`.

```bash
export CUDA_VISIBLE_DEVICES=4
python tools/evaluate.py \
  --config-file configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml \
  --weights output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth \
  --output output/eval/vc_suda_stage_b_1024_teacher8499_segm \
  --batch-size 1 \
  --num-workers 2
```

Success gate:

- Command exits `0`.
- Output includes COCO `bbox_AP` and `segm_AP` metrics.
- `segm_AP` is present, finite, and `> 0.0`; record the exact value in the handoff notes.
- `output/eval/vc_suda_stage_b_1024_teacher8499_segm/coco_instances_results.json` exists and is non-empty.
- The eval config keeps `model.finetune_weights: null`, `runtime.resume: null`, `vc_suda.enabled: false`, and `runtime.eval_iou_types: [bbox, segm]`.

Failure gate:

- Command exits non-zero.
- The evaluator fails on empty predictions.
- `segm_AP` is missing, NaN, not finite, or `<= 0.0`.
- `coco_instances_results.json` is missing or empty.
- Any command or config change tries to use `runtime.resume` or mix training warm-start semantics into eval.

## 2. Stage C Verify

Run this after Stage B segm post-eval passes. This verifies the Stage C config and the final Stage B checkpoint path. It does not start training.

```bash
unset CUDA_VISIBLE_DEVICES
python tools/verify_vc_suda_stage.py \
  --config configs/vc_suda_stage_c_1024_teacher8499.yaml
```

Success gate:

- Command exits `0` and prints `PASS config=configs/vc_suda_stage_c_1024_teacher8499.yaml`.
- `checks=` includes `stage`, `unlabeled_split`, `eval_iou_types`, `unsupervised_weight`, `ema_teacher`, `depth_norm`, `target_unlabeled_val_overlap`, `checkpoint_semantics`, `unlabeled_batch_strip`, and `depth_nonconstant`.
- `details.finetune_weights` resolves to `.../vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth`.
- `runtime.resume` remains `null`.
- `model.finetune_weights` points to `checkpoint_iter_0008999.pth`, not `model_final.pth`, an intermediate checkpoint, or a resume state.
- `target_unlabeled_images > 0`, `val_images > 0`, and target-unlabeled images do not overlap validation images.

Failure gate:

- Command exits non-zero or prints `FAIL`.
- The final checkpoint is missing.
- Any check above is absent.
- `runtime.resume` is not `null`.
- The checkpoint path is not the Stage B `checkpoint_iter_0008999.pth` path.
- Target-unlabeled data overlaps validation data, label fields leak into unlabeled batches, or depth is constant/near-constant.

## 3. Stage C Pseudo Threshold Sweep

Run this after Stage C verify passes. Prefer CPU first because it avoids taking GPUs from other workers. If CPU is too slow and you choose CUDA, wait until Stage B has released GPUs 4-7, then mask one free GPU as shown in the optional GPU command.

CPU command:

```bash
python tools/diagnose_vc_suda_pseudo_labels.py \
  --config configs/vc_suda_stage_c_1024_teacher8499.yaml \
  --weights output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth \
  --max-images 2 \
  --device cpu \
  --num-workers 0 \
  --threshold-sweep 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.7 \
  --output-json output/eval/vc_suda_stage_c_1024_teacher8499_pseudo_sweep_8999.json
```

Optional GPU command, only after GPU 4-7 are released:

```bash
export CUDA_VISIBLE_DEVICES=4
python tools/diagnose_vc_suda_pseudo_labels.py \
  --config configs/vc_suda_stage_c_1024_teacher8499.yaml \
  --weights output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth \
  --max-images 2 \
  --device cuda:0 \
  --num-workers 0 \
  --threshold-sweep 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.7 \
  --output-json output/eval/vc_suda_stage_c_1024_teacher8499_pseudo_sweep_8999.json
```

Success gate:

- Best case: command exits `0` and prints `PASS pseudo_label_diagnostics`.
- Required diagnostic condition: JSON contains `predictions > 0`.
- Required threshold condition: at least one threshold in `threshold_sweep` has `keep_rate > 0`.
- Required empty-image condition: choose a threshold only if its `empty_ratio < 1.0`; lower is better.
- At the configured Stage C gate threshold, pass requires `keep_rate > 0`. If the configured threshold fails but lower thresholds pass, treat this as a threshold-selection decision point, not a training launch approval.
- `output/eval/vc_suda_stage_c_1024_teacher8499_pseudo_sweep_8999.json` exists and records `weights` as the Stage B `checkpoint_iter_0008999.pth` path.

Failure gate:

- `predictions == 0`; Stage C would have no pseudo-label candidates.
- Every threshold has `keep_rate == 0`.
- Every usable threshold has `empty_ratio == 1.0`.
- The configured threshold has `keep_rate == 0`; this blocks Stage C launch at the current config threshold.
- The command cannot load the final Stage B checkpoint.
- The command uses any checkpoint other than `checkpoint_iter_0008999.pth`.

## Final Handoff Decision

- Proceed toward a separate Stage C launch only if Stage B segm post-eval passes, Stage C verify passes, and pseudo diagnostics show `predictions > 0` plus a usable threshold with `keep_rate > 0` and `empty_ratio < 1.0`.
- Block Stage C if segm AP is missing/non-finite, the Stage C config fails verification, pseudo predictions are zero, or the configured threshold keeps zero pseudo-labels.
- This document does not authorize starting Stage C training. It only defines the gates to run after `checkpoint_iter_0008999.pth` appears.
