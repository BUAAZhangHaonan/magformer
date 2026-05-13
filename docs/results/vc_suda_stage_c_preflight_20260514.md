# VC-SUDA Stage C Preflight - 2026-05-14

## Scope

- Host: WS-4029GP-TRT via `ssh 4029`
- Repository: `/home/hdd3/zhanghaonan/magformer`
- Branch: `feature/vc-suda-sim2real`
- Conda env: `magformer`
- Config: `configs/vc_suda_stage_c_1024_teacher8499.yaml`
- This was a preflight only. No formal Stage C training was launched.

## Gate Summary

- Stage C config exists and loads through the normal Pydantic config path.
- `vc_suda.stage: C` and `vc_suda.ema_teacher.enabled: true`.
- `vc_suda.target_unlabeled_ann: annotations/instances_target_unlabeled.json`.
- Effective target-unlabeled split is `train`.
- `vc_suda.unsupervised_weight: 0.1` with `unsupervised_warmup_epochs: 10`.
- `runtime.eval_iou_types: [bbox, segm]`.
- `runtime.resume: null`.
- `model.finetune_weights` is a placeholder for the Stage B final checkpoint: `output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth`.
- Depth normalization keeps the Stage B values: `norm: minmax`, `per_sample_norm: true`, `clip_min: 0.0`, `clip_max: 2.095623016357422`.
- `runtime.skip_depth_sanity: false` for Stage C.

## Validation Commands

Static preflight, allowing the Stage B final checkpoint placeholder because Stage B is still running:

```bash
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate magformer
python tools/verify_vc_suda_stage.py \
  --config configs/vc_suda_stage_c_1024_teacher8499.yaml \
  --skip-batch \
  --allow-missing-finetune
```

Result:

```text
PASS config=configs/vc_suda_stage_c_1024_teacher8499.yaml
checks=stage,unlabeled_split,eval_iou_types,unsupervised_weight,ema_teacher,depth_norm,target_unlabeled_val_overlap,checkpoint_semantics
WARNING model.finetune_weights does not exist yet; allowed because Stage B final checkpoint is pending.
details={"finetune_weights": "/home/hdd3/zhanghaonan/magformer/output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth", "target_unlabeled_images": 200, "target_unlabeled_split": "train", "unsupervised_weight": 0.1, "val_images": 28}
```

One-batch CPU/data preflight:

```bash
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate magformer
python tools/verify_vc_suda_stage.py \
  --config configs/vc_suda_stage_c_1024_teacher8499.yaml \
  --allow-missing-finetune
```

Result:

```text
PASS config=configs/vc_suda_stage_c_1024_teacher8499.yaml
checks=stage,unlabeled_split,eval_iou_types,unsupervised_weight,ema_teacher,depth_norm,target_unlabeled_val_overlap,checkpoint_semantics,unlabeled_batch_strip,depth_nonconstant
WARNING model.finetune_weights does not exist yet; allowed because Stage B final checkpoint is pending.
details={"finetune_weights": "/home/hdd3/zhanghaonan/magformer/output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth", "target_strong_depth": {"max": 1.0467190742492676, "min": -0.04866107553243637, "shape": [1, 1, 1024, 1024], "std": 0.4546217620372772, "unique": 970057}, "target_unlabeled_images": 200, "target_unlabeled_split": "train", "target_weak_depth": {"max": 1.0505133867263794, "min": -0.049213748425245285, "shape": [1, 1, 1024, 1024], "std": 0.4546493589878082, "unique": 969998}, "unsupervised_weight": 0.1, "val_images": 28}
```

Pseudo-label keep-rate diagnostic gate:

```bash
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate magformer
python tools/diagnose_vc_suda_pseudo_labels.py \
  --config configs/vc_suda_stage_c_1024_teacher8499.yaml \
  --weights output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0000999.pth \
  --max-images 2 \
  --device cpu \
  --num-workers 0
```

Result: fail-fast, as intended, because the current Stage B checkpoint keeps zero pseudo-labels at the Stage C threshold.

```text
FAIL keep_rate=0 at threshold=0.7; Stage C would keep zero pseudo-labels.
{"empty_images": 2, "empty_ratio": 1.0, "images": 2, "keep_rate": 0.0, "kept": 0, "kept_per_image": [0, 0], "predictions": 200, "score_distribution": {"count": 200, "max": 0.25181618332862854, "mean": 0.04233705624938011, "median": 0.017960816621780396, "min": 0.0011792480945587158, "p25": 0.004576519131660461, "p75": 0.0579221174120903, "p90": 0.11407066136598587, "p95": 0.1529223471879959, "p99": 0.24780187010765076}, "threshold": {"config": {"epoch": 0, "quality_threshold": 0.7, "use_curriculum": false}, "source": "vc_suda.pseudo_label.quality_threshold", "value": 0.7}}
```

Threshold sweep usage for the current 999 checkpoint:

```bash
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate magformer
python tools/diagnose_vc_suda_pseudo_labels.py \
  --config configs/vc_suda_stage_c_1024_teacher8499.yaml \
  --weights output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0000999.pth \
  --max-images 2 \
  --device cpu \
  --num-workers 0 \
  --threshold-sweep 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.7
```

The command still exits non-zero when the configured Stage C gate keeps zero pseudo-labels at threshold `0.7`.
The JSON is still printed before exit and now includes `threshold_sweep`, with `keep_rate`, `empty_ratio`, and `kept_per_image_mean` for each scanned threshold.
This scan is CPU-only with the same small `--max-images 2` probe, so it does not require rerunning GPU work and does not affect the running Stage B job.
Use the same command with the final Stage B checkpoint later to choose a data-driven Stage C threshold.

Observed 999-checkpoint sweep on the same 2-image CPU probe:

```text
threshold=0.05 keep_rate=0.275 empty_ratio=0.0 kept_per_image_mean=27.5
threshold=0.10 keep_rate=0.130 empty_ratio=0.0 kept_per_image_mean=13.0
threshold=0.15 keep_rate=0.060 empty_ratio=0.0 kept_per_image_mean=6.0
threshold=0.20 keep_rate=0.045 empty_ratio=0.0 kept_per_image_mean=4.5
threshold=0.25 keep_rate=0.010 empty_ratio=0.5 kept_per_image_mean=1.0
threshold=0.30 keep_rate=0.000 empty_ratio=1.0 kept_per_image_mean=0.0
threshold=0.40 keep_rate=0.000 empty_ratio=1.0 kept_per_image_mean=0.0
threshold=0.50 keep_rate=0.000 empty_ratio=1.0 kept_per_image_mean=0.0
threshold=0.60 keep_rate=0.000 empty_ratio=1.0 kept_per_image_mean=0.0
threshold=0.70 keep_rate=0.000 empty_ratio=1.0 kept_per_image_mean=0.0
```

Targeted pytest coverage:

```bash
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate magformer
python -m pytest \
  tests/test_vc_suda_stage_c_preflight.py \
  tests/test_vc_suda_data_protocol.py \
  tests/test_vc_suda_train_entrypoint.py \
  -q
```

Result: passed. Existing Pydantic deprecation warnings for legacy `dpe_enabled` and `dpe_beta` were still present.

## Gate Status

- Config gate: pass.
- Static preflight: pass with one expected warning because the Stage B final checkpoint does not exist yet.
- One-batch unlabeled data gate: pass.
- Pseudo-label keep-rate gate: fail-fast on the current intermediate Stage B checkpoint because `keep_rate=0.0` at threshold `0.7`.
- Formal Stage C launch gate: blocked until Stage B produces the final `checkpoint_iter_0008999.pth` path used by `model.finetune_weights`.
