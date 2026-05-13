# VC-SUDA Stage C Preflight - 2026-05-14

## Scope

- Host: WS-4029GP-TRT via `ssh 4029`
- Repository: `/home/hdd3/zhanghaonan/magformer`
- Branch: `feature/vc-suda-sim2real`
- Conda env: `magformer`
- Config: `configs/vc_suda_stage_c_1024_teacher8499.yaml`
- Output directory: `output/experiments/20260514_stage_c_preflight/`
- This was a preflight only. No formal Stage C training was launched.

## Config Check

`model.finetune_weights` points to the completed Stage B checkpoint:

```text
output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth
```

Stage C pseudo-label gating is now configured for the observed teacher score range:

```text
vc_suda.pseudo_label.quality_threshold: 0.20
vc_suda.pseudo_label.use_curriculum: true
vc_suda.curriculum.start_threshold: 0.20
vc_suda.curriculum.end_threshold: 0.20
vc_suda.unsupervised_weight: 0.1
```

The fixed 0.20 curriculum threshold keeps the unsupervised warmup path active while avoiding a threshold that drops every pseudo-label.

Stage C keeps `vc_suda.ema_teacher.enabled: true` for pseudo-label generation, but now sets `runtime.ema_enabled: false`. This prevents formal eval from swapping in the generic `Trainer` EMA shadow instead of evaluating the student model.

## Strict Stage C Verifier

Command, run without `--allow-missing-finetune`:

```bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate magformer
python tools/verify_vc_suda_stage.py \
  --config configs/vc_suda_stage_c_1024_teacher8499.yaml
```

Result: pass.

```text
PASS config=configs/vc_suda_stage_c_1024_teacher8499.yaml
checks=stage,unlabeled_split,eval_iou_types,unsupervised_weight,ema_teacher,runtime_no_generic_ema,depth_norm,target_unlabeled_val_overlap,checkpoint_semantics,unlabeled_batch_strip,depth_nonconstant
details={"finetune_weights": "/home/hdd3/zhanghaonan/magformer/output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth", "target_unlabeled_images": 200, "target_unlabeled_split": "train", "unsupervised_weight": 0.1, "val_images": 28}
```

The verifier also read one CPU batch and reported non-constant target weak/strong depth tensors.

## Pseudo-Label Diagnostics

Command:

```bash
CUDA_VISIBLE_DEVICES=4 python tools/diagnose_vc_suda_pseudo_labels.py \
  --config configs/vc_suda_stage_c_1024_teacher8499.yaml \
  --max-images 20 \
  --batch-size 1 \
  --num-workers 0 \
  --device cuda \
  --threshold-sweep 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
  --output-json output/experiments/20260514_stage_c_preflight/diagnose_max20_after_gate.json
```

The run loaded the Stage B 8999 checkpoint with `missing keys: 0, unexpected keys: 0`. It emitted the existing SHA256 sidecar warning for the checkpoint, but weight loading completed.

## Default Gate

Configured threshold source: `vc_suda.curriculum`.

Configured threshold value: `0.20`.

Gate defaults:

```text
min_keep_rate: 0.10
max_empty_ratio: 0.05
```

Result: pass.

| Images | Predictions | Kept | Keep Rate | Empty Images | Empty Ratio | Kept/Image Min | Kept/Image Max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 1984 | 253 | 0.127520 | 0/20 | 0.000000 | 5 | 19 |

`threshold=0.20` satisfies both gate checks: `keep_rate=0.127520 >= 0.10` and `empty_ratio=0.000000 <= 0.05`.

Per-image kept counts at `threshold=0.20`:

```text
[19, 11, 11, 15, 7, 13, 8, 5, 10, 18, 9, 17, 17, 16, 11, 14, 15, 8, 11, 18]
```

## Score Distribution

| Images | Predictions | Min | Mean | P25 | Median | P75 | P90 | P95 | P99 | Max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 1984 | 0.000027 | 0.071115 | 0.002505 | 0.032299 | 0.130342 | 0.212471 | 0.233510 | 0.250917 | 0.259158 |

## Threshold Sweep

20-image scan:

| Threshold | Kept | Keep Rate | Empty Ratio | Zero Images | Kept/Image Min | Kept/Image Max | Kept/Image Mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 877 | 0.442036 | 0.000000 | 0 | 13 | 81 | 43.85 |
| 0.10 | 615 | 0.309980 | 0.000000 | 0 | 10 | 59 | 30.75 |
| 0.15 | 429 | 0.216230 | 0.000000 | 0 | 9 | 42 | 21.45 |
| 0.20 | 253 | 0.127520 | 0.000000 | 0 | 5 | 19 | 12.65 |
| 0.25 | 26 | 0.013105 | 0.500000 | 10 | 0 | 6 | 1.30 |
| 0.30 | 0 | 0.000000 | 1.000000 | 20 | 0 | 0 | 0.00 |
| 0.40 | 0 | 0.000000 | 1.000000 | 20 | 0 | 0 | 0.00 |
| 0.50 | 0 | 0.000000 | 1.000000 | 20 | 0 | 0 | 0.00 |
| 0.60 | 0 | 0.000000 | 1.000000 | 20 | 0 | 0 | 0.00 |
| 0.70 | 0 | 0.000000 | 1.000000 | 20 | 0 | 0 | 0.00 |
| 0.80 | 0 | 0.000000 | 1.000000 | 20 | 0 | 0 | 0.00 |
| 0.90 | 0 | 0.000000 | 1.000000 | 20 | 0 | 0 | 0.00 |

The persisted JSON includes `kept_per_image`, `kept_per_image_min`, `kept_per_image_max`, and `zero_image_count` for every sweep threshold.

## Stage C Launch Decision

Stage C now passes the preflight gates for the sampled 20-image pseudo-label diagnostic at the configured threshold `0.20`.

Reason:

- The configured threshold is now `0.20`, sourced from the fixed curriculum threshold.
- At `0.20`, the sampled keep-rate is above the default `0.10` minimum.
- At `0.20`, no sampled target-unlabeled image is empty after filtering.

## Artifacts

- `output/experiments/20260514_stage_c_preflight/diagnose_max20_after_gate.json`
