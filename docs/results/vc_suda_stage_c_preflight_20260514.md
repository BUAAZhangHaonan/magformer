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

`model.finetune_weights` already points to the completed Stage B checkpoint:

```text
output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth
```

No config edit was needed for this item.

## Strict Stage C Verifier

Command, run without `--allow-missing-finetune`:

```bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate magformer
python tools/verify_vc_suda_stage.py \
  --config configs/vc_suda_stage_c_1024_teacher8499.yaml
```

Result: pass.

```text
PASS config=configs/vc_suda_stage_c_1024_teacher8499.yaml
checks=stage,unlabeled_split,eval_iou_types,unsupervised_weight,ema_teacher,depth_norm,target_unlabeled_val_overlap,checkpoint_semantics,unlabeled_batch_strip,depth_nonconstant
details={"finetune_weights": "/home/hdd3/zhanghaonan/magformer/output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth", "target_unlabeled_images": 200, "target_unlabeled_split": "train", "unsupervised_weight": 0.1, "val_images": 28}
```

The verifier also read one CPU batch and reported non-constant target weak/strong depth tensors.

## Pseudo-Label Diagnostics

Both scans used one visible GPU only:

```bash
CUDA_VISIBLE_DEVICES=4 python tools/diagnose_vc_suda_pseudo_labels.py \
  --config configs/vc_suda_stage_c_1024_teacher8499.yaml \
  --max-images 5 \
  --batch-size 1 \
  --num-workers 0 \
  --device cuda \
  --threshold-sweep 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
  --output-json output/experiments/20260514_stage_c_preflight/diagnose_max5.json
```

The 5-image scan failed the configured Stage C gate, so the same scan was repeated with `--max-images 20`:

```bash
CUDA_VISIBLE_DEVICES=4 python tools/diagnose_vc_suda_pseudo_labels.py \
  --config configs/vc_suda_stage_c_1024_teacher8499.yaml \
  --max-images 20 \
  --batch-size 1 \
  --num-workers 0 \
  --device cuda \
  --threshold-sweep 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
  --output-json output/experiments/20260514_stage_c_preflight/diagnose_max20.json
```

Both runs loaded the Stage B 8999 checkpoint with `missing keys: 0, unexpected keys: 0`. Both runs emitted the existing SHA256 sidecar warning for the checkpoint, but weight loading completed.

## Score Distribution

| Probe | Images | Predictions | Min | Mean | P25 | Median | P75 | P90 | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| max5 | 5 | 500 | 0.000190 | 0.057208 | 0.001321 | 0.007580 | 0.090538 | 0.217233 | 0.237702 | 0.253449 | 0.259158 |
| max20 | 20 | 1984 | 0.000027 | 0.071115 | 0.002505 | 0.032299 | 0.130342 | 0.212471 | 0.233510 | 0.250917 | 0.259158 |

## Default Gate

Configured threshold source: `vc_suda.pseudo_label.quality_threshold`.

Configured threshold value: `0.7`.

| Probe | Kept | Keep Rate | Empty Images | Empty Ratio |
| --- | ---: | ---: | ---: | ---: |
| max5 | 0 | 0.000000 | 5/5 | 1.000000 |
| max20 | 0 | 0.000000 | 20/20 | 1.000000 |

## Threshold Sweep

20-image scan:

| Threshold | Kept | Keep Rate | Empty Ratio | Kept/Image Mean |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | 877 | 0.442036 | 0.000000 | 43.85 |
| 0.10 | 615 | 0.309980 | 0.000000 | 30.75 |
| 0.15 | 429 | 0.216230 | 0.000000 | 21.45 |
| 0.20 | 253 | 0.127520 | 0.000000 | 12.65 |
| 0.25 | 26 | 0.013105 | 0.500000 | 1.30 |
| 0.30 | 0 | 0.000000 | 1.000000 | 0.00 |
| 0.40 | 0 | 0.000000 | 1.000000 | 0.00 |
| 0.50 | 0 | 0.000000 | 1.000000 | 0.00 |
| 0.60 | 0 | 0.000000 | 1.000000 | 0.00 |
| 0.70 | 0 | 0.000000 | 1.000000 | 0.00 |
| 0.80 | 0 | 0.000000 | 1.000000 | 0.00 |
| 0.90 | 0 | 0.000000 | 1.000000 | 0.00 |

The practical threshold candidates from this probe are `0.10`, `0.15`, and `0.20`. `0.25` is too sparse because half of the sampled images are empty. Thresholds `0.30` and higher keep nothing.

## Stage C Launch Decision

Do not launch Stage C with the current config.

Reason:

- The configured threshold `0.7` has `keep_rate=0` on both max5 and max20 probes.
- At `0.7`, `empty_ratio=1.0`, so every sampled target-unlabeled image would contribute zero pseudo-labels.
- The tool fail-fast matches the launch risk: current Stage C would train with no kept pseudo-labels from the unsupervised branch.

## Artifacts

- `output/experiments/20260514_stage_c_preflight/diagnose_max5.json`
- `output/experiments/20260514_stage_c_preflight/diagnose_max5.log`
- `output/experiments/20260514_stage_c_preflight/diagnose_max20.json`
- `output/experiments/20260514_stage_c_preflight/diagnose_max20.log`
