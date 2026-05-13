# VC-SUDA Stage C Smoke - 2026-05-14

## Scope

- Host: WS-4029GP-TRT via `ssh 4029`.
- Repository: `/home/hdd3/zhanghaonan/magformer`.
- Branch: `feature/vc-suda-sim2real`.
- Conda env: `magformer`.
- Base config: `configs/vc_suda_stage_c_1024_teacher8499.yaml`.
- Temp smoke config: `output/tmp/vc_suda_stage_c_1024_teacher8499_smoke_20260514.yaml`.
- Smoke output: `output/experiments/vc_suda_stage_c_1024_teacher8499_smoke_20260514`.
- Result: **failed before training iterations**.

## Override Support

`tools/train.py --help` exposes these runtime overrides:

```text
--config / --config-file
--dataset-root
--weights
--finetune-weights
--output-dir
--resume
--eval-only
--gpus
--num-workers
--seed
```

There is no CLI `--max-iter` or generic dotted override support in `tools/train.py`. `magformer.config.load_config()` supports API-level `overrides`, but the train entrypoint does not expose arbitrary overrides. I used a temp config under `output/tmp` instead of editing the formal Stage C config.

Smoke-only config changes:

```text
name: vc_suda_stage_c_1024_teacher8499_smoke_20260514
solver.max_iter: 2
runtime.output_dir: output/experiments/vc_suda_stage_c_1024_teacher8499_smoke_20260514
runtime.gpus: [4, 5, 6, 7]
runtime.ddp_enabled: true
runtime.num_workers: 0
runtime.log_period: 1
runtime.eval_period: 3
runtime.checkpoint_period: 99999
runtime.eval_max_images: 1
runtime.skip_depth_sanity: false
```

The smoke config kept these formal Stage C settings unchanged:

```text
model.finetune_weights: output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth
vc_suda.target_unlabeled_ann: annotations/instances_target_unlabeled.json
data.depth.norm: minmax
data.depth.per_sample_norm: true
data.depth_noise.enabled: true
data.depth_noise.gaussian_std: 0.01
```

## Preflight

Static verifier passed on the formal config and the temp smoke config.

```text
PASS config=configs/vc_suda_stage_c_1024_teacher8499.yaml
checks=stage,unlabeled_split,eval_iou_types,unsupervised_weight,ema_teacher,depth_norm,target_unlabeled_val_overlap,checkpoint_semantics
details={"finetune_weights": "/home/hdd3/zhanghaonan/magformer/output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth", "target_unlabeled_images": 200, "target_unlabeled_split": "train", "unsupervised_weight": 0.1, "val_images": 28}
```

The Stage B warm-start checkpoint exists:

```text
output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth
size: 556M
```

GPU 4-7 were idle before launch:

```text
4, 15 MiB, 0%
5, 15 MiB, 0%
6, 15 MiB, 0%
7, 15 MiB, 0%
```

## Launch Command

Run in tmux session `vc_suda_stage_c_smoke_20260514`:

```bash
torchrun --standalone --nproc_per_node=4 \
  tools/train.py \
  --config output/tmp/vc_suda_stage_c_1024_teacher8499_smoke_20260514.yaml \
  2>&1 | tee output/experiments/vc_suda_stage_c_1024_teacher8499_smoke_20260514/smoke_console.log
```

## Confirmed Before Failure

Warm-start loaded on all four ranks:

```text
[Train] Warm-start loaded model weights from output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth
[Train] Warm-start missing keys: 0, unexpected keys: 0
```

Log counts:

```text
Warm-start loaded: 4
Warm-start missing: 4
SemiSupervisedDataset: 4
target_unlabeled=200: 4
```

The Stage C dataset path was constructed on all four ranks:

```text
[SemiSupervisedDataset] Stage=C, source=1008, target_labeled=25, target_unlabeled=200
```

No label leakage assertion appeared before the failure. The target-unlabeled batch path was entered during depth sanity, because the failure occurred after the first train loader batch was read.

## Failure

The smoke stopped in the training entrypoint depth sanity gate, before `VCSUDADDPTrainer` construction and before any training iteration.

Rank failures:

```text
[Train] Depth sanity preflight failed:
[Train]   - depth outside [0,1]: min=-0.044997, max=1.038527
RuntimeError: Depth sanity preflight failed; aborting before full training.
```

Other ranks reported the same failure, with min around `-0.045966` and max `1.038527`.

Persisted depth sanity report:

```json
{
  "depth": {
    "min": -0.045966,
    "max": 1.038527,
    "mean": 0.54381,
    "std": 0.222067
  },
  "confidence": {},
  "masks": {
    "foreground_ratio": 0.00268
  },
  "should_abort": true,
  "reasons": [],
  "aborted": false
}
```

The JSON has `should_abort: true`, but its `reasons` field is empty because `tools/train.py` writes the report without passing the computed reasons to `write_depth_sanity_report()`. The console log is the source for the exact reason.

## Root Cause

The formal Stage C training transform normalizes depth first, then applies depth noise:

```text
DepthNormalize(... per_sample_norm=True) -> clips normalized depth into [0,1]
DepthNoiseAug(gaussian_std=0.01) -> adds Gaussian noise after normalization
```

This order allows the final tensor entering depth sanity to leave `[0,1]`. The smoke did not change `data.depth_noise`, so this is a formal Stage C launch blocker, not a temp-config artifact.

## Not Reached

These checks were not exercised because the run failed before trainer construction:

- VC-SUDA EMA teacher creation.
- Pseudo-label scorer execution.
- Pseudo loss or pseudo diagnostics in train logs.
- DDP wrapper construction in `VCSUDADDPTrainer`.
- EMA update path.
- Any 2-5 training iteration metric.

## Artifacts

Kept under output paths and not committed:

- `output/tmp/vc_suda_stage_c_1024_teacher8499_smoke_20260514.yaml`
- `output/experiments/vc_suda_stage_c_1024_teacher8499_smoke_20260514/smoke_console.log`
- `output/experiments/vc_suda_stage_c_1024_teacher8499_smoke_20260514/depth_sanity.json`
- `output/experiments/vc_suda_stage_c_1024_teacher8499_smoke_20260514/config_resolved.yaml`
- `output/experiments/vc_suda_stage_c_1024_teacher8499_smoke_20260514/run_metadata.json`

No checkpoint was produced.
