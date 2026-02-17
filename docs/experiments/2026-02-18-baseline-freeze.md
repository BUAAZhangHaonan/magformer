# Baseline Freeze (2026-02-18)

This document freezes the current comparison baseline between MagFormer and Mask2Former.

## Scope

- Dataset: `magformer_datasets/0909_512_0.12K`
- Primary metric: `segm/AP`
- Secondary metrics: `segm/AP50`, `segm/AP75`, `segm/APs`, `segm/APm`, `segm/APl`

## Frozen Inputs

- MagFormer config: `magformer/configs/magformer_aligned_comparison.yaml`
- Mask2Former config: `mask2former/MGM_Mask2Former/configs/mgm_aligned_comparison.yaml`
- Reference high-AP run config snapshot:
  `mask2former/MGM_Mask2Former/output/0909_512_0.12K/20260214_train2000/config.yaml`

## Frozen Baseline Observations

- MagFormer aligned 2k run (existing):
  `magformer/magformer/output/magformer_aligned_comparison/metrics_log.csv`
  - final `segm_AP ~= 0.5636`
- Mask2Former reference 2k run (existing):
  `mask2former/MGM_Mask2Former/output/0909_512_0.12K/20260214_train2000/metrics.json`
  - final `segm/AP ~= 85.64`

## Reproducible Runbook

Use:

```bash
bash scripts/experiments/run_baseline_freeze.sh --dry-run
bash scripts/experiments/run_baseline_freeze.sh --run
```

Behavior:

- creates deterministic output tree under `magformer/output/experiments/baseline_freeze/`
- snapshots both configs into the output tree
- prints all executed commands
- supports `--run` for real execution and `--dry-run` for command preview

## Output Directory Contract

The script generates:

- `magformer/output/experiments/baseline_freeze/config_snapshots/`
- `magformer/output/experiments/baseline_freeze/magformer_aligned_2k/`
- `magformer/output/experiments/baseline_freeze/mask2former_reference_2k/`

This contract is used by later milestones (Track A / Track B runs and reporting).
