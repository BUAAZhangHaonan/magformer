# VC-SUDA Teacher Baseline - 2026-05-14

## Scope

- Host: WS-4029GP-TRT via `ssh 4029`
- Repository: `/home/hdd3/zhanghaonan/magformer`
- Branch: `feature/vc-suda-sim2real`
- Conda env: `magformer`
- Teacher checkpoint: `output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth`
- Teacher config: `configs/vc_suda_stage_b_1024_teacher8499.yaml`
- Formal baseline split for this note: `magformer_datasets/pseudo_real_512`, `annotations/instances_val.json`, `val` split

## Results

COCO AP is shown in raw 0-1 units. The `AP x100` columns match the percent-style numbers used in the main results summary.

| Row | Eval target | Images | Eval mode | Source | bbox AP | bbox AP x100 | segm AP | segm AP x100 | Status |
|---|---:|---:|---|---|---:|---:|---:|---:|---|
| A | `pseudo_real_512` val split | 28 | single-scale, standard `tools/evaluate.py` | `output/experiments/20260514_teacher_baselines/pseudo_real_val_gpu5/evaluate.log` | 0.000433 | 0.0433 | 0.000162 | 0.0162 | Formal baseline for this note |
| B | `target_labeled` sanity | 25 | single-scale, standard `tools/evaluate.py` with temporary eval-only `val_ann` override | `output/experiments/20260514_teacher_baselines/target_labeled_sanity_gpu5/evaluate.log` | 0.000995 | 0.0995 | 0.000369 | 0.0369 | Sanity only, not formal eval |
| C | original 1.5K all split, iter 8499 | 1,566 | single-scale bbox-only historical eval | `output/experiments/20260510_1k_finetune_full_1024_v13/eval_8499_bbox.log` | 0.6942 | 69.42 | - | - | Historical reference, not rerun |
| D | original 1.5K all split, iter 8499 TTA | 1,566 | 3-scale + hflip TTA, NMS | `output/experiments/20260510_1k_finetune_full_1024_v13/eval_tta_8499.log` | 0.7055 | 70.55 | 0.6686 | 66.86 | Historical reference, not rerun |

## Commands Run

Formal pseudo-real val baseline:

```bash
CUDA_VISIBLE_DEVICES=5 python tools/evaluate.py \
  --config-file output/experiments/20260514_teacher_baselines/pseudo_real_val_gpu5/eval_runtime_gpu0.yaml \
  --weights output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth \
  --output output/experiments/20260514_teacher_baselines/pseudo_real_val_gpu5 \
  --batch-size 4 \
  --num-workers 4
```

Target-labeled sanity:

```bash
CUDA_VISIBLE_DEVICES=5 python tools/evaluate.py \
  --config-file output/experiments/20260514_teacher_baselines/target_labeled_sanity_gpu5/eval_runtime_gpu0_target_labeled.yaml \
  --weights output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth \
  --output output/experiments/20260514_teacher_baselines/target_labeled_sanity_gpu5 \
  --batch-size 4 \
  --num-workers 4
```

The temporary eval configs only remap the runtime GPU to logical `0` under `CUDA_VISIBLE_DEVICES=5`; the target-labeled sanity config also changes `data.val_ann` to `annotations/instances_target_labeled.json`. Training code and training configs were not changed.

## Comparability Notes

- Rows A and B are small VC-SUDA protocol splits. They must not be mixed with rows C and D, which use the original 1,566-image all split.
- Row B is a target-labeled sanity check only. It is not the formal validation split and should not be reported as final eval.
- Row C is bbox-only because the historical log records only bbox COCO metrics.
- Row D is TTA and is not comparable to single-scale rows without labeling it as TTA.
- The long TTA run was not rerun. This note only cites the existing log path above.

## Runtime Notes

- GPU 4-7 were checked before eval and were idle, each at about 15 MiB and 0% utilization.
- The successful fresh evals used physical GPU 5 as a single visible GPU.
- A direct run of the original config on physical `cuda:4` failed with a CUDA illegal memory access. The successful path used the same standard eval entry with `CUDA_VISIBLE_DEVICES=5` and logical `runtime.gpus: [0]`, matching the device mapping style from the prior successful eval log.
- Eval outputs were kept under `output/experiments/20260514_teacher_baselines/`. Large prediction JSON files are not intended for commit.
