# R124 Depth Boundary Resume Runbook

Date: 2026-05-18
Repo: `/home/hdd3/zhanghaonan/magformer`

## Current Block

- Do not start R121 or R122 now.
- `nvidia-smi` currently reports GPU0 `Unknown Error`.
- With physical GPUs 4-7 selected, PyTorch currently reports `cuda_available=False` and `device_count=0`.
- This is a CUDA/NVML host state problem, not an R121/R122 config decision.

## Resume Gate

An admin must first restart the NVIDIA driver or reboot the server.

After the admin action, both probes below must pass before any training command is launched.

```bash
cd /home/hdd3/zhanghaonan/magformer
nvidia-smi
```

Expected:

- No GPU has `Unknown Error`.
- GPUs 4, 5, 6, and 7 are listed.
- No unexpected training process is already occupying the target GPUs.

Run one small PyTorch tensor probe per target GPU:

```bash
cd /home/hdd3/zhanghaonan/magformer
for gpu in 4 5 6 7; do
  echo "=== GPU ${gpu} ==="
  CUDA_VISIBLE_DEVICES=${gpu} /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
x = torch.ones((2, 2), device="cuda")
print("tensor_sum", float(x.sum().item()))
print("device_name", torch.cuda.get_device_name(0))
PY
done
```

Expected for every GPU:

- `cuda_available True`
- `device_count 1`
- `tensor_sum 4.0`
- a valid `device_name`

If any probe fails, stop. Escalate the driver/NVML issue again. Do not start R121 or R122.

## R121 Smoke Rerun

Run this only after the CUDA probe passes.

```bash
cd /home/hdd3/zhanghaonan/magformer
tmux new -s r121_depth_boundary_smoke
```

Inside tmux:

```bash
cd /home/hdd3/zhanghaonan/magformer
export CUDA_VISIBLE_DEVICES=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MAGFORMER_MS_DEFORM_ATTN_BACKEND=cuda
export PYTHONUNBUFFERED=1

/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/train.py \
  --config configs/baseline_vc_suda_r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075.yaml \
  --output-dir output/vc_suda/r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075 \
  --gpus 0 \
  --num-workers 0 \
  2>&1 | tee output/vc_suda/r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075/train.log
```

Pass criteria:

- Depth sanity passes.
- The log contains `loss_depth_boundary`.
- There is no OOM.
- There is no NaN.
- The run completes 1 iter.

Quick checks:

```bash
grep -n "depth sanity\\|loss_depth_boundary\\|OutOfMemory\\|CUDA out of memory\\|nan\\|NaN\\|iter" \
  output/vc_suda/r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075/train.log
```

Failure handling:

- CUDA init failure: stop immediately. Re-run the CUDA probes. If PyTorch still cannot see the GPU, escalate driver/NVML again.
- OOM: stop. Do not lower model size or patch code in this resume step. Record the peak memory line and retry only after confirming the GPU is free.
- Depth sanity failure: stop. Check that the config is exactly `configs/baseline_vc_suda_r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075.yaml` and that the dataset files are present. Do not bypass the sanity check.
- `loss_depth_boundary` missing: stop. Check the resolved config and train log. Do not launch R122 until this loss appears in R121.
- NaN: stop. Save the log path and do not continue to R122.

## R122 300iter Launch

Start R122 only if R121 smoke passes all criteria above.

```bash
cd /home/hdd3/zhanghaonan/magformer
tmux new -s r122_depth_boundary_pseudo300
```

Inside tmux:

```bash
cd /home/hdd3/zhanghaonan/magformer
export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MAGFORMER_MS_DEFORM_ATTN_BACKEND=cuda
export PYTHONUNBUFFERED=1

/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  tools/train.py \
  --config configs/baseline_vc_suda_r122_depth_boundary_w001_pseudo300.yaml \
  --output-dir output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300 \
  --gpus 0,1,2,3 \
  --num-workers 2 \
  2>&1 | tee output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300/train.log
```

Monitor:

- Training starts and reaches the first logged iter.
- The log contains `loss_depth_boundary`.
- There is no OOM.
- There is no NaN.
- `output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300/checkpoint_iter_0000099.pth` is generated.

Quick checks:

```bash
grep -n "loss_depth_boundary\\|OutOfMemory\\|CUDA out of memory\\|nan\\|NaN\\|iter" \
  output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300/train.log

ls -lh output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300/checkpoint_iter_0000099.pth
```

## R122 Evaluation

Use checkpoint iter0099 first.

Remaining75:

```bash
cd /home/hdd3/zhanghaonan/magformer
CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch PYTHONUNBUFFERED=1 \
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/evaluate_1024_backmap.py \
  --iou-types bbox,segm \
  --base-config configs/baseline_vc_suda_r122_depth_boundary_w001_pseudo300.yaml \
  --dataset-root magformer_datasets/pseudo_real_512 \
  --ann annotations/instances_target_unlabeled_r114_balanced_minus125.json \
  --split train \
  --weights output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300/checkpoint_iter_0000099.pth \
  --output-dir output/diagnostics/r122_depth_boundary_w001_iter0099_remaining75_1024_backmap_topk200_20260518 \
  --image-size 1024 \
  --batch-size 4 \
  --num-workers 0 \
  --score-threshold 0.05 \
  --mask-threshold 0.5 \
  --inference-topk 200 \
  --max-dets 200 \
  --force-pytorch-msda \
  --dump-inference-stats output/diagnostics/r122_depth_boundary_w001_iter0099_remaining75_1024_backmap_topk200_20260518/inference_stats.json
```

Val28:

```bash
cd /home/hdd3/zhanghaonan/magformer
CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch PYTHONUNBUFFERED=1 \
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/evaluate_1024_backmap.py \
  --iou-types bbox,segm \
  --base-config configs/baseline_vc_suda_r122_depth_boundary_w001_pseudo300.yaml \
  --dataset-root magformer_datasets/pseudo_real_512 \
  --ann annotations/instances_val.json \
  --split val \
  --weights output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300/checkpoint_iter_0000099.pth \
  --output-dir output/diagnostics/r122_depth_boundary_w001_iter0099_val28_1024_backmap_topk200_20260518 \
  --image-size 1024 \
  --batch-size 4 \
  --num-workers 0 \
  --score-threshold 0.05 \
  --mask-threshold 0.5 \
  --inference-topk 200 \
  --max-dets 200 \
  --force-pytorch-msda \
  --dump-inference-stats output/diagnostics/r122_depth_boundary_w001_iter0099_val28_1024_backmap_topk200_20260518/inference_stats.json
```

## R120 Atlas Go/No-Go

Compare R122 iter0099 against the R120 atlas rows, especially tiny/high100 bucket metrics:

- Boundary-F
- IoU
- R75
- FP75

Continue only if all of these hold:

- `remaining75` tiny: Boundary-F improves by at least `+0.02`, IoU improves by at least `+0.02`, and R75 is at least `0.035`.
- `remaining75` high100: Boundary-F improves by at least `+0.02`, R75 is at least `0.270`, and FP75 does not rise.
- `val28` moves in the same direction.
- small and mid50 buckets are protected. They must not regress enough to erase the tiny/high100 gain.

R120 anchor directories:

- `output/diagnostics/r120_error_atlas_20260518/`
- `output/diagnostics/r114_magformer_r113warm_target150_iter2000_remaining75_1024_backmap_topk200_20260518/`
- `output/diagnostics/r114_magformer_r113warm_target150_iter2000_val28_1024_backmap_topk200_20260518/`

After the R122 evaluation writes the R120-atlas-style `bucket_compare.csv`, run the CPU-only comparator before deciding whether to continue:

```bash
cd /home/hdd3/zhanghaonan/magformer
CUDA_VISIBLE_DEVICES="" /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/compare_r122_go_no_go.py \
  --baseline-bucket-csv output/diagnostics/r120_error_atlas_20260518/bucket_compare.csv \
  --candidate-bucket-csv output/diagnostics/r122_depth_boundary_w001_iter0099_bucket_compare_20260518/bucket_compare.csv \
  --baseline-run r114_remaining75 \
  --candidate-run r122_remaining75 \
  --output-json output/diagnostics/r122_depth_boundary_w001_iter0099_go_no_go_20260518/go_no_go.json \
  --output-md output/diagnostics/r122_depth_boundary_w001_iter0099_go_no_go_20260518/go_no_go.md
```

Interpretation:

- Exit `0` means every gate passed.
- Exit `1` means at least one gate failed. Stop and read `go_no_go.md`.
- Exit `2` means the CSV schema or required buckets are missing or ambiguous. Fix the evaluation output, not the comparator.
- This command only reads CSV files and writes the go/no-go report. It does not use GPU and does not start training.

## Do Not Modify Existing Outputs

Do not edit, delete, move, or overwrite these existing directories:

- `output/baseline/r114_magformer_r113warm_target150_balanced_2000/`
- `output/baseline/r115_magformer_r114warm_fulltarget200_oracle_2000/`
- `output/baseline/r115b_magformer_r114warm_fulltarget200_oracle_2000/`
- `output/vc_suda/r118_magformer_r114warm_pseudo300/`
- `output/vc_suda/r121_depth_boundary_w001_smoke_b1_iter1_loss_only/`
- `output/vc_suda/r121_depth_boundary_w001_smoke_g4/`
- `output/diagnostics/r114_magformer_r113warm_target150_iter2000_remaining75_1024_backmap_topk200_20260518/`
- `output/diagnostics/r114_magformer_r113warm_target150_iter2000_val28_1024_backmap_topk200_20260518/`
- `output/diagnostics/r118_iter0099_remaining75_1024_backmap_topk200_20260518/`
- `output/diagnostics/r118_iter0099_val28_1024_backmap_topk200_20260518/`

Use only the new R121 `fg00075` smoke output directory, the R122 training output directory, and the R122 diagnostics directories listed above.
