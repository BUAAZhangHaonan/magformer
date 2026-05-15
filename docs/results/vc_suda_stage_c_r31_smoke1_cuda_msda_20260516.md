# VC-SUDA Stage C R31 CUDA MSDA Smoke1 - 2026-05-16

## Scope

- Host: `WS-4029GP-TRT` via `ssh 4029`.
- Repository: `/home/hdd3/zhanghaonan/magformer`.
- Branch at launch: `feature/vc-suda-sim2real`.
- Base config: `configs/vc_suda_stage_c_r31_target_labeled_plus25_1024_teacher8499.yaml`.
- Smoke config: `configs/vc_suda_stage_c_r31_target_labeled_plus25_1024_smoke1_cuda_msda.yaml`.
- Output: `output/vc_suda/stage_c_r31_target_labeled_plus25_1024_smoke1_cuda_msda`.
- tmux session: `r31_smoke1_cuda_msda_cleanenv_20260516`.

## Config Gate

The smoke config keeps the R31 1024 route unchanged except for runtime bounding:

- `name`: `vc_suda_stage_c_r31_target_labeled_plus25_1024_smoke1_cuda_msda`
- `solver.max_iter: 1`
- `runtime.output_dir`: `output/vc_suda/stage_c_r31_target_labeled_plus25_1024_smoke1_cuda_msda`
- `runtime.num_workers: 0`
- `runtime.log_period: 1`
- `runtime.eval_period: 99999`
- `runtime.checkpoint_period: 99999`
- `runtime.logger.log_dir/run_name`: smoke output/run name

Unchanged fields include:

- `data.image_size: 1024`
- R31 split: `instances_target_labeled_r31_plus25.json` and `instances_target_unlabeled_r31_minus25.json`
- source root/ann: `magformer_datasets/20260318_1K_32254`, `annotations/instances_train.json`
- warm-start: R12 `checkpoint_iter_0000499.pth`
- LR, loss weights, pseudo `max_instances: 100`, and Stage C weights

## Preflight

Command:

```bash
python tools/verify_vc_suda_stage.py \
  --config configs/vc_suda_stage_c_r31_target_labeled_plus25_1024_smoke1_cuda_msda.yaml \
  --stage C \
  --emit-data-evidence \
  --evidence-max-samples 1
```

Result: pass.

Key checks:

- source count: `25654`
- target labeled count: `50`
- target unlabeled count: `175`
- val count: `28`
- checkpoint role: `r12_ckpt499_continuation`
- `eval_period=99999`, `eval_saves_best=false`
- unlabeled weak/strong batch strips labels
- target weak/strong depth tensors are non-constant

## Launch

The run used a clean shell environment and then activated `magformer`. `LD_LIBRARY_PATH` and `CUDA_HOME` were unset after activation.

Command recorded in `train_launch.log`:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
MAGFORMER_MS_DEFORM_ATTN_BACKEND=cuda \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m torch.distributed.run --standalone --nproc_per_node=4 tools/train.py \
  --config configs/vc_suda_stage_c_r31_target_labeled_plus25_1024_smoke1_cuda_msda.yaml \
  --output-dir output/vc_suda/stage_c_r31_target_labeled_plus25_1024_smoke1_cuda_msda \
  --gpus 0,1,2,3 \
  --num-workers 0
```

ENV probe:

- `LD_LIBRARY_PATH_AFTER=<unset>`
- `CUDA_HOME_AFTER=<unset>`
- `CUDA_VISIBLE_DEVICES=4,5,6,7`
- `MAGFORMER_MS_DEFORM_ATTN_BACKEND=cuda`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- PyTorch: `2.5.1+cu124`
- Torch CUDA: `12.4`
- visible CUDA device count: `4`

## Smoke Result

Result: pass, `EXIT_CODE:0`.

Observed path:

- 4 DDP ranks started.
- source, target_labeled, target_unlabeled, and val datasets loaded.
- model built with `50,068,863` parameters.
- R12 ckpt499 warm-start loaded on all ranks.
- warm-start missing/unexpected keys: `0/0`.
- depth sanity wrote `should_abort: false`.
- EMA teacher initialized.
- one Stage C training iteration completed.
- final trainer eval ran automatically at `iter=1` on 28 val images, bbox-only.

Training evidence:

```text
start iter=0/1 eval_period=99999 ckpt_period=99999 log_period=1
iter=0/1 ... loss=24.0133 ... loss_ce=0.0249 loss_dice=0.1052 loss_mask=0.0881
training completed
EXIT_CODE:0
```

Final eval was not a selection metric. It reported bbox AP/AP50/AP75:

```text
0.1222 / 0.4031 / 0.0425
```

## Memory

Two memory sources were recorded:

- trainer CUDA peak: `18940.72 MB`
- external `nvidia-smi` sampler peak:
  - GPU4: `19645 MiB`
  - GPU5: `19645 MiB`
  - GPU6: `20065 MiB`
  - GPU7: `19645 MiB`

Use `~20.1 GiB` as the observed external peak for this 4-card 1024 R31 smoke.

## Error Scan

No matches were found for:

- `Traceback`
- `CUDA error`
- `illegal memory`
- `RuntimeError`
- `OutOfMemory`
- `CUBLAS`
- `non-finite`
- `nan`

Warnings:

- PyTorch printed `expandable_segments not supported on this platform`.
- Rank0 printed the PyTorch 2.4+ warning that the NCCL process group was not explicitly destroyed during normal process exit.
- The R12 checkpoint has no SHA256 sidecar. The checkpoint still loaded on all ranks with `0/0` missing/unexpected keys.

## Decision

The R31 1-iter CUDA MSDA smoke gate passes. It validates the real Stage C training path under clean env and CUDA MSDA. Do not treat the final 28-image bbox eval as an experiment result.
