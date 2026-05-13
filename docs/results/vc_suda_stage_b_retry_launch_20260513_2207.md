# VC-SUDA Stage B Retry Training Launch - 2026-05-13 22:07

## Run
- Host: WS-4029GP-TRT
- Repository: /home/hdd3/zhanghaonan/magformer
- Branch at launch: feature/vc-suda-sim2real
- Commit at launch: f706b28d93588a6aead47d34443650df550fe291
- DDP fix commit: f706b28, `fix: prevent ddp buffer mutation in vc-suda stage b`
- Config: configs/vc_suda_stage_b_1024_teacher8499.yaml
- Stage: VC-SUDA Stage B supervised warmup
- Image size: 1024
- Conda env: magformer

## Warm Start
- Teacher checkpoint: output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth
- Handoff prerequisite: 1-step DDP smoke passed on HEAD f706b28.
- Handoff prerequisite: teacher checkpoint 8499 warm-start matched 774/774.
- Retry launch log confirmation: all four ranks loaded the teacher checkpoint with `Warm-start missing keys: 0, unexpected keys: 0`.
- Non-fatal warning: SHA256 sidecar was not found for the teacher checkpoint.

## Command

    source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
    unset CUDA_VISIBLE_DEVICES
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    conda run --no-capture-output -n magformer \
      torchrun --standalone --nproc_per_node=4 tools/train.py \
      --config configs/vc_suda_stage_b_1024_teacher8499.yaml \
      --output-dir output/experiments/vc_suda_stage_b_1024_teacher8499_g4_7_retry_20260513_2207 \
      --gpus 4,5,6,7 \
      --num-workers 2

## Runtime
- tmux session: vc_suda_stage_b_1024_teacher8499_g4_7_retry_20260513_2207
- Output dir: output/experiments/vc_suda_stage_b_1024_teacher8499_g4_7_retry_20260513_2207
- Main log: output/experiments/vc_suda_stage_b_1024_teacher8499_g4_7_retry_20260513_2207/train.log
- Launch script: output/experiments/vc_suda_stage_b_1024_teacher8499_g4_7_retry_20260513_2207/launch.sh
- GPUs: physical GPU 4,5,6,7. CUDA_VISIBLE_DEVICES was unset.
- Resource limits: OMP_NUM_THREADS=4, MKL_NUM_THREADS=4, NUMEXPR_NUM_THREADS=4, dataloader workers=2 per rank.

## Preflight
- Git status was clean before launch.
- HEAD was f706b28 before launch.
- GPU 4-7 were idle before launch, each using about 15 MiB and 0% utilization.
- Existing tmux sessions were eval_1k and eval_32k; the retry session name did not exist before launch.
- Disk /home/hdd3 had 7.7T free, 44% used.
- System memory had about 233 GiB available before launch.
- Swap was already full before launch: 2.0 GiB used out of 2.0 GiB. vmstat showed no active swap-in or swap-out during checks.

## Stage B Data Protocol
- Formal log showed `SemiSupervisedDataset` with `Stage=B`.
- Source set length: 1008.
- Target labeled set length: 25.
- Target unlabeled set length: 0.
- Validation set length: 28.
- Trainer log showed all ranks as `stage=B`.

## Retry Check
- Checked in the requested 180-300 second window after launch.
- tmux session was still alive.
- Training had passed iter 1 and reached structured metric line `iter=100/9000` at 2026-05-13 22:12:08.
- Error keyword scan returned zero hits for fatal, error, traceback, early stop, and early stopping.
- No immediate early-stop trigger was observed.
- GPU memory during the check was about 16.0 GiB on each of GPUs 4-7, under 90% of 24 GiB.
- CPU was about 24% busy during vmstat sampling.
- System memory was about 29 GiB used out of 251 GiB, with about 214 GiB available.
- Disk /home/hdd3 remained 44% used.

## Process IDs
- tmux pane PID: 3397734.
- conda wrapper PID: 3397746.
- torchrun PID: 3397757.
- train.py rank PIDs using GPUs 4-7: 3397767, 3397768, 3397769, 3397770.

## Previous Failure Context
- The previous formal Stage B launch reached the backward pass and then failed with an inplace gradient modification error.
- That previous failed run was recorded in docs/results/vc_suda_stage_b_training_launch_20260513_2131.md.
- The current retry was launched after HEAD f706b28 fixed the DDP buffer mutation path.
- This retry has passed the point where the previous inplace run failed.

## Status
- Result status: running.
- No code was changed for this record.
- Training was left running in tmux.
