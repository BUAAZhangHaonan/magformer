# VC-SUDA Stage B Training Launch - 2026-05-13 21:31

## Run
- Host: WS-4029GP-TRT
- Repository: /home/hdd3/zhanghaonan/magformer
- Branch at launch: feature/vc-suda-sim2real
- Commit at launch: 889793f
- Config: configs/vc_suda_stage_b_1024_teacher8499.yaml
- Stage: VC-SUDA Stage B supervised warmup
- Image size: 1024
- Conda env: magformer

## Warm Start
- Teacher checkpoint: output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth
- Handoff prerequisite: 1-step smoke passed on HEAD 889793f.
- Handoff prerequisite: warm-start matched 774/774.
- Formal launch log confirmation: all four ranks loaded the teacher checkpoint with `Warm-start missing keys: 0, unexpected keys: 0`.
- Non-fatal warning: SHA256 sidecar was not found for the teacher checkpoint.

## Command

    unset CUDA_VISIBLE_DEVICES
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    conda run --no-capture-output -n magformer \
      torchrun --standalone --nproc_per_node=4 tools/train.py \
      --config configs/vc_suda_stage_b_1024_teacher8499.yaml \
      --output-dir output/experiments/vc_suda_stage_b_1024_teacher8499_g4_7_20260513_2131 \
      --gpus 4,5,6,7 \
      --num-workers 2

The tmux wrapper prepended `/home/hdd3/zhanghaonan/anaconda3/bin` to PATH so that the command still invoked `conda run` inside the tmux non-login shell.

## Runtime
- tmux session: vc_suda_stage_b_1024_teacher8499_g4_7_20260513_2131
- Output dir: output/experiments/vc_suda_stage_b_1024_teacher8499_g4_7_20260513_2131
- Main log: output/experiments/vc_suda_stage_b_1024_teacher8499_g4_7_20260513_2131/train.log
- Launch script: output/experiments/vc_suda_stage_b_1024_teacher8499_g4_7_20260513_2131/launch_command.sh
- GPUs: physical GPU 4,5,6,7. CUDA_VISIBLE_DEVICES was unset.
- Resource limits: OMP_NUM_THREADS=4, MKL_NUM_THREADS=4, NUMEXPR_NUM_THREADS=4, dataloader workers=2 per rank.

## Preflight
- Git status was clean before launch.
- HEAD was 889793f before launch.
- GPU 4-7 were idle before launch, each using about 15 MiB and 0% utilization.
- No compute process was present on GPU 4-7 before launch.
- Disk /home/hdd3 had 7.7T free, 44% used.
- System memory had about 220 GiB available.
- Swap was full: 2.0 GiB used out of 2.0 GiB.
- Existing tmux sessions were eval_1k and eval_32k; the Stage B session name did not exist before launch.

## Stage B Data Protocol
- Formal log showed `SemiSupervisedDataset` with `Stage=B`.
- Source set length: 1008.
- Target labeled set length: 25.
- Target unlabeled set length: 0.
- Validation set length: 28.
- Trainer log showed all ranks as `stage=B`.

## Launch Attempts
- `vc_suda_stage_b_1024_teacher8499_g4_7_20260513_2130` stopped immediately before training because the tmux non-login shell could not find `conda`.
- That failed setup log is kept at output/experiments/vc_suda_stage_b_1024_teacher8499_g4_7_20260513_2130/train.log.
- GPU 4-7 stayed idle after that failed setup attempt.
- `vc_suda_stage_b_1024_teacher8499_g4_7_20260513_2131` started the formal command and entered Stage B training.

## Early Check
- Checked after about 150 seconds.
- tmux session was no longer alive because the train process had exited with error.
- There were no remaining torchrun or train.py processes for the run.
- GPU 4-7 were idle after failure, each using about 15 MiB and 0% utilization.
- The run reached `start iter=0/9000` but did not complete iter 1.
- No immediate early-stop trigger was observed.

## Failure
- Result status: failed launch. Training is not running.
- First failing operation: backward pass in `magformer/engine/vc_suda_trainer.py`, line 511, at `self.scaler.scale(accum_loss).backward()`.
- Error on all ranks: `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`.
- Torch elastic reported `ChildFailedError` after rank failures.
- Per instruction, no repeated retry was attempted after the training error.

## Process IDs
- tmux pane PID during launch: 3381998.
- torchrun parent PID from failure log: 3382022.
- train.py rank PIDs from failure log: 3382074, 3382075, 3382076, 3382077.
- All of these processes had exited by the 150-second check.

## Status
- This document records the formal Stage B launch attempt and the failure mode.
- No code was changed for this record.
- Next action should be a code-level root-cause fix for the inplace gradient modification before another formal Stage B launch.
