# VC-SUDA Stage A Resume Training Launch - 2026-05-13 20:06

## Run
- Host: WS-4029GP-TRT
- Repository: /home/hdd3/zhanghaonan/magformer
- Branch at launch: feature/vc-suda-sim2real
- Commit at launch: a52a8c0
- Config: configs/vc_suda_stage_a_40ep_512.yaml
- Stage: VC-SUDA Stage A
- Image size: 512
- Conda env: magformer

## Resume Source
- Previous short-run output dir: output/experiments/vc_suda_stage_a_512_g4_7_20260513_1935
- Resume checkpoint: output/experiments/vc_suda_stage_a_512_g4_7_20260513_1935/checkpoint_iter_0000394.pth
- Resume checkpoint size at preflight: 913M
- Previous short run failed by unintended early stop at iter 394.
- Early stop is now disabled explicitly by the pushed fix at commit a52a8c0.

## Command

    source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
    unset CUDA_VISIBLE_DEVICES
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export NUMEXPR_NUM_THREADS=4
    export PYTHONUNBUFFERED=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    conda run --no-capture-output -n magformer \
      torchrun --standalone --nproc_per_node=4 tools/train.py \
      --config configs/vc_suda_stage_a_40ep_512.yaml \
      --output-dir output/experiments/vc_suda_stage_a_512_g4_7_resume_earlystopfix_20260513_2006 \
      --gpus 4,5,6,7 \
      --num-workers 2 \
      --resume output/experiments/vc_suda_stage_a_512_g4_7_20260513_1935/checkpoint_iter_0000394.pth

## Runtime
- tmux session: vc_suda_stage_a_resume_g4_7_20260513_2006
- Output dir: output/experiments/vc_suda_stage_a_512_g4_7_resume_earlystopfix_20260513_2006
- Main launch log: output/experiments/vc_suda_stage_a_512_g4_7_resume_earlystopfix_20260513_2006/resume_launch.log
- Metrics CSV: output/experiments/vc_suda_stage_a_512_g4_7_resume_earlystopfix_20260513_2006/metrics_log.csv
- GPUs: physical GPU 4,5,6,7. CUDA_VISIBLE_DEVICES is unset.
- Resource limits: OMP_NUM_THREADS=4, MKL_NUM_THREADS=4, NUMEXPR_NUM_THREADS=4, dataloader workers=2 per rank.

## Preflight
- Git status was clean before launch.
- HEAD was a52a8c0 before launch.
- GPU 4-7 were idle before launch, each using about 15 MiB and 0% utilization.
- Disk /home/hdd3 had 7.7T free, 44% used.
- Existing tmux sessions were eval_1k and eval_32k; the resume session name did not exist before launch.
- The resume checkpoint file existed before launch.

## Early Check
- Checked after about 100 seconds.
- tmux session remained active.
- torchrun parent and four train.py ranks were running.
- nvidia-smi showed active GPU use on physical GPU 4,5,6,7.
- Log showed checkpoint load from iter 394 and `start iter=394/3160`.
- Training progressed past iter 394 and wrote metrics through at least iter 480.
- First resumed eval ran at iter 473 and saved checkpoint_iter_0000473.pth plus model_best.pth.
- No fatal, traceback, CUDA OOM, ChildFailedError, or immediate early-stop trigger was observed in the early check.
- Non-fatal warning observed: PyTorch DDP grad stride performance warning.

## Early Metrics Snapshot
- Iter 460 train loss: 34.5322, lr=0.0002.
- Iter 473 validation: bbox AP=0.0284, bbox AP50=0.1156, segm AP=0.0135, segm AP50=0.0470.
- Iter 480 train loss: 39.7814, lr=0.0002.

## Status
- Result status: formal resume training is running.
- Do not stop the tmux session unless a later operator decides to intervene.
- This document records the launch only. Final training quality should be judged from the completed run output.
