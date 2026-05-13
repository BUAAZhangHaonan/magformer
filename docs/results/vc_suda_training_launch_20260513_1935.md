# VC-SUDA Stage A Early-Stop Short Run - 2026-05-13 19:35

## Run
- Host: WS-4029GP-TRT
- Repository: /home/hdd3/zhanghaonan/magformer
- Branch at launch: feature/vc-suda-sim2real
- Commit at launch: 2390bb8ac191176528931063fd846733330133fc
- Config: configs/vc_suda_stage_a_40ep_512.yaml
- Stage: VC-SUDA Stage A
- Image size: 512
- Conda env: magformer

## Command

    unset CUDA_VISIBLE_DEVICES
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export NUMEXPR_NUM_THREADS=4
    export PYTHONUNBUFFERED=1
    /home/hdd3/zhanghaonan/anaconda3/bin/conda run --no-capture-output -n magformer \
      torchrun --standalone --nproc_per_node=4 tools/train.py \
      --config configs/vc_suda_stage_a_40ep_512.yaml \
      --output-dir output/experiments/vc_suda_stage_a_512_g4_7_20260513_1935 \
      --gpus 4,5,6,7 \
      --num-workers 2

## Runtime
- tmux session: vc_suda_stage_a_512_g4_7_20260513_1935
- Output dir: output/experiments/vc_suda_stage_a_512_g4_7_20260513_1935
- Main log: output/experiments/vc_suda_stage_a_512_g4_7_20260513_1935/train.log
- Launch script: output/experiments/vc_suda_stage_a_512_g4_7_20260513_1935/launch.sh
- Copied config: output/experiments/vc_suda_stage_a_512_g4_7_20260513_1935/config.yaml
- GPUs: physical GPU 4,5,6,7 through config/runtime gpus. CUDA_VISIBLE_DEVICES is unset.
- Resource limits: OMP_NUM_THREADS=4, MKL_NUM_THREADS=4, NUMEXPR_NUM_THREADS=4, dataloader workers=2 per rank.

## Preflight
- Git status was clean before launch.
- HEAD was 2390bb8 before launch.
- GPU 4-7 were idle before launch, each using about 15 MiB and 0% utilization.
- Disk /home/hdd3 had 7.7T free, 44% used.
- tmux session name did not exist before launch.

## Smoke Evidence
- Handoff prerequisite stated that the 1-step smoke had passed on HEAD 2390bb8.
- This launch used the same HEAD, the same VC-SUDA Stage A config family, and a clean worktree.
- Early formal-run evidence after launch: DDP initialized rank 0-3, Stage A started, and training reached iter 80/3160 with metrics written.

## Early Check
- Checked after about 75 seconds.
- tmux session remained active.
- torchrun parent and four train.py ranks were running.
- nvidia-smi pmon showed one python3.11 compute process on each GPU 4,5,6,7.
- Main tmux log showed no Traceback, RuntimeError, CUDA OOM, ChildFailedError, or abort during the early check.
- Non-fatal warnings observed: missing SHA256 sidecar for warm-start checkpoint, deprecated NCCL_ASYNC_ERROR_HANDLING, unsupported expandable_segments, and DDP grad stride performance warnings.
- First eval at iter 78 wrote bbox/segm AP=0.0. This is early-run evidence, not a result claim.

## Follow-up Status
- After the docs write was corrected, the formal tmux run was rechecked and was still running.
- Recheck showed the same tmux session active, four train.py ranks alive, and progress past iter 140/3160.
- During the first docs write attempt, an unquoted heredoc expanded the Markdown command block and started an extra foreground duplicate command outside tmux. That duplicate command failed with CUDA OOM because the formal tmux run already owned GPU 4-7. The formal tmux run stayed active, and its train.log did not contain that fatal error.

## Final Status
- Result status: failed short run. Do not use `vc_suda_stage_a_512_g4_7_20260513_1935` as the formal Stage A result.
- Stop point: iter 394/3160 after the fifth eval.
- Last metric: segm AP 0.724 AP points (`val/segm_AP=0.007240232484208836` on the 0-1 scale).
- Checkpoints kept: `output/experiments/vc_suda_stage_a_512_g4_7_20260513_1935/model_best.pth` and `output/experiments/vc_suda_stage_a_512_g4_7_20260513_1935/checkpoint_iter_0000394.pth`.
- Root cause: `runtime.early_stop: null` was interpreted as an empty config, so Trainer silently used `patience=5`, `min_delta=0.1`, and `target_ap=70.0`. The metric is stored on a 0-1 scale, but these defaults were 0-100-style AP points, so the run plateaued under the default 0.1 delta and stopped early.
- Fix direction: early stop must be opt-in with `runtime.early_stop.enabled: true`; the Stage A config now writes `runtime.early_stop.enabled: false` explicitly.
