# R132 Official Mask2Former 32K R97 Continuation Launch

Date: 2026-05-19
Host: `4029` (`/home/hdd3/zhanghaonan/magformer`)
Branch: `feature/vc-suda-sim2real`
Conda env: `mask2former`
TMUX session: `r132_official_m2f_32k_cache_continue_g67`
Output: `output/baseline/r132_official_m2f_32k_cache_continue_g67`
Source checkpoint: `output/baseline/r97_official_m2f_32k_cache_short/model_final.pth`

## Goal

Continue the official RGB-only Mask2Former 32K source baseline from R97 so it becomes a fair non-MagFormer source baseline before later pseudo-real labeled finetuning.

## Launch

The launch used only physical GPUs 6 and 7:

```bash
cd /home/hdd3/zhanghaonan/magformer
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mask2former
export CUDA_VISIBLE_DEVICES=6,7
bash output/baseline/r132_official_m2f_32k_cache_continue_g67/launch_command.sh
```

The generated launch script uses the documented R92/R95/R97 inline runtime shim: it pre-imports `detectron2._C` and forces `torch.multiprocessing.start_processes(..., start_method="fork")`, so the official wrapper's in-process ECC dataset registration is visible to the two worker processes. No tracked training code was changed.

Effective Detectron2 options:

```text
--resume
--num-gpus 2
--config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml
OUTPUT_DIR /home/hdd3/zhanghaonan/magformer/output/baseline/r132_official_m2f_32k_cache_continue_g67
DATASETS.TRAIN ("ecc20260318_1k_32254_train",)
DATASETS.TEST ("ecc20260318_1k_32254_val",)
MODEL.WEIGHTS /home/hdd3/zhanghaonan/magformer/output/baseline/r97_official_m2f_32k_cache_short/model_final.pth
MODEL.SEM_SEG_HEAD.NUM_CLASSES 1
MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100
TEST.DETECTIONS_PER_IMAGE 100
INPUT.IMAGE_SIZE 1024
SOLVER.IMS_PER_BATCH 4
SOLVER.MAX_ITER 3000
SOLVER.CHECKPOINT_PERIOD 500
TEST.EVAL_PERIOD 500
DATALOADER.NUM_WORKERS 2
```

`last_checkpoint` in the R132 output directory points to the R97 `model_final.pth`, so `--resume` loads model, trainer, and scheduler state. The log confirms:

```text
Loading trainer from .../r97_official_m2f_32k_cache_short/model_final.pth
Loading scheduler from state_dict ...
Starting training from iteration 1500
```

## Verification

Environment probe before launch:

```text
CUDA_VISIBLE_DEVICES 6,7
torch 2.5.1+cu121
cuda_available True
device_count 2
logical 0 NVIDIA GeForce RTX 3090
logical 1 NVIDIA GeForce RTX 3090
```

Runtime evidence after launch:

```text
CUDA_VISIBLE_DEVICES=6,7
CONDA_DEFAULT_ENV=mask2former
World size: 2
iter: 1559 total_loss: 18.95 lr: 0.0001 max_mem: 9381M
```

GPU evidence from `nvidia-smi` after training started:

```text
GPU 6: 10742 MiB, 87%, pid 73756, python
GPU 7: 11676 MiB, 100%, pid 73757, python
```

Physical GPU 4 had a separate `safa` process at the time of verification, and this R132 run had no process on GPU 4 or 5.

## Notes

A first direct wrapper launch failed before training because spawned workers did not inherit dataset registration. Its failed tmux log was moved to `output/baseline/r132_official_m2f_32k_cache_continue_g67.failed_spawn_*.tmux.log`. The active launch uses the already documented fork-shim path and has clean active `log.txt`, `metrics.json`, and `tmux_train.log` under the R132 output directory.
