# R135 official Mask2Former R134 iter4500 target150 smoke

Date: 2026-05-19
Host: `4029` (`/home/hdd3/zhanghaonan/magformer`)
Branch: `feature/vc-suda-sim2real`
Conda env: `mask2former`
TMUX session: `r135_official_m2f_r134iter4500_target150_smoke_g67`
Output: `output/baseline/r135_official_m2f_r134iter4500_target150_smoke_g67`
Normalized COCO dir: `output/diagnostics/r135_official_m2f_target150_coco`

## Source checkpoint choice

Use R134 iter4500, not R134 final, for target150 labeled-only finetuning.

R134 source validation curve from `output/baseline/r134_official_m2f_32k_cache_continue_3000_5000_g67/metrics.json`:

| checkpoint iter | bbox AP | segm AP |
| --- | ---: | ---: |
| 3499 | 50.0111 | 54.7139 |
| 3999 | 53.6355 | 58.8848 |
| 4499 | 55.1434 | 60.2268 |
| 5000 / final | 52.1816 | 58.4591 |

R134 `model_0004499.pth` is the best current source checkpoint by segm AP, so R135 loads:

```text
output/baseline/r134_official_m2f_32k_cache_continue_3000_5000_g67/model_0004499.pth
```

## R135 smoke setup

This is only a finite smoke stage to prove the path and get an initial target labeled-only signal. Do not call it the final RGB target150 baseline.

Dataset:

- Train: `magformer_datasets/pseudo_real_512/annotations/instances_target_labeled_r114_balanced_plus125.json`, 150 images, 9083 annotations, image dir `images/train`.
- Eval: `magformer_datasets/pseudo_real_512/annotations/instances_val.json`, 28 images, 1892 annotations, image dir `images/val_only`.

Key options:

```text
CUDA_VISIBLE_DEVICES=6,7
--num-gpus 2
SOLVER.MAX_ITER 200
SOLVER.IMS_PER_BATCH 4
SOLVER.CHECKPOINT_PERIOD 200
TEST.EVAL_PERIOD 0
MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100
TEST.DETECTIONS_PER_IMAGE 100
MODEL.WEIGHTS /home/hdd3/zhanghaonan/magformer/output/baseline/r134_official_m2f_32k_cache_continue_3000_5000_g67/model_0004499.pth
DATASETS.TRAIN ("eccpseudo_real_512_target150_labeled",)
DATASETS.TEST ("eccpseudo_real_512_val28",)
```

Exact launch command is recorded in:

```text
output/baseline/r135_official_m2f_r134iter4500_target150_smoke_g67/launch_command.sh
output/baseline/r135_official_m2f_r134iter4500_target150_smoke_g67/tmux_command.txt
```

## Verification

Environment probe before launch:

```text
CUDA_VISIBLE_DEVICES 6,7
CONDA_DEFAULT_ENV mask2former
torch 2.5.1+cu121
cuda_available True
device_count 2
logical 0 NVIDIA GeForce RTX 3090
logical 1 NVIDIA GeForce RTX 3090
detectron2_C_ok True
```

Launch log evidence:

```text
[official-mask2former] registered train dataset: eccpseudo_real_512_target150_labeled
[official-mask2former] registered val/test dataset: eccpseudo_real_512_val28
[DetectionCheckpointer] Loading from .../r134_official_m2f_32k_cache_continue_3000_5000_g67/model_0004499.pth ...
Starting training from iteration 0
```

Runtime GPU evidence showed the R135 worker pids only on physical GPUs 6 and 7. Separate `safa` jobs were present on GPUs 4 and 5 and were not touched.

## Smoke result

Status: passed.

- Final training iter: 199, so 200 total iterations.
- Final logged total loss: `22.5895`.
- Checkpoints: `model_0000199.pth` and `model_final.pth`.
- Val28 eval ran as part of the configured final evaluation.
- Val28 bbox AP/AP50/AP75/APs/APm/APl: `16.9757, 45.8431, 8.1479, 19.0126, nan, nan`.
- Val28 segm AP/AP50/AP75/APs/APm/APl: `12.9465, 35.4385, 6.6849, 13.1592, nan, nan`.

No remaining75 eval was started manually.
