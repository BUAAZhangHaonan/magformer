# R95 Official Mask2Former Source Short Training

Date: 2026-05-17
Host: `4029` (`/home/hdd3/zhanghaonan/magformer`)
Conda env: `mask2former`
TMUX session: `r95_official_m2f_source_short`
Output: `output/baseline/r95_official_m2f_source_1k1024_short`
Dataset root: `magformer_datasets/20260318_1K_1566`
Register id: `20260318_1K_1566`

## Goal

Verify that the official RGB-only Mask2Former baseline can learn non-zero source-val AP on the `20260318_1K_1566` 1K_1024 source split before moving to 32K.

## Command

The run used the `mask2former` env, GPU 4-7, and an inline runtime shim that pre-imported compiled `detectron2._C` and forced `torch.multiprocessing.start_processes(..., start_method="fork")`. This matched the prior R92/R94 workaround and did not modify source code.

```bash
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate mask2former
cd /home/hdd3/zhanghaonan/magformer
export CUDA_VISIBLE_DEVICES=4,5,6,7
python -u -c 'import sys, runpy; import torch.multiprocessing as mp; import detectron2, detectron2._C; orig = mp.start_processes; exec("def fork_start_processes(fn, args=(), nprocs=1, join=True, daemon=False, start_method=\"spawn\"):\n    return orig(fn, args, nprocs, join, daemon, start_method=\"fork\")"); mp.start_processes = fork_start_processes; sys.argv = ["baselines/run_official_mask2former_ecc.py", "--register", "20260318_1K_1566", "--dataset-root", "magformer_datasets/20260318_1K_1566", "--normalized-ann-dir", "output/diagnostics/r95_official_mask2former_coco", "--", "--num-gpus", "4", "--config-file", "configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml", "OUTPUT_DIR", "/home/hdd3/zhanghaonan/magformer/output/baseline/r95_official_m2f_source_1k1024_short", "DATASETS.TRAIN", "(\"ecc20260318_1k_1566_train\",)", "DATASETS.TEST", "(\"ecc20260318_1k_1566_val\",)", "MODEL.SEM_SEG_HEAD.NUM_CLASSES", "1", "MODEL.MASK_FORMER.NUM_OBJECT_QUERIES", "100", "TEST.DETECTIONS_PER_IMAGE", "100", "INPUT.IMAGE_SIZE", "1024", "SOLVER.IMS_PER_BATCH", "4", "SOLVER.MAX_ITER", "3000", "SOLVER.CHECKPOINT_PERIOD", "500", "TEST.EVAL_PERIOD", "500", "DATALOADER.NUM_WORKERS", "2", "MODEL.WEIGHTS", "detectron2://ImageNetPretrained/torchvision/R-50.pkl"]; runpy.run_path("baselines/run_official_mask2former_ecc.py", run_name="__main__")'
```

Note: the first tmux launch used `set -u` and stopped at `conda activate` because conda referenced an unset `ADDR2LINE` variable. Training had not started and the output directory did not exist. The command was relaunched in the same requested tmux session name without `set -u`.

## Metrics

Completed evals from `metrics.json` / `log.txt`:

| Iteration | Checkpoint | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | segm APs | segm APm | segm APl |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 499 | `model_0000499.pth` | 0.0000 | 0.0000 | 0.0000 | 7.8451 | 20.7823 | 4.4291 | 0.1600 | 10.3355 | nan |

Best bbox AP: `0.0000` at `model_0000499.pth`.
Best segm AP: `7.8451` at `model_0000499.pth`.

Prediction sanity check for `inference/coco_instances_results.json`:

- Predictions: `14900`
- Predictions with score > 0: `10617`
- Max score: `0.9440508484840393`
- Mean score: `0.5958529979850622`
- Segmentation results present: yes

## Stop Reason

Stopped after the first eval because source-val `segm/AP=7.8451`, which is above the requested `>= 0.10` gate, and predictions were not all zero. The stop was done with Ctrl-C after the gate was verified, so the tmux log ends with an expected `KeyboardInterrupt` after iter 575.

## Conclusion

Pass.

The official Mask2Former RGB-only source-supervised baseline learns non-zero source-val segmentation AP on the 1K_1024 source split. This validates the learning link.

Proceed to 32K: yes, for the official RGB-only source-supervised baseline. This result does not validate semi-supervised training or MagFormer RGB-D.
