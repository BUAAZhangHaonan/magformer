# VC-SUDA Stage B Launch - 2026-05-14

## Run
- Host: WS-4029GP-TRT via ssh 4029.
- Repository: `/home/hdd3/zhanghaonan/magformer`.
- Branch at launch: `feature/vc-suda-sim2real`.
- HEAD at launch: `d30e21b`.
- Config: `configs/vc_suda_stage_b_1024_teacher8499.yaml`.
- Conda env: `magformer`.
- tmux session: `stage_b_1024_teacher8499_20260514_005821`.
- tmux pane PID: `3450814`.
- torchrun PID: `3450822`.
- Train rank PIDs seen at launch: `3450832`, `3450833`, `3450834`, `3450835`.
- Output dir: `output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821`.
- Main log: `output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/train.log`.
- Launch script: `output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/launch.sh`.

## Preflight
- Git status before launch had one existing untracked path: `output/tmp/`.
- GPU 4-7 before launch were idle, each at about 15 MiB used and 0% utilization.
- GPU 4 was checked with `nvidia-smi -i 4 -q -d ECC,POWER,TEMPERATURE,COMPUTE`; no pending repair signal was reported, temperature was 27 C, compute mode was Default.
- Teacher checkpoint existed: `output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth`.

## Config Check
- `model.finetune_weights`: `output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth`.
- Dataset root: `magformer_datasets/pseudo_real_512`.
- Dataset splits: `train_split: train`, `val_split: val`.
- Dataset annotations: `train_ann: annotations/instances_source.json`, `val_ann: annotations/instances_val.json`.
- `runtime.gpus`: `[4, 5, 6, 7]` in the formal config.
- `runtime.early_stop.enabled`: `false`.
- Depth normalization: `data.depth.norm: minmax`, `per_sample_norm: true`, `clip_min: 0.0`, `clip_max: 2.095623016357422`.
- Depth sanity preflight is skipped by config: `runtime.skip_depth_sanity: true`.

## Command

The launch used `CUDA_VISIBLE_DEVICES=4,5,6,7` to limit training to physical GPUs 4-7. Under this mask, the process-visible device IDs are logical `0,1,2,3`, so the training CLI used `--gpus 0,1,2,3` while keeping the formal Stage B config file unchanged.

```bash
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate magformer
export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc_per_node=4 tools/train.py \
  --config configs/vc_suda_stage_b_1024_teacher8499.yaml \
  --output-dir output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821 \
  --gpus 0,1,2,3 \
  --num-workers 2
```

## Initial Log Evidence
- Data loading reached all four ranks.
- Stage B dataset log: `SemiSupervisedDataset] Stage=B, source=1008, target_labeled=25, target_unlabeled=0`.
- Validation set log: `CocoRgbdDataset] Loaded 28 images from val split`.
- Warm-start log on all four ranks: `Warm-start loaded model weights from output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth`.
- Checkpoint key check on all four ranks: `Warm-start missing keys: 0, unexpected keys: 0`.
- Non-fatal warning: SHA256 sidecar was not found for the teacher checkpoint.
- Depth sanity log: `Skipping depth sanity preflight by configuration`.
- Training start log: `start iter=0/9000 eval_period=99999 ckpt_period=500 log_period=20`.

## First Metrics
- First structured line:
  - `[2026-05-14 00:58:45] iter=0/9000 eta=20:14:23 time=8.10s lr=1e-07 loss=96.0030 loss_ce=0.1420 loss_dice=0.0498 loss_mask=0.0455`
- First tqdm loss samples across ranks:
  - `iter=1, loss=98.0728, lr=0.000000`
  - `iter=1, loss=132.6142, lr=0.000000`
  - `iter=1, loss=61.2495, lr=0.000000`
  - `iter=1, loss=96.0030, lr=0.000000`
- Later structured line observed during monitoring:
  - `[2026-05-14 00:59:37] iter=20/9000 eta=06:27:22 time=2.59s lr=2.764e-06 loss=75.6853 loss_ce=0.1010 loss_dice=0.0844 loss_mask=0.0656`

## Monitoring Result
- tmux was still alive at the last check.
- GPU 4-7 were active at the last check:
  - GPU 4: 15693 MiB / 24576 MiB, 72% util, 66 C.
  - GPU 5: 16013 MiB / 24576 MiB, 99% util, 67 C.
  - GPU 6: 16013 MiB / 24576 MiB, 89% util, 62 C.
  - GPU 7: 16045 MiB / 24576 MiB, 75% util, 60 C.
- Error scan count was zero for: `cuda illegal`, `out of memory`, `traceback`, `ChildFailedError`, `RuntimeError`, `NCCL`, and `error`.
- No checkpoint mismatch was observed.
- No DDP buffer mutation bug was observed during the monitored window.
- No depth constant bug was observed in the monitored window; depth sanity itself is skipped by this config.
- No OOM was observed.

## Status
- Result status: running.
- Training was left running in tmux.
- This record is documentation only. No training code or config file was changed.

## Post-Training Segm AP Gate

Run this only after the Stage B training process has finished and the final checkpoint exists. The launch-time final eval is not enough for the target metric because `configs/vc_suda_stage_b_1024_teacher8499.yaml` only sets `runtime.eval_iou_types: [bbox]`.

Use the dedicated post-eval config:

- Config: `configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml`.
- Dataset root: `magformer_datasets/pseudo_real_512`.
- Validation annotation: `annotations/instances_val.json`.
- Validation split: `val`.
- IoU types: `bbox`, `segm`.
- Empty prediction behavior: `tools/evaluate.py` runs with `fail_on_empty=True`.
- Warm-start safety: `model.finetune_weights: null`; pass the final checkpoint through `--weights` so it cannot be confused with training warm-start or resume state.
- Data split safety: `vc_suda.enabled: false`; target labeled and unlabeled annotations are unset for this eval-only config.

Example command, after replacing the checkpoint path if the run directory changes:

```bash
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate magformer
export CUDA_VISIBLE_DEVICES=0
python tools/evaluate.py \
  --config-file configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml \
  --weights output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/model_final.pth \
  --output output/eval/vc_suda_stage_b_1024_teacher8499_segm \
  --batch-size 1 \
  --num-workers 2
```

Expected gate output includes both `bbox_AP` and `segm_AP` in the printed COCO metrics and writes `output/eval/vc_suda_stage_b_1024_teacher8499_segm/coco_instances_results.json`. Do not run this while the Stage B training job is still using GPUs 4-7.
