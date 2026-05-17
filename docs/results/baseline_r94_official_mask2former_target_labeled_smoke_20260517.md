# R94 official Mask2Former target_labeled25 smoke

Date: 2026-05-17
Host: `4029` (`/home/hdd3/zhanghaonan/magformer`)
Conda env: `mask2former`
Branch: `feature/vc-suda-sim2real`

## Goal

Verify the fixed official Mask2Former RGB-only wrapper on pseudo-real target-labeled training:

- Train: `magformer_datasets/pseudo_real_512/annotations/instances_target_labeled.json`, 25 images, 1697 annotations.
- Eval: `magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json`, 200 images, 11750 annotations.
- Image root for both splits: `magformer_datasets/pseudo_real_512/images/train`.
- Config: `configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml` under `baselines/Mask2Former`.
- Limit: 200 training iter. This is a smoke/trend check, not a final baseline.

## Start Point

Chosen start point: `output/baseline/r92_official_mask2former_1k1024_smoke_mask2former_env/model_final.pth`.

Reason: R92 was weak, only 10 iter and AP 0, but it already proved the official wrapper, one-class head, compiled Detectron2, and Mask2Former runtime can start and save a checkpoint in `mask2former`. For this R94 task, that is a better chain-link validation than starting again from ImageNet R50. `MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 200` was kept for checkpoint shape compatibility; `TEST.DETECTIONS_PER_IMAGE` stayed at the requested `100`.

## Static Checks

- Train ann exists: `magformer_datasets/pseudo_real_512/annotations/instances_target_labeled.json`.
- Eval ann exists: `magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json`.
- Image dir exists: `magformer_datasets/pseudo_real_512/images/train`.
- All train/eval `file_name` entries resolve under `images/train`.
- Explicit wrapper args register `eccpseudo_real_512_target_labeled` and `eccpseudo_real_512_target_unlabeled`.

## Wrapper Fix

The target-labeled/unlabeled COCO annotations store `segmentation` as RLE dicts. The official Mask2Former instance mapper used here is polygon-only and ignores `INPUT.MASK_FORMAT bitmask`. Training failed before iter 1 until the wrapper normalized temporary COCO JSON from RLE masks to polygon masks.

Change made in `baselines/run_official_mask2former_ecc.py`:

- Source annotation JSON is not edited.
- Temporary normalized JSON still goes under `output/diagnostics/...`.
- RLE masks are decoded and polygonized before registration.
- Tiny non-empty masks that do not produce a valid contour polygon get a valid bounding-rectangle polygon.

Tests added in `tests/test_official_mask2former_wrapper_registration.py`:

- RLE masks become polygon lists in normalized COCO.
- Tiny non-empty RLE masks still produce a valid polygon.
- Existing explicit split registration tests still pass.

Validation command:

```bash
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate mask2former
cd /home/hdd3/zhanghaonan/magformer
python tests/test_official_mask2former_wrapper_registration.py
```

Real pseudo-real normalized check after the fix:

- `instances_target_labeled.json`: 1697 annotations, empty segmentation 0, bad polygons 0.
- `instances_target_unlabeled.json`: 11750 annotations, empty segmentation 0, bad polygons 0.

## Train Command

Launched in tmux session `r94_official_m2f_target_labeled25_smoke` with GPUs 4-7.

Effective command options:

```bash
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate mask2former
cd /home/hdd3/zhanghaonan/magformer
export CUDA_VISIBLE_DEVICES=4,5,6,7
python baselines/run_official_mask2former_ecc.py \
  --register pseudo_real_512 \
  --dataset-root magformer_datasets/pseudo_real_512 \
  --train-ann annotations/instances_target_labeled.json \
  --val-ann annotations/instances_target_unlabeled.json \
  --train-image-dir images/train \
  --val-image-dir images/train \
  --train-split target_labeled \
  --val-split target_unlabeled \
  --normalized-ann-dir output/diagnostics/r94_official_mask2former_coco \
  -- \
  --num-gpus 4 \
  --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
  OUTPUT_DIR /home/hdd3/zhanghaonan/magformer/output/baseline/r94_official_m2f_target_labeled25_smoke \
  DATASETS.TRAIN '("eccpseudo_real_512_target_labeled",)' \
  DATASETS.TEST '("eccpseudo_real_512_target_unlabeled",)' \
  SOLVER.MAX_ITER 200 \
  SOLVER.IMS_PER_BATCH 4 \
  SOLVER.CHECKPOINT_PERIOD 200 \
  TEST.EVAL_PERIOD 0 \
  INPUT.IMAGE_SIZE 1024 \
  MODEL.SEM_SEG_HEAD.NUM_CLASSES 1 \
  MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 200 \
  MODEL.WEIGHTS /home/hdd3/zhanghaonan/magformer/output/baseline/r92_official_mask2former_1k1024_smoke_mask2former_env/model_final.pth \
  TEST.DETECTIONS_PER_IMAGE 100 \
  DATALOADER.NUM_WORKERS 2
```

A small Python shim pre-imported compiled `detectron2._C` and forced `torch.multiprocessing.start_processes(..., start_method="fork")`, matching the R92 workaround so custom in-process dataset registration is visible to workers.

Train log: `output/baseline/r94_official_m2f_target_labeled25_smoke.tmux.log`.

## Train Result

Status: passed.

- Final checkpoint: `output/baseline/r94_official_m2f_target_labeled25_smoke/model_final.pth`.
- Final iter: 199, so 200 total iterations.
- Final logged total loss: `49.6945`.
- Final logged component losses: `loss_ce=0.0937`, `loss_mask=0.1674`, `loss_dice=4.5472`.
- Earlier loss trend: iter 19 `54.6993`, iter 99 `49.2185`, iter 179 `48.8393`.
- Max memory: `5953M`.
- No OOM, no training Traceback after the wrapper fix, and no non-finite training loss.

The train script also ran its default final eval on target_unlabeled and wrote `output/baseline/r94_official_m2f_target_labeled25_smoke/inference/coco_instances_results.json`.

## Eval Command

Ran separately on GPU 4 with the requested eval output directory:

```bash
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate mask2former
cd /home/hdd3/zhanghaonan/magformer
export CUDA_VISIBLE_DEVICES=4
python baselines/run_official_mask2former_ecc.py \
  --register pseudo_real_512 \
  --dataset-root magformer_datasets/pseudo_real_512 \
  --train-ann annotations/instances_target_labeled.json \
  --val-ann annotations/instances_target_unlabeled.json \
  --train-image-dir images/train \
  --val-image-dir images/train \
  --train-split target_labeled \
  --val-split target_unlabeled \
  --normalized-ann-dir output/diagnostics/r94_official_mask2former_coco_eval \
  -- \
  --num-gpus 1 \
  --eval-only \
  --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
  OUTPUT_DIR /home/hdd3/zhanghaonan/magformer/output/baseline/r94_official_m2f_target_unlabeled200_eval \
  DATASETS.TRAIN '("eccpseudo_real_512_target_labeled",)' \
  DATASETS.TEST '("eccpseudo_real_512_target_unlabeled",)' \
  INPUT.IMAGE_SIZE 1024 \
  MODEL.SEM_SEG_HEAD.NUM_CLASSES 1 \
  MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 200 \
  MODEL.WEIGHTS /home/hdd3/zhanghaonan/magformer/output/baseline/r94_official_m2f_target_labeled25_smoke/model_final.pth \
  TEST.DETECTIONS_PER_IMAGE 100 \
  DATALOADER.NUM_WORKERS 2
```

Eval log: `output/baseline/r94_official_m2f_target_unlabeled200_eval.eval.log`.

## Eval Result

Status: passed.

- Results JSON: `output/baseline/r94_official_m2f_target_unlabeled200_eval/inference/coco_instances_results.json`.
- bbox AP/AP50/AP75/APs/APm/APl: `0.0000, 0.0000, 0.0000, 0.0000, nan, nan`.
- segm AP/AP50/AP75/APs/APm/APl: `0.0000, 0.0000, 0.0000, 0.0000, nan, nan`.

## Conclusion

R94 passed as a short official Mask2Former RGB-only target_labeled25 smoke after the wrapper-level RLE-to-polygon normalization fix. The run verifies that the fixed official wrapper can train on `pseudo_real target_labeled25` and evaluate on `target_unlabeled200` with COCO bbox/segm output.

The AP is still 0.0000, so this is only a chain and trend smoke. It is worth doing a real RGB-only baseline long run next, but only as a proper baseline with a clean config and enough iterations, not as another 200-iter smoke.
