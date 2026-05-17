# R92 official Mask2Former RGB-only smoke

Date: 2026-05-17
Host: `4029` (`/home/hdd3/zhanghaonan/magformer`)
Conda env: `magformer`
Branch: `feature/vc-suda-sim2real`
Start commit: `49312cb839ddf6fd82f1d74c1645e412e666ecd9`

## Goal

Verify the existing official Mask2Former RGB-only baseline wrapper on `20260318_1K_1566` with a 10-iter smoke run, then evaluate the produced checkpoint on `pseudo_real_512` val28 for COCO bbox/segm AP.

This run intentionally did not enter 32K, semi-supervised training, MagFormer RGB-D, or VC-SUDA.

## Preflight

- GPUs 4-7 were idle before launch: each showed about 15 MiB used and 0% utilization, with only `Xorg` listed by `nvidia-smi pmon`.
- Training output path was missing before launch: `output/baseline/r92_official_mask2former_1k1024_smoke`.
- Eval output path was missing before launch: `output/baseline/r92_official_mask2former_pseudoreal_val28_eval`.
- The requested root config path `configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml` does not exist at repo root. The wrapper changes into `baselines/Mask2Former`, where the official config exists at `configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml`.

## Training command

The smoke was launched in tmux session `r92_official_m2f_smoke`:

```bash
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate magformer
cd /home/hdd3/zhanghaonan/magformer
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONUNBUFFERED=1
python baselines/run_official_mask2former_ecc.py \
  --register 20260318_1K_1566 \
  --dataset-root magformer_datasets/20260318_1K_1566 \
  -- \
  --num-gpus 4 \
  --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
  OUTPUT_DIR output/baseline/r92_official_mask2former_1k1024_smoke \
  DATASETS.TRAIN '("ecc20260318_1k_1566_train",)' \
  DATASETS.TEST '("ecc20260318_1k_1566_val",)' \
  SOLVER.MAX_ITER 10 \
  SOLVER.IMS_PER_BATCH 4 \
  SOLVER.CHECKPOINT_PERIOD 10 \
  TEST.EVAL_PERIOD 0 \
  INPUT.IMAGE_SIZE 1024 \
  MODEL.WEIGHTS detectron2://ImageNetPretrained/torchvision/R-50.pkl \
  DATALOADER.NUM_WORKERS 2
```

Log path: `output/baseline/r92_official_mask2former_1k1024_smoke.tmux.log`

## Training result

Status: failed before iter 1.

The process stopped with a Traceback while importing Detectron2's compiled extension:

```text
ImportError: cannot import name '_C' from 'detectron2' (unknown location)
```

The failing import path was:

```text
baselines/Mask2Former/mask2former/evaluation/instance_evaluation.py
baselines/detectron2/detectron2/evaluation/fast_eval_api.py
```

No OOM was observed. No non-finite loss was reached because training did not start.

Checkpoint status: `output/baseline/r92_official_mask2former_1k1024_smoke/model_final.pth` was not produced.

## Eval command

Eval was not run because the requested condition was not met: `model_final.pth` did not exist.

Planned eval target:

- Register: `pseudo_real_512`
- Dataset root: `magformer_datasets/pseudo_real_512`
- Test set: `("eccpseudo_real_512_val",)`
- Output path: `output/baseline/r92_official_mask2former_pseudoreal_val28_eval`
- GPU: 4
- Detections per image: 200
- Image size: 1024

## Eval result

COCO bbox AP: not available.
COCO segm AP: not available.
Results JSON: `output/baseline/r92_official_mask2former_pseudoreal_val28_eval/inference/coco_instances_results.json` was not produced.

## Pass/fail

Pass: no.

Reason: the existing official Mask2Former RGB-only wrapper/runtime could not start training in `conda env magformer` because Detectron2 `_C` was unavailable from the imported `baselines/detectron2` package.

Wrapper/runtime follow-up needed: yes. The current path/import setup needs to use a Detectron2 build with `_C` available, or `baselines/detectron2` must be built for the `magformer` env before this official wrapper can train and produce pseudo-real val28 AP.

Additional eval-layout note: `magformer_datasets/pseudo_real_512/annotations/instances_val.json` has 28 images, while the images are under `images/val_only`; the generic custom COCO registration currently expects `images/val`. After the training import issue is fixed, eval registration should also confirm this val image-root mapping.
