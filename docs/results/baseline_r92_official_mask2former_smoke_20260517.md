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

## Retry in `mask2former` env

Date: 2026-05-17
Host: `4029` (`/home/hdd3/zhanghaonan/magformer`)
Conda env: `mask2former`
Train tmux session: `r92_official_m2f_smoke_mask2former_env`
Train output: `output/baseline/r92_official_mask2former_1k1024_smoke_mask2former_env`
Eval output: `output/baseline/r92_official_mask2former_pseudoreal_val28_eval_mask2former_env`

### Env verification

The retry used the compiled Detectron2 and Mask2Former pieces from `conda env mask2former` plus the official repo-local `mask2former` package:

```text
torch: version=2.5.1+cu121 file=/home/hdd3/zhanghaonan/anaconda3/envs/mask2former/lib/python3.11/site-packages/torch/__init__.py
detectron2: version=0.6 file=/home/hdd3/zhanghaonan/anaconda3/envs/mask2former/lib/python3.11/site-packages/detectron2/__init__.py
detectron2._C: version=n/a file=/home/hdd3/zhanghaonan/anaconda3/envs/mask2former/lib/python3.11/site-packages/detectron2/_C.cpython-311-x86_64-linux-gnu.so
MultiScaleDeformableAttention: version=n/a file=/home/hdd3/zhanghaonan/anaconda3/envs/mask2former/lib/python3.11/site-packages/MultiScaleDeformableAttention.cpython-311-x86_64-linux-gnu.so
mask2former: version=n/a file=/home/hdd3/zhanghaonan/magformer/baselines/Mask2Former/mask2former/__init__.py
```

### Retry command notes

The wrapper was still `baselines/run_official_mask2former_ecc.py`. Two runtime fixes were needed to make the official wrapper work with this local one-class ECC dataset:

- `detectron2` and `detectron2._C` were imported before running the wrapper, so the wrapper did not shadow the compiled conda Detectron2 with `baselines/detectron2`.
- `torch.multiprocessing.start_processes` was forced to `fork`, so the 4-GPU workers inherited the wrapper's in-process custom dataset registration.

The final 10-iter train command used these effective options:

```bash
conda activate mask2former
cd /home/hdd3/zhanghaonan/magformer
export CUDA_VISIBLE_DEVICES=4,5,6,7
python baselines/run_official_mask2former_ecc.py \
  --register 20260318_1K_1566 \
  --dataset-root magformer_datasets/20260318_1K_1566 \
  -- \
  --num-gpus 4 \
  --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
  OUTPUT_DIR /home/hdd3/zhanghaonan/magformer/output/baseline/r92_official_mask2former_1k1024_smoke_mask2former_env \
  DATASETS.TRAIN '("ecc20260318_1k_1566_train",)' \
  DATASETS.TEST '("ecc20260318_1k_1566_val",)' \
  SOLVER.MAX_ITER 10 \
  SOLVER.IMS_PER_BATCH 4 \
  SOLVER.CHECKPOINT_PERIOD 10 \
  TEST.EVAL_PERIOD 0 \
  INPUT.IMAGE_SIZE 1024 \
  MODEL.SEM_SEG_HEAD.NUM_CLASSES 1 \
  MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 200 \
  MODEL.WEIGHTS detectron2://ImageNetPretrained/torchvision/R-50.pkl \
  DATALOADER.NUM_WORKERS 2
```

`MODEL.SEM_SEG_HEAD.NUM_CLASSES 1` matches the registered ECC metadata. `MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 200` is required by the requested `TEST.DETECTIONS_PER_IMAGE 200`; otherwise one-class Mask2Former has only 100 default query scores and top-200 inference fails.

### Retry training result

Status: passed.

- Final checkpoint: `output/baseline/r92_official_mask2former_1k1024_smoke_mask2former_env/model_final.pth`
- Final train log: `output/baseline/r92_official_mask2former_1k1024_smoke_mask2former_env.tmux.log`
- Final logged iter: `iter: 9`
- Final logged loss: `total_loss: 88.64`
- Max memory: `5623M`
- No OOM, Traceback, or non-finite loss was found in the final train log.

### Retry eval command notes

Eval used GPU 4, the new `model_final.pth`, `DATASETS.TEST '("eccpseudo_real_512_val",)'`, `TEST.DETECTIONS_PER_IMAGE 200`, `INPUT.IMAGE_SIZE 1024`, `MODEL.SEM_SEG_HEAD.NUM_CLASSES 1`, and `MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 200`.

One runtime pycocotools compatibility patch was needed because `magformer_datasets/pseudo_real_512/annotations/instances_val.json` lacks top-level `info` and `licenses`; the patch added empty defaults in memory before `COCO.loadRes` and did not edit the dataset file.

### Retry eval result

Status: passed.

- Eval log: `output/baseline/r92_official_mask2former_pseudoreal_val28_eval_mask2former_env.eval.log`
- Results JSON: `output/baseline/r92_official_mask2former_pseudoreal_val28_eval_mask2former_env/inference/coco_instances_results.json`
- COCO bbox AP: `0.0000`
- COCO bbox AP50/AP75/APs/APm/APl: `0.0000,0.0000,0.0000,nan,nan`
- COCO segm AP: `0.0000`
- COCO segm AP50/AP75/APs/APm/APl: `0.0000,0.0000,0.0000,nan,nan`

### Retry pass/fail

Pass: yes.

The `mask2former` env has the compiled extensions needed for the official Mask2Former RGB-only smoke. The final 10-iter train produced `model_final.pth`, and the pseudo-real val28 eval produced COCO bbox/segm AP plus `inference/coco_instances_results.json`.

## R94 wrapper split-registration fix

Date: 2026-05-17
Host: `4029` (`/home/hdd3/zhanghaonan/magformer`)
Conda env: `mask2former`
Branch: `feature/vc-suda-sim2real`

### Wrapper changes

`baselines/run_official_mask2former_ecc.py` now supports explicit official Mask2Former COCO split registration:

- `--train-ann`, `--val-ann`
- `--train-image-dir`, `--val-image-dir`
- `--train-split`, `--val-split`
- `--normalized-ann-dir`

This allows pseudo-real splits such as:

```bash
--train-ann annotations/instances_target_labeled.json \
--val-ann annotations/instances_target_unlabeled.json \
--train-image-dir images/train \
--val-image-dir images/train \
--train-split target_labeled \
--val-split target_unlabeled
```

Registered dataset names become:

- `eccpseudo_real_512_target_labeled`
- `eccpseudo_real_512_target_unlabeled`

The wrapper fails immediately if a requested annotation file or image directory is missing. It does not edit the source annotation JSON. If `info` or `licenses` is missing, it writes a temporary normalized JSON under `output/diagnostics/...` and registers that path in Detectron2 metadata.

### Validation commands

Lightweight regression script:

```bash
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate mask2former
cd /home/hdd3/zhanghaonan/magformer
python tests/test_official_mask2former_wrapper_registration.py
```

Real pseudo-real registration check:

```bash
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate mask2former
cd /home/hdd3/zhanghaonan/magformer
python - <<'PY'
from pathlib import Path
from pycocotools.coco import COCO
from detectron2.data import DatasetCatalog, MetadataCatalog
from baselines.run_official_mask2former_ecc import register_official_mask2former_datasets
train_name, val_name = register_official_mask2former_datasets(
    register="pseudo_real_512",
    dataset_root="magformer_datasets/pseudo_real_512",
    train_ann="annotations/instances_target_labeled.json",
    val_ann="annotations/instances_target_unlabeled.json",
    train_image_dir="images/train",
    val_image_dir="images/train",
    train_split="target_labeled",
    val_split="target_unlabeled",
    normalized_ann_dir="output/diagnostics/official_mask2former_coco_registration_test",
)
records = DatasetCatalog.get(val_name)
metadata = MetadataCatalog.get(val_name)
coco = COCO(metadata.json_file)
print(train_name, val_name, len(records), isinstance(coco.dataset.get("info"), dict), isinstance(coco.dataset.get("licenses"), list), Path(records[0]["file_name"]).exists())
PY
```

Direct wrapper eval-only dry run:

```bash
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate mask2former
cd /home/hdd3/zhanghaonan/magformer
CUDA_VISIBLE_DEVICES=4 python baselines/run_official_mask2former_ecc.py \
  --register pseudo_real_512 \
  --dataset-root magformer_datasets/pseudo_real_512 \
  --train-ann annotations/instances_target_labeled.json \
  --val-ann annotations/instances_target_unlabeled.json \
  --train-image-dir images/train \
  --val-image-dir images/train \
  --train-split target_labeled \
  --val-split target_unlabeled \
  --normalized-ann-dir output/diagnostics/official_mask2former_coco_direct_cli \
  -- \
  --num-gpus 1 \
  --eval-only \
  --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
  OUTPUT_DIR /home/hdd3/zhanghaonan/magformer/output/diagnostics/official_mask2former_target_unlabeled_cli_eval_dry \
  DATASETS.TRAIN '("eccpseudo_real_512_target_labeled",)' \
  DATASETS.TEST '("eccpseudo_real_512_target_unlabeled",)' \
  INPUT.IMAGE_SIZE 1024 \
  MODEL.SEM_SEG_HEAD.NUM_CLASSES 1 \
  MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 200 \
  MODEL.WEIGHTS /home/hdd3/zhanghaonan/magformer/output/baseline/r92_official_mask2former_1k1024_smoke_mask2former_env/model_final.pth \
  TEST.DETECTIONS_PER_IMAGE 200 \
  DATALOADER.NUM_WORKERS 0
```

### Validation results

- Regression script passed. It verified explicit `target_unlabeled` registration, normalized metadata, source JSON unchanged, and missing annotation raises `FileNotFoundError`.
- Real registration loaded `eccpseudo_real_512_target_unlabeled` with 200 images. The normalized JSON had both `info` and `licenses`, and the first image path existed.
- Direct wrapper eval-only completed on `target_unlabeled`. It reached `Start inference on 200 batches`, saved `inference/coco_instances_results.json`, and completed COCO bbox and segm evaluation without `COCO.loadRes` `info/licenses` KeyError.
- bbox AP: `0.0000`; segm AP: `0.0000`. AP quality was not the goal of this dry run.
