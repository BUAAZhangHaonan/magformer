# R96 official Mask2Former 32K SQLite cache smoke

Date: 2026-05-17

## Goal

R96 adds train-side SQLite cache support to the official RGB-only Mask2Former wrapper. The goal is to avoid loading the 9.9G `annotations/instances_train.json` for the 32K source train split.

This change is registration-only. It does not add semi-supervision, pseudo labels, VC-SUDA logic, or model changes.

## Design

- `baselines/run_official_mask2former_ecc.py` now accepts `--train-cache`.
- When `--train-cache` is set, the train split is registered with `DatasetCatalog.register`.
- The callable reads the SQLite cache in read-only mode and returns Detectron2 standard dicts:
  - record fields: `file_name`, `height`, `width`, `image_id`, `annotations`
  - annotation fields: `bbox`, `bbox_mode=BoxMode.XYWH_ABS`, `segmentation`, `category_id=0`, `iscrowd`
- Metadata is set with:
  - `thing_classes=["component"]`
  - `thing_dataset_id_to_contiguous_id={1: 0}`
  - `evaluator_type="coco"`
  - `image_root=<train image root>`
- The SQLite path is not stored as `json_file`.
- JSON train/val registration keeps the existing path and normalization behavior.
- Missing cache or image root raises immediately. There is no fallback to `instances_train.json`.

## Validation

Compile:

```bash
python -m py_compile baselines/run_official_mask2former_ecc.py tests/test_official_mask2former_wrapper_registration.py
```

Result: pass.

Unit test:

```bash
python tests/test_official_mask2former_wrapper_registration.py
```

Result: pass.

The test adds a SQLite fixture and checks:

- returned record fields
- `category_id` maps from COCO id `1` to contiguous id `0`
- `bbox_mode` is `BoxMode.XYWH_ABS`
- polygon segmentation is preserved
- train metadata has no `json_file`
- missing SQLite cache raises
- missing train image root raises

Real 32K cache read-only registration check:

```bash
python - <<'PY'
from pathlib import Path
from detectron2.data import DatasetCatalog, MetadataCatalog
from baselines import run_official_mask2former_ecc as wrapper

root = Path("magformer_datasets/20260318_1K_32254").resolve()
train_name, val_name = wrapper.register_official_mask2former_datasets(
    register="20260318_1K_32254_r96_cache_check",
    dataset_root=str(root),
    train_cache="cache/coco_loader/instances_train.sqlite",
    val_ann="annotations/instances_val.json",
    train_image_dir="images/train",
    val_image_dir="images/val",
    train_split="train",
    val_split="val",
    normalized_ann_dir="output/diagnostics/official_mask2former_coco",
)
metadata = MetadataCatalog.get(train_name)
records = DatasetCatalog.get(train_name)
print(len(records))
print([Path(r["file_name"]).exists() for r in records[:3]])
print(hasattr(metadata, "json_file"))
print(metadata.thing_dataset_id_to_contiguous_id)
PY
```

Result:

- train dataset: `ecc20260318_1k_32254_train`
- val dataset: `ecc20260318_1k_32254_val`
- records: `25654`
- first three image paths exist: `True, True, True`
- train metadata has no `json_file`: `False`
- mapping: `{1: 0}`
- no `instances_train.json` was passed to train registration.

## Smoke

TMUX session: `r96_official_m2f_32k_cache_smoke`

Environment:

- conda env: `mask2former`
- GPUs: `CUDA_VISIBLE_DEVICES=4,5,6,7`
- config: `configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml`
- train cache: `magformer_datasets/20260318_1K_32254/cache/coco_loader/instances_train.sqlite`
- dataset root: `magformer_datasets/20260318_1K_32254`
- output: `output/baseline/r96_official_m2f_32k_cache_smoke`
- log: `output/baseline/r96_official_m2f_32k_cache_smoke.tmux.log`

Key options:

```text
MODEL.WEIGHTS detectron2://ImageNetPretrained/torchvision/R-50.pkl
MODEL.SEM_SEG_HEAD.NUM_CLASSES 1
MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100
INPUT.IMAGE_SIZE 1024
TEST.DETECTIONS_PER_IMAGE 100
SOLVER.IMS_PER_BATCH 4
SOLVER.MAX_ITER 10
TEST.EVAL_PERIOD 0
SOLVER.CHECKPOINT_PERIOD 10
DATALOADER.NUM_WORKERS 2
```

Smoke result:

- Train cache registration printed `registered train dataset: ecc20260318_1k_32254_train`.
- Detectron2 serialized `25654` train elements.
- Training reached `iter: 9`, so 10 iterations completed.
- Final smoke loss:
  - `total_loss: 81.63`
  - `loss_ce: 1.601`
  - `loss_mask: 1.242`
  - `loss_dice: 4.926`
- Training time after loader setup: `0:00:13`.
- Peak CUDA memory in the training log: `5495M`.
- Checkpoints were written under `output/baseline/r96_official_m2f_32k_cache_smoke/` and are not part of the commit.
- The official trainer also ran its default final val eval on `instances_val.json`. This produced bbox and segm AP `0.0000`, which is expected for a 10-iter ImageNet-start smoke and is not used as a quality gate.
- Error scan found no `Traceback`, `RuntimeError`, CUDA OOM, `Killed`, `MemoryError`, or `Exception`.

The smoke command line contains `--train-cache cache/coco_loader/instances_train.sqlite`. Log grep found no `instances_train.json` load line. The only large train dataset evidence is `Serializing 25654 elements`, from the SQLite-backed DatasetCatalog callable.

## R97 decision

Pass for R96. The wrapper can enter R97 32K short training or long training from a registration and smoke standpoint.

Residual note: the official trainer runs final eval even with `TEST.EVAL_PERIOD 0` when `DATASETS.TEST` is set. For future train-only smoke, use an empty test dataset or a wrapper option that skips final eval if wall time matters.
