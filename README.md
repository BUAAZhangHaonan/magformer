# MAGFormer - Multi-modal Adaptive Gated Transformer

MAGFormer is a pure PyTorch RGB-D instance segmentation model designed for dense clutter scenes. The shipped repo currently targets single-class segmentation in COCO-format RGB-D datasets. It uses a multi-modal gated fusion module to combine RGB and depth features and a transformer decoder for mask prediction.

## Key Features

- Dual backbones: Swin (RGB) + ConvNeXt (Depth)
- Multi-modal gated fusion with depth priors
- Single-class COCO RGB-D dataset support (ECCD-compatible)
- Unified training, evaluation, and visualization

## Current Canonical Results

The current live publication scope in this repo is the `1024/512` suite.

- Canonical manifest: `output/analysis/2026-04-12-live-metrics-manifest.json`
- Canonical report: `output/analysis/2026-04-12-all-models-metrics-1024-512-sorted-by-segm-ap.md`

Older experiment summaries under `docs/experiments/` are kept as historical snapshots and should not be treated as the current canonical source.

## Project Structure

```
magformer/
├── magformer/              # Core package
│   ├── config/            # YAML + Pydantic configuration
│   ├── data/              # COCO RGB-D dataset + transforms
│   ├── engine/            # Trainer + evaluator
│   ├── models/            # MAGFormer model and common modules
│   └── visualization/     # Mask + box visualization
├── tools/                  # Train/eval/inference entrypoints
├── configs/                # YAML configs
├── requirements.txt
└── README.md
```

## Installation

```bash
conda create -n magformer python=3.10 -y
conda activate magformer
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .
```

## Dataset Format

```
dataset_root/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── depth/
│   ├── depth_npy/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── depth_noise_mask/
└── annotations/
    ├── instances_train.json
    ├── instances_val.json
    └── instances_test.json
```

## Quick Start

### Training

```bash
python tools/train.py \
    --config-file configs/magformer.yaml \
    --dataset-root /path/to/eccd
```

If you want to warm-start from an earlier MAGFormer checkpoint, pass it explicitly:

```bash
python tools/train.py \
    --config-file configs/magformer.yaml \
    --dataset-root /path/to/eccd \
    --finetune-weights /path/to/model_best.pth
```

### Evaluation

```bash
python tools/evaluate.py \
    --config-file configs/magformer.yaml \
    --dataset-root /path/to/eccd \
    --weights output/checkpoint_iter_0005000.pth
```

### Inference

```bash
python tools/inference.py \
    --config-file configs/magformer.yaml \
    --weights output/checkpoint_iter_0005000.pth \
    --image /path/to/image.png \
    --depth /path/to/depth.npy \
    --output output/vis.png
```

## License

Apache License 2.0
