# MAGFormer - Multi-modal Adaptive Gated MaskFormer

MAGFormer is an RGB-D instance segmentation model designed for dense clutter scenes. It features a novel Multi-modal Gated Fusion (MGM) module that adaptively combines RGB and depth features based on scene content.

## Key Features

- **Dual Backbone Architecture**: Swin Transformer for RGB, ConvNeXt for depth
- **Multi-modal Gated Fusion**: Learnable confidence-based feature fusion
- **Depth Prior Extraction**: Gradient, variance, and edge-aware features
- **Mask2Former Framework**: Transformer-based mask prediction
- **COCO RGB-D Format**: Standard dataset format support

## Project Structure

```
magformer/
├── mask2former/           # MAGFormer model implementation
│   ├── modeling/
│   │   ├── backbone/      # RGB & Depth backbones
│   │   ├── meta_arch/    # MGMMaskFormer model
│   │   ├── mgm/          # MGM fusion module
│   │   └── head/         # Segmentation heads
│   ├── data/
│   │   ├── datasets/      # Dataset registration
│   │   └── dataset_mappers/  # Data preprocessing
│   └── evaluation/       # Evaluation metrics
├── baselines/            # Baseline methods
│   ├── uoais/          # UOAIS baseline
│   ├── msmformer/       # MSMFormer baseline
│   └── ucn/            # UCN baseline
├── configs/             # Configuration files
├── tools/              # Training/inference scripts
├── requirements.txt
└── setup.py
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+
- PyTorch 1.12+

### Install with pip

```bash
git clone https://github.com/your-org/magformer.git
cd magformer
pip install -e .
```

### Install with conda

```bash
conda create -n magformer python=3.10
conda activate magformer
pip install torch torchvision
pip install -e .
```

## Quick Start

### Training

```bash
python tools/train.py \
    --config configs/mgm_swin_convnext.yaml \
    --num-gpus 4
```

### Evaluation

```bash
python tools/evaluate.py \
    --config configs/mgm_swin_convnext.yaml \
    --model-path output/model_final.pth
```

### Demo

```bash
python tools/demo.py \
    --config configs/mgm_swin_convnext.yaml \
    --model-path output/model_final.pth \
    --input images/
    --depth-dir depth/
```

## Dataset

MAGFormer expects datasets in COCO RGB-D instance segmentation format:

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

### Register Custom Dataset

```python
from detectron2.data import MetadataCatalog, DatasetCatalog
from mask2former.data.datasets.register_coco_rgbd_instance import register_coco_instances_rgbd

register_coco_instances_rgbd(
    "my_dataset_train",
    {},
    "path/to/annotations/instances_train.json",
    "path/to/images/train",
    "path/to/depth/depth_npy/train"
)
```

## Model Architecture

### Multi-modal Gated Fusion (MGM)

The core innovation of MAGFormer is the MGM module:

1. **Depth Prior Extractor**: Computes depth priors
   - Gradient magnitude
   - Depth variance
   - Valid/hole masks
   - RGB-edge consistency

2. **Confidence Predictor**: Predicts per-pixel fusion confidence
   - Uses both RGB and depth features
   - Incorporates depth priors
   - Temperature annealing during training

3. **Gated Fusion**: Combines features
   ```
   fused = m * depth + (1 - m) * rgb + residual
   ```

### Configuration

Key configuration parameters in `configs/mgm_swin_convnext.yaml`:

```yaml
MODEL:
  MASK_FORMER:
    NUM_OBJECT_QUERIES: 200
    DEC_LAYERS: 10

  MGM:
    TEMP_INIT: 2.0
    TEMP_FINAL: 1.0
    RESIDUAL_ALPHA: 0.05
    LOSS_ENTROPY_W: 0.05

INPUT:
  IMAGE_SIZE: 512
  DEPTH_SCALE: 0.001
  DEPTH_NORM: "minmax"
```

## Baseline Comparison

MAGFormer includes three state-of-the-art baseline methods:

### UOAIS (ICRA 2022)
Hierarchical occlusion modeling for amodal segmentation

```bash
python baselines/uoais/train.py --config ...
```

### MSMFormer (ICRA 2024)
Mean Shift Mask Transformer with hypersphere attention

```bash
python baselines/msmformer/train.py --config ...
```

### UCN (CoRL 2020)
Learning RGB-D feature embeddings with mean shift clustering

```bash
python baselines/ucn/train.py --config ...
```

## Results

| Method | Dataset | mAP | Boundary F | FPS |
|--------|----------|------|------------|------|
| UCN | ECCD | 45.2 | 72.1 | 25 |
| UOAIS | ECCD | 52.3 | 78.4 | 15 |
| MSMFormer | ECCD | 58.7 | 82.1 | 12 |
| **MAGFormer** | **ECCD** | **64.5** | **87.3** | **18** |

## Citation

If you use MAGFormer in your research, please cite:

```bibtex
@inproceedings{magformer2026,
  title={MAGFormer: Instance Segmentation in Dense Clutter via Modality Arbitration and Decoupled Realistic Synthesis},
  author={...},
  booktitle={ICRA},
  year={2026}
}
```

## License

Apache License 2.0

## Acknowledgments

- [Detectron2](https://github.com/facebookresearch/detectron2)
- [Mask2Former](https://github.com/facebookresearch/Mask2Former)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)

## Contact

For issues and questions, please open a GitHub issue.
