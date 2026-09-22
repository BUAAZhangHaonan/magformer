#!/usr/bin/env python3
import os, sys, torch
from torch.utils.data import DataLoader
from pathlib import Path

os.chdir('/home/g203-4028/magformer')
sys.path.insert(0, '/home/g203-4028/magformer')

from magformer.config import load_config, set_seed
from magformer.data import CocoRgbdDataset
from magformer.data.transforms import RGBDTransform
from magformer.data.collate import collate_fn
from magformer.engine.eval_runtime import run_inference_evaluation
from magformer.models import build_model
from magformer.engine.utils import load_checkpoint

CKPT = '/home/g203-4028/magformer/output/experiments/20260513_1k_finetune_full_1024_v14/checkpoint_iter_0011999.pth'
OUTDIR = '/home/g203-4028/magformer/output/experiments/20260513_1k_finetune_full_1024_v14/eval_iter11999'

config = load_config('/home/g203-4028/magformer/configs/finetune_1k_full_1024_v14.yaml', overrides={
    'model': {'weights': CKPT},
    'runtime': {'output_dir': OUTDIR, 'gpus': [0]}
})
device = torch.device('cuda:0')
set_seed(42)

data_cfg = config.data
dataset = CocoRgbdDataset(
    dataset_root=data_cfg.dataset_root, ann_file=data_cfg.val_ann,
    split=getattr(data_cfg, 'val_split', 'all'), transform=None, is_train=False)
dataset.transform = RGBDTransform(
    image_size=data_cfg.image_size, min_scale=data_cfg.min_scale, max_scale=data_cfg.max_scale,
    random_flip='none', rgb_brightness=0.0, rgb_contrast=0.0, rgb_saturation=0.0, rgb_hue=0.0,
    depth_scale=data_cfg.depth.scale, depth_shift=data_cfg.depth.shift,
    depth_clip_min=data_cfg.depth.clip_min, depth_clip_max=data_cfg.depth.clip_max,
    depth_norm=data_cfg.depth.norm,
    depth_per_sample_norm=getattr(data_cfg.depth, 'per_sample_norm', True), is_train=False)

loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
model = build_model(config)
load_checkpoint(CKPT, model, strict=True)
model = model.to(device)
model.eval()

result = run_inference_evaluation(model, loader, coco_gt=dataset.coco, device=device,
    output_dir=Path(OUTDIR), amp_enabled=False,
    category_ids=list(getattr(dataset, 'category_ids', [])) or None, fail_on_empty=True)
print(result.coco_metrics)
