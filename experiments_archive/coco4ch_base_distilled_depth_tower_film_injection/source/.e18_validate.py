from pathlib import Path
import sys

import torch

ROOT = Path('/home/hdd3/zhanghaonan/magformer')
SOURCE = ROOT / 'source/E18-retrain_rdi_lite_and_mbv3l_from_pretrained-20260808'
EXP = ROOT / 'scripts/E18-retrain_rdi_lite_and_mbv3l_from_pretrained-20260808'
sys.path.insert(0, str(SOURCE))

from magformer.config import load_config
from tools.train import build_datasets, build_data_loaders, build_model, load_finetune_weights

for name in ('rdi_lite', 'rdi_mbv3l'):
    cfg = load_config(str(EXP / f'{name}.yaml'))
    assert cfg.runtime.eval_period == 5000
    assert cfg.runtime.checkpoint_period == 5000
    assert cfg.runtime.resume is None
    assert Path(cfg.data.dataset_root, cfg.data.train_ann).is_file()
    assert Path(cfg.data.dataset_root, cfg.data.val_ann).is_file()
    assert Path(cfg.model.finetune_weights).is_file()
    if name == 'rdi_mbv3l':
        assert Path(cfg.model.magformer.depth_backbone.weights).is_file()

    cfg.runtime.num_workers = 0
    train_ds, val_ds = build_datasets(cfg)
    train_loader, val_loader = build_data_loaders(
        cfg, train_ds, val_ds, batch_size=1, num_workers=0, is_distributed=False
    )
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    assert train_batch['images'].shape[0] == 1
    assert val_batch['images'].shape[0] == 1

    model = build_model(cfg, torch.device('cpu'))
    result = load_finetune_weights(model, cfg.model.finetune_weights, strict=False)
    print(f'{name}: train_batch={tuple(train_batch["images"].shape)} val_batch={tuple(val_batch["images"].shape)} missing={len(result["missing_keys"])} unexpected={len(result["unexpected_keys"])}')

print('E18_CONFIG_DATA_MODEL_SMOKE_OK')
