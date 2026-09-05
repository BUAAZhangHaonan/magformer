from __future__ import annotations

import pytest
import torch
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config


class _ToyMSMFormer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Conv2d(3, 4, kernel_size=1)
        self.head = torch.nn.Conv2d(4, 2, kernel_size=1)


def test_msmformer_uses_vendor_style_adamw_optimizer() -> None:
    import sys
    from pathlib import Path

    from baselines.run_msmformer_ecc import build_msmformer_optimizer

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "baselines" / "msmformer" / "MSMFormer"))
    from meanshiftformer.config import add_meanshiftformer_config

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_meanshiftformer_config(cfg)
    cfg.SOLVER.BASE_LR = 1.0e-4
    cfg.SOLVER.WEIGHT_DECAY = 5.0e-2
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.OPTIMIZER = "ADAMW"

    model = _ToyMSMFormer()
    optimizer = build_msmformer_optimizer(cfg, model)

    assert isinstance(optimizer, torch.optim.AdamW)

    lr_by_param_id = {
        id(param): group["lr"]
        for group in optimizer.param_groups
        for param in group["params"]
    }
    assert lr_by_param_id[id(model.backbone.weight)] == pytest.approx(1.0e-5)
    assert lr_by_param_id[id(model.backbone.bias)] == pytest.approx(1.0e-5)
    assert lr_by_param_id[id(model.head.weight)] == pytest.approx(1.0e-4)
    assert lr_by_param_id[id(model.head.bias)] == pytest.approx(1.0e-4)
