from types import SimpleNamespace
import torch
import torch.nn as nn

from tools.train import build_optimizer


class DummyTrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_backbone = nn.Sequential(nn.Conv2d(3, 8, 1), nn.GroupNorm(4, 8))
        self.depth_backbone = nn.Sequential(nn.Conv2d(1, 8, 1), nn.GroupNorm(4, 8))
        self.fusion = nn.Sequential(nn.Conv2d(8, 8, 1), nn.GroupNorm(4, 8))
        self.decoder = nn.Linear(8, 8)


def _cfg():
    solver = SimpleNamespace(
        base_lr=1e-4,
        backbone_multiplier=0.1,
        rgb_backbone_multiplier=0.1,
        depth_backbone_multiplier=0.2,
        mgm_multiplier=2.0,
        weight_decay=0.05,
        weight_decay_norm=0.0,
        weight_decay_embed=0.0,
        optimizer="ADAMW",
    )
    return SimpleNamespace(solver=solver)


def _param_lr(optimizer):
    mapping = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            mapping[id(p)] = group["lr"]
    return mapping


def test_optimizer_lr_groups_match_mgm_reference():
    model = DummyTrainModel()
    optimizer = build_optimizer(model, _cfg())
    p2lr = _param_lr(optimizer)

    rgb_lr = p2lr[id(model.rgb_backbone[0].weight)]
    depth_lr = p2lr[id(model.depth_backbone[0].weight)]
    mgm_lr = p2lr[id(model.fusion[0].weight)]
    other_lr = p2lr[id(model.decoder.weight)]

    assert rgb_lr == 1e-5
    assert depth_lr == 2e-5
    assert mgm_lr == 2e-4
    assert other_lr == 1e-4
