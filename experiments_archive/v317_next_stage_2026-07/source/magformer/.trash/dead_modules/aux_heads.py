# -*- coding: utf-8 -*-
"""
Co-DETR style auxiliary detection heads for MagFormer.

Lightweight FCOS-style heads that consume pixel decoder multi-scale features.
Training-only -- discarded at inference for zero cost.

The pixel decoder (MSDeformAttnPixelDecoder) produces 3 FPN levels
corresponding to strides [8, 16, 32]. Each level gets a shared FCOS head
that predicts per-pixel classification logits, LTRB box offsets, and
centerness. These are supervised with FCOS-style focal + GIoU + centerness
losses to provide denser gradient signal to the pixel decoder encoder.
"""

import torch
import torch.nn as nn
from typing import Dict, List


class FCOSAuxHead(nn.Module):
    """Single-scale FCOS head for auxiliary detection supervision.

    Consists of separate classification and regression towers, each with
    num_convs 3x3 conv + GroupNorm + ReLU layers, followed by task-
    specific prediction layers.
    """

    def __init__(self, in_channels: int, num_classes: int, num_convs: int = 2):
        super().__init__()
        cls_tower: list[nn.Module] = []
        reg_tower: list[nn.Module] = []
        for _ in range(num_convs):
            cls_tower.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            cls_tower.append(nn.GroupNorm(8, in_channels))
            cls_tower.append(nn.ReLU(inplace=True))
            reg_tower.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            reg_tower.append(nn.GroupNorm(8, in_channels))
            reg_tower.append(nn.ReLU(inplace=True))
        self.cls_tower = nn.Sequential(*cls_tower)
        self.reg_tower = nn.Sequential(*reg_tower)
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.reg_pred = nn.Conv2d(in_channels, 4, 3, padding=1)  # l,t,r,b
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)
        self._init_weights()

    def _init_weights(self):
        for modules in [self.cls_tower, self.reg_tower]:
            for m in modules:
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.zeros_(m.bias)
        # Prior-prob init for focal-loss style training
        prior_prob = 0.01
        import math
        bias_val = -math.log((1.0 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_val)
        nn.init.normal_(self.cls_pred.weight, std=0.01)
        for m in [self.reg_pred, self.centerness]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        cls_feat = self.cls_tower(features)
        reg_feat = self.reg_tower(features)
        return {
            "cls_logits": self.cls_pred(cls_feat),
            "bbox_pred": self.reg_pred(reg_feat),
            "centerness": self.centerness(reg_feat),
        }


class CoDETRAuxModule(nn.Module):
    """Multi-scale Co-DETR auxiliary module.

    Adds a shared FCOS-style head on each FPN level from the pixel decoder.
    All predictions are supervised with FCOS-style losses during training.
    At inference, this module is skipped entirely (zero inference cost).
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 1,
        num_convs: int = 2,
    ):
        super().__init__()
        # Shared head across all FPN levels
        self.head = FCOSAuxHead(in_channels, num_classes, num_convs)
        self.num_classes = num_classes

    def forward(
        self, multi_scale_features: List[torch.Tensor],
    ) -> Dict[str, List[torch.Tensor]]:
        """Process each FPN level through the shared head."""
        all_cls: list[torch.Tensor] = []
        all_bbox: list[torch.Tensor] = []
        all_center: list[torch.Tensor] = []
        for feat in multi_scale_features:
            out = self.head(feat)
            all_cls.append(out["cls_logits"])
            all_bbox.append(out["bbox_pred"])
            all_center.append(out["centerness"])
        return {
            "aux_cls_logits": all_cls,
            "aux_bbox_pred": all_bbox,
            "aux_centerness": all_center,
        }
