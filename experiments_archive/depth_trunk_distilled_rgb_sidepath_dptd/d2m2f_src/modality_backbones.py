"""Modality-decomposition backbones for the paper table.

Both take the standard 4ch RGBD mapper output (already z-scored per channel by
the meta-arch) and expose a single modality to the (3ch, COCO-init) Swin:

- D2SwinTransformerRGBOnly:  slice rgb channels, drop depth
- D2SwinTransformerDepthOnly: replicate the z-scored depth channel 3x
"""
import torch

from detectron2.modeling import BACKBONE_REGISTRY

from mask2former.modeling.backbone.swin import D2SwinTransformer


@BACKBONE_REGISTRY.register()
class D2SwinTransformerRGBOnly(D2SwinTransformer):
    def forward(self, x):
        assert x.dim() == 4 and x.shape[1] == 4, f"RGBOnly expects (N,4,H,W), got {x.shape}"
        return super().forward(x[:, :3].contiguous())


@BACKBONE_REGISTRY.register()
class D2SwinTransformerDepthOnly(D2SwinTransformer):
    def forward(self, x):
        assert x.dim() == 4 and x.shape[1] == 4, f"DepthOnly expects (N,4,H,W), got {x.shape}"
        return super().forward(x[:, 3:4].repeat(1, 3, 1, 1).contiguous())
