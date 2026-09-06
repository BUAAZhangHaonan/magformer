from __future__ import annotations

import torch

from magformer.models.magformer.arch import MagFormerArch


class _RgbBackbone(torch.nn.Module):
    def forward(self, images: torch.Tensor):
        b = images.shape[0]
        # Minimal multi-scale feature dict
        return {
            "res2": torch.zeros((b, 8, 8, 8), dtype=images.dtype, device=images.device),
            "res3": torch.zeros((b, 8, 4, 4), dtype=images.dtype, device=images.device),
            "res4": torch.zeros((b, 8, 2, 2), dtype=images.dtype, device=images.device),
            "res5": torch.zeros((b, 8, 1, 1), dtype=images.dtype, device=images.device),
        }


class _DepthBackboneRaises(torch.nn.Module):
    def forward(self, depths: torch.Tensor):
        raise AssertionError("depth_backbone should not be called when disabled")


class _FusionRaises(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("fusion module should not be called when disabled")


class _PixelDecoder(torch.nn.Module):
    def forward(
        self,
        features,
        confidence_maps=None,
        depth_modulation_maps=None,
        depth_raw=None,
        padding_mask=None,
    ):
        # Pass-through minimal contract used by MagFormerArch
        mem = features["res2"]
        padding = torch.zeros(
            (mem.shape[0], mem.shape[2], mem.shape[3]),
            dtype=torch.bool,
            device=mem.device,
        )
        return {
            "memory": mem,
            "mask_features": mem,
            "multi_scale_features": [mem],
            "multi_scale_pos": [torch.zeros_like(mem)],
            "multi_scale_padding_masks": [padding],
        }


class _TransformerDecoder(torch.nn.Module):
    def forward(
        self,
        memory,
        mask_features,
        multi_scale_features=None,
        multi_scale_pos=None,
        multi_scale_padding_masks=None,
        pos_key=None,
        depth_raw=None,
    ):
        b = memory.shape[0]
        # num_classes=1 => logits last dim should be 2 (incl. no-object)
        return {
            "pred_logits": torch.zeros((b, 2, 2), dtype=memory.dtype, device=memory.device),
            "pred_masks": torch.zeros((b, 2, 8, 8), dtype=memory.dtype, device=memory.device),
        }


def test_modality_fusion_enabled_false_skips_depth_and_fusion() -> None:
    arch = MagFormerArch(
        rgb_backbone=_RgbBackbone(),
        depth_backbone=_DepthBackboneRaises(),
        fusion_module=_FusionRaises(),
        pixel_decoder=_PixelDecoder(),
        transformer_decoder=_TransformerDecoder(),
        num_classes=1,
        num_queries=2,
        hidden_dim=8,
        pixel_mean=[0.0, 0.0, 0.0],
        pixel_std=[1.0, 1.0, 1.0],
    )
    arch.modality_fusion_enabled = False
    arch.depth_backbone_enabled = True
    arch.eval()

    images = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    depths = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    _ = arch(images, depths)


def test_depth_backbone_enabled_false_implies_no_fusion() -> None:
    arch = MagFormerArch(
        rgb_backbone=_RgbBackbone(),
        depth_backbone=_DepthBackboneRaises(),
        fusion_module=_FusionRaises(),
        pixel_decoder=_PixelDecoder(),
        transformer_decoder=_TransformerDecoder(),
        num_classes=1,
        num_queries=2,
        hidden_dim=8,
        pixel_mean=[0.0, 0.0, 0.0],
        pixel_std=[1.0, 1.0, 1.0],
    )
    arch.modality_fusion_enabled = True
    arch.depth_backbone_enabled = False
    arch.eval()

    images = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    depths = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    _ = arch(images, depths)
