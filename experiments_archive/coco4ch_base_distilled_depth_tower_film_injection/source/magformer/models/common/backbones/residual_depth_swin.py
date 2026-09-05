# -*- coding: utf-8 -*-
"""Swin-T with stage-entry residual depth injection."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .d2_swin import D2SwinBackbone


_STAGE_KEYS = ("res2", "res3", "res4", "res5")
_STAGE_STRIDES = (4, 8, 16, 32)


class _DepthDownBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class LightDepthPyramid(nn.Module):
    """Minimal depth pyramid used by the residual-depth fusion."""

    def __init__(self, channels: Sequence[int] = (24, 48, 96, 192)) -> None:
        super().__init__()
        channels = tuple(int(value) for value in channels)
        if channels != (24, 48, 96, 192):
            raise ValueError(
                "Residual-depth light depth channels are fixed to (24, 48, 96, 192), "
                f"got {channels}"
            )
        self.out_features = list(_STAGE_KEYS)
        self._stage_out_channels = dict(zip(_STAGE_KEYS, channels))
        self._stage_out_strides = dict(zip(_STAGE_KEYS, _STAGE_STRIDES))
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
        )
        self.down_blocks = nn.ModuleList(
            [
                _DepthDownBlock(channels[index], channels[index + 1])
                for index in range(len(channels) - 1)
            ]
        )

    def forward(self, depth: torch.Tensor) -> Dict[str, torch.Tensor]:
        features: Dict[str, torch.Tensor] = {}
        x = self.stem(depth)
        features["res2"] = x
        for index, block in enumerate(self.down_blocks, start=1):
            x = block(x)
            features[_STAGE_KEYS[index]] = x
        return features

    @property
    def output_shape(self) -> Dict[str, tuple[int, int, int]]:
        return {
            name: (
                self._stage_out_channels[name],
                self._stage_out_strides[name],
                self._stage_out_strides[name],
            )
            for name in self.out_features
        }


class ResidualDepthSwinBackbone(D2SwinBackbone):
    """Inject a depth residual at every Swin stage entrance.

    The four 1x1 projections are zero-initialized, so construction preserves
    the exact RGB-only Swin function.  There is deliberately no additional
    scalar gate or sample-dependent confidence path.
    """

    def __init__(
        self,
        *,
        depth_encoder: nn.Module,
        depth_channels: Sequence[int],
        **swin_kwargs,
    ) -> None:
        super().__init__(**swin_kwargs)
        depth_channels = tuple(int(value) for value in depth_channels)
        rgb_channels = tuple(self.model.num_features)
        if len(depth_channels) != len(_STAGE_KEYS):
            raise ValueError(
                f"Residual-depth requires four depth channel values, got {depth_channels}"
            )
        if rgb_channels != (96, 192, 384, 768):
            raise ValueError(
                "Residual-depth currently supports the canonical Swin-T channels "
                f"(96, 192, 384, 768), got {rgb_channels}"
            )

        encoder_channels = tuple(
            int(depth_encoder._stage_out_channels[key]) for key in _STAGE_KEYS
        )
        encoder_strides = tuple(
            int(depth_encoder._stage_out_strides[key]) for key in _STAGE_KEYS
        )
        if encoder_channels != depth_channels:
            raise ValueError(
                "Configured depth channels do not match encoder output: "
                f"configured={depth_channels}, actual={encoder_channels}"
            )
        if encoder_strides != _STAGE_STRIDES:
            raise ValueError(
                "Residual-depth depth encoder must output strides (4, 8, 16, 32), "
                f"got {encoder_strides}"
            )

        self.depth_encoder = depth_encoder
        self.depth_channels = depth_channels
        self.depth_projections = nn.ModuleDict(
            {
                key: nn.Conv2d(depth_ch, rgb_ch, kernel_size=1, bias=True)
                for key, depth_ch, rgb_ch in zip(
                    _STAGE_KEYS, depth_channels, rgb_channels
                )
            }
        )
        for projection in self.depth_projections.values():
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    @staticmethod
    def _pad_to_stride(
        rgb: torch.Tensor,
        depth: torch.Tensor,
        depth_valid_mask: torch.Tensor,
        stride: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = rgb.shape[-2:]
        pad_h = (-height) % stride
        pad_w = (-width) % stride
        if pad_h == 0 and pad_w == 0:
            return rgb, depth, depth_valid_mask
        padding = (0, pad_w, 0, pad_h)
        return (
            F.pad(rgb, padding, value=0.0),
            F.pad(depth, padding, value=0.0),
            F.pad(depth_valid_mask, padding, value=False),
        )

    @staticmethod
    def _validate_inputs(
        rgb: torch.Tensor,
        depth: torch.Tensor,
        depth_valid_mask: torch.Tensor,
    ) -> None:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"RGB input must have shape (B,3,H,W), got {rgb.shape}")
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError(
                f"Depth input must have shape (B,1,H,W), got {depth.shape}"
            )
        if depth_valid_mask.ndim != 4 or depth_valid_mask.shape[1] != 1:
            raise ValueError(
                "depth_valid_mask must have shape (B,1,H,W), "
                f"got {depth_valid_mask.shape}"
            )
        if rgb.shape[0] != depth.shape[0] or rgb.shape[-2:] != depth.shape[-2:]:
            raise ValueError(
                f"RGB/depth shape mismatch: rgb={rgb.shape}, depth={depth.shape}"
            )
        if depth_valid_mask.shape != depth.shape:
            raise ValueError(
                "depth_valid_mask must exactly match depth shape: "
                f"mask={depth_valid_mask.shape}, depth={depth.shape}"
            )
        if depth_valid_mask.dtype is not torch.bool:
            raise TypeError(
                f"depth_valid_mask must be bool, got {depth_valid_mask.dtype}"
            )

    def _inject_stage(
        self,
        key: str,
        x: torch.Tensor,
        depth_feature: torch.Tensor,
        depth_valid_mask: torch.Tensor,
        expected_hw: tuple[int, int],
    ) -> torch.Tensor:
        stride = self._stage_out_strides[key]
        if depth_feature.shape[-2:] != expected_hw:
            raise RuntimeError(
                f"Residual-depth depth feature {key} has spatial shape "
                f"{depth_feature.shape[-2:]}, expected {expected_hw}"
            )
        with torch.autocast(
            device_type=depth_feature.device.type,
            enabled=False,
        ):
            normalized = F.group_norm(
                depth_feature.float(),
                num_groups=1,
                weight=None,
                bias=None,
                eps=1.0e-5,
            )
            residual = self.depth_projections[key](normalized)
            valid = F.max_pool2d(
                depth_valid_mask.float(),
                kernel_size=stride,
                stride=stride,
            )
            if valid.shape[-2:] != expected_hw:
                raise RuntimeError(
                    f"Residual-depth valid mask {key} has spatial shape {valid.shape[-2:]}, "
                    f"expected {expected_hw}"
                )
            residual = residual * valid
            residual_tokens = residual.flatten(2).transpose(1, 2)
            if residual_tokens.shape != x.shape:
                raise RuntimeError(
                    f"Residual-depth token shape mismatch at {key}: "
                    f"rgb={x.shape}, depth={residual_tokens.shape}"
                )
            return x.float() + residual_tokens

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        depth_valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self._validate_inputs(rgb, depth, depth_valid_mask)
        rgb, depth, depth_valid_mask = self._pad_to_stride(
            rgb, depth, depth_valid_mask
        )
        depth = torch.where(depth_valid_mask, depth, torch.zeros_like(depth))
        depth_features = self.depth_encoder(depth)
        missing = [key for key in _STAGE_KEYS if key not in depth_features]
        if missing:
            raise KeyError(f"Residual-depth depth encoder is missing features: {missing}")

        model = self.model
        x = model.patch_embed(rgb)
        height, width = x.shape[-2:]
        if model.ape:
            absolute_pos_embed = F.interpolate(
                model.absolute_pos_embed,
                size=(height, width),
                mode="bicubic",
            )
            x = x + absolute_pos_embed
        x = x.flatten(2).transpose(1, 2)
        x = model.pos_drop(x)

        outputs: Dict[str, torch.Tensor] = {}
        for index, layer in enumerate(model.layers):
            key = _STAGE_KEYS[index]
            x = self._inject_stage(
                key,
                x,
                depth_features[key],
                depth_valid_mask,
                (height, width),
            )
            x_out, out_h, out_w, x, height, width = layer(x, height, width)
            if index in model.out_indices:
                norm_layer = getattr(model, f"norm{index}")
                x_out = norm_layer(x_out)
                outputs[key] = (
                    x_out.view(-1, out_h, out_w, model.num_features[index])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
        return outputs

    def residual_depth_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for module in (self.depth_encoder, self.depth_projections)
            for parameter in module.parameters()
        )
