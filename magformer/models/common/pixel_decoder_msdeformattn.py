# -*- coding: utf-8 -*-
"""MSDeformAttn pixel decoder for MAGFormer."""

from typing import Dict, List, Optional
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .layers import PositionEmbeddingSine, DepthPosEncoding, _get_clones, _get_activation_fn
from ..ops import MSDeformAttn


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forward_ffn(src)
        return src


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device),
                torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * h)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * w)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        num_feature_levels=4,
        enc_n_points=4,
    ):
        super().__init__()
        encoder_layer = MSDeformAttnTransformerEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            enc_n_points,
        )
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        nn.init.normal_(self.level_embed)

    @staticmethod
    def get_valid_ratio(mask):
        _, h, w = mask.shape
        valid_h = torch.sum(~mask[:, :, 0], 1)
        valid_w = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_h.float() / h
        valid_ratio_w = valid_w.float() / w
        return torch.stack([valid_ratio_w, valid_ratio_h], -1)

    def forward(self, srcs, pos_embeds):
        masks = [
            torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
            for x in srcs
        ]
        src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes = [], [], [], []

        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            src_flatten.append(src)
            mask_flatten.append(mask)
            lvl_pos_embed_flatten.append(lvl_pos_embed)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        memory = self.encoder(
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            lvl_pos_embed_flatten,
            mask_flatten,
        )
        return memory, spatial_shapes, level_start_index


class MSDeformAttnPixelDecoder(nn.Module):
    def __init__(
        self,
        in_features: List[str],
        in_channels: Dict[str, int],
        hidden_dim: int = 256,
        mask_dim: int = 256,
        transformer_dropout: float = 0.1,
        transformer_nheads: int = 8,
        transformer_dim_feedforward: int = 1024,
        transformer_enc_layers: int = 6,
        common_stride: int = 4,
        dpe_enabled: bool = False,
    ):
        super().__init__()
        self.in_features = list(in_features)
        self.feature_channels = [in_channels[k] for k in self.in_features]
        self.feature_strides = [2 ** (i + 2) for i in range(len(self.in_features))]

        self.transformer_in_features = self.in_features
        self.transformer_num_feature_levels = len(self.transformer_in_features)

        self.input_proj = nn.ModuleList()
        for in_ch in [in_channels[k] for k in self.transformer_in_features[::-1]]:
            groups = min(32, hidden_dim)
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, hidden_dim, kernel_size=1),
                    nn.GroupNorm(groups, hidden_dim),
                )
            )

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=hidden_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=max(transformer_enc_layers, 1),
            num_feature_levels=self.transformer_num_feature_levels,
        )

        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.dpe_enabled = dpe_enabled
        self.depth_pe = DepthPosEncoding(hidden_dim=hidden_dim) if dpe_enabled else None

        self.mask_features = nn.Conv2d(hidden_dim, mask_dim, kernel_size=1)
        nn.init.xavier_uniform_(self.mask_features.weight)
        nn.init.constant_(self.mask_features.bias, 0)

        self.maskformer_num_feature_levels = 3
        self.common_stride = common_stride

        stride = min(self.feature_strides)
        self.num_fpn_levels = max(int(np.log2(stride) - np.log2(self.common_stride)), 0)
        lateral_convs = []
        output_convs = []
        for idx, in_ch in enumerate(self.feature_channels[: self.num_fpn_levels]):
            groups = min(32, hidden_dim)
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False),
                nn.GroupNorm(groups, hidden_dim),
            )
            output_conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(groups, hidden_dim),
                nn.ReLU(inplace=True),
            )
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        self.lateral_convs = nn.ModuleList(lateral_convs[::-1])
        self.output_convs = nn.ModuleList(output_convs[::-1])

    def forward(self, features: Dict[str, torch.Tensor], confidence_maps=None, depth_raw: Optional[torch.Tensor] = None):
        del confidence_maps

        srcs, pos = [], []
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()
            proj_x = self.input_proj[idx](x)
            pos_embed = self.pe_layer(proj_x)
            if self.dpe_enabled and depth_raw is not None:
                pos_embed = pos_embed + self.depth_pe(depth_raw, proj_x.shape[-2:])
            srcs.append(proj_x)
            pos.append(pos_embed)

        memory, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = memory.shape[0]

        split_sizes = [
            (level_start_index[i + 1] - level_start_index[i] if i < self.transformer_num_feature_levels - 1 else memory.shape[1] - level_start_index[i])
            for i in range(self.transformer_num_feature_levels)
        ]
        y = torch.split(memory, split_sizes, dim=1)

        out = []
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        for idx, f in enumerate(self.in_features[: self.num_fpn_levels][::-1]):
            x = features[f].float()
            cur_fpn = self.lateral_convs[idx](x)
            y_top = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y_top = self.output_convs[idx](y_top)
            out.append(y_top)

        multi_scale_features = []
        multi_scale_pos = []
        for i, o in enumerate(out):
            if i < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                multi_scale_pos.append(self.pe_layer(o))

        return {
            "mask_features": self.mask_features(out[-1]),
            "memory": out[0],
            "multi_scale_features": multi_scale_features,
            "multi_scale_pos": multi_scale_pos,
        }
