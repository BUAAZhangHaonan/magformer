# -*- coding: utf-8 -*-
"""
MSDeformAttn pixel decoder for MAGFormer.

支持深度位置编码 (DPE) 调制和 pos_key_list 输出。
"""

from typing import Dict, List, Optional
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# 使用 torch.amp.autocast 替代 torch.cuda.amp.autocast
try:
    from torch.amp import autocast
    _autocast_device_type = "cuda"
except ImportError:
    from torch.cuda.amp import autocast as _old_autocast
    _autocast_device_type = None

    def autocast(device_type="cuda", enabled=True):
        return _old_autocast(enabled=enabled)

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
                torch.linspace(0.5, h - 0.5, h,
                               dtype=torch.float32, device=device),
                torch.linspace(0.5, w - 0.5, w,
                               dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / \
                (valid_ratios[:, None, lvl, 1] * h)
            ref_x = ref_x.reshape(-1)[None] / \
                (valid_ratios[:, None, lvl, 0] * w)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device)
        for layer in self.layers:
            output = layer(output, pos, reference_points,
                           spatial_shapes, level_start_index, padding_mask)
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
        self.encoder = MSDeformAttnTransformerEncoder(
            encoder_layer, num_encoder_layers)
        self.level_embed = nn.Parameter(
            torch.Tensor(num_feature_levels, d_model))
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

    def forward(self, srcs, pos_embeds, masks: Optional[List[torch.Tensor]] = None):
        if masks is None:
            masks = [
                torch.zeros((x.size(0), x.size(2), x.size(3)),
                            device=x.device, dtype=torch.bool)
                for x in srcs
            ]
        else:
            if len(masks) != len(srcs):
                raise ValueError(
                    f"masks length {len(masks)} must match srcs length {len(srcs)}")
        src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes = [], [], [], []

        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            if mask.shape != (bs, h, w):
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} must be (B,H,W)={(bs, h, w)} at lvl={lvl}")
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
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
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
    """
    MSDeformAttn Pixel Decoder。

    支持：
    - 深度位置编码 (DPE) 调制
    - 输出 pos_key_list 用于 Transformer 解码器的 key_pos
    """

    def __init__(
        self,
        in_features: List[str],
        in_channels: Dict[str, int],
        transformer_in_features: Optional[List[str]] = None,
        hidden_dim: int = 256,
        mask_dim: int = 256,
        transformer_dropout: float = 0.1,
        transformer_nheads: int = 8,
        transformer_dim_feedforward: int = 1024,
        transformer_enc_layers: int = 6,
        common_stride: int = 4,
        dpe_enabled: bool = False,
        dpe_beta: float = 10.0,
    ):
        super().__init__()
        self.in_features = list(in_features)
        self.feature_channels = [in_channels[k] for k in self.in_features]
        self.feature_strides = [2 ** (i + 2)
                                for i in range(len(self.in_features))]
        stride_map = {
            name: stride for name, stride in zip(self.in_features, self.feature_strides)
        }

        if transformer_in_features is None:
            self.transformer_in_features = self.in_features
        else:
            self.transformer_in_features = list(transformer_in_features)
            missing = [
                k for k in self.transformer_in_features if k not in stride_map]
            if missing:
                raise ValueError(
                    f"transformer_in_features must be a subset of in_features, missing: {missing}"
                )
        self.transformer_feature_strides = [
            stride_map[k] for k in self.transformer_in_features]
        self.transformer_num_feature_levels = len(self.transformer_in_features)

        # 跟踪 decoder level 名称
        self.decoder_level_names = self.transformer_in_features[::-1][:3]

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
        self.depth_pe = DepthPosEncoding(
            hidden_dim=hidden_dim, beta=dpe_beta) if dpe_enabled else None

        self.mask_features = nn.Conv2d(hidden_dim, mask_dim, kernel_size=1)
        nn.init.xavier_uniform_(self.mask_features.weight)
        nn.init.constant_(self.mask_features.bias, 0)

        self.maskformer_num_feature_levels = 3
        self.common_stride = common_stride

        # Match Mask2Former: FPN levels are computed from encoder input strides
        # (e.g. encoder uses res3-res5 => min stride 8 => one extra res2 FPN level).
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = max(
            int(np.log2(stride) - np.log2(self.common_stride)), 0)
        lateral_convs = []
        output_convs = []
        for idx, in_ch in enumerate(self.feature_channels[: self.num_fpn_levels]):
            groups = min(32, hidden_dim)
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False),
                nn.GroupNorm(groups, hidden_dim),
            )
            output_conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                          stride=1, padding=1, bias=False),
                nn.GroupNorm(groups, hidden_dim),
                nn.ReLU(inplace=True),
            )
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        self.lateral_convs = nn.ModuleList(lateral_convs[::-1])
        self.output_convs = nn.ModuleList(output_convs[::-1])

        # Match Mask2Former / detectron2 c2_xavier_fill initialization for FPN convs.
        for convs in [self.lateral_convs, self.output_convs]:
            for seq in convs:
                for m in seq.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

    @autocast(device_type="cuda", enabled=False)
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        confidence_maps: Optional[Dict[str, torch.Tensor]] = None,
        depth_modulation_maps: Optional[Dict[str, torch.Tensor]] = None,
        depth_raw: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        depth_valid_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播。

        Args:
            features: 多尺度特征字典
            confidence_maps: legacy 置信度图字典（用于 DPE 调制）
            depth_modulation_maps: 通用深度调制图字典；当存在时优先于 confidence_maps
            depth_raw: 原始深度图 (B, 1, H, W)
            padding_mask: 填充掩码 (B, H, W)
            depth_valid_masks: raw depth validity masks (B, 1, H, W)

        Returns:
            包含 mask_features, memory, multi_scale_features, multi_scale_pos, pos_key_list 的字典
        """
        srcs, pos_2d_list = [], []
        masks = None
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()
            proj_x = self.input_proj[idx](x)
            pos_embed = self.pe_layer(proj_x)

            srcs.append(proj_x)
            pos_2d_list.append(pos_embed)

        if padding_mask is not None:
            # padding_mask: (B, H, W), True indicates padding. Downsample per level with nearest.
            if padding_mask.dtype != torch.bool:
                padding_mask = padding_mask.to(torch.bool)
            masks = []
            for s in srcs:
                h, w = s.shape[-2:]
                m = F.interpolate(
                    padding_mask[:, None].float(),
                    size=(h, w),
                    mode="nearest",
                ).to(torch.bool)
                masks.append(m[:, 0])

        memory, spatial_shapes, level_start_index = self.transformer(
            srcs, pos_2d_list, masks=masks)
        bs = memory.shape[0]

        split_sizes = [
            (level_start_index[i + 1] - level_start_index[i] if i <
             self.transformer_num_feature_levels - 1 else memory.shape[1] - level_start_index[i])
            for i in range(self.transformer_num_feature_levels)
        ]
        y = torch.split(memory, split_sizes, dim=1)

        out = []
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(
                bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        for idx, f in enumerate(self.in_features[: self.num_fpn_levels][::-1]):
            x = features[f].float()
            cur_fpn = self.lateral_convs[idx](x)
            y_top = cur_fpn + \
                F.interpolate(out[-1], size=cur_fpn.shape[-2:],
                              mode="bilinear", align_corners=False)
            y_top = self.output_convs[idx](y_top)
            out.append(y_top)

        multi_scale_features = []
        multi_scale_pos = []
        for i, o in enumerate(out):
            if i < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                multi_scale_pos.append(self.pe_layer(o))

        # 计算深度调制位置编码 (pos_key_list)
        pos_key_list = None
        modulation_maps = self._resolve_dpe_modulation_maps(
            depth_raw=depth_raw,
            confidence_maps=confidence_maps,
            depth_modulation_maps=depth_modulation_maps,
            depth_valid_masks=depth_valid_masks,
        )
        if self.dpe_enabled and depth_raw is not None and modulation_maps is not None:
            pos_key_list = []
            # 计算基础深度位置编码
            depth_pe_base = self.depth_pe(depth_raw, padding_mask)

            for i, feature_level in enumerate(multi_scale_features):
                # 获取对应尺度的置信度图
                feature_name = self.decoder_level_names[i] if i < len(
                    self.decoder_level_names) else None
                conf_map = None
                if feature_name is not None and feature_name in modulation_maps:
                    conf_map = modulation_maps[feature_name]
                elif len(modulation_maps) > 0:
                    # 回退：使用第一个可用的置信度图
                    first_key = list(modulation_maps.keys())[0]
                    conf_map = modulation_maps[first_key]

                if conf_map is not None:
                    h, w = feature_level.shape[-2:]
                    # 调整置信度图大小
                    conf_map_resized = F.interpolate(conf_map, size=(
                        h, w), mode="bilinear", align_corners=False)
                    # 限制置信度范围
                    conf_map_clamped = conf_map_resized.clamp(0.0, 1.0)

                    # 调整深度 PE 大小
                    depth_pe_scaled = F.interpolate(depth_pe_base, size=(
                        h, w), mode="bilinear", align_corners=False)

                    # 调制：pos_key = pos_2d + conf_map * depth_pe
                    pos_2d = multi_scale_pos[i] if i < len(
                        multi_scale_pos) else self.pe_layer(feature_level)
                    pos_key = pos_2d + conf_map_clamped * depth_pe_scaled
                    pos_key_list.append(pos_key)
                else:
                    pos_key_list.append(None)

        result = {
            "mask_features": self.mask_features(out[-1]),
            "memory": out[0],
            "multi_scale_features": multi_scale_features,
            "multi_scale_pos": multi_scale_pos,
        }

        # 只有当 pos_key_list 有有效值时才添加
        if pos_key_list is not None and any(pk is not None for pk in pos_key_list):
            result["pos_key_list"] = pos_key_list

        return result

    @staticmethod
    def _resolve_dpe_modulation_maps(
        *,
        depth_raw: Optional[torch.Tensor],
        confidence_maps: Optional[Dict[str, torch.Tensor]],
        depth_modulation_maps: Optional[Dict[str, torch.Tensor]],
        depth_valid_masks: Optional[torch.Tensor],
    ) -> Optional[Dict[str, torch.Tensor]]:
        if depth_modulation_maps:
            return depth_modulation_maps
        if confidence_maps:
            return confidence_maps
        if depth_valid_masks is None:
            return None
        return {"depth_valid": depth_valid_masks.to(dtype=torch.float32)}
