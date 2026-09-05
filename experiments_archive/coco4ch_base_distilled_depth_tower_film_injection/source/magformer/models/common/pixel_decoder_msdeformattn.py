# -*- coding: utf-8 -*-
"""
MSDeformAttn pixel decoder for MAGFormer.

支持深度位置编码 (DPE) 调制和 pos_key_list 输出。
"""

from typing import Dict, List, Optional
import numpy as np
import torch
from torch import nn
import torch.utils.checkpoint as cp
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
from .layers.film import FiLMModulation
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
    def __init__(self, encoder_layer, num_layers, use_checkpoint: bool = False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.use_checkpoint = use_checkpoint

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
            if self.use_checkpoint and self.training:
                output = cp.checkpoint(
                    lambda x, *args, _layer=layer: _layer(x, *args),
                    output, pos, reference_points, spatial_shapes, level_start_index, padding_mask,
                    use_reentrant=False,
                )
            else:
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
        use_checkpoint: bool = False,
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
            encoder_layer, num_encoder_layers, use_checkpoint=use_checkpoint)
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
        use_checkpoint: bool = False,
        maskformer_num_feature_levels: int = 3,
        film_config=None,
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
        self.decoder_level_names = self.transformer_in_features[::-1][:maskformer_num_feature_levels]

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
            use_checkpoint=use_checkpoint,
        )

        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.dpe_enabled = dpe_enabled
        self.depth_pe = DepthPosEncoding(
            hidden_dim=hidden_dim, beta=dpe_beta) if dpe_enabled else None

        self.mask_features = nn.Conv2d(hidden_dim, mask_dim, kernel_size=1)
        nn.init.xavier_uniform_(self.mask_features.weight)
        nn.init.constant_(self.mask_features.bias, 0)

        self.maskformer_num_feature_levels = maskformer_num_feature_levels
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

        # FiLM modulation (config-gated, identity init -> zero regression risk)
        self._film_enabled = False
        if film_config is not None and getattr(film_config, 'enabled', False):
            self._film_enabled = True
            film_hidden = getattr(film_config, 'hidden_dim', 256) or 256
            self.film_layers = nn.ModuleList([
                FiLMModulation(feature_dim=hidden_dim, cond_dim=hidden_dim, hidden_dim=film_hidden)
                for _ in range(maskformer_num_feature_levels)
            ])
            self.depth_cond_proj = nn.Linear(1, hidden_dim)

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        confidence_maps: Optional[Dict[str, torch.Tensor]] = None,
        depth_modulation_maps: Optional[Dict[str, torch.Tensor]] = None,
        depth_raw: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播。

        Args:
            features: 多尺度特征字典
            confidence_maps: legacy 置信度图字典（用于 DPE 调制）
            depth_modulation_maps: 通用深度调制图字典；当存在时优先于 confidence_maps
            depth_raw: 原始深度图 (B, 1, H, W)
            padding_mask: 填充掩码 (B, H, W)

        Returns:
            包含 mask_features, memory, multi_scale_features, multi_scale_pos, pos_key_list 的字典
        """
        srcs, pos_2d_list, masks = [], [], []
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()
            level_mask = self._resize_padding_mask(
                padding_mask,
                x,
                level_name=f,
            )
            x = x.masked_fill(level_mask.unsqueeze(1), 0)
            proj_x = self.input_proj[idx](x)
            proj_x = proj_x.masked_fill(level_mask.unsqueeze(1), 0)
            pos_embed = self.pe_layer(proj_x, level_mask)

            srcs.append(proj_x)
            pos_2d_list.append(pos_embed)
            masks.append(level_mask)

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
            level = z.transpose(1, 2).view(
                bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]
            )
            out.append(level.masked_fill(masks[i].unsqueeze(1), 0))

        out_padding_masks = list(masks)

        for idx, f in enumerate(self.in_features[: self.num_fpn_levels][::-1]):
            x = features[f].float()
            level_mask = self._resize_padding_mask(
                padding_mask,
                x,
                level_name=f,
            )
            x = x.masked_fill(level_mask.unsqueeze(1), 0)
            cur_fpn = self.lateral_convs[idx](x)
            cur_fpn = cur_fpn.masked_fill(level_mask.unsqueeze(1), 0)
            y_top = cur_fpn + \
                F.interpolate(out[-1], size=cur_fpn.shape[-2:],
                              mode="bilinear", align_corners=False)
            y_top = self.output_convs[idx](y_top)
            out.append(y_top.masked_fill(level_mask.unsqueeze(1), 0))
            out_padding_masks.append(level_mask)

        multi_scale_features = []
        multi_scale_pos = []
        multi_scale_padding_masks = []
        for i, o in enumerate(out):
            if i < self.maskformer_num_feature_levels:
                level_mask = out_padding_masks[i]
                multi_scale_features.append(o)
                multi_scale_padding_masks.append(level_mask)
                multi_scale_pos.append(self.pe_layer(o, level_mask))

        # 计算深度调制位置编码 (pos_key_list)
        pos_key_list = None
        modulation_maps = self._resolve_dpe_modulation_maps(
            depth_raw=depth_raw,
            confidence_maps=confidence_maps,
            depth_modulation_maps=depth_modulation_maps,
        )
        if self.dpe_enabled and depth_raw is not None and modulation_maps is not None:
            pos_key_list = []
            # 计算基础深度位置编码
            depth_pe_base = self.depth_pe(depth_raw, padding_mask)

            for i, feature_level in enumerate(multi_scale_features):
                # 获取对应尺度的置信度图
                feature_name = self.decoder_level_names[i] if i < len(
                    self.decoder_level_names) else None
                conf_map = self._select_dpe_modulation_map(
                    feature_name=feature_name,
                    modulation_maps=modulation_maps,
                )

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
                        multi_scale_pos) else self.pe_layer(
                            feature_level, multi_scale_padding_masks[i]
                        )
                    pos_key = pos_2d + conf_map_clamped * depth_pe_scaled
                    pos_key = pos_key.masked_fill(
                        multi_scale_padding_masks[i].unsqueeze(1), 0
                    )
                    pos_key_list.append(pos_key)
                else:
                    pos_key_list.append(None)

        # Apply FiLM modulation to multi-scale features (depth-conditioned)
        if self._film_enabled and depth_raw is not None:
            if padding_mask is None:
                depth_global = depth_raw.mean(dim=[2, 3])
            else:
                valid = (~padding_mask).unsqueeze(1).to(depth_raw.dtype)
                valid_count = valid.sum(dim=[2, 3]).clamp_min(1)
                depth_global = (depth_raw * valid).sum(dim=[2, 3]) / valid_count
            depth_cond = self.depth_cond_proj(depth_global)  # (B, hidden_dim)
            for lvl in range(len(multi_scale_features)):
                multi_scale_features[lvl] = self.film_layers[lvl](
                    multi_scale_features[lvl], depth_cond)
                multi_scale_features[lvl] = multi_scale_features[lvl].masked_fill(
                    multi_scale_padding_masks[lvl].unsqueeze(1), 0
                )

        result = {
            "mask_features": self.mask_features(out[-1]),
            "memory": out[0],
            "multi_scale_features": multi_scale_features,
            "multi_scale_pos": multi_scale_pos,
            "multi_scale_padding_masks": multi_scale_padding_masks,
        }

        # 只有当 pos_key_list 有有效值时才添加
        if pos_key_list is not None and any(pk is not None for pk in pos_key_list):
            result["pos_key_list"] = pos_key_list

        return result

    @staticmethod
    def _resize_padding_mask(
        padding_mask: Optional[torch.Tensor],
        feature: torch.Tensor,
        *,
        level_name: str,
    ) -> torch.Tensor:
        """Return a strict boolean padding mask aligned with one feature level."""
        batch_size, _, height, width = feature.shape
        if padding_mask is None:
            return torch.zeros(
                (batch_size, height, width),
                dtype=torch.bool,
                device=feature.device,
            )
        if padding_mask.dtype != torch.bool:
            raise TypeError(
                f"padding_mask must be bool, got {padding_mask.dtype} at {level_name}"
            )
        if padding_mask.device != feature.device:
            raise ValueError(
                f"padding_mask device {padding_mask.device} must match feature device "
                f"{feature.device} at {level_name}"
            )
        if padding_mask.ndim != 3 or padding_mask.shape[0] != batch_size:
            raise ValueError(
                f"padding_mask shape {tuple(padding_mask.shape)} must be "
                f"(B,H,W) with B={batch_size} at {level_name}"
            )

        resized = F.interpolate(
            padding_mask.unsqueeze(1).float(),
            size=(height, width),
            mode="nearest",
        ).squeeze(1).to(torch.bool)
        all_padding = resized.flatten(1).all(dim=1)
        if all_padding.any():
            indices = all_padding.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                f"all-padding feature level {level_name} for samples {indices}"
            )
        return resized

    @staticmethod
    def _resolve_dpe_modulation_maps(
        *,
        depth_raw: Optional[torch.Tensor],
        confidence_maps: Optional[Dict[str, torch.Tensor]],
        depth_modulation_maps: Optional[Dict[str, torch.Tensor]],
    ) -> Optional[Dict[str, torch.Tensor]]:
        if depth_modulation_maps:
            return depth_modulation_maps
        if confidence_maps:
            return confidence_maps
        if depth_raw is None:
            return None
        valid_mask = (torch.isfinite(depth_raw) & (depth_raw > 0)).float()
        if valid_mask.numel() == 0:
            return None
        if float(valid_mask.max().item()) <= 0.0:
            valid_mask = torch.zeros_like(depth_raw, dtype=torch.float32)
        return {"depth_valid": valid_mask}

    @staticmethod
    def _select_dpe_modulation_map(
        *,
        feature_name: Optional[str],
        modulation_maps: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if feature_name is None:
            raise KeyError("DPE decoder level has no feature name")
        if feature_name in modulation_maps:
            return modulation_maps[feature_name]
        if set(modulation_maps) == {"depth_valid"}:
            return modulation_maps["depth_valid"]
        available = sorted(modulation_maps)
        raise KeyError(
            f"Missing DPE modulation map for '{feature_name}'; available scales: {available}"
        )
