# -*- coding: utf-8 -*-
"""
Multi-scale masked transformer decoder for MAGFormer.

支持 key_pos 参数用于深度位置编码调制。
支持 deformable cross-attention 替代 dense cross-attention。
"""

from typing import List, Optional, Dict
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from .dynamic_query import DynamicContentQueryModule
from magformer.models.common.layers.film import FiLMModulation
from magformer.models.common.layers.geometry_prior import DepthGeometryPrior


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}")


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, tgt_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return self.norm(tgt)


class MGMCrossAttentionLayer(nn.Module):
    """
    支持独立 key_pos 的交叉注意力层。

    与标准 CrossAttentionLayer 的区别：
    - 标准层：key 位置编码使用 memory 的位置编码 (pos)
    - MGM 层：key 位置编码可独立指定 (key_pos)，用于深度位置编码调制

    这对于将深度调制位置编码传递给交叉注意力是必需的。
    """

    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        key_pos: Optional[Tensor] = None,
    ):
        """
        Args:
            tgt: 目标查询 (L_q, B, C)
            memory: 内存特征 (S, B, C)
            memory_mask: 注意力掩码
            memory_key_padding_mask: 键填充掩码
            pos: 内存位置编码（当 key_pos 为 None 时使用）
            query_pos: 查询位置编码
            key_pos: 独立的键位置编码（用于深度调制位置编码）

        Returns:
            更新后的目标 (L_q, B, C)
        """
        # 如果提供了 key_pos，则使用它；否则使用 pos
        effective_key_pos = key_pos if key_pos is not None else pos

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, effective_key_pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        return self.norm(tgt)


# 保留原始名称作为别名
CrossAttentionLayer = MGMCrossAttentionLayer


class DeformableCrossAttentionLayer(nn.Module):
    """
    Deformable cross-attention layer using multi-scale deformable attention.

    Replaces dense MGMCrossAttentionLayer with sparse, content-adaptive sampling.
    Each query predicts sampling offsets and attention weights from a reference
    point, attending to only a small set of points across all feature levels.
    This is far more memory-efficient and better for small object detection.
    """

    def __init__(self, d_model, n_levels, n_heads=8, n_points=4, dropout=0.0):
        super().__init__()
        from magformer.models.ops.modules.ms_deform_attn import MSDeformAttn
        self.deformable_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        input_flatten: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        input_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            query: (L_q, B, C) - query content embeddings
            reference_points: (B, L_q, n_levels, 2) - normalized [0,1] reference points
            input_flatten: (B, sum(H_l*W_l), C) - all levels concatenated
            spatial_shapes: (n_levels, 2) - [(H_0,W_0), ...]
            level_start_index: (n_levels,) - start indices per level
            input_padding_mask: (B, sum(H_l*W_l)) - optional padding mask (True=padding)

        Returns:
            output: (L_q, B, C)
        """
        query_t = query.transpose(0, 1)  # (B, L_q, C)
        output = self.deformable_attn(
            query=query_t,
            reference_points=reference_points,
            input_flatten=input_flatten,
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
            input_padding_mask=input_padding_mask,
        )
        output = query + self.dropout(output.transpose(0, 1))  # residual, back to (L_q, B, C)
        return self.norm(output)


def compute_reference_points_from_masks(
    outputs_mask: Tensor,
    num_feature_levels: int,
) -> Tensor:
    """
    Compute reference points from mask prediction centroids.

    For each query, computes the weighted centroid of its predicted mask
    (using sigmoid probabilities as weights). The centroid is then replicated
    across all feature levels.

    Args:
        outputs_mask: (B, Q, H, W) - raw mask predictions (logits)
        num_feature_levels: number of feature levels to replicate for

    Returns:
        reference_points: (B, Q, num_feature_levels, 2) - normalized [0,1]
    """
    attn_probs = outputs_mask.sigmoid()  # (B, Q, H, W)
    B, Q, H, W = attn_probs.shape
    device = attn_probs.device

    y_grid = torch.linspace(0, 1, H, device=device, dtype=attn_probs.dtype)
    x_grid = torch.linspace(0, 1, W, device=device, dtype=attn_probs.dtype)
    y_grid, x_grid = torch.meshgrid(y_grid, x_grid, indexing='ij')

    # Weighted centroid
    prob_sum = attn_probs.flatten(2).sum(-1, keepdim=True).clamp(min=1e-6)  # (B, Q, 1)
    ref_x = (attn_probs * x_grid).flatten(2).sum(-1, keepdim=True) / prob_sum  # (B, Q, 1)
    ref_y = (attn_probs * y_grid).flatten(2).sum(-1, keepdim=True) / prob_sum  # (B, Q, 1)
    ref = torch.cat([ref_x, ref_y], dim=-1).squeeze(2)  # (B, Q, 2)

    # Replicate for all levels
    ref = ref.unsqueeze(2).expand(-1, -1, num_feature_levels, -1)  # (B, Q, n_levels, 2)
    return ref  # look-forward-twice: allow gradient flow through reference points


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        return self.norm(tgt)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiScaleMaskedTransformerDecoder(nn.Module):
    """
    多尺度掩码 Transformer 解码器。

    支持 key_pos 参数用于将深度调制位置编码传递给交叉注意力层。
    支持 deformable cross-attention 替代 dense cross-attention (use_deformable_cross_attn=True)。

    Adaptive attention masking for small objects:
    When mask_attn_topk_ratio is set, instead of using a hard 0.5 sigmoid threshold
    (which produces all-True masks for low-confidence small objects), we use a top-K
    approach per query. For each query, we keep only the top-K pixels with highest
    sigmoid scores as the attended region. This guarantees every query always has a
    focused attention region, regardless of absolute confidence levels.

    K is computed as: max(mask_attn_topk_min, predicted_mask_area * mask_attn_topk_ratio)
    where predicted_mask_area = number of pixels with sigmoid > 0.5 for that query.
    This way, small objects get a proportionally reasonable attention region.
    """

    _version = 2

    def __init__(
        self,
        num_queries: int,
        hidden_dim: int,
        nheads: int,
        dim_feedforward: int,
        num_layers: int,
        num_classes: int,
        mask_dim: int,
        dropout: float = 0.1,
        pre_norm: bool = False,
        enforce_input_project: bool = False,
        num_feature_levels: int = 3,
        mask_attn_topk_ratio: Optional[float] = None,
        mask_attn_topk_min: int = 64,
        use_checkpoint: bool = False,
        use_deformable_cross_attn: bool = False,
        deformable_n_points: int = 4,
        encoder_query_selection: bool = False,
        dcqm_enabled: bool = False,
        look_forward_twice: bool = False,
        film_config=None,
        geometry_config=None,
        depth_edge_config=None,
    ):
        super().__init__()
        self.mask_classification = True
        self.num_heads = nheads
        self.num_layers = num_layers
        self.num_feature_levels = num_feature_levels
        self.mask_attn_topk_ratio = mask_attn_topk_ratio
        self.mask_attn_topk_min = mask_attn_topk_min
        self.use_checkpoint = use_checkpoint
        self.use_deformable_cross_attn = use_deformable_cross_attn
        self.encoder_query_selection = encoder_query_selection
        self.hidden_dim = hidden_dim

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(hidden_dim, nheads,
                                   dropout=dropout, normalize_before=pre_norm)
            )
            if use_deformable_cross_attn:
                self.transformer_cross_attention_layers.append(
                    DeformableCrossAttentionLayer(
                        hidden_dim, num_feature_levels, nheads,
                        deformable_n_points, dropout)
                )
            else:
                # 使用 MGMCrossAttentionLayer 支持独立的 key_pos
                self.transformer_cross_attention_layers.append(
                    MGMCrossAttentionLayer(
                        hidden_dim, nheads, dropout=dropout, normalize_before=pre_norm)
                )
            self.transformer_ffn_layers.append(
                FFNLayer(hidden_dim, dim_feedforward=dim_feedforward,
                         dropout=dropout, normalize_before=pre_norm)
            )

        # Reference point head: maps decoder output to 2D coords for positional matching
        self.ref_point_head = nn.Linear(hidden_dim, 2)

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.dcqm_enabled = dcqm_enabled
        self.look_forward_twice = look_forward_twice
        if self.dcqm_enabled:
            self.dcqm = DynamicContentQueryModule(
                num_queries=num_queries,
                hidden_dim=hidden_dim,
                num_feature_levels=self.num_feature_levels,
            )

        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if enforce_input_project:
                self.input_proj.append(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1))
                nn.init.xavier_uniform_(self.input_proj[-1].weight)
                nn.init.constant_(self.input_proj[-1].bias, 0)
            else:
                self.input_proj.append(nn.Identity())

        if self.encoder_query_selection:
            self.enc_output = nn.Linear(hidden_dim, hidden_dim)
            self.enc_output_norm = nn.LayerNorm(hidden_dim)

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.box_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # Mask2Former 标配 init: focal bias + xavier uniform
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_embed.bias, bias_value)
        nn.init.xavier_uniform_(self.class_embed.weight)
        for m in [self.mask_embed, self.box_embed]:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        # --- FiLM cross-attention modulation (config-gated) ---
        self.film_enabled = film_config is not None and getattr(film_config, 'enabled', False)
        if self.film_enabled:
            film_hidden = getattr(film_config, 'hidden_dim', 256)
            self.film_layers = nn.ModuleList([
                FiLMModulation(feature_dim=hidden_dim, cond_dim=hidden_dim, hidden_dim=film_hidden)
                for _ in range(self.num_layers)
            ])

        # --- Depth geometry attention bias (config-gated) ---
        self.geometry_enabled = geometry_config is not None and getattr(geometry_config, 'enabled', False)
        if self.geometry_enabled:
            decay_range = getattr(geometry_config, 'decay_range', (0.75, 1.0))
            self.geometry_prior = DepthGeometryPrior(
                num_heads=nheads, decay_range=decay_range
            )

        # --- Depth edge attention mask (config-gated) ---
        self.depth_edge_enabled = depth_edge_config is not None and getattr(depth_edge_config, 'enabled', False)
        if self.depth_edge_enabled:
            # Small conv to project 1-channel edge map to bias
            self.edge_proj = nn.Sequential(
                nn.Conv2d(1, nheads, kernel_size=1),
            )
            nn.init.zeros_(self.edge_proj[-1].weight)
            nn.init.zeros_(self.edge_proj[-1].bias)

    def _validate_padding_masks(
        self,
        multi_scale_features: List[Tensor],
        multi_scale_padding_masks: Optional[List[Tensor]],
    ) -> List[Tensor]:
        if multi_scale_padding_masks is None:
            raise ValueError("multi_scale_padding_masks is required")
        if len(multi_scale_features) != self.num_feature_levels:
            raise ValueError(
                f"expected {self.num_feature_levels} feature levels, got "
                f"{len(multi_scale_features)}"
            )
        if len(multi_scale_padding_masks) != self.num_feature_levels:
            raise ValueError(
                f"expected {self.num_feature_levels} padding-mask levels, got "
                f"{len(multi_scale_padding_masks)}"
            )

        validated = []
        for level, (feature, mask) in enumerate(
            zip(multi_scale_features, multi_scale_padding_masks)
        ):
            batch_size, _, height, width = feature.shape
            if mask.dtype != torch.bool:
                raise TypeError(
                    f"padding mask at level {level} must be bool, got {mask.dtype}"
                )
            if mask.device != feature.device:
                raise ValueError(
                    f"padding mask device {mask.device} must match feature device "
                    f"{feature.device} at level {level}"
                )
            if mask.shape != (batch_size, height, width):
                raise ValueError(
                    f"padding mask shape {tuple(mask.shape)} must be "
                    f"{(batch_size, height, width)} at level {level}"
                )
            all_padding = mask.flatten(1).all(dim=1)
            if all_padding.any():
                indices = all_padding.nonzero(as_tuple=False).flatten().tolist()
                raise ValueError(
                    f"all-padding feature level {level} for samples {indices}"
                )
            validated.append(mask)
        return validated

    def _combine_semantic_and_padding_mask(
        self,
        semantic_mask: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        """Union semantic and padding masks, recovering only valid keys."""
        if semantic_mask.dtype != torch.bool:
            raise TypeError(
                f"semantic attention mask must be bool, got {semantic_mask.dtype}"
            )
        batch_size, num_keys = padding_mask.shape
        if semantic_mask.shape[0] != batch_size * self.num_heads:
            raise ValueError(
                f"semantic mask batch-head dimension {semantic_mask.shape[0]} must be "
                f"B*nheads={batch_size * self.num_heads}"
            )
        if semantic_mask.shape[-1] != num_keys:
            raise ValueError(
                f"semantic mask key count {semantic_mask.shape[-1]} must match "
                f"padding mask key count {num_keys}"
            )

        num_queries = semantic_mask.shape[1]
        expanded_padding = padding_mask[:, None, None, :].expand(
            batch_size, self.num_heads, num_queries, num_keys
        ).reshape(batch_size * self.num_heads, num_queries, num_keys)
        combined = semantic_mask | expanded_padding
        all_masked = combined.all(dim=-1)
        if all_masked.any():
            combined = torch.where(
                all_masked.unsqueeze(-1),
                expanded_padding,
                combined,
            )
        if combined.all(dim=-1).any():
            raise RuntimeError("attention-mask recovery left a query with no valid keys")
        return combined

    def _forward_decoder_layer(self, i, output, src_i, pos_i, pos_key_i, query_embed, attn_mask, depth_bias=None):
        """Single decoder layer: cross-attn + self-attn + FFN (dense mode).

        Args:
            depth_bias: Optional (B*nheads, Nq, Nk) additive float bias for geometry/edge.
        """
        # Inject additive depth bias into attn_mask
        if depth_bias is not None:
            if not depth_bias.dtype.is_floating_point:
                raise TypeError(f"depth_bias must be floating point, got {depth_bias.dtype}")
            if depth_bias.shape != attn_mask.shape:
                raise ValueError(
                    f"depth_bias shape {tuple(depth_bias.shape)} must match attention "
                    f"mask shape {tuple(attn_mask.shape)}"
                )
            if depth_bias.device != attn_mask.device:
                raise ValueError("depth_bias and attention mask must share a device")
            attn_mask_for_cross = depth_bias.masked_fill(attn_mask, float('-inf'))
        else:
            attn_mask_for_cross = attn_mask

        output = self.transformer_cross_attention_layers[i](
            output,
            src_i,
            memory_mask=attn_mask_for_cross,
            memory_key_padding_mask=None,
            pos=pos_i,
            query_pos=query_embed,
            key_pos=pos_key_i,
        )
        output = self.transformer_self_attention_layers[i](
            output,
            tgt_mask=None,
            tgt_key_padding_mask=None,
            query_pos=query_embed,
        )
        output = self.transformer_ffn_layers[i](output)
        return output

    def _forward_decoder_layer_deformable(
        self, i, output, reference_points, src_flatten, spatial_shapes,
        level_start_index, query_embed, input_padding_mask,
    ):
        """Single decoder layer: deformable cross-attn + self-attn + FFN."""
        output = self.transformer_cross_attention_layers[i](
            output,
            reference_points=reference_points,
            input_flatten=src_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            input_padding_mask=input_padding_mask,
        )
        output = self.transformer_self_attention_layers[i](
            output,
            tgt_mask=None,
            tgt_key_padding_mask=None,
            query_pos=query_embed,
        )
        output = self.transformer_ffn_layers[i](output)
        return output

    def _prepare_deformable_inputs(
        self,
        multi_scale_features: List[Tensor],
        multi_scale_pos: Optional[List[Tensor]],
        pos_key: Optional[List[Tensor]],
        multi_scale_padding_masks: List[Tensor],
    ):
        """Prepare flattened multi-scale features and spatial metadata for deformable attention."""
        src_flatten_list = []
        padding_mask_list = []
        spatial_shapes_list = []

        for i in range(self.num_feature_levels):
            x = multi_scale_features[i]
            level_mask = multi_scale_padding_masks[i]
            x = x.masked_fill(level_mask.unsqueeze(1), 0)
            x = self.input_proj[i](x)  # (B, C, H_i, W_i)
            x = x.masked_fill(level_mask.unsqueeze(1), 0)
            h, w = x.shape[-2:]
            spatial_shapes_list.append((h, w))
            x = x.flatten(2)  # (B, C, H_i*W_i)
            x = x + self.level_embed.weight[i][None, :, None]

            # Add depth positional encoding if available (scaled down)
            if pos_key is not None and i < len(pos_key) and pos_key[i] is not None:
                pk = pos_key[i]
                if pk.dim() == 4:
                    pk = pk.flatten(2)  # (B, C, H_i*W_i)
                x = x + pk * 0.1  # scale down to avoid overwhelming features

            src_flatten_list.append(x)
            padding_mask_list.append(level_mask.flatten(1))

        # Concatenate all levels: (B, C, sum(S_i)) -> (B, sum(S_i), C)
        src_flatten = torch.cat(src_flatten_list, dim=2).transpose(1, 2)
        input_padding_mask = torch.cat(padding_mask_list, dim=1)

        spatial_shapes = torch.tensor(
            spatial_shapes_list,
            device=src_flatten.device,
            dtype=torch.long,
        )
        level_start_index = torch.cat([
            torch.zeros(1, device=src_flatten.device, dtype=torch.long),
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ])

        return src_flatten, spatial_shapes, level_start_index, input_padding_mask

    def forward(
        self,
        memory: torch.Tensor,
        mask_features: Tensor,
        multi_scale_features: Optional[List[Tensor]] = None,
        multi_scale_pos: Optional[List[Tensor]] = None,
        multi_scale_padding_masks: Optional[List[Tensor]] = None,
        pos_key: Optional[List[Tensor]] = None,
        dn_query_embed: Optional[Tensor] = None,
        dn_query_feat: Optional[Tensor] = None,
        depth_raw: Optional[torch.Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        前向传播。

        Args:
            memory: 内存特征（未使用，保留用于兼容）
            mask_features: 掩码特征 (B, C, H, W)
            multi_scale_features: 多尺度特征列表
            multi_scale_pos: 多尺度位置编码列表
            pos_key: 深度调制位置编码列表（用于 key_pos）

        Returns:
            包含 pred_logits, pred_masks 和 aux_outputs 的字典
        """
        if multi_scale_features is None or len(multi_scale_features) == 0:
            raise ValueError("multi_scale_features is required for padding-aware decoding")
        multi_scale_padding_masks = self._validate_padding_masks(
            multi_scale_features,
            multi_scale_padding_masks,
        )

        src = []
        projected_features = []
        pos = []
        pos_key_list = []
        flattened_padding_masks = []
        size_list = []

        for i in range(self.num_feature_levels):
            x = multi_scale_features[i]
            level_mask = multi_scale_padding_masks[i]
            size_list.append(x.shape[-2:])
            x = x.masked_fill(level_mask.unsqueeze(1), 0)
            x = self.input_proj[i](x)
            x = x.masked_fill(level_mask.unsqueeze(1), 0)
            projected_features.append(x)
            x = x.flatten(2)
            x = x + self.level_embed.weight[i][None, :, None]
            src.append(x.permute(2, 0, 1))
            flattened_padding_masks.append(level_mask.flatten(1))

            if multi_scale_pos is not None and multi_scale_pos[i] is not None:
                pos_i = multi_scale_pos[i].masked_fill(
                    level_mask.unsqueeze(1), 0
                ).flatten(2).permute(2, 0, 1)
            else:
                pos_i = torch.zeros_like(src[-1])
            pos.append(pos_i)

            # 处理深度调制位置编码 (pos_key)
            if pos_key is not None and i < len(pos_key) and pos_key[i] is not None:
                pk = pos_key[i]
                if pk.dim() == 4:
                    pk = pk.masked_fill(level_mask.unsqueeze(1), 0)
                    pk = pk.flatten(2).permute(2, 0, 1)
                pos_key_list.append(pk)
            else:
                pos_key_list.append(None)

        _, bs, _ = src[0].shape

        if self.encoder_query_selection:
            all_src = torch.cat(src, dim=0)  # (S_total, B, C)
            all_pos = torch.cat(pos, dim=0)  # (S_total, B, C)
            all_padding = torch.cat(flattened_padding_masks, dim=1)
            memory_flat = self.enc_output_norm(self.enc_output(all_src))
            enc_logits = self.class_embed(memory_flat.transpose(0, 1))  # (B, S_total, num_classes+1)
            enc_scores = enc_logits[:, :, 0].masked_fill(all_padding, float('-inf'))
            num_select = min(self.query_embed.weight.shape[0], enc_scores.shape[1])
            valid_counts = (~all_padding).sum(dim=1)
            if (valid_counts < num_select).any():
                raise ValueError(
                    "encoder query selection requires at least "
                    f"{num_select} valid pixels per sample, got {valid_counts.tolist()}"
                )
            _, topk_indices = torch.topk(enc_scores, num_select, dim=1)
            gather_idx = topk_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            _gathered_src = torch.gather(all_src.transpose(0, 1), 1, gather_idx)
            output = _gathered_src.transpose(0, 1) if self.look_forward_twice else _gathered_src.detach().transpose(0, 1)
            _gathered_pos = torch.gather(all_pos.transpose(0, 1), 1, gather_idx)
            query_embed = _gathered_pos.transpose(0, 1) if self.look_forward_twice else _gathered_pos.detach().transpose(0, 1)
            num_regular_queries = num_select
        else:
            num_regular_queries = self.query_embed.weight.shape[0]
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            if self.dcqm_enabled:
                output = self.dcqm(
                    projected_features,
                    self.query_feat.weight,
                    multi_scale_padding_masks,
                )
            else:
                output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        # DN-DETR: concatenate denoising queries with regular queries
        dn_enabled = dn_query_embed is not None and dn_query_feat is not None
        if dn_enabled:
            # dn_query_embed: (num_dn, B, C), dn_query_feat: (num_dn, B, C)
            query_embed = torch.cat([query_embed, dn_query_embed], dim=0)
            output = torch.cat([output, dn_query_feat], dim=0)

        # --- Deformable mode: prepare flattened features and reference points ---
        if self.use_deformable_cross_attn:
            src_flatten, spatial_shapes, level_start_index, input_padding_mask = self._prepare_deformable_inputs(
                multi_scale_features,
                multi_scale_pos,
                pos_key,
                multi_scale_padding_masks,
            )
            # Initial reference points from query positional embeddings
            reference_points = self.ref_point_head(
                query_embed.transpose(0, 1)  # (B, N_q, C)
            ).sigmoid()  # (B, N_q, 2)
            reference_points = reference_points.unsqueeze(2).expand(
                -1, -1, self.num_feature_levels, -1
            )  # (B, N_q, n_levels, 2)

        predictions_class = []
        predictions_mask = []
        predictions_boxes = []
        outputs_class, outputs_mask, attn_mask, outputs_coord = self.forward_prediction_heads(
            output,
            mask_features,
            size_list[0],
            multi_scale_padding_masks[0],
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        predictions_boxes.append(outputs_coord)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels

            if self.use_deformable_cross_attn:
                # Deformable cross-attention path
                if self.use_checkpoint and self.training:
                    output = cp.checkpoint(
                        self._forward_decoder_layer_deformable,
                        i, output, reference_points, src_flatten,
                        spatial_shapes, level_start_index, query_embed,
                        input_padding_mask,
                        use_reentrant=False,
                    )
                else:
                    output = self._forward_decoder_layer_deformable(
                        i, output, reference_points, src_flatten,
                        spatial_shapes, level_start_index, query_embed,
                        input_padding_mask,
                    )
            else:
                # Dense cross-attention path (original)
                attn_mask = self._combine_semantic_and_padding_mask(
                    attn_mask,
                    flattened_padding_masks[level_index],
                )

                # FiLM modulation on memory features before cross-attention
                src_level = src[level_index]  # (S, B, C) where S = H*W
                if self.film_enabled:
                    # src_level shape: (S, B, C). FiLM expects (B, C, H, W).
                    S, B, C = src_level.shape
                    feat_h = size_list[level_index][0]
                    feat_w = size_list[level_index][1]
                    src_4d = src_level.permute(1, 2, 0).reshape(B, C, feat_h, feat_w)
                    level_mask = multi_scale_padding_masks[level_index]
                    valid = (~level_mask).unsqueeze(1).to(src_4d.dtype)
                    valid_count = valid.sum(dim=[2, 3]).clamp_min(1)
                    cond = (src_4d * valid).sum(dim=[2, 3]) / valid_count
                    src_4d = self.film_layers[i](src_4d, cond)
                    src_4d = src_4d.masked_fill(level_mask.unsqueeze(1), 0)
                    src_level = src_4d.reshape(B, C, S).permute(2, 0, 1)

                # Compute depth bias (geometry + edge) if enabled and depth_raw available
                depth_bias = None
                if (self.geometry_enabled or self.depth_edge_enabled) and depth_raw is not None:
                    feat_h = size_list[level_index][0]
                    feat_w = size_list[level_index][1]
                    Nk = feat_h * feat_w
                    Nq = output.shape[0]  # (Nq, B, C)
                    B = output.shape[1]

                    # Start with zeros
                    bias = torch.zeros(B * self.num_heads, Nq, Nk,
                                       device=attn_mask.device, dtype=attn_mask.dtype
                                       if attn_mask.dtype.is_floating_point
                                       else src_level.dtype)

                    # Geometry prior bias
                    if self.geometry_enabled:
                        geo_bias = self.geometry_prior(
                            depth_raw, feature_stride=1,
                            num_patches_h=feat_h, num_patches_w=feat_w,
                        )  # (B, heads, Nk)
                        # Scale down to be a small additive bias
                        geo_bias = geo_bias - 1.0  # center around 0 (since beta^d in [0.75, 1.0])
                        # Expand to (B*heads, Nq, Nk)
                        geo_bias = geo_bias.unsqueeze(2).expand(-1, -1, Nq, -1)
                        geo_bias = geo_bias.reshape(B * self.num_heads, Nq, Nk)
                        bias = bias + geo_bias * 0.1

                    # Depth edge attention mask
                    if self.depth_edge_enabled:
                        # Sobel edge detection on depth map
                        depth_input = depth_raw  # (B, 1, H, W)
                        # Resize to feature map size
                        depth_resized = F.interpolate(
                            depth_input, size=(feat_h, feat_w),
                            mode='bilinear', align_corners=False,
                        )  # (B, 1, feat_h, feat_w)
                        # Sobel kernels
                        sobel_x = torch.tensor(
                            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=depth_resized.dtype, device=depth_resized.device,
                        ).view(1, 1, 3, 3)
                        sobel_y = sobel_x.transpose(2, 3)
                        gx = F.conv2d(F.pad(depth_resized, (1,1,1,1), mode='reflect'), sobel_x)
                        gy = F.conv2d(F.pad(depth_resized, (1,1,1,1), mode='reflect'), sobel_y)
                        edge_mag = (gx.pow(2) + gy.pow(2)).sqrt()  # (B, 1, feat_h, feat_w)
                        # Project to multi-head bias
                        edge_bias = self.edge_proj(edge_mag)  # (B, heads, feat_h, feat_w)
                        edge_bias = edge_bias.flatten(2)  # (B, heads, Nk)
                        # Expand to (B*heads, Nq, Nk)
                        edge_bias = edge_bias.unsqueeze(2).expand(-1, -1, Nq, -1)
                        edge_bias = edge_bias.reshape(B * self.num_heads, Nq, Nk)
                        bias = bias + edge_bias

                    depth_bias = bias

                # Attention + FFN with optional gradient checkpointing
                if self.use_checkpoint and self.training:
                    output = cp.checkpoint(
                        self._forward_decoder_layer,
                        i, output, src_level, pos[level_index],
                        pos_key_list[level_index], query_embed, attn_mask, depth_bias,
                        use_reentrant=False,
                    )
                else:
                    output = self._forward_decoder_layer(
                        i, output, src_level, pos[level_index],
                        pos_key_list[level_index], query_embed, attn_mask, depth_bias,
                    )

            outputs_class, outputs_mask, attn_mask, outputs_coord = self.forward_prediction_heads(
                output,
                mask_features,
                size_list[(i + 1) % self.num_feature_levels],
                multi_scale_padding_masks[(i + 1) % self.num_feature_levels],
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_boxes.append(outputs_coord)

            # Update reference points from mask predictions (deformable mode)
            if self.use_deformable_cross_attn:
                reference_points = compute_reference_points_from_masks(
                    outputs_mask, self.num_feature_levels,
                )
                # Handle DN queries: their reference points stay at initial positions
                if dn_enabled:
                    dn_ref = self.ref_point_head(
                        query_embed[num_regular_queries:].transpose(0, 1)
                    ).sigmoid().unsqueeze(2).expand(
                        -1, -1, self.num_feature_levels, -1
                    ).detach()
                    regular_ref = reference_points[:, :num_regular_queries]
                    reference_points = torch.cat([regular_ref, dn_ref], dim=1)

        # Extract normalized query embeddings for contrastive loss (regular queries only)
        final_output = self.decoder_norm(output).transpose(0, 1)  # (B, Nq_total, C)
        if dn_enabled:
            # Split: first num_regular_queries are matching queries, rest are DN queries
            query_emb = F.normalize(final_output[:, :num_regular_queries], dim=-1)
        else:
            query_emb = F.normalize(final_output, dim=-1)

        # Predicted reference points for positional matching cost
        reference_points = self.ref_point_head(final_output).sigmoid()  # (B, Nq, 2)

        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "pred_boxes": predictions_boxes[-1],
            "aux_outputs": self._set_aux_loss(predictions_class, predictions_mask, predictions_boxes),
            "query_embeddings": query_emb,
            "reference_points": reference_points,
        }

        # DN-DETR: store metadata for loss computation
        if dn_enabled:
            out["dn_num_regular_queries"] = num_regular_queries
            # Also add to aux_outputs
            for i, aux in enumerate(out["aux_outputs"]):
                out["aux_outputs"][i]["dn_num_regular_queries"] = num_regular_queries

        return out

    def forward_prediction_heads(
        self,
        output,
        mask_features,
        attn_mask_target_size,
        attn_mask_target_padding_mask,
    ):
        decoder_output = self.decoder_norm(output).transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum(
            "bqc,bchw->bqhw", mask_embed, mask_features)
        outputs_coord = self.box_embed(decoder_output).sigmoid()

        attn_mask = F.interpolate(
            outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        batch_size, _, target_height, target_width = attn_mask.shape
        if attn_mask_target_padding_mask.shape != (
            batch_size,
            target_height,
            target_width,
        ):
            raise ValueError(
                "attention target padding mask shape "
                f"{tuple(attn_mask_target_padding_mask.shape)} must be "
                f"{(batch_size, target_height, target_width)}"
            )
        valid_pixels = ~attn_mask_target_padding_mask

        if self.mask_attn_topk_ratio is not None:
            # Adaptive top-K attention masking for small objects.
            #
            # Problem: the standard 0.5 threshold produces all-True masks for
            # low-confidence small object predictions. The safety fallback then
            # resets them to all-False (attend everywhere), losing focused attention.
            #
            # Solution: for each query, keep the top-K highest sigmoid pixels as
            # attended (mask=False). K is proportional to predicted mask area with
            # a minimum floor. This guarantees every query always has a focused
            # attention region, regardless of absolute confidence levels.
            attn_probs = attn_mask.sigmoid()  # (B, Q, H, W)
            B, Q, H, W = attn_probs.shape
            num_pixels = H * W

            # Compute predicted mask area per query: pixels with sigmoid > 0.5
            pred_area = (
                (attn_probs > 0.5) & valid_pixels.unsqueeze(1)
            ).flatten(2).sum(dim=-1)

            # K per query: max(topk_min, predicted_area * ratio)
            # Clamp to at most num_pixels to avoid degenerate cases
            k_per_query = torch.clamp(
                (pred_area.float() * self.mask_attn_topk_ratio).long(),
                min=self.mask_attn_topk_min,
            )  # (B, Q)
            valid_counts = valid_pixels.flatten(1).sum(dim=1, keepdim=True)
            k_per_query = torch.minimum(k_per_query, valid_counts)

            # Flatten spatial dims: (B, Q, H*W)
            attn_probs_flat = attn_probs.flatten(2)

            # For each query, find the top-K pixels.
            # topk returns (values, indices) along last dim.
            # We batch this over B and Q by flattening to (B*Q, H*W).
            padding_2d = attn_mask_target_padding_mask[:, None].expand(
                B, Q, H, W
            ).reshape(B * Q, num_pixels)
            probs_2d = attn_probs_flat.reshape(B * Q, num_pixels).masked_fill(
                padding_2d, float('-inf')
            )
            k_vals = k_per_query.reshape(B * Q)

            # Build the mask per query.
            # For each row in probs_2d, we want the top-k_j pixels to be False
            # (attend), rest True (mask out).
            # Since k varies per query, we use a vectorized approach:
            # Sort in descending order, then the k-th value is the threshold.
            sorted_probs, _ = probs_2d.sort(dim=-1, descending=True)  # (B*Q, H*W)

            # Gather the k-th value for each query (k is 1-indexed, so index k-1).
            # Clamp k to valid range.
            k_idx = (k_vals - 1).clamp(0, num_pixels - 1)  # (B*Q,)
            kth_values = sorted_probs.gather(1, k_idx.unsqueeze(1)).squeeze(1)  # (B*Q,)

            # Attention mask: True = mask out. Pixels with prob >= kth_value are
            # attended (False). But we need to handle ties carefully: if many pixels
            # have the same value as the k-th, we might keep more than k. That's fine --
            # it's strictly better than masking them out.
            attn_mask_2d = probs_2d < kth_values.unsqueeze(1)  # (B*Q, H*W)

            # Reshape to (B, Q, H, W), then expand for multi-head:
            # (B, Q, H, W) -> (B, num_heads, Q, H*W) -> (B*num_heads, Q, H*W)
            attn_mask = attn_mask_2d.reshape(B, Q, H, W)
            attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
                1, self.num_heads, 1, 1).flatten(0, 1).bool()
        else:
            # Original Mask2Former behavior: hard 0.5 sigmoid threshold.
            attn_mask = (
                attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(
                    1, self.num_heads, 1, 1).flatten(0, 1) < 0.5
            ).bool()

        expanded_padding = attn_mask_target_padding_mask[:, None, None].expand(
            batch_size,
            self.num_heads,
            attn_mask.shape[1],
            target_height,
            target_width,
        ).reshape(
            batch_size * self.num_heads,
            attn_mask.shape[1],
            target_height * target_width,
        )
        attn_mask = attn_mask | expanded_padding

        attn_mask = attn_mask.detach()
        return outputs_class, outputs_mask, attn_mask, outputs_coord

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_boxes):
        return [
            {"pred_logits": a, "pred_masks": b, "pred_boxes": c}
            for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_boxes[:-1])
        ]

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        重写以处理旧版本 checkpoint 的加载兼容性。
        版本 2 添加了 key_pos 支持。
        Deformable cross-attention 新增的参数 (ref_point_head, deformable sampling
        offsets/weights) 在旧 checkpoint 中不存在, 自动使用 strict=False 加载。
        """
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # 旧版本 checkpoint，跳过 key_pos 相关检查
            strict = False
        # When deformable cross-attention is enabled but loading from a non-deformable
        # checkpoint, the new parameters won't exist. Allow missing keys silently.
        if self.use_deformable_cross_attn:
            strict = False
        # FiLM / geometry / edge modules: allow missing keys for backward compat
        if self.film_enabled or self.geometry_enabled or self.depth_edge_enabled:
            strict = False
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys, error_msgs)
