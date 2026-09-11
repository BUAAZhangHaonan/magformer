# -*- coding: utf-8 -*-
"""
Modality Fusion Module (原 MGM)

多模态融合模块，用于融合 RGB 和深度特征。
支持: sa_gate (channel-only), sa_gate_spatial, dccg, dccg_cross_only
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from .cmx_fusion import FeatureRectifyModule as _CMX_FRM
from .cmx_fusion import FeatureFusionModule as _CMX_FFM


def _bilinear(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """统一的双线性插值（align_corners=False）"""
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _make_group_norm(num_channels: int, preferred_groups: int = 8) -> nn.GroupNorm:
    for num_groups in (preferred_groups, 16, 8, 4, 2, 1):
        if num_groups <= num_channels and num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups, num_channels)
    return nn.GroupNorm(1, num_channels)


# =============================================================================
# 深度先验提取
# =============================================================================
class DepthPriorExtractor(nn.Module):
    """
    深度先验提取（不可学习、无梯度）：
    - 梯度幅值（Sobel）
    - 局部方差（盒滤）
    - 有效/空洞掩码（基于深度范围）
    - （可选）RGB-深度边缘一致性
    """

    def __init__(
        self,
        var_kernel: int = 5,
        z_min: float = 0.0,
        z_max: float = 1.0,
        use_rgb_edge: bool = True,
        robust_norm: bool = True,
        robust_norm_method: str = "minmax",
    ):
        super().__init__()
        self.k = int(var_kernel)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.use_rgb_edge = bool(use_rgb_edge)
        self.robust_norm = bool(robust_norm)
        self.robust_norm_method = robust_norm_method

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    @torch.no_grad()
    def _robust_norm(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x_flat = x.view(batch, -1)

        if self.robust_norm and self.robust_norm_method == "quantile":
            x_flat_f32 = x_flat.float()
            p95 = torch.quantile(x_flat_f32, 0.95, dim=1, keepdim=True)
            p05 = torch.quantile(x_flat_f32, 0.05, dim=1, keepdim=True)
            norm_flat = (x_flat_f32 - p05) / (p95 - p05 + 1e-6)
            norm_flat = norm_flat.clamp_(0, 1).to(x.dtype)
        else:
            mn = x_flat.min(dim=1, keepdim=True).values
            mx = x_flat.max(dim=1, keepdim=True).values
            norm_flat = (x_flat - mn) / (mx - mn + 1e-6)
            norm_flat = norm_flat.clamp_(0, 1)
        return norm_flat.view_as(x)

    @torch.no_grad()
    def _compute_grad(self, x: torch.Tensor) -> torch.Tensor:
        x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
        w_x = self.sobel_x.to(x_pad.dtype)
        w_y = self.sobel_y.to(x_pad.dtype)
        gx = F.conv2d(x_pad, w_x, stride=1, padding=0)
        gy = F.conv2d(x_pad, w_y, stride=1, padding=0)
        grad = torch.sqrt(gx**2 + gy**2 + 1e-6)
        return self._robust_norm(grad)

    @torch.no_grad()
    def _compute_var(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.k
        pad = kernel // 2
        mu = F.avg_pool2d(x, kernel_size=kernel, stride=1,
                          padding=pad, count_include_pad=False)
        var = (
            F.avg_pool2d(x**2, kernel_size=kernel, stride=1,
                         padding=pad, count_include_pad=False)
            - mu**2
        )
        return self._robust_norm(var.clamp_min_(0.0))

    @torch.no_grad()
    def _valid_and_hole(self, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = ((depth > self.z_min) & (depth < self.z_max)).float()
        hole = 1.0 - valid
        return valid, hole

    @torch.no_grad()
    def _edge_consistency(self, rgb: Optional[torch.Tensor], depth: torch.Tensor) -> torch.Tensor:
        if not self.use_rgb_edge or rgb is None:
            return torch.ones_like(depth)

        if rgb.shape[1] == 3:
            gray = 0.299 * rgb[:, 0:1] + 0.587 * \
                rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        else:
            gray = rgb[:, :1]
        if gray.max() > 1.0:
            gray = gray / 255.0

        grad_rgb = self._compute_grad(gray)
        grad_dep = self._compute_grad(depth)
        return (1.0 - (grad_rgb - grad_dep).abs()).clamp_(0, 1)

    @torch.no_grad()
    def forward(
        self, depth_raw: torch.Tensor, rgb: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        depth = depth_raw.float()
        priors = {
            "gradient": self._compute_grad(depth),
            "variance": self._compute_var(depth),
        }
        valid, hole = self._valid_and_hole(depth)
        priors["valid"] = valid
        priors["hole"] = hole
        priors["edge_consistency"] = self._edge_consistency(rgb, depth)
        return priors


# =============================================================================
# DCCG: Depth-Confidence Cross-Modal Gate
# =============================================================================
class DepthConfidenceEstimator(nn.Module):
    """从 raw depth map 生成逐像素置信度 [0,1].
    conf 接近 1 = depth 可靠, 接近 0 = depth 不可靠.
    极轻量: 1x1→GN→GELU→3x3DW→GN→GELU→1x1→Sigmoid
    """
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 1, bias=False),
            _make_group_norm(hidden_dim, 8),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            _make_group_norm(hidden_dim, 8),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1, bias=True),
        )
        # 初始化使 Sigmoid 输出约 0.5 (中性起点)
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def logits(self, depth_raw: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """Return raw logits (before sigmoid), resized to target_size."""
        depth = _bilinear(depth_raw.float(), target_size)
        return self.net(depth)

    def forward(self, depth_raw: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        return torch.sigmoid(self.logits(depth_raw, target_size))


class CrossModalSpatialAttention(nn.Module):
    """双向跨模态空间注意力.
    每个模态用另一个模态的 channel 统计来引导自己的 spatial attention.
    """
    def __init__(self, channels: int):
        super().__init__()
        # Channel gating: GAP → 1x1 → Sigmoid
        self.channel_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        # Spatial attention after channel gating: 3x3 DW → GN → GELU → 1x1 → Sigmoid
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            _make_group_norm(channels, 8),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=True),
        )
        # 零初始化最后一层, 初始状态等价于 uniform attention (sigmoid(0)=0.5)
        nn.init.normal_(self.spatial_attn[-1].weight, std=0.01)
        nn.init.zeros_(self.spatial_attn[-1].bias)

    def forward(self, target_feat: torch.Tensor, guide_feat: torch.Tensor) -> torch.Tensor:
        # guide 的 channel 统计 → broadcast → 调制 target
        guide_gate = self.channel_gate(F.adaptive_avg_pool2d(guide_feat, 1))
        modulated = target_feat * guide_gate
        attn_logits = self.spatial_attn(modulated)
        return torch.sigmoid(attn_logits)


# =============================================================================
# 多模态融合
# =============================================================================
class ModalityFusionModule(nn.Module):
    """多模态融合模块. 支持 sa_gate / sa_gate_spatial / dccg / dccg_cross_only."""

    def __init__(
        self,
        feature_dims: Optional[List[int]] = None,
        image_feature_dims: Optional[List[int]] = None,
        depth_feature_dims: Optional[List[int]] = None,
        scale_keys: Optional[List[str]] = None,
        residual_alpha: float = 0.0,
        temp_init: float = 1.5,
        temp_final: float = 1.0,
        temp_steps: int = 3000,
        clamp_min: float = 0.05,
        clamp_max: float = 0.95,
        loss_entropy_weight: float = 0.01,
        noise_mask_weight: float = 0.0,
        hidden_dim: int = 256,
        prior_enabled: bool = True,
        prior_use_grad: bool = True,
        prior_use_var: bool = True,
        prior_use_valid_hole: bool = True,
        prior_use_rgb_edge: bool = False,
        prior_var_kernel: int = 5,
        prior_z_min: float = 0.0,
        prior_z_max: float = 1.0,
        robust_norm: bool = True,
        robust_norm_method: str = "minmax",
        prior_compute_on: str = "res3",
        post_fuse_norm: bool = True,
        mode: str = "sa_gate",
        fuse_scales: Optional[List[str]] = None,
        prior_names: Optional[List[str]] = None,
        cross_attn_heads: int = 8,
        cross_attn_downsample: int = 8,
        sa_gate_spatial: bool = False,
        prior_ablation_zero: Optional[List[str]] = None,
        dccg_conf_hidden: int = 16,
        dccg_use_confidence: bool = True,
    ):
        super().__init__()

        if scale_keys is None:
            raise ValueError("scale_keys must be provided")

        self.image_feature_dims = list(
            image_feature_dims or feature_dims or [])
        if not self.image_feature_dims:
            raise ValueError(
                "image_feature_dims or feature_dims must be provided")
        self.depth_feature_dims = list(
            depth_feature_dims or self.image_feature_dims)
        self.scale_keys = list(scale_keys)
        self.fuse_scales = list(fuse_scales or self.scale_keys)
        self.residual_alpha = float(residual_alpha)
        self.prior_enabled = bool(prior_enabled)
        self.prior_use_grad = bool(prior_use_grad)
        self.prior_use_var = bool(prior_use_var)
        self.prior_use_valid_hole = bool(prior_use_valid_hole)
        self.prior_use_rgb_edge = bool(prior_use_rgb_edge)
        self.prior_compute_on = str(prior_compute_on)
        self.post_fuse_norm = bool(post_fuse_norm)
        self.mode = str(mode).lower()
        supported_modes = {"sa_gate", "dccg", "dccg_cross_only", "cmx"}
        if self.mode not in supported_modes:
            raise ValueError(
                f"Unsupported fusion mode: {self.mode}. "
                f"Supported modes are {sorted(supported_modes)}"
            )
        self.prior_names = list(prior_names or [])
        self.sa_gate_spatial = bool(sa_gate_spatial)
        self.prior_ablation_zero = set(prior_ablation_zero or [])
        self.dccg_use_confidence = bool(dccg_use_confidence)
        # ---- DCCG: temperature anneal / clamp / entropy (Fix 3 / C1) ----
        self.temp_init = float(temp_init)
        self.temp_final = float(temp_final)
        self.temp_steps = max(int(temp_steps), 1)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.loss_entropy_weight = float(loss_entropy_weight)
        self.noise_mask_weight = float(noise_mask_weight)
        self.register_buffer('_dccg_step', torch.zeros(1, dtype=torch.long))
        self._scale_to_index = {key: idx for idx,
                                key in enumerate(self.scale_keys)}

        if len(self.image_feature_dims) != len(self.scale_keys):
            raise ValueError(
                "Image feature dimensions and scale keys must match.")
        if len(self.depth_feature_dims) != len(self.scale_keys):
            raise ValueError(
                "Depth feature dimensions and scale keys must match.")
        invalid_scales = sorted(set(self.fuse_scales) - set(self.scale_keys))
        if invalid_scales:
            raise ValueError(
                f"fuse_scales must be a subset of scale_keys, got extra {invalid_scales}")
        if self.prior_enabled and self.prior_compute_on != "full" and self.prior_compute_on not in self.scale_keys:
            raise ValueError(
                f"`prior_compute_on` ('{self.prior_compute_on}') must be 'full' or one of `scale_keys` ({self.scale_keys})"
            )

        self.prior_extractor = DepthPriorExtractor(
            var_kernel=prior_var_kernel,
            z_min=prior_z_min,
            z_max=prior_z_max,
            use_rgb_edge=self.prior_use_rgb_edge,
            robust_norm=robust_norm,
            robust_norm_method=robust_norm_method,
        )

        self._prior_channels = 0
        if self.prior_enabled:
            if self.prior_use_grad:
                self._prior_channels += 1
            if self.prior_use_var:
                self._prior_channels += 1
            if self.prior_use_valid_hole:
                self._prior_channels += 2
            if self.prior_use_rgb_edge:
                self._prior_channels += 1

        self.align_image = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(ch, ch, 1, bias=False),
                              _make_group_norm(ch, 8))
                for ch in self.image_feature_dims
            ]
        )
        self.align_depth = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(depth_ch, image_ch, 1, bias=False),
                    _make_group_norm(image_ch, 8),
                )
                for image_ch, depth_ch in zip(self.image_feature_dims, self.depth_feature_dims)
            ]
        )
        self.post_norm = (
            nn.ModuleList([_make_group_norm(ch, 8)
                          for ch in self.image_feature_dims])
            if self.post_fuse_norm
            else None
        )

        # ---- SA-Gate layers (for sa_gate / sa_gate_spatial modes) ----
        self.sa_gate_layers = nn.ModuleDict()
        self.sa_gate_spatial_layers = nn.ModuleDict()

        is_dccg_mode = self.mode in ("dccg", "dccg_cross_only")
        is_cmx_mode = self.mode == "cmx"

        if self.residual_alpha != 0.0:
            raise ValueError(
                "residual_alpha is not implemented by ModalityFusionModule; set it to 0.0"
            )
        if self.noise_mask_weight != 0.0:
            raise ValueError(
                "noise_mask_weight is not implemented by ModalityFusionModule; set it to 0.0"
            )
        if is_dccg_mode and self.prior_enabled:
            raise ValueError(
                "DCCG does not consume depth priors; set modality_fusion.prior.enabled=false"
            )

        for key in self.fuse_scales:
            idx = self._scale_to_index[key]
            channels = self.image_feature_dims[idx]

            if not is_dccg_mode and not is_cmx_mode:
                # Original SA-Gate layers (only create for non-DCCG / non-CMX modes)
                in_ch = channels * 3 + self._prior_channels
                hidden = max(8, channels // 4)
                layer = nn.Sequential(
                    nn.Conv2d(in_ch, hidden, 1, bias=True),
                    nn.GELU(),
                    nn.Conv2d(hidden, channels * 2, 1, bias=True),
                )
                nn.init.zeros_(layer[-1].weight)
                nn.init.zeros_(layer[-1].bias[:channels])
                nn.init.constant_(layer[-1].bias[channels:], -2.0)
                self.sa_gate_layers[key] = layer

                if self.sa_gate_spatial:
                    spatial_layer = nn.Sequential(
                        nn.Conv2d(in_ch, hidden, 1, bias=False),
                        nn.GroupNorm(min(hidden, 32) if hidden % min(hidden, 32) == 0 else 1, hidden),
                        nn.GELU(),
                        nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
                        nn.GroupNorm(min(hidden, 32) if hidden % min(hidden, 32) == 0 else 1, hidden),
                        nn.GELU(),
                        nn.Conv2d(hidden, channels * 2, 1, bias=True),
                    )
                    nn.init.zeros_(spatial_layer[-1].weight)
                    nn.init.zeros_(spatial_layer[-1].bias[:channels])
                    nn.init.constant_(spatial_layer[-1].bias[channels:], -2.0)
                    self.sa_gate_spatial_layers[key] = spatial_layer

        # ---- CMX layers (for cmx mode) ----
        # CMX original code uses reduction=1, but that explodes params for high-dim
        # stages (dim=768 ChannelWeights = 14M). Use proportional reduction so each
        # scale FRM stays ~0.5-1M params, matching DCCG budget.
        if is_cmx_mode:
            self.cmx_frms = nn.ModuleDict()
            self.cmx_ffms = nn.ModuleDict()
            for key in self.fuse_scales:
                idx = self._scale_to_index[key]
                channels = self.image_feature_dims[idx]
                # reduction grows with channels: 96->1, 192->2, 384->4, 768->8
                reduction = max(1, channels // 96)
                # num_heads must divide (channels // reduction) for cross-attn
                eff_dim = channels // reduction
                num_heads = 8 if eff_dim % 8 == 0 else (4 if eff_dim % 4 == 0 else 1)
                self.cmx_frms[key] = _CMX_FRM(dim=channels, reduction=reduction)
                self.cmx_ffms[key] = _CMX_FFM(dim=channels, reduction=reduction, num_heads=num_heads)

        # ---- DCCG layers (for dccg / dccg_cross_only modes) ----
        if is_dccg_mode:
            if self.dccg_use_confidence:
                self.depth_confidence = DepthConfidenceEstimator(
                    hidden_dim=dccg_conf_hidden)

            self.cross_modal_rgb_attn = nn.ModuleDict()
            self.cross_modal_dep_attn = nn.ModuleDict()
            for key in self.fuse_scales:
                idx = self._scale_to_index[key]
                channels = self.image_feature_dims[idx]
                self.cross_modal_rgb_attn[key] = CrossModalSpatialAttention(channels)
                self.cross_modal_dep_attn[key] = CrossModalSpatialAttention(channels)

        self._prior_missing_warned = False

        # Initialize alignment layers
        for seq in self.align_image:
            conv = seq[0]
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            gn = seq[1]
            nn.init.ones_(gn.weight)
            nn.init.zeros_(gn.bias)
        for seq in self.align_depth:
            conv = seq[0]
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            gn = seq[1]
            nn.init.ones_(gn.weight)
            nn.init.zeros_(gn.bias)

    def _prepare_priors_ms(
        self,
        depth_raw: torch.Tensor,
        rgb_image: Optional[torch.Tensor],
        target_sizes: Dict[str, Tuple[int, int]],
        override_compute_on: Optional[str] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        if not self.prior_enabled or not target_sizes:
            return {}

        effective_compute_on = self.prior_compute_on if override_compute_on is None else override_compute_on
        if effective_compute_on != "full" and effective_compute_on not in target_sizes:
            effective_compute_on = "full"

        if effective_compute_on == "full":
            compute_res_depth = depth_raw
            compute_res_rgb = rgb_image
        else:
            height, width = target_sizes[effective_compute_on]
            compute_res_depth = _bilinear(depth_raw, (height, width))
            compute_res_rgb = _bilinear(
                rgb_image, (height, width)) if rgb_image is not None else None

        priors_single = self.prior_extractor(
            compute_res_depth, compute_res_rgb)
        priors_ms = {key: {} for key in target_sizes}
        for prior_name, prior_tensor in priors_single.items():
            for key, (height, width) in target_sizes.items():
                priors_ms[key][prior_name] = _bilinear(
                    prior_tensor, (height, width))
        return priors_ms

    def _stack_priors(self, key: str, priors_ms: Dict[str, Dict[str, torch.Tensor]]) -> Optional[torch.Tensor]:
        if not self.prior_enabled or key not in priors_ms:
            return None

        parts = []
        if self.prior_use_grad and "gradient" in priors_ms[key]:
            p = priors_ms[key]["gradient"]
            parts.append(torch.zeros_like(p) if "gradient" in self.prior_ablation_zero else p)
        if self.prior_use_var and "variance" in priors_ms[key]:
            p = priors_ms[key]["variance"]
            parts.append(torch.zeros_like(p) if "variance" in self.prior_ablation_zero else p)
        if self.prior_use_valid_hole:
            if "valid" in priors_ms[key]:
                p = priors_ms[key]["valid"]
                parts.append(torch.zeros_like(p) if "valid" in self.prior_ablation_zero else p)
            if "hole" in priors_ms[key]:
                p = priors_ms[key]["hole"]
                parts.append(torch.zeros_like(p) if "hole" in self.prior_ablation_zero else p)
        if self.prior_use_rgb_edge and "edge_consistency" in priors_ms[key]:
            p = priors_ms[key]["edge_consistency"]
            parts.append(torch.zeros_like(p) if "edge_consistency" in self.prior_ablation_zero else p)
        if not parts:
            return None
        return torch.cat(parts, dim=1)

    def _zero_prior_stack_like(self, reference: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            reference.shape[0],
            self._prior_channels,
            reference.shape[-2],
            reference.shape[-1],
            dtype=reference.dtype,
            device=reference.device,
        )

    def _current_temp(self):
        """Anneal schedule: temp_init -> temp_final over temp_steps.

        Merged fast path:
        - training: computed as a GPU tensor from the ``_dccg_step`` buffer so
          the fusion forward performs no host synchronization (also avoids a
          graph break under torch.compile).
        - eval: the value is constant, so it is cached as a python float after
          the first call; a per-call ``.item()`` would abort CUDA-graph
          capture of the inference path.
        """
        if not self.training:
            cached = getattr(self, "_temp_eval_cache", None)
            if cached is not None:
                return cached
        step = self._dccg_step
        progress = (step.to(torch.float32) / float(self.temp_steps)).clamp(max=1.0)
        temp = self.temp_init + (self.temp_final - self.temp_init) * progress
        temp = temp.clamp(min=1e-6)
        if not self.training:
            self._temp_eval_cache = float(temp.detach().cpu())
        return temp

    def advance_optimizer_step(self) -> None:
        """Advance DCCG temperature only after a successful optimizer update."""
        if self.training and self.mode.startswith("dccg") and self.dccg_use_confidence:
            self._dccg_step += 1

    def _apply_sa_gate(
        self,
        *,
        key: str,
        img_a: torch.Tensor,
        dep_a: torch.Tensor,
        prior_stack: Optional[torch.Tensor],
    ) -> torch.Tensor:
        shared = 0.5 * (img_a + dep_a)
        rgb_specific = img_a - shared
        dep_specific = dep_a - shared

        if self.sa_gate_spatial:
            parts = [img_a, dep_a, shared]
            if prior_stack is not None:
                parts.append(prior_stack)
            gate_input = torch.cat(parts, dim=1)
            gate_out = self.sa_gate_spatial_layers[key](gate_input)
            rgb_logits, dep_logits = torch.chunk(gate_out, 2, dim=1)
            weights = torch.softmax(torch.stack(
                [rgb_logits, dep_logits], dim=1), dim=1)
            rgb_weight = weights[:, 0]
            dep_weight = weights[:, 1]
        else:
            pooled_parts = [
                F.adaptive_avg_pool2d(img_a, 1),
                F.adaptive_avg_pool2d(dep_a, 1),
                F.adaptive_avg_pool2d(shared, 1),
            ]
            if prior_stack is not None:
                pooled_parts.append(F.adaptive_avg_pool2d(prior_stack, 1))
            rgb_logits, dep_logits = torch.chunk(
                self.sa_gate_layers[key](torch.cat(pooled_parts, dim=1)), 2, dim=1)
            weights = torch.softmax(torch.stack(
                [rgb_logits, dep_logits], dim=1), dim=1)
            rgb_weight = weights[:, 0]
            dep_weight = weights[:, 1]

        return shared + rgb_weight * rgb_specific + dep_weight * dep_specific

    def _apply_dccg(
        self,
        *,
        key: str,
        img_a: torch.Tensor,
        dep_a: torch.Tensor,
        depth_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Depth-Confidence Cross-Modal Gate fusion. Returns (out, conf)."""
        h, w = img_a.shape[-2:]

        # Step 1: Shared/Private decomposition
        shared = 0.5 * (img_a + dep_a)
        rgb_private = img_a - shared
        dep_private = dep_a - shared

        # Step 2: Bidirectional cross-modal spatial attention
        # RGB 用 depth 的 channel 统计引导自己的 spatial attention
        rgb_attn = self.cross_modal_rgb_attn[key](img_a, dep_a)   # [B,C,H,W]
        # Depth 用 RGB 的 channel 统计引导自己的 spatial attention
        dep_attn = self.cross_modal_dep_attn[key](dep_a, img_a)   # [B,C,H,W]

        # Step 3: Confidence-gated fusion (or cross-only without confidence)
        if self.dccg_use_confidence:
            # Temperature anneal + clamp (Fix 3 / C1): confidence estimator learns
            # from raw depth, sigmoid(logits/temp) clamped to safe range.
            temp = self._current_temp()
            conf_logits = self.depth_confidence.logits(depth_raw, (h, w))
            conf = torch.sigmoid(conf_logits / temp)
            conf = conf.clamp(self.clamp_min, self.clamp_max)
            # depth 可靠 → dep 权重高; depth 不可靠 → RGB 权重高
            rgb_gate = rgb_attn * (1.0 - conf)
            dep_gate = dep_attn * conf
        else:
            # dccg_cross_only: 纯跨模态 attention，无 confidence 调制
            conf = torch.ones_like(img_a[:, :1])
            rgb_gate = rgb_attn
            dep_gate = dep_attn

        # 归一化
        gate_sum = rgb_gate + dep_gate + 1e-6
        rgb_w = rgb_gate / gate_sum
        dep_w = dep_gate / gate_sum

        out = shared + rgb_w * rgb_private + dep_w * dep_private
        return out, conf

    def _apply_cmx(
        self,
        key: str,
        img_a: torch.Tensor,
        dep_a: torch.Tensor,
    ) -> torch.Tensor:
        """CMX (T-ITS 2023) cross-modal fusion: FRM rectify + FFM fuse."""
        r_img, r_dep = self.cmx_frms[key](img_a, dep_a)
        fused = self.cmx_ffms[key](r_img, r_dep)
        return fused

    def forward(
        self,
        image_features: Dict[str, torch.Tensor],
        depth_features: Dict[str, torch.Tensor],
        depth_raw: torch.Tensor,
        rgb_image: Optional[torch.Tensor] = None,
        depth_noise_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        target_sizes = {
            key: value.shape[-2:]
            for key, value in image_features.items()
            if key in self.fuse_scales
        }
        if not target_sizes:
            return dict(image_features), {}, {}

        effective_prior_compute_on = self.prior_compute_on
        if (
            self.prior_enabled
            and self.prior_compute_on != "full"
            and self.prior_compute_on not in target_sizes
        ):
            if not self._prior_missing_warned:
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"[MGM] prior_compute_on='{self.prior_compute_on}' not present in this batch features. Fallback to 'full'."
                )
            self._prior_missing_warned = True
            effective_prior_compute_on = "full"

        is_dccg_mode = self.mode in ("dccg", "dccg_cross_only")
        is_cmx_mode = self.mode == "cmx"

        priors_ms = self._prepare_priors_ms(
            depth_raw,
            rgb_image,
            target_sizes,
            override_compute_on=effective_prior_compute_on,
        )

        fused: Dict[str, torch.Tensor] = {}
        confidence_maps: Dict[str, torch.Tensor] = {}
        confs_for_entropy = []
        for idx, key in enumerate(self.scale_keys):
            if key not in image_features:
                continue
            if key not in self.fuse_scales or key not in depth_features:
                fused[key] = image_features[key]
                continue

            img_a = self.align_image[idx](image_features[key])
            dep_a = self.align_depth[idx](depth_features[key])

            if self.prior_enabled and self.prior_use_valid_hole and not is_dccg_mode and not is_cmx_mode:
                valid_mask = priors_ms.get(key, {}).get("valid")
                if valid_mask is not None:
                    dep_a = dep_a * valid_mask

            if is_dccg_mode:
                out, conf = self._apply_dccg(
                    key=key,
                    img_a=img_a,
                    dep_a=dep_a,
                    depth_raw=depth_raw,
                )
                if self.dccg_use_confidence:
                    confidence_maps[key] = conf
                if (
                    self.training
                    and self.dccg_use_confidence
                    and self.loss_entropy_weight > 0
                ):
                    confs_for_entropy.append(conf.flatten())
            elif is_cmx_mode:
                out = self._apply_cmx(
                    key=key,
                    img_a=img_a,
                    dep_a=dep_a,
                )
            else:
                prior_stack = self._stack_priors(key, priors_ms)
                if prior_stack is None and self._prior_channels > 0:
                    prior_stack = self._zero_prior_stack_like(img_a)
                out = self._apply_sa_gate(
                    key=key,
                    img_a=img_a,
                    dep_a=dep_a,
                    prior_stack=prior_stack,
                )

            if self.post_norm:
                out = self.post_norm[idx](out)
            fused[key] = out

        losses: Dict[str, torch.Tensor] = {}
        if is_dccg_mode and self.dccg_use_confidence:
            if self.loss_entropy_weight > 0 and confs_for_entropy:
                all_conf = torch.cat(confs_for_entropy, dim=0)
                eps = 1e-7
                all_conf_c = all_conf.clamp(eps, 1.0 - eps)
                # Binary entropy; minimizing pushes conf toward 0 or 1 (decisive gating)
                entropy = -(all_conf_c * all_conf_c.log()
                            + (1.0 - all_conf_c) * (1.0 - all_conf_c).log())
                losses["loss_entropy"] = entropy.mean() * self.loss_entropy_weight
        return fused, confidence_maps, losses
