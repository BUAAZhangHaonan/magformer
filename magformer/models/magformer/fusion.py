# -*- coding: utf-8 -*-
"""
Modality Fusion Module (原 MGM)

多模态融合模块，用于融合 RGB 和深度特征。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bilinear(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """统一的双线性插值（align_corners=False）"""
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _make_group_norm(num_channels: int, preferred_groups: int = 8) -> nn.GroupNorm:
    for num_groups in (preferred_groups, 16, 8, 4, 2, 1):
        if num_groups <= num_channels and num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups, num_channels)
    return nn.GroupNorm(1, num_channels)


def _pick_num_heads(embed_dim: int, requested_heads: int) -> int:
    limit = max(1, min(int(requested_heads), int(embed_dim)))
    for heads in range(limit, 0, -1):
        if embed_dim % heads == 0:
            return heads
    return 1


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
        mu = F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=pad, count_include_pad=False)
        var = (
            F.avg_pool2d(x**2, kernel_size=kernel, stride=1, padding=pad, count_include_pad=False)
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
            gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
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
# 置信度预测器
# =============================================================================


class ConfidencePredictor(nn.Module):
    """模态融合置信度预测器。"""

    def __init__(
        self,
        image_feature_dims: List[int],
        depth_feature_dims: Optional[List[int]],
        scale_keys: List[str],
        hidden_dim: int = 128,
        temp_init: float = 1.5,
        clamp_min: float = 0.05,
        clamp_max: float = 0.95,
        prior_in_channels: int = 5,
    ):
        super().__init__()
        self.image_feature_dims = list(image_feature_dims)
        self.depth_feature_dims = list(depth_feature_dims or image_feature_dims)
        self.scale_keys = list(scale_keys)
        self.hidden = int(hidden_dim)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self._prior_channels = int(prior_in_channels)

        assert len(self.image_feature_dims) == len(self.scale_keys)
        assert len(self.depth_feature_dims) == len(self.scale_keys)
        assert hidden_dim % 16 == 0, "hidden_dim must be divisible by 16"

        self.register_buffer("temp", torch.tensor(float(temp_init), dtype=torch.float32))

        self.proj_image = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, self.hidden, 1, bias=False),
                    _make_group_norm(self.hidden, 16),
                    nn.GELU(),
                )
                for ch in self.image_feature_dims
            ]
        )
        self.proj_depth = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, self.hidden, 1, bias=False),
                    _make_group_norm(self.hidden, 16),
                    nn.GELU(),
                )
                for ch in self.depth_feature_dims
            ]
        )

        self.proj_prior = None
        if self._prior_channels > 0:
            self.proj_prior = nn.Sequential(
                nn.Conv2d(self._prior_channels, self.hidden, 1, bias=False),
                _make_group_norm(self.hidden, 16),
                nn.GELU(),
            )

        in_ch_head = self.hidden * 3 if self._prior_channels > 0 else self.hidden * 2
        self.head = nn.Sequential(
            nn.Conv2d(in_ch_head, self.hidden, 3, padding=1, bias=False),
            _make_group_norm(self.hidden, 16),
            nn.GELU(),
            nn.Conv2d(self.hidden, self.hidden // 2, 3, padding=1, bias=False),
            _make_group_norm(self.hidden // 2, 8),
            nn.GELU(),
            nn.Conv2d(self.hidden // 2, 1, 1),
        )

        last = self.head[-1]
        if isinstance(last, nn.Conv2d):
            if last.bias is not None:
                nn.init.constant_(last.bias, -4.0)
            nn.init.normal_(last.weight, std=1e-3)

    def set_temperature(self, t: float) -> None:
        self.temp.fill_(float(t))

    def forward(
        self,
        image_features: Dict[str, torch.Tensor],
        depth_features: Dict[str, torch.Tensor],
        priors_ms: Dict[str, Dict[str, torch.Tensor]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        m_maps: Dict[str, torch.Tensor] = {}
        logits_maps: Dict[str, torch.Tensor] = {}

        for idx, key in enumerate(self.scale_keys):
            if key not in image_features or key not in depth_features:
                continue

            img_p = self.proj_image[idx](image_features[key])
            dep_p = self.proj_depth[idx](depth_features[key])
            features_to_cat = [img_p, dep_p]

            prior_stack = priors_ms.get(key, {}).get("stack")
            if prior_stack is not None and self.proj_prior is not None:
                if prior_stack.shape[1] != self._prior_channels:
                    raise RuntimeError(
                        f"Runtime prior channels {prior_stack.shape[1]} != initialized {self._prior_channels}"
                    )
                features_to_cat.append(self.proj_prior(prior_stack))

            logits = self.head(torch.cat(features_to_cat, dim=1)) / self.temp.clamp(1e-6)
            m = torch.sigmoid(logits)
            if self.clamp_max > self.clamp_min:
                m = m * (self.clamp_max - self.clamp_min) + self.clamp_min
            else:
                m = m.clamp(min=self.clamp_min, max=self.clamp_max)
            m_maps[key] = m
            logits_maps[key] = logits

        return m_maps, logits_maps


# =============================================================================
# 多模态融合
# =============================================================================


class ModalityFusionModule(nn.Module):
    """多模态融合模块，兼容 legacy MGM 与 light-depth Batch 1 新模式。"""

    def __init__(
        self,
        feature_dims: Optional[List[int]] = None,
        image_feature_dims: Optional[List[int]] = None,
        depth_feature_dims: Optional[List[int]] = None,
        scale_keys: Optional[List[str]] = None,
        residual_alpha: float = 0.05,
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
        mode: str = "legacy_gated",
        fuse_scales: Optional[List[str]] = None,
        prior_names: Optional[List[str]] = None,
        cross_attn_heads: int = 8,
        cross_attn_downsample: int = 8,
    ):
        super().__init__()

        if scale_keys is None:
            raise ValueError("scale_keys must be provided")

        self.image_feature_dims = list(image_feature_dims or feature_dims or [])
        if not self.image_feature_dims:
            raise ValueError("image_feature_dims or feature_dims must be provided")
        self.depth_feature_dims = list(depth_feature_dims or self.image_feature_dims)
        self.scale_keys = list(scale_keys)
        self.fuse_scales = list(fuse_scales or self.scale_keys)
        self.residual_alpha = float(residual_alpha)
        self.temp_init = float(temp_init)
        self.temp_final = float(temp_final)
        self.temp_steps = int(temp_steps)
        self._cur_step = 0
        self.loss_entropy_weight = float(loss_entropy_weight)
        self.noise_mask_weight = float(noise_mask_weight)
        self.prior_enabled = bool(prior_enabled)
        self.prior_use_grad = bool(prior_use_grad)
        self.prior_use_var = bool(prior_use_var)
        self.prior_use_valid_hole = bool(prior_use_valid_hole)
        self.prior_use_rgb_edge = bool(prior_use_rgb_edge)
        self.prior_compute_on = str(prior_compute_on)
        self.post_fuse_norm = bool(post_fuse_norm)
        self.mode = str(mode).lower()
        self.prior_names = list(prior_names or [])
        self.cross_attn_heads = int(cross_attn_heads)
        self.cross_attn_downsample = max(1, int(cross_attn_downsample))
        self._scale_to_index = {key: idx for idx, key in enumerate(self.scale_keys)}

        if len(self.image_feature_dims) != len(self.scale_keys):
            raise ValueError("Image feature dimensions and scale keys must match.")
        if len(self.depth_feature_dims) != len(self.scale_keys):
            raise ValueError("Depth feature dimensions and scale keys must match.")
        invalid_scales = sorted(set(self.fuse_scales) - set(self.scale_keys))
        if invalid_scales:
            raise ValueError(f"fuse_scales must be a subset of scale_keys, got extra {invalid_scales}")
        if self.mode not in {"legacy_gated", "direct_add", "gated_add", "film", "cross_attn"}:
            raise ValueError(f"Unsupported fusion mode: {self.mode}")
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

        pred_scale_keys = (
            [key for key in self.scale_keys if key in self.fuse_scales]
            if self.mode in {"legacy_gated", "gated_add"}
            else []
        )
        self.conf_pred = None
        if pred_scale_keys:
            pred_indices = [self._scale_to_index[key] for key in pred_scale_keys]
            self.conf_pred = ConfidencePredictor(
                image_feature_dims=[self.image_feature_dims[idx] for idx in pred_indices],
                depth_feature_dims=[self.depth_feature_dims[idx] for idx in pred_indices],
                scale_keys=pred_scale_keys,
                hidden_dim=hidden_dim,
                temp_init=temp_init,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
                prior_in_channels=self._prior_channels,
            )
        self._pred_scale_keys = pred_scale_keys

        self.align_image = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(ch, ch, 1, bias=False), _make_group_norm(ch, 8))
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
            nn.ModuleList([_make_group_norm(ch, 8) for ch in self.image_feature_dims])
            if self.post_fuse_norm
            else None
        )
        self.film_layers = nn.ModuleDict()
        self.cross_attn_layers = nn.ModuleDict()
        self.cross_attn_norms = nn.ModuleDict()
        self.cross_attn_prior_proj = nn.ModuleDict()
        if self.mode == "film":
            for key in self.fuse_scales:
                idx = self._scale_to_index[key]
                in_ch = self.image_feature_dims[idx] + self._prior_channels
                layer = nn.Conv2d(in_ch, self.image_feature_dims[idx] * 2, 1)
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                self.film_layers[key] = layer
        elif self.mode == "cross_attn":
            for key in self.fuse_scales:
                idx = self._scale_to_index[key]
                embed_dim = self.image_feature_dims[idx]
                num_heads = _pick_num_heads(embed_dim, self.cross_attn_heads)
                self.cross_attn_layers[key] = nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    batch_first=True,
                    dropout=0.0,
                )
                self.cross_attn_norms[key] = nn.LayerNorm(embed_dim)
                if self._prior_channels > 0:
                    self.cross_attn_prior_proj[key] = nn.Conv2d(self._prior_channels, embed_dim, 1)

        self._prior_missing_warned = False

        for seq in self.align_image:
            conv = seq[0]
            nn.init.eye_(conv.weight.view(conv.weight.shape[0], -1))
            gn = seq[1]
            nn.init.ones_(gn.weight)
            nn.init.zeros_(gn.bias)
        for seq in self.align_depth:
            conv = seq[0]
            nn.init.eye_(conv.weight.view(conv.weight.shape[0], -1))
            gn = seq[1]
            nn.init.ones_(gn.weight)
            nn.init.zeros_(gn.bias)
        for layer in self.cross_attn_prior_proj.values():
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _update_temperature(self) -> None:
        if self.training and self.conf_pred is not None:
            if self._cur_step <= self.temp_steps and self.temp_steps > 0:
                progress = float(self._cur_step) / float(self.temp_steps)
                temp = self.temp_init + (self.temp_final - self.temp_init) * progress
                self.conf_pred.set_temperature(temp)
            self._cur_step += 1

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
            compute_res_rgb = _bilinear(rgb_image, (height, width)) if rgb_image is not None else None

        priors_single = self.prior_extractor(compute_res_depth, compute_res_rgb)
        priors_ms = {key: {} for key in target_sizes}
        for prior_name, prior_tensor in priors_single.items():
            for key, (height, width) in target_sizes.items():
                priors_ms[key][prior_name] = _bilinear(prior_tensor, (height, width))
        return priors_ms

    def _stack_priors(self, key: str, priors_ms: Dict[str, Dict[str, torch.Tensor]]) -> Optional[torch.Tensor]:
        if not self.prior_enabled or key not in priors_ms:
            return None

        parts = []
        if self.prior_use_grad and "gradient" in priors_ms[key]:
            parts.append(priors_ms[key]["gradient"])
        if self.prior_use_var and "variance" in priors_ms[key]:
            parts.append(priors_ms[key]["variance"])
        if self.prior_use_valid_hole:
            if "valid" in priors_ms[key]:
                parts.append(priors_ms[key]["valid"])
            if "hole" in priors_ms[key]:
                parts.append(priors_ms[key]["hole"])
        if self.prior_use_rgb_edge and "edge_consistency" in priors_ms[key]:
            parts.append(priors_ms[key]["edge_consistency"])
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

    def _downsample_for_cross_attn(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.cross_attn_downsample <= 1:
            return tensor
        height, width = tensor.shape[-2:]
        target_h = max(1, height // self.cross_attn_downsample)
        target_w = max(1, width // self.cross_attn_downsample)
        if target_h == height and target_w == width:
            return tensor
        return F.adaptive_avg_pool2d(tensor, (target_h, target_w))

    def _apply_cross_attn(
        self,
        *,
        key: str,
        img_a: torch.Tensor,
        dep_a: torch.Tensor,
        prior_stack: Optional[torch.Tensor],
    ) -> torch.Tensor:
        kv_map = dep_a
        if prior_stack is not None and key in self.cross_attn_prior_proj:
            kv_map = kv_map + self.cross_attn_prior_proj[key](prior_stack)
        kv_map = self._downsample_for_cross_attn(kv_map)

        query_tokens = img_a.flatten(2).transpose(1, 2)
        kv_tokens = kv_map.flatten(2).transpose(1, 2)
        attn_out, _ = self.cross_attn_layers[key](
            query_tokens,
            kv_tokens,
            kv_tokens,
            need_weights=False,
        )
        attn_out = self.cross_attn_norms[key](query_tokens + attn_out)
        return attn_out.transpose(1, 2).reshape_as(img_a)

    def forward(
        self,
        image_features: Dict[str, torch.Tensor],
        depth_features: Dict[str, torch.Tensor],
        depth_raw: torch.Tensor,
        rgb_image: Optional[torch.Tensor] = None,
        depth_noise_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if self.training:
            self._update_temperature()

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

        priors_ms = self._prepare_priors_ms(
            depth_raw,
            rgb_image,
            target_sizes,
            override_compute_on=effective_prior_compute_on,
        )

        if self.conf_pred is not None:
            priors_for_pred: Dict[str, Dict[str, torch.Tensor]] = {}
            for key in self._pred_scale_keys:
                if key not in image_features or key not in depth_features:
                    continue
                prior_stack = self._stack_priors(key, priors_ms)
                if prior_stack is None and self._prior_channels > 0:
                    prior_stack = self._zero_prior_stack_like(image_features[key])
                if prior_stack is not None:
                    priors_for_pred[key] = {"stack": prior_stack}
            m_maps, logits_maps = self.conf_pred(image_features, depth_features, priors_for_pred)
        else:
            m_maps, logits_maps = {}, {}

        fused: Dict[str, torch.Tensor] = {}
        for idx, key in enumerate(self.scale_keys):
            if key not in image_features:
                continue
            if key not in self.fuse_scales or key not in depth_features:
                fused[key] = image_features[key]
                continue

            img_a = self.align_image[idx](image_features[key])
            dep_a = self.align_depth[idx](depth_features[key])

            if self.prior_enabled and self.prior_use_valid_hole:
                valid_mask = priors_ms.get(key, {}).get("valid")
                if valid_mask is not None:
                    dep_a = dep_a * valid_mask

            if self.mode == "legacy_gated":
                m = m_maps[key]
                out = m * (dep_a + self.residual_alpha * img_a) + (1.0 - m) * img_a
            elif self.mode == "gated_add":
                m = m_maps[key]
                out = img_a + m * dep_a
            elif self.mode == "direct_add":
                out = img_a + dep_a
            elif self.mode == "film":
                prior_stack = self._stack_priors(key, priors_ms)
                if prior_stack is None and self._prior_channels > 0:
                    prior_stack = self._zero_prior_stack_like(img_a)
                context = dep_a if prior_stack is None else torch.cat([dep_a, prior_stack], dim=1)
                gamma, beta = torch.chunk(self.film_layers[key](context), 2, dim=1)
                out = img_a * (1.0 + torch.tanh(gamma)) + beta
            elif self.mode == "cross_attn":
                prior_stack = self._stack_priors(key, priors_ms)
                if prior_stack is None and self._prior_channels > 0:
                    prior_stack = self._zero_prior_stack_like(img_a)
                out = self._apply_cross_attn(
                    key=key,
                    img_a=img_a,
                    dep_a=dep_a,
                    prior_stack=prior_stack,
                )
            else:
                raise ValueError(f"Unsupported fusion mode: {self.mode}")

            if self.post_norm:
                out = self.post_norm[idx](out)
            fused[key] = out

        losses: Dict[str, torch.Tensor] = {}
        if self.training and m_maps:
            if self.loss_entropy_weight > 0:
                ent_terms = [
                    -(
                        m.clamp(1e-6, 1.0 - 1e-6) * torch.log(m.clamp(1e-6, 1.0 - 1e-6))
                        + (1.0 - m.clamp(1e-6, 1.0 - 1e-6))
                        * torch.log(1.0 - m.clamp(1e-6, 1.0 - 1e-6))
                    ).mean()
                    for m in m_maps.values()
                ]
                losses["loss_mgm_entropy"] = self.loss_entropy_weight * torch.stack(ent_terms).mean()

            if self.noise_mask_weight > 0 and depth_noise_mask is not None:
                bces = []
                for key, logits in logits_maps.items():
                    target_noise_mask = _bilinear(depth_noise_mask.float(), logits.shape[-2:])
                    target_noise_mask_dilated = F.max_pool2d(
                        target_noise_mask, kernel_size=3, stride=1, padding=1
                    )
                    bces.append(
                        F.binary_cross_entropy_with_logits(logits, 1.0 - target_noise_mask_dilated)
                    )
                if bces:
                    losses["loss_mgm_noise"] = self.noise_mask_weight * torch.stack(bces).mean()

        return fused, m_maps, losses
