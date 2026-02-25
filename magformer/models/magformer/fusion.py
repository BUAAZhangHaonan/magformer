# -*- coding: utf-8 -*-
"""
Modality Fusion Module (原 MGM)

多模态门控融合模块，用于融合 RGB 和深度特征。
"""

import logging
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _bilinear(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """统一的双线性插值（align_corners=False）"""
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


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
        self.robust_norm_method = robust_norm_method  # "quantile" or "minmax"

        # Sobel核
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    @torch.no_grad()
    def _robust_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        归一化到[0,1]，逐样本独立；默认用 min-max（快且稳定）。
        在 AMP 下：
        - quantile 统计对半精度不稳定 -> 统一转 float32 计算后再 cast 回原 dtype
        """
        B = x.shape[0]
        x_flat = x.view(B, -1)

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
        """计算梯度幅值"""
        x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
        w_x = self.sobel_x.to(x_pad.dtype)
        w_y = self.sobel_y.to(x_pad.dtype)
        gx = F.conv2d(x_pad, w_x, stride=1, padding=0)
        gy = F.conv2d(x_pad, w_y, stride=1, padding=0)
        g = torch.sqrt(gx**2 + gy**2 + 1e-6)
        return self._robust_norm(g)

    @torch.no_grad()
    def _compute_var(self, x: torch.Tensor) -> torch.Tensor:
        """计算局部方差"""
        k = self.k
        pad = k // 2
        mu = F.avg_pool2d(
            x, kernel_size=k, stride=1, padding=pad, count_include_pad=False
        )
        var = (
            F.avg_pool2d(
                x**2, kernel_size=k, stride=1, padding=pad, count_include_pad=False
            )
            - mu**2
        )
        var = var.clamp_min_(0.0)
        return self._robust_norm(var)

    @torch.no_grad()
    def _valid_and_hole(self, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算有效区域和空洞掩码"""
        # Using strict inequality assumes invalid/saturated depths are exactly 0.0 or 1.0.
        # This is generally a safe and robust assumption for normalized depth maps.
        valid = ((d > self.z_min) & (d < self.z_max)).float()
        hole = 1.0 - valid
        return valid, hole

    @torch.no_grad()
    def _edge_consistency(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """计算 RGB-深度边缘一致性"""
        if not self.use_rgb_edge or rgb is None:
            return torch.ones_like(depth)

        if rgb.shape[1] == 3:
            gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        else:
            gray = rgb[:, :1]

        if gray.max() > 1.0:
            gray = gray / 255.0

        g_rgb = self._compute_grad(gray)
        g_dep = self._compute_grad(depth)
        return (1.0 - (g_rgb - g_dep).abs()).clamp_(0, 1)

    @torch.no_grad()
    def forward(
        self, depth_raw: torch.Tensor, rgb: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        提取深度先验。

        Args:
            depth_raw: (B, 1, H, W) 深度图
            rgb: (B, 3, H, W) RGB 图像 (可选)

        Returns:
            先验字典 {gradient, variance, valid, hole, edge_consistency}
        """
        d = depth_raw if depth_raw.dim() == 4 else depth_raw.unsqueeze(1)

        priors = {
            "gradient": self._compute_grad(d),
            "variance": self._compute_var(d),
        }
        valid, hole = self._valid_and_hole(d)
        priors["valid"] = valid
        priors["hole"] = hole
        priors["edge_consistency"] = self._edge_consistency(rgb, d)

        return priors


# =============================================================================
# 置信度预测器
# =============================================================================


class ConfidencePredictor(nn.Module):
    """模态融合置信度预测器"""

    def __init__(
        self,
        feature_dims: List[int],
        scale_keys: List[str],
        hidden_dim: int = 128,
        temp_init: float = 1.5,
        clamp_min: float = 0.05,
        clamp_max: float = 0.95,
        prior_in_channels: int = 5,
    ):
        super().__init__()
        self.feature_dims = list(feature_dims)
        self.scale_keys = list(scale_keys)
        self.hidden = int(hidden_dim)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        assert len(feature_dims) == len(
            scale_keys
        ), "Feature dimensions and scale keys must match."
        assert hidden_dim % 16 == 0, "hidden_dim must be divisible by 16"

        self.register_buffer(
            "temp", torch.tensor(float(temp_init), dtype=torch.float32)
        )

        self.proj_image = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, self.hidden, 1, bias=False),
                    nn.GroupNorm(16, self.hidden),
                    nn.GELU(),
                )
                for ch in self.feature_dims
            ]
        )
        self.proj_depth = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, self.hidden, 1, bias=False),
                    nn.GroupNorm(16, self.hidden),
                    nn.GELU(),
                )
                for ch in self.feature_dims
            ]
        )

        self.proj_prior = None
        if prior_in_channels > 0:
            self.proj_prior = nn.Sequential(
                nn.Conv2d(prior_in_channels, self.hidden, 1, bias=False),
                nn.GroupNorm(16, self.hidden),
                nn.GELU(),
            )
        self._prior_channels = prior_in_channels

        in_ch_head = self.hidden * 3 if self._prior_channels > 0 else self.hidden * 2
        self.head = nn.Sequential(
            nn.Conv2d(in_ch_head, self.hidden, 3, padding=1, bias=False),
            nn.GroupNorm(16, self.hidden),
            nn.GELU(),
            nn.Conv2d(self.hidden, self.hidden // 2, 3, padding=1, bias=False),
            nn.GroupNorm(8, self.hidden // 2),
            nn.GELU(),
            nn.Conv2d(self.hidden // 2, 1, 1),
        )
        # Warm-start stability: start with low depth confidence so that enabling
        # fusion does not immediately destroy the RGB-pretrained feature
        # distribution.
        last = self.head[-1]
        if isinstance(last, nn.Conv2d):
            if last.bias is not None:
                nn.init.constant_(last.bias, -4.0)
            nn.init.normal_(last.weight, std=1e-3)

    def set_temperature(self, t: float) -> None:
        """设置温度参数"""
        self.temp.fill_(float(t))

    def forward(
        self,
        image_features: Dict[str, torch.Tensor],
        depth_features: Dict[str, torch.Tensor],
        priors_ms: Dict[str, Dict[str, torch.Tensor]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        预测融合置信度。

        Args:
            image_features: RGB 特征字典
            depth_features: 深度特征字典
            priors_ms: 多尺度先验字典

        Returns:
            (m_maps, logits_maps) 置信度和 logits 字典
        """
        m_maps: Dict[str, torch.Tensor] = {}
        logits_maps: Dict[str, torch.Tensor] = {}

        for i, key in enumerate(self.scale_keys):
            if key not in image_features or key not in depth_features:
                continue

            img_p = self.proj_image[i](image_features[key])
            dep_p = self.proj_depth[i](depth_features[key])
            features_to_cat = [img_p, dep_p]

            prior_stack = priors_ms.get(key, {}).get("stack", None)

            if prior_stack is not None and self.proj_prior is not None:
                if prior_stack.shape[1] != self._prior_channels:
                    raise RuntimeError(
                        f"Runtime prior channels {prior_stack.shape[1]} != initialized {self._prior_channels}"
                    )
                features_to_cat.append(self.proj_prior(prior_stack))

            x = torch.cat(features_to_cat, dim=1)
            raw_logits = self.head(x)
            logits = raw_logits / self.temp.clamp(1e-6)
            m = torch.sigmoid(logits)
            # Keep m in [clamp_min, clamp_max] without hard clamping (preserves gradients).
            if self.clamp_max > self.clamp_min:
                scale = self.clamp_max - self.clamp_min
                m = m * scale + self.clamp_min
            else:
                m = m.clamp(min=self.clamp_min, max=self.clamp_max)
            m_maps[key] = m
            logits_maps[key] = logits

        return m_maps, logits_maps


# =============================================================================
# 多模态门控融合 (原 MGM)
# =============================================================================


class ModalityFusionModule(nn.Module):
    """
    多模态门控融合模块 (原 MultiModalGatedFusion/MGM)。

    功能:
    - 深度先验提取
    - 置信度预测
    - 门控融合 RGB 和深度特征
    - 温度退火
    - 熵损失和噪声掩码损失
    """

    def __init__(
        self,
        feature_dims: List[int],
        scale_keys: List[str],
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
    ):
        super().__init__()

        if prior_enabled and prior_compute_on != "full":
            assert (
                prior_compute_on in scale_keys
            ), f"`prior_compute_on` ('{prior_compute_on}') must be 'full' or one of `scale_keys` ({scale_keys})"

        self.feature_dims = list(feature_dims)
        self.scale_keys = list(scale_keys)
        self.residual_alpha = float(residual_alpha)
        self.temp_init = float(temp_init)
        self.temp_final = float(temp_final)
        self.temp_steps = int(temp_steps)
        self._cur_step = 0
        self.loss_entropy_weight = float(loss_entropy_weight)
        self.noise_mask_weight = float(noise_mask_weight)
        self.prior_enabled = prior_enabled
        self.prior_use_grad = prior_use_grad
        self.prior_use_var = prior_use_var
        self.prior_use_valid_hole = prior_use_valid_hole
        self.prior_use_rgb_edge = prior_use_rgb_edge
        self.prior_compute_on = prior_compute_on
        self.post_fuse_norm = post_fuse_norm

        self.prior_extractor = DepthPriorExtractor(
            var_kernel=prior_var_kernel,
            z_min=prior_z_min,
            z_max=prior_z_max,
            use_rgb_edge=self.prior_use_rgb_edge,
            robust_norm=robust_norm,
            robust_norm_method=robust_norm_method,
        )

        prior_ch = 0
        if self.prior_enabled:
            if self.prior_use_grad:
                prior_ch += 1
            if self.prior_use_var:
                prior_ch += 1
            if self.prior_use_valid_hole:
                prior_ch += 2
            if self.prior_use_rgb_edge:
                prior_ch += 1

        self.conf_pred = ConfidencePredictor(
            feature_dims=self.feature_dims,
            scale_keys=self.scale_keys,
            hidden_dim=hidden_dim,
            temp_init=temp_init,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            prior_in_channels=prior_ch,
        )

        self.align_image = nn.ModuleList(
            [
                nn.Identity()
                for ch in self.feature_dims
            ]
        )
        self.align_depth = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(ch, ch, 1, bias=False), nn.GroupNorm(8, ch))
                for ch in self.feature_dims
            ]
        )

        self.post_norm = (
            nn.ModuleList([nn.GroupNorm(8, ch) for ch in self.feature_dims])
            if self.post_fuse_norm
            else None
        )

        self._prior_missing_warned = False

    def _update_temperature(self) -> None:
        """在每个训练步骤中更新状态并应用温度调度。"""
        if self.training:
            if self._cur_step <= self.temp_steps and self.temp_steps > 0:
                p = float(self._cur_step) / float(self.temp_steps)
                t = self.temp_init + (self.temp_final - self.temp_init) * p
                self.conf_pred.set_temperature(t)
            self._cur_step += 1

    def _prepare_priors_ms(
        self,
        depth_raw: torch.Tensor,
        rgb_image: Optional[torch.Tensor],
        target_sizes: Dict[str, Tuple[int, int]],
        override_compute_on: Optional[str] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """准备多尺度先验"""
        if not self.prior_enabled:
            return {}

        # 若 forward 已经提供 override，则优先；否则沿用 self.prior_compute_on
        base_compute_on = (
            self.prior_compute_on
            if override_compute_on is None
            else override_compute_on
        )

        # 再次兜底（双保险）
        effective_compute_on = base_compute_on
        if effective_compute_on != "full" and effective_compute_on not in target_sizes:
            effective_compute_on = "full"

        if effective_compute_on == "full":
            compute_res_rgb = rgb_image
            compute_res_depth = depth_raw
        else:
            h0, w0 = target_sizes[effective_compute_on]
            compute_res_depth = _bilinear(depth_raw, (h0, w0))
            compute_res_rgb = (
                _bilinear(rgb_image, (h0, w0)) if rgb_image is not None else None
            )

        all_priors_single_res = self.prior_extractor(
            compute_res_depth, compute_res_rgb)

        priors_ms = {key: {} for key in target_sizes}
        for prior_name, prior_tensor in all_priors_single_res.items():
            for key, (h, w) in target_sizes.items():
                priors_ms[key][prior_name] = _bilinear(prior_tensor, (h, w))

        return priors_ms

    def forward(
        self,
        image_features: Dict[str, torch.Tensor],
        depth_features: Dict[str, torch.Tensor],
        depth_raw: torch.Tensor,
        rgb_image: Optional[torch.Tensor] = None,
        depth_noise_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        前向传播。

        Args:
            image_features: RGB 特征字典
            depth_features: 深度特征字典
            depth_raw: 原始深度图 (B, 1, H, W)
            rgb_image: 原始 RGB 图像 (可选)
            depth_noise_mask: 噪声掩码 (B, 1, H, W)

        Returns:
            (fused_features, m_maps, losses) 融合特征、置信度和损失
        """
        if self.training:
            self._update_temperature()

        target_sizes = {k: v.shape[-2:] for k, v in image_features.items()}

        # 如果当前 batch 缺失 prior_compute_on 对应 level，回退到 full
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

        priors_for_pred = {}
        if self.prior_enabled:
            for key in target_sizes:
                prior_list = []
                if self.prior_use_grad:
                    prior_list.append(priors_ms[key]["gradient"])
                if self.prior_use_var:
                    prior_list.append(priors_ms[key]["variance"])
                if self.prior_use_valid_hole:
                    prior_list.append(priors_ms[key]["valid"])
                    prior_list.append(priors_ms[key]["hole"])
                if self.prior_use_rgb_edge:
                    prior_list.append(priors_ms[key]["edge_consistency"])

                if prior_list:
                    priors_for_pred[key] = {
                        "stack": torch.cat(prior_list, dim=1)}

        m_maps, logits_maps = self.conf_pred(
            image_features, depth_features, priors_for_pred)

        fused: Dict[str, torch.Tensor] = {}
        for i, key in enumerate(self.scale_keys):
            if key not in image_features:
                continue

            img_a = self.align_image[i](image_features[key])

            if key not in depth_features or key not in m_maps:
                fused[key] = img_a
                continue

            dep_a = self.align_depth[i](depth_features[key])
            m = m_maps[key]

            if self.prior_enabled and self.prior_use_valid_hole:
                dep_a = dep_a * priors_ms[key]["valid"]

            if self.post_norm:
                # Normalize depth features only (normalizing fused RGB+D features
                # breaks RGB-pretrained weights).
                dep_a = self.post_norm[i](dep_a)
            # 严格无放大版残差
            out = m * (dep_a + self.residual_alpha * img_a) + (1.0 - m) * img_a
            fused[key] = out

        losses: Dict[str, torch.Tensor] = {}
        if self.training and m_maps:
            if self.loss_entropy_weight > 0:
                ent_terms = [
                    -(
                        m.clamp(1e-6, 1.0 - 1e-6) *
                        torch.log(m.clamp(1e-6, 1.0 - 1e-6))
                        + (1.0 - m.clamp(1e-6, 1.0 - 1e-6))
                        * torch.log(1.0 - m.clamp(1e-6, 1.0 - 1e-6))
                    ).mean()
                    for m in m_maps.values()
                ]
                losses["loss_mgm_entropy"] = (
                    self.loss_entropy_weight * torch.stack(ent_terms).mean()
                )

            # The entropy loss correctly encourages m to be binary.
            if self.noise_mask_weight > 0 and depth_noise_mask is not None:
                bces = []
                for key, m in m_maps.items():
                    logits = logits_maps.get(key, None)
                    if logits is None:
                        continue
                    # 1. 上采样噪声掩码到当前特征图尺寸
                    target_noise_mask = _bilinear(
                        depth_noise_mask.float(), m.shape[-2:]
                    )
                    # 2. 最大池化膨胀1像素
                    target_noise_mask_dilated = F.max_pool2d(
                        target_noise_mask, kernel_size=3, stride=1, padding=1
                    )
                    # 3. 使用 logits 上的 with_logits 版本（AMP安全）
                    bce = F.binary_cross_entropy_with_logits(
                        logits, 1.0 - target_noise_mask_dilated
                    )
                    bces.append(bce)

                if bces:
                    losses["loss_mgm_noise"] = (
                        self.noise_mask_weight * torch.stack(bces).mean()
                    )

        return fused, m_maps, losses
