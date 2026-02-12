# -*- coding: utf-8 -*-
"""
Modality Fusion Module (原 MGM)

多模态门控融合模块，用于融合 RGB 和深度特征。
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 深度先验提取
# =============================================================================


class DepthPriorExtractor(nn.Module):
    """
    深度先验提取 (不可学习、无梯度)。

    提取:
    - 梯度幅值 (Sobel)
    - 局部方差 (盒滤波)
    - 有效/空洞掩码
    - RGB-深度边缘一致性
    """

    def __init__(
        self,
        var_kernel: int = 5,
        z_min: float = 0.0,
        z_max: float = 1.0,
        use_rgb_edge: bool = False,
    ):
        super().__init__()
        self.k = int(var_kernel)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.use_rgb_edge = bool(use_rgb_edge)

        # Sobel 核
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    @torch.no_grad()
    def _compute_grad(self, x: torch.Tensor) -> torch.Tensor:
        """计算梯度幅值"""
        x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(x_pad, self.sobel_x.to(x_pad.dtype), stride=1, padding=0)
        gy = F.conv2d(x_pad, self.sobel_y.to(x_pad.dtype), stride=1, padding=0)
        g = torch.sqrt(gx**2 + gy**2 + 1e-6)
        return self._normalize(g)

    @torch.no_grad()
    def _compute_var(self, x: torch.Tensor) -> torch.Tensor:
        """计算局部方差"""
        k = self.k
        pad = k // 2
        mu = F.avg_pool2d(x, kernel_size=k, stride=1,
                          padding=pad, count_include_pad=False)
        var = F.avg_pool2d(x**2, kernel_size=k, stride=1,
                           padding=pad, count_include_pad=False) - mu**2
        var = var.clamp_min_(0.0)
        return self._normalize(var)

    @torch.no_grad()
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """归一化到 [0,1]"""
        B = x.shape[0]
        x_flat = x.view(B, -1)
        mn = x_flat.min(dim=1, keepdim=True).values
        mx = x_flat.max(dim=1, keepdim=True).values
        norm = (x_flat - mn) / (mx - mn + 1e-6)
        return norm.clamp_(0, 1).view_as(x)

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

        # 有效/空洞掩码
        valid = ((d > self.z_min) & (d < self.z_max)).float()
        hole = 1.0 - valid
        priors["valid"] = valid
        priors["hole"] = hole

        # 边缘一致性
        if self.use_rgb_edge and rgb is not None:
            gray = 0.299 * rgb[:, 0:1] + 0.587 * \
                rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
            if gray.max() > 1.0:
                gray = gray / 255.0
            g_rgb = self._compute_grad(gray)
            g_dep = self._compute_grad(d)
            priors["edge_consistency"] = (
                1.0 - (g_rgb - g_dep).abs()).clamp(0, 1)
        else:
            priors["edge_consistency"] = torch.ones_like(d)

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
            scale_keys), "Feature dimensions and scale keys must match"

        # 温度参数
        self.register_buffer("temp", torch.tensor(
            float(temp_init), dtype=torch.float32))

        # 投影层
        self.proj_image = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, self.hidden, 1, bias=False),
                nn.GroupNorm(16, self.hidden),
                nn.GELU(),
            )
            for ch in self.feature_dims
        ])

        self.proj_depth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, self.hidden, 1, bias=False),
                nn.GroupNorm(16, self.hidden),
                nn.GELU(),
            )
            for ch in self.feature_dims
        ])

        # 先验投影
        if prior_in_channels > 0:
            self.proj_prior = nn.Sequential(
                nn.Conv2d(prior_in_channels, self.hidden, 1, bias=False),
                nn.GroupNorm(16, self.hidden),
                nn.GELU(),
            )
        else:
            self.proj_prior = None

        # 预测头
        in_ch_head = self.hidden * 3 if self.proj_prior is not None else self.hidden * 2
        self.head = nn.Sequential(
            nn.Conv2d(in_ch_head, self.hidden, 3, padding=1, bias=False),
            nn.GroupNorm(16, self.hidden),
            nn.GELU(),
            nn.Conv2d(self.hidden, self.hidden // 2, 3, padding=1, bias=False),
            nn.GroupNorm(8, self.hidden // 2),
            nn.GELU(),
            nn.Conv2d(self.hidden // 2, 1, 1),
        )

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
        m_maps = {}
        logits_maps = {}

        for i, key in enumerate(self.scale_keys):
            if key not in image_features or key not in depth_features:
                continue

            img_p = self.proj_image[i](image_features[key])
            dep_p = self.proj_depth[i](depth_features[key])
            features_to_cat = [img_p, dep_p]

            prior_stack = priors_ms.get(key, {}).get("stack", None)
            if prior_stack is not None and self.proj_prior is not None:
                features_to_cat.append(self.proj_prior(prior_stack))

            x = torch.cat(features_to_cat, dim=1)
            raw_logits = self.head(x)
            logits = raw_logits / self.temp.clamp(1e-6)
            m = torch.sigmoid(logits)
            m_maps[key] = m.clamp(self.clamp_min, self.clamp_max)
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
        prior_compute_on: str = "res3",
        post_fuse_norm: bool = True,
    ):
        super().__init__()

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
        self.prior_compute_on = prior_compute_on
        self.prior_use_grad = prior_use_grad
        self.prior_use_var = prior_use_var
        self.prior_use_valid_hole = prior_use_valid_hole
        self.prior_use_rgb_edge = prior_use_rgb_edge
        self.post_fuse_norm = post_fuse_norm

        # 深度先验提取器
        prior_ch = 0
        if self.prior_enabled:
            self.prior_extractor = DepthPriorExtractor(
                var_kernel=prior_var_kernel,
                z_min=prior_z_min,
                z_max=prior_z_max,
                use_rgb_edge=prior_use_rgb_edge,
            )
            if prior_use_grad:
                prior_ch += 1
            if prior_use_var:
                prior_ch += 1
            if prior_use_valid_hole:
                prior_ch += 2
            if prior_use_rgb_edge:
                prior_ch += 1

        # 置信度预测器
        self.conf_pred = ConfidencePredictor(
            feature_dims=self.feature_dims,
            scale_keys=self.scale_keys,
            hidden_dim=hidden_dim,
            temp_init=temp_init,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            prior_in_channels=prior_ch,
        )

        # 对齐层
        self.align_image = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 1, bias=False),
                nn.GroupNorm(8, ch),
            )
            for ch in self.feature_dims
        ])

        self.align_depth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 1, bias=False),
                nn.GroupNorm(8, ch),
            )
            for ch in self.feature_dims
        ])

        # 后归一化
        if self.post_fuse_norm:
            self.post_norm = nn.ModuleList(
                [nn.GroupNorm(8, ch) for ch in self.feature_dims])
        else:
            self.post_norm = None

    def _update_temperature(self) -> None:
        """在每个训练步骤中更新温度"""
        if self.training:
            if self._cur_step <= self.temp_steps and self.temp_steps > 0:
                p = float(self._cur_step) / float(self.temp_steps)
                t = self.temp_init + (self.temp_final - self.temp_init) * p
                self.conf_pred.set_temperature(t)
            self._cur_step += 1

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

        # 准备多尺度先验
        priors_ms = self._prepare_priors_ms(depth_raw, rgb_image, target_sizes)

        # 准备先验用于预测
        priors_for_pred = {}
        if self.prior_enabled:
            for key in target_sizes:
                prior_list = []
                # 梯度
                if self.prior_use_grad and "gradient" in priors_ms[key]:
                    prior_list.append(priors_ms[key]["gradient"])
                # 方差
                if self.prior_use_var and "variance" in priors_ms[key]:
                    prior_list.append(priors_ms[key]["variance"])
                # 有效/空洞
                if self.prior_use_valid_hole and "valid" in priors_ms[key] and "hole" in priors_ms[key]:
                    prior_list.append(priors_ms[key]["valid"])
                    prior_list.append(priors_ms[key]["hole"])
                # 边缘一致性
                if self.prior_use_rgb_edge and "edge_consistency" in priors_ms[key]:
                    prior_list.append(priors_ms[key]["edge_consistency"])

                if prior_list:
                    priors_for_pred[key] = {
                        "stack": torch.cat(prior_list, dim=1)}

        # 预测置信度
        m_maps, logits_maps = self.conf_pred(
            image_features, depth_features, priors_for_pred)

        # 融合特征
        fused = {}
        for i, key in enumerate(self.scale_keys):
            if key not in image_features:
                continue

            img_a = self.align_image[i](image_features[key])

            if key not in depth_features or key not in m_maps:
                fused[key] = self.post_norm[i](
                    img_a) if self.post_norm else img_a
                continue

            dep_a = self.align_depth[i](depth_features[key])
            m = m_maps[key]

            # 有效区域掩码
            if self.prior_enabled and "valid" in priors_ms[key]:
                dep_a = dep_a * priors_ms[key]["valid"]

            # 门控融合
            fused_base = m * dep_a + (1.0 - m) * img_a
            out = m * (dep_a + self.residual_alpha * img_a) + (1.0 - m) * img_a

            if self.post_norm:
                out = self.post_norm[i](out)
            fused[key] = out

        # 计算损失
        losses = {}
        if self.training and m_maps:
            # 熵损失
            if self.loss_entropy_weight > 0:
                ent_terms = [
                    -(
                        m.clamp(1e-6, 1.0 - 1e-6) *
                        torch.log(m.clamp(1e-6, 1.0 - 1e-6)) +
                        (1.0 - m.clamp(1e-6, 1.0 - 1e-6)) *
                        torch.log(1.0 - m.clamp(1e-6, 1.0 - 1e-6))
                    ).mean()
                    for m in m_maps.values()
                ]
                losses["loss_mgm_entropy"] = self.loss_entropy_weight * \
                    torch.stack(ent_terms).mean()

            # 噪声掩码损失
            if self.noise_mask_weight > 0 and depth_noise_mask is not None:
                bces = []
                for key, m in m_maps.items():
                    logits = logits_maps.get(key, None)
                    if logits is None:
                        continue
                    target_noise_mask = F.interpolate(
                        depth_noise_mask.float(), m.shape[-2:], mode="bilinear", align_corners=False
                    )
                    target_noise_mask = F.max_pool2d(
                        target_noise_mask, kernel_size=3, stride=1, padding=1)
                    bce = F.binary_cross_entropy_with_logits(
                        logits, 1.0 - target_noise_mask)
                    bces.append(bce)

                if bces:
                    losses["loss_mgm_noise"] = self.noise_mask_weight * \
                        torch.stack(bces).mean()

        return fused, m_maps, losses

    def _prepare_priors_ms(
        self,
        depth_raw: torch.Tensor,
        rgb_image: Optional[torch.Tensor],
        target_sizes: Dict[str, Tuple[int, int]],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """准备多尺度先验"""
        if not self.prior_enabled:
            return {}

        # 确定计算分辨率
        compute_on = self.prior_compute_on
        if compute_on != "full" and compute_on not in target_sizes:
            compute_on = "full"

        if compute_on == "full":
            compute_res_depth = depth_raw
            compute_res_rgb = rgb_image
        else:
            h0, w0 = target_sizes[compute_on]
            compute_res_depth = F.interpolate(
                depth_raw, (h0, w0), mode="bilinear", align_corners=False)
            compute_res_rgb = (
                F.interpolate(rgb_image, (h0, w0),
                              mode="bilinear", align_corners=False)
                if rgb_image is not None else None
            )

        # 提取先验
        all_priors_single_res = self.prior_extractor(
            compute_res_depth, compute_res_rgb)

        # 上采样到各尺度
        priors_ms = {key: {} for key in target_sizes}
        for prior_name, prior_tensor in all_priors_single_res.items():
            for key, (h, w) in target_sizes.items():
                # prior_tensor 可能是 (B, 1, H, W) 或 (B, H, W)
                if prior_tensor.dim() == 3:
                    prior_tensor_4d = prior_tensor.unsqueeze(1)
                else:
                    prior_tensor_4d = prior_tensor
                resized = F.interpolate(
                    prior_tensor_4d, (h, w), mode="bilinear", align_corners=False)
                # 保持 (B, 1, H, W) 格式用于后续 torch.cat(dim=1)
                priors_ms[key][prior_name] = resized

        return priors_ms
