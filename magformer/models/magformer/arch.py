# -*- coding: utf-8 -*-
"""
MAGFormer Architecture

纯 PyTorch 实现的 MAGFormer 主架构。
整合双骨干网络、模态融合和 Transformer 解码器。
"""

from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MagFormerArch(nn.Module):
    """
    MAGFormer 主架构。

    架构:
        1. RGB Backbone (Swin Transformer)
        2. Depth Backbone (ConvNeXt)
        3. Modality Fusion Module
        4. Pixel Decoder + Transformer Decoder

    输入:
        - RGB 图像: (B, 3, H, W)
        - Depth 图像: (B, 1, H, W)

    输出:
        - pred_logits: (B, N_queries, C) 类别预测
        - pred_masks: (B, N_queries, H, W) 掩码预测
    """

    def __init__(
        self,
        rgb_backbone: nn.Module,
        depth_backbone: nn.Module,
        fusion_module: nn.Module,
        pixel_decoder: nn.Module,
        transformer_decoder: nn.Module,
        num_classes: int = 1,
        num_queries: int = 100,
        hidden_dim: int = 256,
        pixel_mean: List[float] = [123.675, 116.280, 103.530],
        pixel_std: List[float] = [58.395, 57.120, 57.375],
        size_divisibility: int = 32,
    ):
        """
        Args:
            rgb_backbone: RGB 骨干网络
            depth_backbone: 深度骨干网络
            fusion_module: 模态融合模块
            pixel_decoder: 像素解码器
            transformer_decoder: Transformer 解码器
            num_classes: 类别数
            num_queries: 对象查询数
            hidden_dim: 隐藏维度
            pixel_mean: RGB 均值 (归一化)
            pixel_std: RGB 标准差 (归一化)
            size_divisibility: 尺寸整除因子
        """
        super().__init__()

        self.rgb_backbone = rgb_backbone
        self.depth_backbone = depth_backbone
        self.fusion = fusion_module
        self.pixel_decoder = pixel_decoder
        self.decoder = transformer_decoder
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.size_divisibility = size_divisibility
        # Ablation / debug switches (wired from config in `from_config`).
        self.modality_fusion_enabled: bool = True
        self.depth_backbone_enabled: bool = True

        # 归一化参数
        self.register_buffer("pixel_mean", torch.tensor(
            pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(
            pixel_std).view(-1, 1, 1), False)

        from ..common import HungarianMatcher, SetCriterion

        self.criterion = SetCriterion(
            num_classes=num_classes,
            matcher=HungarianMatcher(
                cost_class=1.0,
                cost_mask=5.0,
                cost_dice=1.0,
                num_points=12544,
            ),
            weight_dict={
                "loss_ce": 1.0,
                "loss_mask": 5.0,
                "loss_dice": 1.0,
            },
            eos_coef=0.1,
            losses=("labels", "masks"),
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
        )

    @classmethod
    def from_config(cls, config: Any) -> "MagFormerArch":
        from ..common import (
            SwinTransformer,
            D2SwinBackbone,
            ConvNeXtDepth,
            SimplePixelDecoder,
            SimpleTransformerDecoder,
            MSDeformAttnPixelDecoder,
            MultiScaleMaskedTransformerDecoder,
        )
        from .fusion import ModalityFusionModule

        # Support both call styles:
        # - full config: cfg.model.magformer.*
        # - legacy nested model config directly: cfg.*
        if hasattr(config, "model") and getattr(getattr(config, "model"), "magformer", None) is not None:
            root_cfg = config
            model_cfg = config.model.magformer
        else:
            root_cfg = config
            model_cfg = config

        use_rgb_pretrained = model_cfg.rgb_backbone.pretrained and model_cfg.rgb_backbone.weights is None
        swin_backend = getattr(model_cfg.swin, "backend", "d2")
        if swin_backend == "d2":
            rgb_backbone = D2SwinBackbone(
                embed_dim=model_cfg.swin.embed_dim,
                depths=model_cfg.swin.depths,
                num_heads=model_cfg.swin.num_heads,
                window_size=model_cfg.swin.window_size,
                drop_path_rate=model_cfg.swin.drop_path_rate,
                out_features=model_cfg.swin.out_features,
                pretrained=use_rgb_pretrained,
                weights_path=model_cfg.rgb_backbone.weights,
                img_size=model_cfg.swin.pretrain_img_size,
            )
        else:
            rgb_backbone = SwinTransformer(
                embed_dim=model_cfg.swin.embed_dim,
                depths=model_cfg.swin.depths,
                num_heads=model_cfg.swin.num_heads,
                window_size=model_cfg.swin.window_size,
                drop_path_rate=model_cfg.swin.drop_path_rate,
                out_features=model_cfg.swin.out_features,
                pretrained=use_rgb_pretrained,
                weights_path=model_cfg.rgb_backbone.weights,
                img_size=model_cfg.swin.pretrain_img_size,
            )

        depth_out_features = getattr(
            model_cfg.convnext, "out_features", model_cfg.swin.out_features)
        use_depth_pretrained = model_cfg.depth_backbone.pretrained and model_cfg.depth_backbone.weights is None
        depth_backbone = ConvNeXtDepth(
            depths=model_cfg.convnext.depths,
            dims=model_cfg.convnext.dims,
            drop_path_rate=model_cfg.convnext.drop_path_rate,
            layer_scale=model_cfg.convnext.layer_scale,
            out_features=depth_out_features,
            pretrained=use_depth_pretrained,
            weights_path=model_cfg.depth_backbone.weights,
        )

        robust_norm_enabled_raw = getattr(model_cfg.modality_fusion, "robust_norm_enabled", None)
        if robust_norm_enabled_raw is None:
            robust_norm_enabled = getattr(model_cfg.modality_fusion.prior, "robust_norm", True)
        else:
            robust_norm_enabled = robust_norm_enabled_raw

        robust_norm_method_raw = getattr(model_cfg.modality_fusion, "robust_norm_method", None)
        if robust_norm_enabled_raw is None and hasattr(model_cfg.modality_fusion.prior, "robust_norm_method"):
            robust_norm_method = getattr(model_cfg.modality_fusion.prior, "robust_norm_method")
        elif robust_norm_method_raw is None:
            robust_norm_method = getattr(model_cfg.modality_fusion.prior, "robust_norm_method", "minmax")
        else:
            robust_norm_method = robust_norm_method_raw

        fusion = ModalityFusionModule(
            feature_dims=model_cfg.modality_fusion.feature_dims,
            scale_keys=model_cfg.modality_fusion.scale_keys,
            residual_alpha=model_cfg.modality_fusion.residual_alpha,
            temp_init=model_cfg.modality_fusion.temp_init,
            temp_final=model_cfg.modality_fusion.temp_final,
            temp_steps=model_cfg.modality_fusion.temp_steps,
            clamp_min=model_cfg.modality_fusion.clamp_min,
            clamp_max=model_cfg.modality_fusion.clamp_max,
            loss_entropy_weight=model_cfg.modality_fusion.loss_entropy_w,
            noise_mask_weight=model_cfg.modality_fusion.noise_mask_weight,
            hidden_dim=model_cfg.modality_fusion.hidden_dim,
            prior_enabled=model_cfg.modality_fusion.prior.enabled,
            prior_use_grad=model_cfg.modality_fusion.prior.use_gradient,
            prior_use_var=model_cfg.modality_fusion.prior.use_variance,
            prior_use_valid_hole=model_cfg.modality_fusion.prior.use_valid_hole,
            prior_use_rgb_edge=model_cfg.modality_fusion.prior.use_rgb_edge,
            prior_var_kernel=model_cfg.modality_fusion.prior.var_kernel,
            prior_z_min=model_cfg.modality_fusion.prior.z_min,
            prior_z_max=model_cfg.modality_fusion.prior.z_max,
            robust_norm=bool(robust_norm_enabled),
            robust_norm_method=str(robust_norm_method),
            prior_compute_on=model_cfg.modality_fusion.prior.compute_on,
            post_fuse_norm=model_cfg.modality_fusion.post_fuse_norm,
        )

        in_channels = rgb_backbone._stage_out_channels

        pixel_decoder_name = getattr(model_cfg.sem_seg_head, "pixel_decoder_name", "SimplePixelDecoder")
        transformer_decoder_name = getattr(model_cfg.mask_former, "transformer_decoder_name", "SimpleTransformerDecoder")

        if pixel_decoder_name not in {"SimplePixelDecoder", "MSDeformAttnPixelDecoder"}:
            raise ValueError(f"Unsupported pixel decoder: {pixel_decoder_name}")

        if transformer_decoder_name not in {"SimpleTransformerDecoder", "MultiScaleMaskedTransformerDecoder"}:
            raise ValueError(f"Unsupported transformer decoder: {transformer_decoder_name}")

        if pixel_decoder_name == "MSDeformAttnPixelDecoder":
            transformer_in_features = getattr(
                model_cfg.sem_seg_head,
                "deformable_transformer_encoder_in_features",
                model_cfg.sem_seg_head.in_features,
            )
            dpe_cfg = getattr(model_cfg, "dpe", None)
            dpe_enabled_raw = getattr(dpe_cfg, "enabled", None)
            if dpe_enabled_raw is None:
                dpe_enabled_raw = getattr(root_cfg, "dpe_enabled", False)
            dpe_beta_raw = getattr(dpe_cfg, "beta", None)
            if dpe_beta_raw is None:
                dpe_beta_raw = getattr(root_cfg, "dpe_beta", 10.0)
            dpe_enabled = bool(dpe_enabled_raw)
            dpe_beta = float(dpe_beta_raw)
            pixel_decoder = MSDeformAttnPixelDecoder(
                in_features=model_cfg.sem_seg_head.in_features,
                in_channels=in_channels,
                transformer_in_features=transformer_in_features,
                hidden_dim=model_cfg.mask_former.hidden_dim,
                mask_dim=model_cfg.sem_seg_head.mask_dim,
                transformer_dropout=model_cfg.mask_former.dropout,
                transformer_nheads=model_cfg.mask_former.nheads,
                transformer_dim_feedforward=int(
                    getattr(
                        model_cfg.sem_seg_head,
                        "transformer_dim_feedforward",
                        1024,
                    )
                ),
                # Align with Mask2Former: SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS
                transformer_enc_layers=int(getattr(model_cfg.sem_seg_head, "transformer_enc_layers", 0)),
                common_stride=model_cfg.sem_seg_head.common_stride,
                dpe_enabled=dpe_enabled,
                dpe_beta=dpe_beta,
            )
        else:
            pixel_decoder = SimplePixelDecoder(
                in_features=model_cfg.sem_seg_head.in_features,
                in_channels=in_channels,
                hidden_dim=model_cfg.mask_former.hidden_dim,
                mask_dim=model_cfg.sem_seg_head.mask_dim,
            )

        # Align with Mask2Former/MGM semantics:
        # config dec_layers includes the initial learnable-query prediction.
        # Actual transformer decoder layers = dec_layers - 1.
        dec_layers_cfg = int(model_cfg.mask_former.dec_layers)
        if dec_layers_cfg < 1:
            raise ValueError(f"mask_former.dec_layers must be >= 1, got {dec_layers_cfg}")
        decoder_num_layers = dec_layers_cfg - 1

        if transformer_decoder_name == "MultiScaleMaskedTransformerDecoder":
            # Keep parity with original Mask2Former/MGM implementation:
            # decoder layers are instantiated with dropout=0.0.
            transformer_decoder = MultiScaleMaskedTransformerDecoder(
                num_queries=model_cfg.mask_former.num_object_queries,
                hidden_dim=model_cfg.mask_former.hidden_dim,
                nheads=model_cfg.mask_former.nheads,
                dim_feedforward=model_cfg.mask_former.dim_feedforward,
                num_layers=decoder_num_layers,
                num_classes=model_cfg.sem_seg_head.num_classes,
                mask_dim=model_cfg.sem_seg_head.mask_dim,
                dropout=0.0,
                pre_norm=model_cfg.mask_former.pre_norm,
                enforce_input_project=False,
                num_feature_levels=3,
            )
        else:
            transformer_decoder = SimpleTransformerDecoder(
                num_queries=model_cfg.mask_former.num_object_queries,
                hidden_dim=model_cfg.mask_former.hidden_dim,
                nheads=model_cfg.mask_former.nheads,
                dim_feedforward=model_cfg.mask_former.dim_feedforward,
                num_layers=decoder_num_layers,
                num_classes=model_cfg.sem_seg_head.num_classes,
                mask_dim=model_cfg.sem_seg_head.mask_dim,
                dropout=model_cfg.mask_former.dropout,
            )

        model = cls(
            rgb_backbone=rgb_backbone,
            depth_backbone=depth_backbone,
            fusion_module=fusion,
            pixel_decoder=pixel_decoder,
            transformer_decoder=transformer_decoder,
            num_classes=model_cfg.sem_seg_head.num_classes,
            num_queries=model_cfg.mask_former.num_object_queries,
            hidden_dim=model_cfg.mask_former.hidden_dim,
            pixel_mean=model_cfg.pixel_mean,
            pixel_std=model_cfg.pixel_std,
            size_divisibility=model_cfg.sem_seg_head.common_stride,
        )
        model.modality_fusion_enabled = bool(getattr(model_cfg.modality_fusion, "enabled", True))
        model.depth_backbone_enabled = bool(getattr(model_cfg.depth_backbone, "enabled", True))
        model._sync_criterion_from_config(model_cfg)
        return model

    def _sync_criterion_from_config(self, config: Any) -> None:
        from ..common import HungarianMatcher, SetCriterion

        mask_former = config.mask_former
        class_w = float(mask_former.class_weight)
        mask_w = float(mask_former.mask_weight)
        dice_w = float(mask_former.dice_weight)
        eos_coef = float(mask_former.no_object_weight)
        num_points = int(mask_former.train_num_points)

        matcher = HungarianMatcher(
            cost_class=class_w,
            cost_mask=mask_w,
            cost_dice=dice_w,
            num_points=num_points,
        )

        weight_dict = {
            "loss_ce": class_w,
            "loss_mask": mask_w,
            "loss_dice": dice_w,
        }
        if getattr(mask_former, "deep_supervision", False):
            num_aux = max(int(mask_former.dec_layers) - 1, 0)
            for i in range(num_aux):
                weight_dict.update({
                    f"loss_ce_{i}": class_w,
                    f"loss_mask_{i}": mask_w,
                    f"loss_dice_{i}": dice_w,
                })

        self.criterion = SetCriterion(
            num_classes=self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=eos_coef,
            losses=("labels", "masks"),
            num_points=num_points,
            oversample_ratio=float(mask_former.oversample_ratio),
            importance_sample_ratio=float(mask_former.importance_sample_ratio),
            balanced_ce=bool(getattr(mask_former, "balanced_ce", False)),
            balanced_ce_min_fg_ratio=float(getattr(mask_former, "balanced_ce_min_fg_ratio", 0.01)),
        )

    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device

    def forward(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        targets: Optional[List[Dict[str, Any]]] = None,
        padding_masks: Optional[torch.Tensor] = None,
        depth_noise_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播。

        Args:
            images: (B, 3, H, W) RGB 图像
            depths: (B, 1, H, W) 深度图
            targets: 目标列表 (训练时)
            padding_masks: (B, H, W) 填充掩码，True 表示 padding 区域
            depth_noise_masks: (B, 1, H, W) 深度噪声掩码（可选）

        Returns:
            训练时返回损失字典，推理时返回预测字典
        """
        B, _, H, W = images.shape

        # 归一化输入
        images_norm = (images - self.pixel_mean) / self.pixel_std

        # 提取多尺度特征
        rgb_features = self.rgb_backbone(images_norm)  # Dict[str, Tensor]
        fusion_enabled = bool(getattr(self, "modality_fusion_enabled", True)) and bool(
            getattr(self, "depth_backbone_enabled", True)
        )

        if fusion_enabled:
            depth_features = self.depth_backbone(depths)  # Dict[str, Tensor]
            fused_features, confidence_maps, fusion_losses = self.fusion(
                image_features=rgb_features,
                depth_features=depth_features,
                depth_raw=depths,
                rgb_image=images_norm,
                depth_noise_mask=depth_noise_masks,
            )
        else:
            fused_features = rgb_features
            confidence_maps = None
            fusion_losses = {}

        decoder_inputs = self.pixel_decoder(
            features=fused_features,
            confidence_maps=confidence_maps,
            depth_raw=depths,
            padding_mask=padding_masks,
        )

        # 获取深度调制位置编码 (pos_key_list) 用于 key_pos
        pos_key_list = decoder_inputs.get("pos_key_list", None)

        outputs = self.decoder(
            memory=decoder_inputs["memory"],
            mask_features=decoder_inputs["mask_features"],
            multi_scale_features=decoder_inputs.get("multi_scale_features", None),
            multi_scale_pos=decoder_inputs.get("multi_scale_pos", None),
            pos_key=pos_key_list,
        )

        if self.training:
            if targets is None:
                return {"total_loss": torch.tensor(0.0, device=self.device)}

            processed_targets = self._prepare_targets(targets)
            losses = self.criterion(outputs, processed_targets)
            if fusion_losses:
                losses.update(fusion_losses)
                fusion_total = None
                for value in fusion_losses.values():
                    if torch.is_tensor(value):
                        fusion_total = value if fusion_total is None else fusion_total + value
                if fusion_total is not None:
                    losses["total_loss"] = losses["total_loss"] + fusion_total
            return losses
        else:
            # 推理模式: 后处理预测
            return self._inference(outputs, images.shape)

    @torch.no_grad()
    def collect_preflight_diagnostics(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        padding_masks: Optional[torch.Tensor] = None,
        depth_noise_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        images_norm = (images - self.pixel_mean) / self.pixel_std
        rgb_features = self.rgb_backbone(images_norm)
        fusion_enabled = bool(getattr(self, "modality_fusion_enabled", True)) and bool(
            getattr(self, "depth_backbone_enabled", True)
        )

        if fusion_enabled:
            depth_features = self.depth_backbone(depths)
            fused_features, confidence_maps, _ = self.fusion(
                image_features=rgb_features,
                depth_features=depth_features,
                depth_raw=depths,
                rgb_image=images_norm,
                depth_noise_mask=depth_noise_masks,
            )
        else:
            fused_features = rgb_features
            confidence_maps = None

        decoder_inputs = self.pixel_decoder(
            features=fused_features,
            confidence_maps=confidence_maps,
            depth_raw=depths,
            padding_mask=padding_masks,
        )
        outputs = self.decoder(
            memory=decoder_inputs["memory"],
            mask_features=decoder_inputs["mask_features"],
            multi_scale_features=decoder_inputs.get("multi_scale_features", None),
            multi_scale_pos=decoder_inputs.get("multi_scale_pos", None),
            pos_key=decoder_inputs.get("pos_key_list", None),
        )
        return {
            "confidence_maps": confidence_maps,
            "pred_masks": outputs.get("pred_masks"),
        }

    def _prepare_targets(
        self,
        targets: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        prepared = []
        for target in targets:
            labels = target.get("labels", torch.zeros(0, dtype=torch.long, device=self.device)).long()
            if labels.numel() > 0 and labels.min().item() >= 1:
                labels = labels - 1
            labels = labels.clamp(min=0, max=max(self.num_classes - 1, 0))

            masks = target.get("masks", torch.zeros(0, device=self.device)).float()
            if masks.ndim == 2:
                masks = masks.unsqueeze(0)

            prepared_target = {
                "labels": labels,
                "masks": masks,
            }
            prepared.append(prepared_target)
        return prepared

    @torch.no_grad()
    def _inference(
        self,
        outputs: Dict[str, torch.Tensor],
        image_shape: Tuple[int, ...],
    ) -> Dict[str, Any]:
        """
        推理后处理。

        Args:
            outputs: 解码器输出
            image_shape: 原始图像形状

        Returns:
            预测结果字典
        """
        pred_logits = outputs.get("pred_logits", None)
        pred_masks = outputs.get("pred_masks", None)

        if pred_logits is None or pred_masks is None:
            return {}

        B, Nq, _ = pred_logits.shape
        H_img, W_img = image_shape[-2:]

        if pred_masks.shape[-2:] != (H_img, W_img):
            pred_masks = F.interpolate(pred_masks, size=(
                H_img, W_img), mode="bilinear", align_corners=False)

        class_scores = F.softmax(pred_logits, dim=-1)[..., :-1]
        num_classes = class_scores.shape[-1]
        topk = min(100, Nq * max(num_classes, 1))
        top_scores, top_indices = class_scores.flatten(1).topk(topk, dim=1)

        labels = torch.arange(num_classes, device=pred_logits.device).unsqueeze(0).repeat(Nq, 1).flatten(0, 1)

        batch_predictions = []
        for i in range(B):
            query_indices = top_indices[i] // max(num_classes, 1)
            class_indices = labels[top_indices[i]] if num_classes > 0 else torch.zeros_like(query_indices)
            masks = pred_masks[i, query_indices]
            mask_probs = masks.sigmoid()
            binary_masks = (mask_probs > 0.5).float()
            mask_scores = (mask_probs.flatten(1) * binary_masks.flatten(1)).sum(1) / (binary_masks.flatten(1).sum(1) + 1e-6)
            final_scores = top_scores[i] * mask_scores
            batch_pred = {
                "image_id": i,
                "scores": final_scores.detach().cpu().numpy(),
                "category_ids": class_indices.detach().cpu().numpy(),
                "masks": mask_probs.detach().cpu().numpy(),  # Return probabilities, not logits
            }
            batch_predictions.append(batch_pred)

        return {
            "predictions": batch_predictions,
            "pred_logits": pred_logits.detach().cpu(),
            "pred_masks": pred_masks.detach().cpu(),
        }


def build_magformer(config: Dict[str, Any]) -> MagFormerArch:
    """兼容旧接口，转发到 from_config。"""
    return MagFormerArch.from_config(config)
