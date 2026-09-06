# -*- coding: utf-8 -*-
"""
MAGFormer Architecture

纯 PyTorch 实现的 MAGFormer 主架构。
整合双骨干网络、模态融合和 Transformer 解码器。
"""

import inspect
import math
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.box_ops import masks_to_boxes_cxcywh


class MagFormerArch(nn.Module):
    """
    MAGFormer 主架构。

    This project currently supports exactly one foreground class. Multi-class is not implemented.

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
        rgb_backbone: Optional[nn.Module],
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
        inference_topk: int = 100,
        depth_only: bool = False,
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
            inference_topk: 推理时保留的top-k预测数量 (默认100)
        """
        super().__init__()

        if int(num_classes) != 1:
            raise ValueError(
                "This project currently supports exactly one foreground class. Multi-class is not implemented. "
                f"Set num_classes=1 (got {num_classes})."
            )

        self.rgb_backbone = rgb_backbone
        self.depth_backbone = depth_backbone
        # depth_only ablation: skip RGB backbone + fusion forward path
        self.depth_only = bool(depth_only)
        self.fusion = fusion_module
        self.pixel_decoder = pixel_decoder
        self.decoder = transformer_decoder
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.size_divisibility = size_divisibility
        self.inference_topk = inference_topk
        # Ablation / debug switches (wired from config in `from_config`).
        self.modality_fusion_enabled: bool = True
        self.depth_backbone_enabled: bool = True
        self.residual_depth_fusion_enabled: bool = False

        # 归一化参数
        self.register_buffer("pixel_mean", torch.tensor(
            pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(
            pixel_std).view(-1, 1, 1), False)

        from ..common.matcher import HungarianMatcher
        from ..common.criterion import SetCriterion

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
            losses=("labels", "masks", "boxes"),
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
        )

        # AGPE module (wired in from_config)
        self.agpe_enabled = False
        self.agpe = None

        # DN-DETR query denoising (wired in from_config)
        self.dn_enabled = False
        self.dn_scalar = 5
        self.dn_box_noise_scale = 0.4
        self.dn_label_noise_ratio = 0.2
        self.dn_loss_weight = 1.0
        self.dn_contrastive_weight = 0.5

    def on_successful_optimizer_step(self) -> None:
        """Advance model-owned schedules after parameters were actually updated."""
        hook = getattr(self.fusion, "advance_optimizer_step", None)
        if callable(hook):
            hook()

    @classmethod
    def from_config(cls, config: Any) -> "MagFormerArch":
        from ..common import (
            SwinTransformer,
            D2SwinBackbone,
            ResidualDepthSwinBackbone,
            LightDepthPyramid,
            build_depth_backbone,
            SimplePixelDecoder,
            SimpleTransformerDecoder,
            MSDeformAttnPixelDecoder,
            MultiScaleMaskedTransformerDecoder,
        )
        from .fusion import ModalityFusionModule

        # Support both call styles:
        # - full config: cfg.model.magformer.*
        # - legacy nested model config directly: cfg.*
        full_model_cfg = None
        if hasattr(config, "model") and getattr(getattr(config, "model"), "magformer", None) is not None:
            full_model_cfg = config.model
            model_cfg = config.model.magformer
        else:
            model_cfg = config

        num_classes = int(getattr(model_cfg.sem_seg_head, "num_classes", 1))
        if num_classes != 1:
            raise ValueError(
                "This project currently supports exactly one foreground class. Multi-class is not implemented. "
                f"Set model.magformer.sem_seg_head.num_classes=1 (got {num_classes})."
            )

        depth_only = bool(
            getattr(model_cfg, "depth_only", False)
            or getattr(config, "depth_only", False)
        )

        if depth_only:
            rgb_backbone = None
            use_rgb_pretrained = False
            swin_backend = getattr(model_cfg.swin, "backend", "d2")
        else:
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
                    use_checkpoint=getattr(model_cfg.swin, "use_checkpoint", False),
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

        depth_mode = str(getattr(model_cfg, "depth_mode", "legacy"))
        configured_fuse_scales = list(
            getattr(model_cfg.modality_fusion, "fuse_scales", None) or [])
        depth_out_features = list(
            getattr(model_cfg.depth_backbone, "out_features", None)
            or (
                configured_fuse_scales
                if depth_mode != "legacy" and configured_fuse_scales
                else getattr(model_cfg.convnext, "out_features", model_cfg.swin.out_features)
            )
        )
        residual_depth_fusion_cfg = getattr(
            model_cfg, "residual_depth_fusion", None
        )
        residual_depth_fusion_enabled = bool(
            getattr(residual_depth_fusion_cfg, "enabled", False)
        )
        if (
            residual_depth_fusion_enabled
            and str(residual_depth_fusion_cfg.encoder) == "light_depth_pyramid"
        ):
            depth_backbone = LightDepthPyramid()
        else:
            depth_backbone = build_depth_backbone(
                depth_mode=depth_mode,
                backbone_cfg=model_cfg.depth_backbone,
                convnext_cfg=getattr(model_cfg, "convnext", None),
                default_out_features=depth_out_features,
            )

        robust_norm_enabled_raw = getattr(
            model_cfg.modality_fusion, "robust_norm_enabled", None)
        if robust_norm_enabled_raw is None:
            robust_norm_enabled = getattr(
                model_cfg.modality_fusion.prior, "robust_norm", True)
        else:
            robust_norm_enabled = robust_norm_enabled_raw

        robust_norm_method_raw = getattr(
            model_cfg.modality_fusion, "robust_norm_method", None)
        if robust_norm_enabled_raw is None and hasattr(model_cfg.modality_fusion.prior, "robust_norm_method"):
            robust_norm_method = getattr(
                model_cfg.modality_fusion.prior, "robust_norm_method")
        elif robust_norm_method_raw is None:
            robust_norm_method = getattr(
                model_cfg.modality_fusion.prior, "robust_norm_method", "minmax")
        else:
            robust_norm_method = robust_norm_method_raw

        configured_priors = getattr(model_cfg.modality_fusion, "priors", None)
        if configured_priors is None:
            prior_use_grad = model_cfg.modality_fusion.prior.use_gradient
            prior_use_var = model_cfg.modality_fusion.prior.use_variance
            prior_use_valid_hole = model_cfg.modality_fusion.prior.use_valid_hole
            prior_use_rgb_edge = model_cfg.modality_fusion.prior.use_rgb_edge
            resolved_prior_names = []
        else:
            normalized_priors = {str(name).strip().lower()
                                 for name in configured_priors}
            prior_use_grad = bool({"edge", "gradient"} & normalized_priors)
            prior_use_var = bool({"variance", "var"} & normalized_priors)
            prior_use_valid_hole = bool(
                {"valid-hole", "valid_hole", "valid", "hole"} & normalized_priors
            )
            prior_use_rgb_edge = bool(
                {"rgb-edge", "rgb_edge", "edge_consistency"} & normalized_priors
            )
            resolved_prior_names = list(configured_priors)

        fusion_scale_keys = list(model_cfg.modality_fusion.scale_keys)
        # depth_only: rgb_backbone is None; fall back to depth channels so the
        # (unused) fusion module can still be constructed without error.
        _channels_source = (
            depth_backbone.output_shape
            if depth_only or rgb_backbone is None
            else rgb_backbone.output_shape
        )
        rgb_channels = {
            name: int(shape[0]) for name, shape in _channels_source.items()}
        depth_channels = {
            name: int(shape[0]) for name, shape in getattr(depth_backbone, "output_shape", {}).items()
        }
        _ablation_zero = list(getattr(getattr(model_cfg.modality_fusion, "prior", {}), "ablation_zero", []) or [])
        fusion = ModalityFusionModule(
            image_feature_dims=[rgb_channels[key]
                                for key in fusion_scale_keys],
            depth_feature_dims=[depth_channels.get(
                key, rgb_channels[key]) for key in fusion_scale_keys],
            scale_keys=fusion_scale_keys,
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
            prior_use_grad=prior_use_grad,
            prior_use_var=prior_use_var,
            prior_use_valid_hole=prior_use_valid_hole,
            prior_use_rgb_edge=prior_use_rgb_edge,
            prior_var_kernel=model_cfg.modality_fusion.prior.var_kernel,
            prior_z_min=model_cfg.modality_fusion.prior.z_min,
            prior_z_max=model_cfg.modality_fusion.prior.z_max,
            robust_norm=bool(robust_norm_enabled),
            robust_norm_method=str(robust_norm_method),
            prior_compute_on=model_cfg.modality_fusion.prior.compute_on,
            post_fuse_norm=model_cfg.modality_fusion.post_fuse_norm,
            mode=getattr(model_cfg.modality_fusion, "mode", "legacy_gated"),
            fuse_scales=list(getattr(model_cfg.modality_fusion,
                             "fuse_scales", None) or fusion_scale_keys),
            prior_names=resolved_prior_names,
            cross_attn_heads=int(
                getattr(model_cfg.modality_fusion, "cross_attn_heads", 8)),
            cross_attn_downsample=int(
                getattr(model_cfg.modality_fusion, "cross_attn_downsample", 8)),
            sa_gate_spatial=bool(getattr(model_cfg.modality_fusion, "sa_gate_spatial", False)),
            prior_ablation_zero=_ablation_zero,
            dccg_conf_hidden=int(model_cfg.modality_fusion.dccg_conf_hidden),
            dccg_use_confidence=bool(model_cfg.modality_fusion.dccg_use_confidence),
        )

        if residual_depth_fusion_enabled:
            if depth_only:
                raise ValueError(
                    "Residual depth fusion is incompatible with depth_only=true"
                )
            if model_cfg.rgb_backbone.weights is not None:
                raise ValueError(
                    "Residual depth fusion requires "
                    "model.magformer.rgb_backbone.weights=null; "
                    "the RGB path must be initialized by the full-model "
                    "model.finetune_weights checkpoint so loading is checked "
                    "as one explicit contract"
                )
            if full_model_cfg is not None and not full_model_cfg.finetune_weights:
                raise ValueError(
                    "Residual depth fusion requires an explicit "
                    "model.finetune_weights checkpoint"
                )
            if bool(getattr(model_cfg.modality_fusion, "enabled", True)):
                raise ValueError(
                    "Residual depth fusion requires "
                    "model.magformer.modality_fusion.enabled=false"
                )
            if bool(getattr(getattr(model_cfg, "dpe", None), "enabled", False)):
                raise ValueError(
                    "Residual depth fusion requires "
                    "model.magformer.dpe.enabled=false"
                )
            if swin_backend != "d2":
                raise ValueError(
                    "Residual depth fusion requires "
                    "model.magformer.swin.backend='d2'"
                )

            encoder_name = str(residual_depth_fusion_cfg.encoder)
            if encoder_name == "light_depth_pyramid":
                depth_encoder = depth_backbone
            elif encoder_name == "mobilenetv3_large":
                configured_name = str(model_cfg.depth_backbone.name).lower()
                if "mobilenetv3_large" not in configured_name:
                    raise ValueError(
                        "Residual depth fusion with MobileNetV3-L requires "
                        "depth_backbone.name=mobilenetv3_large"
                    )
                if not getattr(model_cfg.depth_backbone, "weights", None):
                    raise ValueError(
                        "Residual depth fusion with MobileNetV3-L requires "
                        "an explicit independent "
                        "depth_backbone.weights checkpoint"
                    )
                depth_encoder = depth_backbone
            else:
                raise ValueError(
                    "Unsupported residual depth fusion encoder: "
                    f"{encoder_name}"
                )

            residual_depth_feature_channels = [
                int(depth_encoder._stage_out_channels[key])
                for key in ("res2", "res3", "res4", "res5")
            ]
            rgb_backbone = ResidualDepthSwinBackbone(
                depth_encoder=depth_encoder,
                depth_channels=residual_depth_feature_channels,
                embed_dim=model_cfg.swin.embed_dim,
                depths=model_cfg.swin.depths,
                num_heads=model_cfg.swin.num_heads,
                window_size=model_cfg.swin.window_size,
                drop_path_rate=model_cfg.swin.drop_path_rate,
                out_features=model_cfg.swin.out_features,
                pretrained=use_rgb_pretrained,
                weights_path=model_cfg.rgb_backbone.weights,
                img_size=model_cfg.swin.pretrain_img_size,
                use_checkpoint=getattr(model_cfg.swin, "use_checkpoint", False),
            )
            depth_backbone = nn.Identity()
            fusion = nn.Identity()

        if depth_only:
            in_channels = depth_backbone._stage_out_channels
        else:
            in_channels = rgb_backbone._stage_out_channels

        pixel_decoder_name = getattr(
            model_cfg.sem_seg_head, "pixel_decoder_name", "SimplePixelDecoder")
        transformer_decoder_name = getattr(
            model_cfg.mask_former, "transformer_decoder_name", "SimpleTransformerDecoder")

        if pixel_decoder_name not in {"SimplePixelDecoder", "MSDeformAttnPixelDecoder"}:
            raise ValueError(
                f"Unsupported pixel decoder: {pixel_decoder_name}")

        if transformer_decoder_name not in {"SimpleTransformerDecoder", "MultiScaleMaskedTransformerDecoder"}:
            raise ValueError(
                f"Unsupported transformer decoder: {transformer_decoder_name}")

        if pixel_decoder_name == "MSDeformAttnPixelDecoder":
            transformer_in_features = getattr(
                model_cfg.sem_seg_head,
                "deformable_transformer_encoder_in_features",
                model_cfg.sem_seg_head.in_features,
            )
            dpe_cfg = getattr(model_cfg, "dpe", None)
            dpe_enabled = bool(getattr(dpe_cfg, "enabled", None) or False)
            dpe_beta = float(getattr(dpe_cfg, "beta", None) or 10.0)
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
                transformer_enc_layers=int(
                    getattr(model_cfg.sem_seg_head, "transformer_enc_layers", 0)),
                common_stride=model_cfg.sem_seg_head.common_stride,
                dpe_enabled=dpe_enabled,
                dpe_beta=dpe_beta,
                use_checkpoint=bool(getattr(model_cfg.sem_seg_head, 'pixel_decoder_use_checkpoint', False)),
                maskformer_num_feature_levels=int(getattr(model_cfg.mask_former, 'num_feature_levels', 3)),
                film_config=getattr(model_cfg, 'film', None),
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
            raise ValueError(
                f"mask_former.dec_layers must be >= 1, got {dec_layers_cfg}")
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
                dropout=float(getattr(model_cfg.mask_former, 'decoder_dropout', 0.0)),
                pre_norm=model_cfg.mask_former.pre_norm,
                enforce_input_project=False,
                num_feature_levels=int(getattr(model_cfg.mask_former, 'num_feature_levels', 3)),
                mask_attn_topk_ratio=getattr(model_cfg.mask_former, "mask_attn_topk_ratio", None),
                mask_attn_topk_min=int(getattr(model_cfg.mask_former, "mask_attn_topk_min", 64)),
                use_checkpoint=bool(getattr(model_cfg.mask_former, "decoder_use_checkpoint", False)),
                use_deformable_cross_attn=bool(getattr(model_cfg.mask_former, "use_deformable_cross_attn", False)),
                deformable_n_points=int(getattr(model_cfg.mask_former, "deformable_n_points", 4)),
                encoder_query_selection=bool(getattr(model_cfg.mask_former, "encoder_query_selection", False)),
                dcqm_enabled=bool(getattr(model_cfg.mask_former, "dcqm_enabled", False)),
                look_forward_twice=bool(getattr(model_cfg.mask_former, "look_forward_twice", False)),
                film_config=getattr(model_cfg, 'film', None),
                geometry_config=getattr(model_cfg, 'geometry_attn', None),
                depth_edge_config=getattr(model_cfg, 'depth_edge_attn', None),
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

        pixel_mean = model_cfg.pixel_mean
        pixel_std = model_cfg.pixel_std

        # Fallback: if magformer sub-config has ImageNet defaults, check parent ModelConfig
        _IMAGENET_MEAN = [123.675, 116.28, 103.53]
        _IMAGENET_STD = [58.395, 57.12, 57.375]
        _parent_cfg = getattr(config, 'model', None)
        if _parent_cfg is not None:
            if list(pixel_mean) == _IMAGENET_MEAN and hasattr(_parent_cfg, 'pixel_mean'):
                pixel_mean = _parent_cfg.pixel_mean
            if list(pixel_std) == _IMAGENET_STD and hasattr(_parent_cfg, 'pixel_std'):
                pixel_std = _parent_cfg.pixel_std

        model = cls(
            rgb_backbone=rgb_backbone,
            depth_backbone=depth_backbone,
            fusion_module=fusion,
            pixel_decoder=pixel_decoder,
            transformer_decoder=transformer_decoder,
            num_classes=model_cfg.sem_seg_head.num_classes,
            num_queries=model_cfg.mask_former.num_object_queries,
            hidden_dim=model_cfg.mask_former.hidden_dim,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            size_divisibility=model_cfg.sem_seg_head.common_stride,
            inference_topk=int(getattr(model_cfg.mask_former, "inference_topk", 100)),
            depth_only=depth_only,
        )
        fusion_enabled_cfg = bool(getattr(model_cfg.modality_fusion, "enabled", True))
        if depth_only:
            # depth-only ablation disables fusion regardless of config
            fusion_enabled_cfg = False
        model.modality_fusion_enabled = fusion_enabled_cfg
        model.depth_backbone_enabled = bool(
            getattr(model_cfg.depth_backbone, "enabled", True))
        model.residual_depth_fusion_enabled = residual_depth_fusion_enabled
        model.depth_mode = depth_mode
        model.depth_only = bool(depth_only)
        if depth_only and not model.depth_backbone_enabled:
            raise RuntimeError(
                "depth_only=True 但 depth_backbone_enabled=False，无可用特征"
            )
        model._sync_criterion_from_config(model_cfg)

        # Wire AGPE from config
        agpe_cfg = model_cfg if hasattr(model_cfg, 'agpe_enabled') else getattr(model_cfg, 'magformer', model_cfg)
        model.agpe_enabled = getattr(agpe_cfg, 'agpe_enabled', False)
        if model.agpe_enabled:
            from .agpe_module import AGPEModule
            feature_dims = getattr(agpe_cfg, 'feature_dims', [96, 192, 384, 768])
            model.agpe = AGPEModule(
                feature_dims,
                reduction=int(getattr(agpe_cfg, 'agpe_reduction', 16)),
                spatial_kernel=int(getattr(agpe_cfg, 'agpe_spatial_kernel', 7)),
            )
            print(f'[AGPE] Enabled: reduction={getattr(agpe_cfg, "agpe_reduction", 16)}, '
                  f'kernel={getattr(agpe_cfg, "agpe_spatial_kernel", 7)}')

        # Wire DN-DETR from config
        dn_cfg = getattr(model_cfg.mask_former, "dn_enabled", None)
        if dn_cfg is None:
            dn_cfg = getattr(model_cfg.mask_former, "dn_enabled", False)
        model.dn_enabled = bool(getattr(model_cfg.mask_former, 'dn_enabled', False))
        if model.dn_enabled:
            model.dn_scalar = int(getattr(model_cfg.mask_former, 'dn_scalar', 5))
            model.dn_box_noise_scale = float(getattr(model_cfg.mask_former, 'dn_box_noise_scale', 0.4))
            model.dn_label_noise_ratio = float(getattr(model_cfg.mask_former, 'dn_label_noise_ratio', 0.2))
            model.dn_loss_weight = float(getattr(model_cfg.mask_former, 'dn_loss_weight', 1.0))
            model.dn_contrastive_weight = float(getattr(model_cfg.mask_former, 'dn_contrastive_weight', 0.5))
            print(f'[DN-DETR] Enabled: scalar={model.dn_scalar}, '
                  f'box_noise={model.dn_box_noise_scale}, '
                  f'label_noise={model.dn_label_noise_ratio}, '
                  f'loss_weight={model.dn_loss_weight}, '
                  f'contrastive_weight={model.dn_contrastive_weight}')

        return model

    def _sync_criterion_from_config(self, config: Any) -> None:
        from ..common.matcher import HungarianMatcher
        from ..common.criterion import SetCriterion

        mask_former = config.mask_former
        class_w = float(mask_former.class_weight)
        mask_w = float(mask_former.mask_weight)
        dice_w = float(mask_former.dice_weight)
        eos_coef = float(mask_former.no_object_weight)
        matcher_num_points = int(mask_former.matcher_num_points)
        loss_num_points = int(mask_former.loss_num_points)

        scale_balanced = bool(getattr(mask_former, "scale_balanced", False))
        matcher = HungarianMatcher(
            cost_class=class_w,
            cost_mask=mask_w,
            cost_dice=dice_w,
            num_points=matcher_num_points,
            scale_balanced=scale_balanced,
            cost_bbox=float(mask_former.cost_bbox),
            cost_giou=float(mask_former.cost_giou),
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

        # Box supervision weights (L1 + GIoU) for hybrid matching
        weight_dict["loss_bbox"] = 5.0
        weight_dict["loss_giou"] = 2.0
        if getattr(mask_former, "deep_supervision", False):
            num_aux = max(int(mask_former.dec_layers) - 1, 0)
            for i in range(num_aux):
                weight_dict[f"loss_bbox_{i}"] = 5.0
                weight_dict[f"loss_giou_{i}"] = 2.0

        # DN-DETR config for criterion
        dn_enabled = bool(getattr(mask_former, "dn_enabled", False))
        dn_loss_weight = float(getattr(mask_former, "dn_loss_weight", 1.0))
        dn_contrastive_weight = float(getattr(mask_former, "dn_contrastive_weight", 0.5))
        if dn_enabled:
            weight_dict["loss_dn_ce"] = 2.0 * dn_loss_weight
            weight_dict["loss_dn_mask"] = 5.0 * dn_loss_weight
            weight_dict["loss_dn_dice"] = 5.0 * dn_loss_weight
            weight_dict["loss_dn_neg_ce"] = dn_contrastive_weight

        self.criterion = SetCriterion(
            num_classes=self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=eos_coef,
            losses=("labels", "masks", "boxes"),
            num_points=loss_num_points,
            oversample_ratio=float(mask_former.oversample_ratio),
            importance_sample_ratio=float(mask_former.importance_sample_ratio),
            balanced_ce=bool(getattr(mask_former, "balanced_ce", False)),
            balanced_ce_min_fg_ratio=float(
                getattr(mask_former, "balanced_ce_min_fg_ratio", 0.01)),
            dn_enabled=dn_enabled,
            dn_loss_weight=dn_loss_weight,
            dn_contrastive_weight=dn_contrastive_weight,
            scale_adaptive_alpha=float(getattr(mask_former, "scale_adaptive_alpha", 0.0)),
            use_uncertainty_weighting=bool(getattr(mask_former, "use_uncertainty_weighting", False)),
        )

    @staticmethod
    def _pixel_decoder_forward_kwargs(
        pixel_decoder: nn.Module,
        *,
        features: Dict[str, torch.Tensor],
        confidence_maps: Optional[Dict[str, torch.Tensor]],
        depth_modulation_maps: Optional[Dict[str, torch.Tensor]],
        depth_raw: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "features": features,
            "confidence_maps": confidence_maps,
            "depth_raw": depth_raw,
            "padding_mask": padding_mask,
        }
        try:
            signature = inspect.signature(pixel_decoder.forward)
        except (TypeError, ValueError):
            signature = None
        if signature is not None and "depth_modulation_maps" in signature.parameters:
            kwargs["depth_modulation_maps"] = depth_modulation_maps
        return kwargs

    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device

    @property
    def query_embed(self):
        """Proxy to decoder's query_embed for CQD access."""
        return self.decoder.query_embed

    def _extract_dual_features(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        depth_valid_masks: Optional[torch.Tensor] = None,
        depth_noise_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
        """Extract and fuse RGB + depth features with parallel backbone via CUDA streams."""
        images_norm = (images - self.pixel_mean) / self.pixel_std

        if getattr(self, "depth_only", False):
            depth_features = self.depth_backbone(depths)
            return depth_features, None, {}

        if getattr(self, "residual_depth_fusion_enabled", False):
            if depth_valid_masks is None:
                raise ValueError(
                    "Residual depth fusion requires explicit depth_valid_masks"
                )
            return (
                self.rgb_backbone(images_norm, depths, depth_valid_masks),
                None,
                {},
            )

        fusion_enabled = bool(getattr(self, "modality_fusion_enabled", True)) and bool(
            getattr(self, "depth_backbone_enabled", True)
        )

        if fusion_enabled and images.is_cuda:
            # Both backbones consume tensors produced on the caller's stream.
            # DDP buffer broadcasts also complete on that stream immediately
            # before forward, so establish the producer-to-consumer dependency
            # before either auxiliary stream starts reading model state or input.
            current_stream = torch.cuda.current_stream(device=images.device)
            rgb_stream = torch.cuda.Stream(device=images.device)
            depth_stream = torch.cuda.Stream(device=images.device)
            rgb_stream.wait_stream(current_stream)
            depth_stream.wait_stream(current_stream)

            with torch.cuda.stream(rgb_stream):
                rgb_features = self.rgb_backbone(images_norm)
            with torch.cuda.stream(depth_stream):
                depth_features = self.depth_backbone(depths)

            # Fusion runs on the caller's stream.  Wait for both producers and
            # record that stream on every produced tensor so the CUDA caching
            # allocator cannot recycle auxiliary-stream storage while fusion is
            # still consuming it asynchronously.
            current_stream.wait_stream(rgb_stream)
            current_stream.wait_stream(depth_stream)
            for feature_dict in (rgb_features, depth_features):
                for feature in feature_dict.values():
                    feature.record_stream(current_stream)
        else:
            rgb_features = self.rgb_backbone(images_norm)
            if fusion_enabled:
                depth_features = self.depth_backbone(depths)

        if fusion_enabled:
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

        return fused_features, confidence_maps, fusion_losses


    def forward(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        targets: Optional[List[Dict[str, Any]]] = None,
        padding_masks: Optional[torch.Tensor] = None,
        depth_valid_masks: Optional[torch.Tensor] = None,
        depth_noise_masks: Optional[torch.Tensor] = None,
        return_features: bool = False,
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

        # Parallel dual backbone feature extraction
        fused_features, confidence_maps, fusion_losses = self._extract_dual_features(
            images, depths, depth_valid_masks, depth_noise_masks
        )

        # AGPE: enhance fused features before pixel decoder
        if getattr(self, 'agpe_enabled', False) and self.agpe is not None:
            _agpe_keys = ['res2', 'res3', 'res4', 'res5']
            _agpe_feats = [fused_features[k] for k in _agpe_keys if k in fused_features]
            if len(_agpe_feats) == self.agpe.num_levels:
                _agpe_enhanced = self.agpe(_agpe_feats)
                for _k, _f in zip(_agpe_keys, _agpe_enhanced):
                    fused_features[_k] = _f

        decoder_inputs = self.pixel_decoder(
            **self._pixel_decoder_forward_kwargs(
                self.pixel_decoder,
                features=fused_features,
                confidence_maps=confidence_maps,
                depth_modulation_maps=confidence_maps,
                depth_raw=depths,
                padding_mask=padding_masks,
            )
        )

        # 获取深度调制位置编码 (pos_key_list) 用于 key_pos
        pos_key_list = decoder_inputs.get("pos_key_list", None)

        if self.training:
            if targets is None:
                raise ValueError("MAGFormer training requires non-empty targets")

            processed_targets = self._prepare_targets(targets)

            # Build GT tensors for CQD from processed_targets
            gt_labels, gt_masks, gt_pad_mask = self._build_gt_tensors(processed_targets, H, W)

            # Generate DN queries before the single decoder call (avoid double decode)
            dn_query_embed = None
            dn_query_feat = None
            dn_meta = None
            if getattr(self, 'dn_enabled', False):
                dn_query_embed, dn_query_feat, dn_meta = self._generate_dn_queries(
                    None, None, gt_labels, gt_masks, gt_pad_mask)

            # Single decoder call with optional DN queries
            outputs = self.decoder(
                memory=decoder_inputs["memory"],
                mask_features=decoder_inputs["mask_features"],
                multi_scale_features=decoder_inputs.get(
                    "multi_scale_features", None),
                multi_scale_pos=decoder_inputs.get("multi_scale_pos", None),
                multi_scale_padding_masks=decoder_inputs["multi_scale_padding_masks"],
                pos_key=pos_key_list,
                dn_query_embed=dn_query_embed,
                dn_query_feat=dn_query_feat,
                depth_raw=depths,
            )

            if dn_meta is not None:
                outputs["dn_meta"] = dn_meta
                outputs["dn_enabled"] = True

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
            # Inference path (also handles return_features for feature extraction)
            if return_features:
                outputs = self.decoder(
                    memory=decoder_inputs["memory"],
                    mask_features=decoder_inputs["mask_features"],
                    multi_scale_features=decoder_inputs.get(
                        "multi_scale_features", None),
                    multi_scale_pos=decoder_inputs.get("multi_scale_pos", None),
                    multi_scale_padding_masks=decoder_inputs["multi_scale_padding_masks"],
                    pos_key=pos_key_list,
                    depth_raw=depths,
                )
                outputs["features"] = decoder_inputs["mask_features"]
                return outputs
            return self.forward_inference_exported(
                images=images,
                depths=depths,
                padding_masks=padding_masks,
                depth_valid_masks=depth_valid_masks,
                depth_noise_masks=depth_noise_masks,
            )

    def _build_gt_tensors(self, processed_targets, H, W):
        """Build padded GT label, mask, and pad-mask tensors from processed targets.

        Args:
            processed_targets: list of dicts with 'labels' and 'masks'
            H, W: spatial dims for mask padding

        Returns:
            gt_labels: (B, max_gt) long tensor
            gt_masks: (B, max_gt, H, W) float tensor
            gt_pad_mask: (B, max_gt) bool, True=valid
        """
        B = len(processed_targets)
        device = self.device

        max_gt = max(len(t["labels"]) for t in processed_targets)
        max_gt = max(max_gt, 1)

        gt_labels_padded = torch.full((B, max_gt), self.num_classes, dtype=torch.long, device=device)
        gt_masks_padded = torch.zeros(B, max_gt, H, W, device=device)
        gt_pad_mask = torch.zeros(B, max_gt, dtype=torch.bool, device=device)  # True=valid

        for b, t in enumerate(processed_targets):
            labels = t["labels"]
            masks = t["masks"]
            n = labels.shape[0]
            if n == 0:
                continue
            n = min(n, max_gt)
            gt_labels_padded[b, :n] = labels[:n]
            gt_pad_mask[b, :n] = True
            # masks may have different spatial size; resize if needed
            m = masks[:n]
            if m.shape[-2:] != (H, W):
                m = F.interpolate(m.unsqueeze(0).float(), size=(H, W), mode="nearest").squeeze(0)
            gt_masks_padded[b, :n] = m

        return gt_labels_padded, gt_masks_padded, gt_pad_mask

    @torch.inference_mode()
    def forward_inference_decoder_outputs(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        padding_masks: Optional[torch.Tensor] = None,
        depth_valid_masks: Optional[torch.Tensor] = None,
        depth_noise_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        fused_features, confidence_maps, _ = self._extract_dual_features(
            images, depths, depth_valid_masks, depth_noise_masks
        )

        # AGPE: enhance fused features before pixel decoder (inference path)
        if getattr(self, 'agpe_enabled', False) and self.agpe is not None:
            _agpe_keys = ['res2', 'res3', 'res4', 'res5']
            _agpe_feats = [fused_features[k] for k in _agpe_keys if k in fused_features]
            if len(_agpe_feats) == self.agpe.num_levels:
                _agpe_enhanced = self.agpe(_agpe_feats)
                for _k, _f in zip(_agpe_keys, _agpe_enhanced):
                    fused_features[_k] = _f

        decoder_inputs = self.pixel_decoder(
            **self._pixel_decoder_forward_kwargs(
                self.pixel_decoder,
                features=fused_features,
                confidence_maps=confidence_maps,
                depth_modulation_maps=confidence_maps,
                depth_raw=depths,
                padding_mask=padding_masks,
            )
        )
        pos_key_list = decoder_inputs.get("pos_key_list", None)
        return self.decoder(
            memory=decoder_inputs["memory"],
            mask_features=decoder_inputs["mask_features"],
            multi_scale_features=decoder_inputs.get("multi_scale_features", None),
            multi_scale_pos=decoder_inputs.get("multi_scale_pos", None),
            multi_scale_padding_masks=decoder_inputs["multi_scale_padding_masks"],
            pos_key=pos_key_list,
            depth_raw=depths,
        )

    @torch.inference_mode()
    def forward_inference_raw(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        padding_masks: Optional[torch.Tensor] = None,
        depth_valid_masks: Optional[torch.Tensor] = None,
        depth_noise_masks: Optional[torch.Tensor] = None,
        include_raw_tensors: bool = False,
        move_predictions_to_cpu: bool = True,
    ) -> Dict[str, Any]:
        outputs = self.forward_inference_decoder_outputs(
            images=images,
            depths=depths,
            padding_masks=padding_masks,
            depth_valid_masks=depth_valid_masks,
            depth_noise_masks=depth_noise_masks,
        )
        return self._inference_raw(
            outputs,
            images.shape,
            include_raw_tensors=include_raw_tensors,
            move_predictions_to_cpu=move_predictions_to_cpu,
            inference_topk=self.inference_topk,
        )

    @torch.inference_mode()
    def forward_inference_exported(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        padding_masks: Optional[torch.Tensor] = None,
        depth_valid_masks: Optional[torch.Tensor] = None,
        depth_noise_masks: Optional[torch.Tensor] = None,
        include_raw_tensors: bool = False,
        move_raw_tensors_to_cpu: bool = False,
    ) -> Dict[str, Any]:
        raw = self.forward_inference_raw(
            images=images,
            depths=depths,
            padding_masks=padding_masks,
            depth_valid_masks=depth_valid_masks,
            depth_noise_masks=depth_noise_masks,
            include_raw_tensors=include_raw_tensors,
            move_predictions_to_cpu=True,
        )
        return self._export_inference_predictions(
            raw,
            include_raw_tensors=include_raw_tensors,
            move_raw_tensors_to_cpu=move_raw_tensors_to_cpu,
        )

    @torch.inference_mode()
    def collect_preflight_diagnostics(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        padding_masks: Optional[torch.Tensor] = None,
        depth_valid_masks: Optional[torch.Tensor] = None,
        depth_noise_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        fused_features, confidence_maps, _ = self._extract_dual_features(
            images, depths, depth_valid_masks, depth_noise_masks
        )

        # AGPE: enhance fused features before pixel decoder (inference path)
        if getattr(self, 'agpe_enabled', False) and self.agpe is not None:
            _agpe_keys = ['res2', 'res3', 'res4', 'res5']
            _agpe_feats = [fused_features[k] for k in _agpe_keys if k in fused_features]
            if len(_agpe_feats) == self.agpe.num_levels:
                _agpe_enhanced = self.agpe(_agpe_feats)
                for _k, _f in zip(_agpe_keys, _agpe_enhanced):
                    fused_features[_k] = _f

        decoder_inputs = self.pixel_decoder(
            **self._pixel_decoder_forward_kwargs(
                self.pixel_decoder,
                features=fused_features,
                confidence_maps=confidence_maps,
                depth_modulation_maps=confidence_maps,
                depth_raw=depths,
                padding_mask=padding_masks,
            )
        )
        outputs = self.decoder(
            memory=decoder_inputs["memory"],
            mask_features=decoder_inputs["mask_features"],
            multi_scale_features=decoder_inputs.get(
                "multi_scale_features", None),
            multi_scale_pos=decoder_inputs.get("multi_scale_pos", None),
            multi_scale_padding_masks=decoder_inputs["multi_scale_padding_masks"],
            pos_key=decoder_inputs.get("pos_key_list", None),
            depth_raw=depths,
        )
        return {
            "confidence_maps": confidence_maps,
            "pred_masks": outputs.get("pred_masks"),
        }

    @staticmethod
    def _sinusoidal_box_pe(boxes, hidden_dim):
        """4D sinusoidal positional encoding for cxcywh boxes.
        Args:
            boxes: (..., 4) tensor of cxcywh in [0, 1]
            hidden_dim: output dimension
        Returns:
            (..., hidden_dim) positional encoding
        """
        half_dim = hidden_dim // 4 // 2
        freqs = torch.exp(-math.log(10000.0) / (half_dim - 1) * torch.arange(half_dim, device=boxes.device, dtype=boxes.dtype))
        # boxes: (..., 4), freqs: (half_dim,)
        args = boxes.unsqueeze(-1) * freqs * 2 * math.pi  # (..., 4, half_dim)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (..., 4, 2*half_dim)
        # Interleave the 4 coordinates' embeddings
        embedding = embedding.reshape(*boxes.shape[:-1], -1)  # (..., 4 * 2 * half_dim)
        # Ensure output dim matches hidden_dim
        if embedding.shape[-1] < hidden_dim:
            padding = torch.zeros(*embedding.shape[:-1], hidden_dim - embedding.shape[-1], device=boxes.device, dtype=boxes.dtype)
            embedding = torch.cat([embedding, padding], dim=-1)
        return embedding[..., :hidden_dim]

    @staticmethod
    def _boxes_from_masks(masks):
        """Convert binary masks to normalized cxcywh boxes."""
        return masks_to_boxes_cxcywh(masks)

    def _generate_dn_queries(
        self,
        target_queries,
        target_masks,
        gt_labels,
        gt_masks,
        gt_pad_mask,
    ):
        """DINO-style Contrastive Query Denoising.

        Generates positive (noisy GT) and negative (shuffled GT) queries for training.
        Returns None during inference (no queries provided).

        Args:
            target_queries: not used (kept for interface compat)
            target_masks: not used (kept for interface compat)
            gt_labels: (B, max_gt) padded GT labels
            gt_masks: (B, max_gt, H, W) padded GT masks
            gt_pad_mask: (B, max_gt) bool, True=valid

        Returns:
            dn_query_embed: (num_dn, B, C) positional embeddings
            dn_query_feat: (num_dn, B, C) content features
            dn_meta: dict with num_positives, num_negatives, dn_scalar, gt_pad_mask
        """
        if not self.dn_enabled or not self.training:
            return None, None, None

        B, max_gt, H, W = gt_masks.shape
        hidden_dim = self.query_embed.embedding_dim
        device = gt_masks.device

        # Collect valid GT boxes across batch
        all_gt_boxes = []
        all_gt_labels = []
        batch_num_valid = []

        for b in range(B):
            valid = gt_pad_mask[b]  # (max_gt,)
            num_valid = valid.sum().item()
            if num_valid == 0:
                batch_num_valid.append(0)
                continue

            valid_masks = gt_masks[b][valid]  # (num_valid, H, W)
            boxes = self._boxes_from_masks(valid_masks)  # (num_valid, 4)
            labels = gt_labels[b][valid]  # (num_valid,)

            all_gt_boxes.append(boxes)
            all_gt_labels.append(labels)
            batch_num_valid.append(num_valid)

        max_valid = max(batch_num_valid) if batch_num_valid else 0
        if max_valid == 0:
            return None, None, None

        num_pos = max_valid * self.dn_scalar  # positive samples per batch element
        num_neg = max_valid  # negative samples per batch element
        num_dn = num_pos + num_neg

        # Build padded GT boxes and labels: (B, max_valid, 4) and (B, max_valid)
        gt_boxes_padded = torch.zeros(B, max_valid, 4, device=device)
        gt_labels_padded = torch.full((B, max_valid), self.num_classes, device=device, dtype=torch.long)

        for b in range(B):
            if batch_num_valid[b] > 0:
                gt_boxes_padded[b, :batch_num_valid[b]] = all_gt_boxes[b]
                gt_labels_padded[b, :batch_num_valid[b]] = all_gt_labels[b]

        # --- Positive queries: GT box + Gaussian noise ---
        noise = torch.randn_like(gt_boxes_padded.unsqueeze(1).expand(B, self.dn_scalar, max_valid, 4)) * self.dn_box_noise_scale
        noisy_boxes_pos = (gt_boxes_padded.unsqueeze(1) + noise).clamp(0, 1)  # (B, dn_scalar, max_valid, 4)
        noisy_boxes_pos = noisy_boxes_pos.reshape(B, num_pos, 4)  # (B, num_pos, 4)

        pos_pe = self._sinusoidal_box_pe(noisy_boxes_pos, hidden_dim)  # (B, num_pos, C)

        # --- Negative queries: cyclically shifted GT boxes ---
        # Roll GT boxes by 1 so each query is assigned a wrong GT's box
        neg_boxes = torch.roll(gt_boxes_padded, shifts=1, dims=1)  # (B, max_valid, 4)

        neg_pe = self._sinusoidal_box_pe(neg_boxes, hidden_dim)  # (B, num_neg, C)

        # --- Content features ---
        # Use mean of learned query embeddings as base content
        base_content = self.query_embed.weight.mean(dim=0)  # (C,)

        pos_feat = base_content.unsqueeze(0).unsqueeze(0).expand(B, num_pos, -1).clone()
        pos_feat = pos_feat + torch.randn_like(pos_feat) * 0.1  # small noise

        neg_feat = base_content.unsqueeze(0).unsqueeze(0).expand(B, num_neg, -1).clone()
        neg_feat = neg_feat + torch.randn_like(neg_feat) * 0.1

        # --- Concatenate pos + neg ---
        dn_query_embed = torch.cat([pos_pe, neg_pe], dim=1)  # (B, num_dn, C)
        dn_query_feat = torch.cat([pos_feat, neg_feat], dim=1)  # (B, num_dn, C)

        # Transpose to (num_dn, B, C) as expected by decoder
        dn_query_embed = dn_query_embed.permute(1, 0, 2)
        dn_query_feat = dn_query_feat.permute(1, 0, 2)

        # Build dn_meta for loss computation
        dn_meta = {
            "num_positives": num_pos,
            "num_negatives": num_neg,
            "dn_scalar": self.dn_scalar,
            "max_valid": max_valid,
            "batch_num_valid": batch_num_valid,  # list of ints
            "gt_labels_padded": gt_labels_padded,  # (B, max_valid)
            "gt_pad_mask": gt_pad_mask,  # (B, max_gt)
        }

        return dn_query_embed, dn_query_feat, dn_meta

    def _prepare_targets(
        self,
        targets: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        prepared = []
        for target in targets:
            labels = target.get("labels", torch.zeros(
                0, dtype=torch.long, device=self.device)).long()
            labels = labels.clamp(min=0, max=max(self.num_classes - 1, 0))

            masks = target.get("masks", torch.zeros(
                0, device=self.device)).float()
            if masks.ndim == 2:
                masks = masks.unsqueeze(0)

            prepared_target = {
                "labels": labels,
                "masks": masks,
            }
            prepared.append(prepared_target)
        return prepared

    @staticmethod
    @torch.inference_mode()
    def _inference_raw(
        outputs: Dict[str, torch.Tensor],
        image_shape: Tuple[int, ...],
        include_raw_tensors: bool = False,
        move_predictions_to_cpu: bool = True,
        inference_topk: int = 100,
    ) -> Dict[str, Any]:
        pred_logits = outputs.get("pred_logits", None)
        pred_masks = outputs.get("pred_masks", None)

        if pred_logits is None or pred_masks is None:
            raise RuntimeError(
                "Decoder output must contain 'pred_logits' and 'pred_masks'; "
                f"got keys={sorted(outputs.keys())}"
            )

        B, Nq, _ = pred_logits.shape
        H_img, W_img = image_shape[-2:]

        def _detach(t: torch.Tensor) -> torch.Tensor:
            t = t.detach()
            return t.cpu() if move_predictions_to_cpu else t

        class_scores = pred_logits.sigmoid()[..., :-1]
        num_classes = class_scores.shape[-1]

        if num_classes <= 0:
            empty_predictions = []
            for i in range(B):
                empty_pred = {
                    "image_id": i,
                    "scores": _detach(pred_logits.new_zeros((0,))),
                    "category_ids": _detach(
                        pred_logits.new_zeros((0,), dtype=torch.long)
                    ),
                    "masks": _detach(
                        pred_masks.new_zeros((0, H_img, W_img), dtype=torch.uint8)
                    ),
                }
                if include_raw_tensors:
                    empty_pred["mask_probs"] = _detach(
                        pred_masks.new_zeros((0, H_img, W_img))
                    )
                    empty_pred["mask_logits"] = _detach(
                        pred_masks.new_zeros((0, H_img, W_img))
                    )
                empty_predictions.append(empty_pred)
            result: Dict[str, Any] = {"predictions": empty_predictions}
            if include_raw_tensors:
                result["pred_logits"] = pred_logits.detach()
                result["pred_masks"] = pred_masks.detach()
            return result

        topk = min(inference_topk, Nq * max(num_classes, 1))
        top_scores, top_indices = class_scores.flatten(1).topk(topk, dim=1)

        labels = torch.arange(num_classes, device=pred_logits.device).unsqueeze(
            0).repeat(Nq, 1).flatten(0, 1)

        # Vectorized batch processing
        query_indices = top_indices // max(num_classes, 1)
        class_indices = labels[top_indices] if num_classes > 0 else torch.zeros_like(
            query_indices)

        # Gather masks: (B, Nq, H, W) -> (B, topk, H, W)
        masks = pred_masks.gather(
            1,
            query_indices.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, pred_masks.shape[-2], pred_masks.shape[-1])
        )

        # Batched interpolation
        if masks.shape[-2:] != (H_img, W_img):
            masks = F.interpolate(
                masks.reshape(B * topk, 1, masks.shape[-2], masks.shape[-1]),
                size=(H_img, W_img),
                mode="bilinear",
                align_corners=False,
            ).reshape(B, topk, H_img, W_img)

        # Batched sigmoid + threshold + scoring
        mask_probs = masks.sigmoid()
        binary_masks = mask_probs > 0.5
        mask_scores = (mask_probs.flatten(2) * binary_masks.float().flatten(2)).sum(2) / (
            binary_masks.float().flatten(2).sum(2) + 1e-6
        )
        final_scores = top_scores * mask_scores

        binary_masks_uint8 = binary_masks.to(torch.uint8)

        batch_predictions = []
        for i in range(B):
            batch_pred = {
                "image_id": i,
                "scores": _detach(final_scores[i]),
                "category_ids": _detach(class_indices[i]),
                "masks": _detach(binary_masks_uint8[i]),
            }
            if include_raw_tensors:
                batch_pred["mask_probs"] = _detach(mask_probs[i])
                batch_pred["mask_logits"] = _detach(masks[i])
            batch_predictions.append(batch_pred)

        result: Dict[str, Any] = {"predictions": batch_predictions}
        if include_raw_tensors:
            result["pred_logits"] = pred_logits.detach()
            result["pred_masks"] = pred_masks.detach()
        return result


    @staticmethod
    def _export_inference_predictions(
        raw_outputs: Dict[str, Any],
        *,
        include_raw_tensors: bool = False,
        move_raw_tensors_to_cpu: bool = False,
    ) -> Dict[str, Any]:
        if "predictions" not in raw_outputs:
            raise KeyError("raw_outputs must contain a 'predictions' key")
        exported_predictions = []
        for pred in raw_outputs.get("predictions", []):
            exported_predictions.append(
                {
                    "image_id": int(pred["image_id"]),
                    "scores": pred["scores"].detach().cpu().numpy()
                    if torch.is_tensor(pred["scores"])
                    else np.asarray(pred["scores"]),
                    "category_ids": pred["category_ids"].detach().cpu().numpy()
                    if torch.is_tensor(pred["category_ids"])
                    else np.asarray(pred["category_ids"]),
                    "masks": pred["masks"].detach().cpu().numpy()
                    if torch.is_tensor(pred["masks"])
                    else np.asarray(pred["masks"]),
                }
            )

        result = {"predictions": exported_predictions}
        if include_raw_tensors:
            pred_logits = raw_outputs.get("pred_logits")
            pred_masks = raw_outputs.get("pred_masks")
            if torch.is_tensor(pred_logits):
                pred_logits = pred_logits.detach()
                if move_raw_tensors_to_cpu:
                    pred_logits = pred_logits.cpu()
            if torch.is_tensor(pred_masks):
                pred_masks = pred_masks.detach()
                if move_raw_tensors_to_cpu:
                    pred_masks = pred_masks.cpu()
            result["pred_logits"] = pred_logits
            result["pred_masks"] = pred_masks
        return result


def build_magformer(config: Dict[str, Any]) -> MagFormerArch:
    """兼容旧接口，转发到 from_config。"""
    return MagFormerArch.from_config(config)
