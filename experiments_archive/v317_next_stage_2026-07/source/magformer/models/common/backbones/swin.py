# -*- coding: utf-8 -*-
"""
Swin Transformer Backbone

基于 timm 的 Swin Transformer 骨干网络实现。
用于 RGB 图像特征提取。
"""

from typing import List, Tuple, Dict, Optional, Any
import torch
import torch.nn as nn
import timm

from ....engine.utils import load_torch_checkpoint


class SwinTransformer(nn.Module):
    """
    Swin Transformer 骨干网络。

    使用 timm 的预训练 Swin Transformer 作为 RGB 图像特征提取器。
    """

    def __init__(
        self,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.3,
        out_features: List[str] = ["res2", "res3", "res4", "res5"],
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        img_size: int = 512,
    ):
        """
        Args:
            embed_dim: 嵌入维度
            depths: 每层深度
            num_heads: 注意力头数
            window_size: 窗口尺寸
            mlp_ratio: MLP 扩展比例
            drop_path_rate: Stochastic Depth 比率
            out_features: 输出特征层名称
            pretrained: 是否使用预训练权重
            weights_path: 预训练权重路径
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.out_features = out_features

        model_name = self._get_model_name(embed_dim)

        out_indices = tuple(range(len(out_features)))
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained and (weights_path is None),
            features_only=True,
            out_indices=out_indices,
            img_size=img_size,
            # Val/Test must keep original resolution (e.g. 1024x1024) to match COCO GT.
            # Set strict_img_size=False so the same backbone can run on both 512 (train crop)
            # and 1024 (val) without PatchEmbed assertions.
            strict_img_size=False,
        )

        # 加载自定义权重
        if weights_path is not None:
            self._load_weights(weights_path)

        feature_info = self.model.feature_info
        channels = feature_info.channels()
        strides = feature_info.reduction()
        self._stage_out_channels = {
            name: ch for name, ch in zip(out_features, channels)}
        self._stage_out_strides = {
            name: st for name, st in zip(out_features, strides)}

    def _get_model_name(self, embed_dim: int) -> str:
        """根据嵌入维度选择 timm 模型名称"""
        if embed_dim == 96:
            return "swin_tiny_patch4_window7_224"
        if embed_dim == 128:
            return "swin_small_patch4_window7_224"
        if embed_dim == 192:
            return "swin_base_patch4_window7_224"
        return "swin_tiny_patch4_window7_224"

    def _load_weights(self, weights_path: str) -> None:
        """加载自定义权重"""
        state_dict = load_torch_checkpoint(weights_path, map_location="cpu")

        # 处理不同的键名格式
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # 过滤不匹配的键
        model_state = self.model.state_dict()
        filtered_state = {
            k: v for k, v in state_dict.items() if k in model_state
        }

        self.model.load_state_dict(filtered_state, strict=False)

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播。

        Args:
            x: (B, 3, H, W) RGB 图像

        Returns:
            多尺度特征字典 {层名: 特征图}
        """
        features = self.model(x)

        result = {}
        for i, feat in enumerate(features):
            stage_name = self.out_features[i]
            # timm 的 Swin 可能输出 (B, H, W, C)，需要转为 (B, C, H, W)
            if feat.dim() == 4 and feat.shape[-1] != feat.shape[-2]:
                # 检查是否为 channels-last 格式
                expected_ch = self._stage_out_channels[stage_name]
                if feat.shape[-1] == expected_ch and feat.shape[1] != expected_ch:
                    feat = feat.permute(0, 3, 1, 2).contiguous()
            result[stage_name] = feat

        return result

    @property
    def output_shape(self) -> Dict[str, Tuple[int, int, int, int]]:
        """返回输出形状 (通道数, 高度步长, 宽度步长)"""
        return {
            name: (
                self._stage_out_channels[name], self._stage_out_strides[name], self._stage_out_strides[name])
            for name in self.out_features
        }


# 便捷函数
def build_swin_backbone(
    config: Dict[str, Any]
) -> SwinTransformer:
    """
    根据配置构建 Swin 骨干网络。

    Args:
        config: Swin 配置字典

    Returns:
        SwinTransformer 模型
    """
    return SwinTransformer(
        embed_dim=config.get("embed_dim", 96),
        depths=config.get("depths", [2, 2, 6, 2]),
        num_heads=config.get("num_heads", [3, 6, 12, 24]),
        window_size=config.get("window_size", 7),
        mlp_ratio=config.get("mlp_ratio", 4.0),
        drop_path_rate=config.get("drop_path_rate", 0.3),
        out_features=config.get(
            "out_features", ["res2", "res3", "res4", "res5"]),
        pretrained=config.get("pretrained", True),
        weights_path=config.get("weights", None),
    )
