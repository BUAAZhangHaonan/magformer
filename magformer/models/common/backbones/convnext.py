# -*- coding: utf-8 -*-
"""
ConvNeXt Depth Backbone

基于 timm 的 ConvNeXt 骨干网络实现，修改输入通道为1以处理深度图。
"""

from typing import List, Tuple, Dict, Optional, Any
import torch
import torch.nn as nn
import timm


class ConvNeXtDepth(nn.Module):
    """
    ConvNeXt 骨干网络 (用于深度图)。

    修改第一层卷积以接受单通道深度输入。
    """

    def __init__(
        self,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale: float = 1e-6,
        out_features: List[str] = ["res2", "res3", "res4", "res5"],
        pretrained: bool = False,
        weights_path: Optional[str] = None,
    ):
        """
        Args:
            depths: 每层深度
            dims: 每层维度
            drop_path_rate: Drop Path 比率
            layer_scale: Layer Scale 初始值
            out_features: 输出特征层名称
            pretrained: 是否使用预训练权重
            weights_path: 预训练权重路径
        """
        super().__init__()

        self.depths = depths
        self.dims = dims
        self.drop_path_rate = drop_path_rate
        self.layer_scale = layer_scale
        self.out_features = out_features

        # 构建 ConvNeXt 模型名称
        model_name = self._get_model_name(dims)

        out_indices = tuple(range(len(out_features)))
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained and (weights_path is None),
            features_only=True,
            out_indices=out_indices,
            in_chans=1,
        )

        # 加载自定义权重
        if weights_path is not None:
            self._load_weights(weights_path)

        # 输出通道与步幅映射
        feature_info = self.model.feature_info
        channels = feature_info.channels()
        strides = feature_info.reduction()
        self._stage_out_channels = {name: ch for name, ch in zip(out_features, channels)}
        self._stage_out_strides = {name: st for name, st in zip(out_features, strides)}

    def _get_model_name(self, dims: List[int]) -> str:
        """根据维度确定模型名称"""
        if dims[0] == 64:
            return "convnext_tiny"
        elif dims[0] == 96:
            return "convnext_tiny"
        elif dims[0] == 128:
            return "convnext_small"
        elif dims[0] == 192:
            return "convnext_base"
        elif dims[0] == 256:
            return "convnext_large"
        else:
            return "convnext_tiny"  # 默认

    def _load_weights(self, weights_path: str) -> None:
        """加载自定义权重"""
        state_dict = torch.load(weights_path, map_location="cpu")

        # 处理不同的键名格式
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        model_state = self.model.state_dict()
        filtered_state = {
            k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape
        }
        self.model.load_state_dict(filtered_state, strict=False)

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播。

        Args:
            x: (B, 1, H, W) 深度图

        Returns:
            多尺度特征字典 {层名: 特征图}
        """
        features = self.model(x)

        result = {}
        for i, feat in enumerate(features):
            stage_name = self.out_features[i]
            # timm 可能输出 (B, H, W, C)，需要转为 (B, C, H, W)
            if feat.dim() == 4:
                expected_ch = self._stage_out_channels[stage_name]
                if feat.shape[-1] == expected_ch and feat.shape[1] != expected_ch:
                    feat = feat.permute(0, 3, 1, 2).contiguous()
            result[stage_name] = feat

        return result

    @property
    def output_shape(self) -> Dict[str, Tuple[int, int, int, int]]:
        """返回输出形状 (通道数, 高度步长, 宽度步长)"""
        return {
            name: (self._stage_out_channels[name], self._stage_out_strides[name], self._stage_out_strides[name])
            for name in self.out_features
        }


# 便捷函数
def build_convnext_depth(
    config: Dict[str, Any]
) -> ConvNeXtDepth:
    """
    根据配置构建 ConvNeXt 深度骨干网络。

    Args:
        config: ConvNeXt 配置字典

    Returns:
        ConvNeXtDepth 模型
    """
    return ConvNeXtDepth(
        depths=config.get("depths", [3, 3, 9, 3]),
        dims=config.get("dims", [96, 192, 384, 768]),
        drop_path_rate=config.get("drop_path_rate", 0.0),
        layer_scale=config.get("layer_scale", 1e-6),
        out_features=config.get("out_features", ["res2", "res3", "res4", "res5"]),
        pretrained=config.get("pretrained", False),
        weights_path=config.get("weights", None),
    )
