# -*- coding: utf-8 -*-
"""
MAGFormer Configuration Schema

使用 Pydantic 定义的配置类，提供类型验证和默认值。
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .vc_suda_schema import VCSUDAConfig


# =============================================================================
# 数据配置 (DataConfig)
# =============================================================================
class DepthConfig(BaseModel):
    """深度数据处理配置"""

    format: str = Field(default="I", description="深度图格式 (I=uint16)")
    scale: float = Field(default=0.001, description="深度缩放因子 (mm->m)")
    shift: float = Field(default=0.0, description="深度平移量")
    clip_min: float = Field(default=0.0, description="深度截断下限 (米)")
    clip_max: float = Field(default=1.0, description="深度截断上限 (米)")
    norm: Union[str, List[float]] = Field(
        default="minmax", description="归一化方法: 'none', 'minmax' 或 [min, max]"
    )
    per_sample_norm: bool = Field(
        default=True,
        description="是否对每个样本进行独立的 min-max 归一化到 [0, 1]（对于深度值已经在很窄范围内的数据至关重要）"
    )

    model_config = ConfigDict(extra="allow")


class DepthNoiseConfig(BaseModel):
    """深度噪声增强配置"""

    enabled: bool = Field(default=False, description="是否启用深度噪声")
    gaussian_std: float = Field(default=0.0, description="高斯噪声标准差")
    speckle_std: float = Field(default=0.0, description="散斑噪声标准差")
    drop_prob: float = Field(default=0.0, description="随机丢弃深度概率")
    drop_val: float = Field(default=0.0, description="丢弃填充值")

    model_config = ConfigDict(extra="allow")


class NoiseMaskConfig(BaseModel):
    """噪声掩码监督配置"""

    enabled: bool = Field(default=True, description="是否启用噪声掩码监督")
    check_dir: bool = Field(default=True, description="是否检查目录存在")

    model_config = ConfigDict(extra="allow")


class RGBPhotoAugConfig(BaseModel):
    """RGB 光度增强配置"""

    enabled: bool = Field(default=False, description="是否启用光度增强")
    brightness: float = Field(default=0.0, description="亮度调整范围 [0,1]")
    contrast: float = Field(default=0.0, description="对比度调整范围 [0,1]")
    saturation: float = Field(default=0.0, description="饱和度调整范围 [0,1]")
    hue: float = Field(default=0.0, description="色调调整范围 [0,0.5]")

    model_config = ConfigDict(extra="allow")


class DataConfig(BaseModel):
    """数据加载和增强配置"""

    # 数据路径
    dataset_root: str = Field(..., description="数据集根目录")
    train_ann: str = Field(
        default="annotations/instances_train.json", description="训练标注文件")
    val_ann: str = Field(
        default="annotations/instances_val.json", description="验证标注文件")
    test_ann: Optional[str] = Field(default=None, description="测试标注文件")
    train_split: str = Field(default="train", description="训练图像/深度子目录")
    val_split: str = Field(default="val", description="验证图像/深度子目录")
    test_split: str = Field(default="test", description="测试图像/深度子目录")
    class_names: List[str] = Field(
        default_factory=lambda: ["component"],
        description="类别名称列表（无数据集元信息时用于可视化）",
    )

    # 数据增强
    image_size: int = Field(default=1024, description="目标图像尺寸")
    min_scale: float = Field(default=0.1, description="最小缩放比例")
    max_scale: float = Field(default=2.0, description="最大缩放比例")
    random_flip: str = Field(
        default="horizontal", description="随机翻转: 'horizontal', 'vertical', 'none'")
    size_divisibility: int = Field(default=32, description="尺寸整除因子")

    # 格式设置
    format: str = Field(default="RGB", description="RGB 图像格式")

    # RGB 增强
    rgb_photo_aug: RGBPhotoAugConfig = Field(
        default_factory=RGBPhotoAugConfig, description="RGB 光度增强配置"
    )

    # 深度配置
    depth: DepthConfig = Field(
        default_factory=DepthConfig, description="深度数据配置")
    depth_noise: DepthNoiseConfig = Field(
        default_factory=DepthNoiseConfig, description="深度噪声配置"
    )
    noise_mask: NoiseMaskConfig = Field(
        default_factory=NoiseMaskConfig, description="噪声掩码配置"
    )

    model_config = ConfigDict(extra="allow")


# =============================================================================
# 模型配置 (ModelConfig)
# =============================================================================
class BackboneConfig(BaseModel):
    """骨干网络配置"""

    name: str = Field(..., description="骨干网络名称")
    weights: Optional[str] = Field(default=None, description="预训练权重路径")
    freeze_at: int = Field(default=0, description="冻结层数")
    pretrained: bool = Field(
        default=True, description="是否使用预训练权重 (当 weights 为 None 时)")
    out_features: Optional[List[str]] = Field(
        default=None, description="输出特征层名称")

    model_config = ConfigDict(extra="allow")


class SwinConfig(BaseModel):
    """Swin Transformer 配置"""

    pretrain_img_size: int = Field(default=224, description="预训练图像尺寸")
    patch_size: int = Field(default=4, description="Patch 尺寸")
    embed_dim: int = Field(default=96, description="嵌入维度")
    depths: List[int] = Field(default=[2, 2, 6, 2], description="每层深度")
    num_heads: List[int] = Field(default=[3, 6, 12, 24], description="注意力头数")
    window_size: int = Field(default=7, description="窗口尺寸")
    mlp_ratio: float = Field(default=4.0, description="MLP 扩展比例")
    qkv_bias: bool = Field(default=True, description="QKV 偏置")
    drop_rate: float = Field(default=0.0, description="Dropout 比率")
    attn_drop_rate: float = Field(default=0.0, description="注意力 Dropout 比率")
    drop_path_rate: float = Field(
        default=0.3, description="Stochastic Depth 比率")
    ape: bool = Field(default=False, description="绝对位置编码")
    patch_norm: bool = Field(default=True, description="Patch 归一化")
    out_features: List[str] = Field(
        default=["res2", "res3", "res4", "res5"], description="输出特征层"
    )

    model_config = ConfigDict(extra="allow")


class ConvNeXtConfig(BaseModel):
    """ConvNeXt 配置"""

    depths: List[int] = Field(default=[3, 3, 9, 3], description="每层深度")
    dims: List[int] = Field(default=[96, 192, 384, 768], description="每层维度")
    drop_path_rate: float = Field(default=0.0, description="Drop Path 比率")
    layer_scale: float = Field(default=1e-6, description="Layer Scale 初始值")

    model_config = ConfigDict(extra="allow")


class DepthPriorConfig(BaseModel):
    """深度先验提取配置"""

    enabled: bool = Field(default=True, description="是否启用深度先验")
    use_gradient: bool = Field(default=True, description="使用梯度先验")
    use_variance: bool = Field(default=True, description="使用方差先验")
    use_valid_hole: bool = Field(default=True, description="使用有效/空洞掩码")
    use_rgb_edge: bool = Field(default=False, description="使用 RGB 边缘一致性")
    var_kernel: int = Field(default=5, description="方差计算核大小")
    z_min: float = Field(default=0.0, description="深度有效范围下限")
    z_max: float = Field(default=1.0, description="深度有效范围上限")
    compute_on: str = Field(
        default="res3", description="计算分辨率: 'full', 'res2', 'res3', 'res4', 'res5'")

    model_config = ConfigDict(extra="allow")


class ModalityFusionConfig(BaseModel):
    """模态融合模块配置 (原 MGM)"""

    enabled: bool = Field(default=True, description="是否启用模态融合")
    mode: str = Field(default="legacy_gated",
                      description="融合模式: 'legacy_gated', 'direct_add', 'gated_add', 'film', 'cross_attn'")
    residual_alpha: float = Field(default=0.05, description="残差连接系数")
    loss_entropy_w: float = Field(default=0.01, description="熵损失权重")
    temp_init: float = Field(default=1.5, description="初始温度")
    temp_final: float = Field(default=1.0, description="最终温度")
    temp_steps: int = Field(default=3000, description="温度衰减步数")
    clamp_min: float = Field(default=0.05, description="置信度下限")
    clamp_max: float = Field(default=0.95, description="置信度上限")
    noise_mask_weight: float = Field(default=0.0, description="噪声掩码损失权重")
    hidden_dim: int = Field(default=256, description="隐藏层维度")
    feature_dims: List[int] = Field(
        default=[96, 192, 384, 768], description="各尺度特征维度"
    )
    scale_keys: List[str] = Field(
        default=["res2", "res3", "res4", "res5"], description="尺度键名"
    )
    fuse_scales: Optional[List[str]] = Field(
        default=None, description="实际执行融合的尺度键名；None 时回退到 scale_keys")
    priors: Optional[List[str]] = Field(
        default=None,
        description="轻量融合路径使用的先验名称列表，例如 ['edge', 'valid-hole', 'variance']。None 时回退到 legacy prior 布尔开关。",
    )
    cross_attn_heads: int = Field(default=8, description="cross-attn 模式的注意力头数")
    cross_attn_downsample: int = Field(
        default=8, description="cross-attn 模式中 key/value 的空间下采样倍率")
    post_fuse_norm: bool = Field(default=True, description="融合后归一化")

    # 深度先验
    prior: DepthPriorConfig = Field(
        default_factory=DepthPriorConfig, description="深度先验配置"
    )

    # 鲁棒归一化
    robust_norm_enabled: Optional[bool] = Field(
        default=None,
        description=(
            "鲁棒归一化开关。None 表示未显式设置，将回退到 legacy "
            "modality_fusion.prior.robust_norm（若存在）。"
        ),
    )
    robust_norm_method: str = Field(
        default="minmax", description="归一化方法: 'minmax', 'quantile'")

    model_config = ConfigDict(extra="allow")


class MaskFormerConfig(BaseModel):
    """Transformer 头部配置"""

    transformer_decoder_name: str = Field(
        default="MultiScaleMaskedTransformerDecoder", description="Transformer Decoder 名称"
    )
    deep_supervision: bool = Field(default=True, description="深度监督")
    no_object_weight: float = Field(default=0.1, description="无对象损失权重")
    class_weight: float = Field(default=1.0, description="分类损失权重")
    dice_weight: float = Field(default=1.0, description="Dice 损失权重")
    mask_weight: float = Field(default=5.0, description="Mask 损失权重")
    nheads: int = Field(default=8, description="注意力头数")
    dropout: float = Field(default=0.1, description="Dropout 比率")
    dim_feedforward: int = Field(default=2048, description="FFN 维度")
    enc_layers: int = Field(default=0, description="编码器层数")
    dec_layers: int = Field(default=6, description="解码器层数")
    pre_norm: bool = Field(default=False, description="预归一化")
    hidden_dim: int = Field(default=256, description="隐藏维度")
    num_object_queries: int = Field(default=100, description="对象查询数")

    # 训练设置
    train_num_points: int = Field(default=12544, description="训练采样点数")
    oversample_ratio: float = Field(default=3.0, description="过采样比例")
    importance_sample_ratio: float = Field(default=0.75, description="重要性采样比例")

    # Mask loss options
    balanced_ce: bool = Field(
        default=False, description="是否启用 class-balanced BCE mask loss（默认对齐 Mask2Former: False）")
    balanced_ce_min_fg_ratio: float = Field(
        default=0.01, description="balanced BCE 的最小前景比例（防止极端权重）")

    # 测试设置
    object_mask_threshold: float = Field(default=0.0, description="对象 Mask 阈值")
    overlap_threshold: float = Field(default=0.0, description="重叠阈值")

    model_config = ConfigDict(extra="allow")


class SemSegHeadConfig(BaseModel):
    """语义分割头部配置"""

    name: str = Field(default="MaskFormerHead", description="头部名称")
    ignore_value: int = Field(default=255, description="忽略值")
    num_classes: int = Field(default=1, description="类别数")
    loss_weight: float = Field(default=1.0, description="损失权重")
    convs_dim: int = Field(default=256, description="卷积维度")
    mask_dim: int = Field(default=256, description="Mask 维度")
    norm: str = Field(default="GN", description="归一化类型")
    transformer_enc_layers: int = Field(
        default=0, description="Transformer 编码器层数")
    pixel_decoder_name: str = Field(
        default="MSDeformAttnPixelDecoder", description="Pixel Decoder 名称"
    )
    in_features: List[str] = Field(
        default=["res2", "res3", "res4", "res5"], description="输入特征"
    )
    deformable_transformer_encoder_in_features: List[str] = Field(
        default=["res3", "res4", "res5"],
        description="Deformable encoder 输入特征（对齐 Mask2Former）",
    )
    transformer_dim_feedforward: int = Field(
        default=1024,
        description="Pixel Decoder Deformable Transformer Encoder 的 FFN 维度",
    )
    common_stride: int = Field(default=4, description="公共步长")

    model_config = ConfigDict(extra="allow")


class DPEConfig(BaseModel):
    """Depth Position Encoding 配置"""

    enabled: Optional[bool] = Field(
        default=None,
        description=(
            "是否启用 DPE。Canonical key: model.magformer.dpe.enabled. "
            "Legacy flat root-level dpe_enabled is normalized before validation."
        ),
    )
    beta: Optional[float] = Field(
        default=None,
        description=(
            "DPE Beta 参数。Canonical key: model.magformer.dpe.beta. "
            "Legacy flat root-level dpe_beta is normalized before validation."
        ),
    )

    model_config = ConfigDict(extra="allow")


class MagFormerModelConfig(BaseModel):
    """MAGFormer 模型配置"""

    depth_mode: str = Field(
        default="legacy",
        description="深度分支语义模式：'legacy' 保持旧重型配置兼容，'light' 表示轻量实验线默认值；不决定具体 backbone 实现文件。",
    )
    rgb_backbone: BackboneConfig = Field(..., description="RGB 骨干网络配置")
    depth_backbone: BackboneConfig = Field(..., description="深度骨干网络配置")

    # Swin 配置 (内嵌在 rgb_backbone 中)
    swin: Optional[SwinConfig] = Field(
        default=None, description="Swin Transformer 详细配置")

    # ConvNeXt 配置 (内嵌在 depth_backbone 中)
    convnext: Optional[ConvNeXtConfig] = Field(
        default=None, description="ConvNeXt 详细配置"
    )

    # 模态融合
    modality_fusion: ModalityFusionConfig = Field(
        default_factory=ModalityFusionConfig, description="模态融合模块配置"
    )

    # Transformer 解码器
    mask_former: MaskFormerConfig = Field(
        default_factory=MaskFormerConfig, description="Transformer 解码器配置"
    )

    # Sem Seg Head
    sem_seg_head: SemSegHeadConfig = Field(
        default_factory=SemSegHeadConfig, description="语义分割头部配置"
    )

    # DPE 配置（canonical nested path; legacy flat keys normalize before validation）
    dpe: DPEConfig = Field(default_factory=DPEConfig,
                           description="Depth Position Encoding 配置")

    # 像素归一化
    pixel_mean: List[float] = Field(
        default=[123.675, 116.280, 103.530], description="RGB 均值"
    )
    pixel_std: List[float] = Field(
        default=[58.395, 57.120, 57.375], description="RGB 标准差"
    )

    model_config = ConfigDict(extra="allow")


class BaseModelConfig(BaseModel):
    """基线模型配置 (UCN, MSMFormer, UOAIS)"""

    type: str = Field(..., description="模型类型: 'ucn', 'msmformer', 'uoa_is'")
    backbone: BackboneConfig = Field(..., description="骨干网络配置")
    num_classes: int = Field(default=1, description="类别数")

    model_config = ConfigDict(extra="allow")


class ModelConfig(BaseModel):
    """模型配置 (联合所有模型类型)"""

    meta_architecture: str = Field(
        default="MagFormer", description="元架构: 'MagFormer' 或基线模型名"
    )
    weights: Optional[str] = Field(default=None, description="模型权重路径")
    finetune_weights: Optional[str] = Field(default=None, description="微调权重路径")

    # MAGFormer 配置
    magformer: Optional[MagFormerModelConfig] = Field(
        default=None, description="MAGFormer 配置"
    )

    # Baseline 配置
    baseline: Optional[BaseModelConfig] = Field(
        default=None, description="基线模型配置"
    )

    @model_validator(mode="after")
    def validate_model_config(self):
        """验证模型配置"""
        arch = getattr(self, "meta_architecture", "MagFormer")

        if arch == "MagFormer":
            if getattr(self, "magformer", None) is None:
                self.magformer = MagFormerModelConfig()
        else:
            if getattr(self, "baseline", None) is None:
                self.baseline = BaseModelConfig(type=arch)

        return self

    model_config = ConfigDict(extra="allow")


# =============================================================================
# 求解器配置 (SolverConfig)
# =============================================================================
class SolverConfig(BaseModel):
    """训练求解器配置"""

    ims_per_batch: int = Field(default=16, description="批大小")
    base_lr: float = Field(default=0.0001, description="基础学习率")
    max_iter: int = Field(default=368750, description="最大迭代数")
    warmup_factor: float = Field(default=1.0, description="预热因子")
    warmup_iters: int = Field(default=10, description="预热迭代数")
    weight_decay: float = Field(default=0.05, description="权重衰减")
    weight_decay_norm: float = Field(default=0.0, description="归一化层权重衰减")
    weight_decay_embed: float = Field(default=0.0, description="嵌入层权重衰减")
    optimizer: str = Field(default="ADAMW", description="优化器类型")
    backbone_multiplier: float = Field(default=0.1, description="骨干网络学习率倍数")
    rgb_backbone_multiplier: Optional[float] = Field(
        default=None,
        description="RGB backbone 学习率倍数。None 时回退到 backbone_multiplier。",
    )
    depth_backbone_multiplier: Optional[float] = Field(
        default=None,
        description="Depth backbone 学习率倍数。None 时回退到 backbone_multiplier * 2.0。",
    )
    mgm_multiplier: float = Field(default=2.0, description="MGM/Fusion 学习率倍数")

    # 学习率调度
    lr_scheduler: str = Field(
        default="poly", description="学习率调度器: 'poly', 'multistep'")
    warmup_method: str = Field(
        default="linear", description="warmup 方法: 'linear', 'constant'")
    steps: List[int] = Field(default=[327778, 355092],
                             description="Step 调度器步数")
    gamma: float = Field(default=0.1, description="Step 调度器衰减系数")

    # 梯度裁剪
    clip_gradients: bool = Field(default=True, description="是否裁剪梯度")
    clip_type: str = Field(default="full_model", description="裁剪类型")
    clip_value: float = Field(default=0.01, description="裁剪值")
    norm_type: float = Field(default=2.0, description="范数类型")

    # AMP
    amp_enabled: bool = Field(default=True, description="是否启用混合精度")

    model_config = ConfigDict(extra="allow")


# =============================================================================
# 运行时配置 (RuntimeConfig)
# =============================================================================
class LoggerConfig(BaseModel):
    """日志配置"""

    type: str = Field(default="tensorboard",
                      description="日志类型: 'tensorboard', 'wandb', 'both'")
    log_dir: str = Field(default="output/logs", description="日志目录")
    project: str = Field(default="magformer", description="项目名称 (WandB)")
    entity: Optional[str] = Field(default=None, description="实体名称 (WandB)")
    run_name: Optional[str] = Field(default=None, description="日志运行名称")

    model_config = ConfigDict(extra="forbid")


class RuntimeConfig(BaseModel):
    """运行时配置"""

    # GPU 设置
    device: str = Field(default="cuda", description="设备: 'cuda', 'cpu'")
    gpus: List[int] = Field(default=[0], description="使用的 GPU ID")

    # 数据加载
    num_workers: int = Field(default=4, description="数据加载进程数")

    # 输出
    output_dir: str = Field(default="output", description="输出目录")

    # 训练设置
    seed: int = Field(default=42, description="随机种子")
    log_period: int = Field(default=100, description="日志周期")
    eval_period: int = Field(default=5000, description="评估周期")
    checkpoint_period: int = Field(default=5000, description="检查点保存周期")
    checkpoint_max_keep: Optional[int] = Field(
        default=2,
        description="最多保留的 numbered checkpoints 数量；None 或 <=0 表示保留全部",
    )
    resume: Optional[str] = Field(default=None, description="恢复检查点路径")
    skip_depth_sanity: bool = Field(default=False, description="是否跳过训练前 depth sanity 预检")

    # Optional runtime controls used by existing training configs.
    grad_accum_steps: int = Field(default=1, description="梯度累积步数")
    early_stop: Optional[Any] = Field(default=None, description="提前停止配置")
    eval_iou_types: Optional[List[str]] = Field(default=None, description="COCO 评估 IoU 类型")
    eval_max_images: Optional[int] = Field(default=None, description="最多评估图像数")
    eval_batch_size: int = Field(default=1, ge=1, description="验证/评估 DataLoader batch size")
    eval_saves_best: bool = Field(default=True, description="评估指标是否更新 model_best.pth")
    ema_enabled: bool = Field(default=False, description="是否启用普通 EMA")
    ema_decay: Optional[float] = Field(default=None, description="普通 EMA decay")
    ema_warmup_iters: Optional[int] = Field(default=None, description="普通 EMA 预热迭代数")
    contrastive_enabled: bool = Field(default=True, description="是否启用对比学习运行时开关")
    contrastive_weight: Optional[float] = Field(default=None, description="对比学习权重")
    contrastive_temperature: Optional[float] = Field(default=None, description="对比学习温度")

    # 日志
    logger: LoggerConfig = Field(
        default_factory=LoggerConfig, description="日志配置")

    # DDP
    ddp_enabled: bool = Field(default=False, description="是否启用分布式训练")
    find_unused_parameters: bool = Field(default=False, description="查找未使用参数")

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# 主配置 (MagFormerConfig)
# =============================================================================
class MagFormerConfig(BaseModel):
    """MAGFormer 主配置类"""

    name: str = Field(default="magformer", description="配置名称")

    # 子配置
    data: DataConfig = Field(..., description="数据配置")
    model: ModelConfig = Field(..., description="模型配置")
    solver: SolverConfig = Field(
        default_factory=SolverConfig, description="求解器配置")
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig, description="运行时配置")

    # 版本
    version: float = Field(default=2.0, description="配置版本")
    vc_suda: VCSUDAConfig = Field(default_factory=VCSUDAConfig, description="VC-SUDA domain adaptation config")

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_flat_keys(cls, values):
        from .validation import normalize_legacy_config_dict

        return normalize_legacy_config_dict(values)

    @model_validator(mode="after")
    def validate_vc_suda_data_protocol(self):
        """Reject VC-SUDA unlabeled target manifests that reuse eval annotations."""
        target_unlabeled_ann = self.vc_suda.target_unlabeled_ann
        if self.vc_suda.stage in {"C", "D", "E"} and target_unlabeled_ann:
            target_key = self._normalize_ann_key(target_unlabeled_ann)
            eval_ann_fields = {
                "val_ann": self.data.val_ann,
                "test_ann": self.data.test_ann,
            }
            for field_name, ann in eval_ann_fields.items():
                if ann and target_key == self._normalize_ann_key(ann):
                    raise ValueError(
                        "vc_suda.target_unlabeled_ann must not match "
                        f"data.{field_name}; unlabeled training data cannot reuse eval GT."
                    )
        return self

    @staticmethod
    def _normalize_ann_key(path: str) -> str:
        return str(path).replace("\\", "/").strip().lstrip("./")

    model_config = ConfigDict(extra="allow")


# =============================================================================
# 辅助函数
# =============================================================================
def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归合并配置字典。

    Args:
        base_config: 基础配置
        override_config: 覆盖配置

    Returns:
        合并后的配置
    """
    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result
