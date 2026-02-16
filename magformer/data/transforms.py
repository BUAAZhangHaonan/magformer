# -*- coding: utf-8 -*-
"""
RGB-D Data Transformations

同步的 RGB 和深度图数据增强，支持几何变换和光度增强。
"""

import random
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import cv2
import torch


# =============================================================================
# 基础变换类
# =============================================================================


class Transform:
    """变换基类"""

    def __call__(self, result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        应用变换。

        Args:
            result: 包含 'image', 'depth', 'masks' 等字段的字典

        Returns:
            变换后的字典
        """
        raise NotImplementedError


class Compose:
    """组合多个变换"""

    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for t in self.transforms:
            result = t(result)
        return result

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


# =============================================================================
# 几何变换 (同步应用于 RGB 和 Depth)
# =============================================================================


class InitContentMask(Transform):
    """
    初始化 content mask（1 表示有效内容，0 表示 padding）。

    该 mask 会随几何增强同步变换，用于构造 transformer 的 padding mask。
    """

    def __call__(self, result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if "content_mask" in result:
            return result
        if "image" not in result:
            return result
        h, w = result["image"].shape[:2]
        result["content_mask"] = np.ones((h, w), dtype=bool)
        return result


class RandomFlip(Transform):
    """随机水平或垂直翻转"""

    def __init__(self, horizontal: bool = True, vertical: bool = False, prob: float = 0.5):
        """
        Args:
            horizontal: 是否允许水平翻转
            vertical: 是否允许垂直翻转
            prob: 翻转概率
        """
        self.horizontal = horizontal
        self.vertical = vertical
        self.prob = prob

    def __call__(self, result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        h, w = None, None
        if "image" in result:
            h, w = result["image"].shape[:2]
        elif "depth" in result:
            h, w = result["depth"].shape[:2]

        # 水平翻转
        if self.horizontal and random.random() < self.prob:
            if "image" in result:
                result["image"] = np.fliplr(result["image"]).copy()
            if "depth" in result:
                result["depth"] = np.fliplr(result["depth"]).copy()
            if "masks" in result:
                result["masks"] = np.fliplr(result["masks"]).copy()
            if "content_mask" in result:
                result["content_mask"] = np.fliplr(result["content_mask"]).copy()
            if "noise_mask" in result:
                result["noise_mask"] = np.fliplr(result["noise_mask"]).copy()
            if "boxes" in result and w is not None:
                boxes = result["boxes"].copy()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                result["boxes"] = boxes

        # 垂直翻转
        if self.vertical and random.random() < self.prob:
            if "image" in result:
                result["image"] = np.flipud(result["image"]).copy()
            if "depth" in result:
                result["depth"] = np.flipud(result["depth"]).copy()
            if "masks" in result:
                result["masks"] = np.flipud(result["masks"]).copy()
            if "content_mask" in result:
                result["content_mask"] = np.flipud(result["content_mask"]).copy()
            if "noise_mask" in result:
                result["noise_mask"] = np.flipud(result["noise_mask"]).copy()
            if "boxes" in result and h is not None:
                boxes = result["boxes"].copy()
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
                result["boxes"] = boxes

        return result


class ResizeScale(Transform):
    """
    随机缩放 (LSJ-style)。

    语义对齐 detectron2 的 `T.ResizeScale(min_scale,max_scale,target_height,target_width)`：
    1) 采样 scale ~ U(min_scale, max_scale)
    2) 计算缩放后的“目标框”尺寸：target_size * scale
    3) 在保持长宽比的前提下，将原图缩放到能放入该目标框的最大尺寸
    """

    def __init__(
        self,
        min_scale: float = 0.1,
        max_scale: float = 2.0,
        target_size: int = 1024,
    ):
        """
        Args:
            min_scale: 最小缩放比例
            max_scale: 最大缩放比例
            target_size: 目标尺寸 (短边)
        """
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.target_size = target_size

    def __call__(self, result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if "image" not in result:
            return result

        # 1) sample scale
        scale = random.uniform(self.min_scale, self.max_scale)

        # 2) scaled target box
        target_h = max(int(round(self.target_size * scale)), 1)
        target_w = max(int(round(self.target_size * scale)), 1)

        # 3) resize to fit inside scaled target box (keep aspect)
        orig_h, orig_w = result["image"].shape[:2]
        resize_scale = min(float(target_h) / float(orig_h), float(target_w) / float(orig_w))
        new_h = max(int(round(orig_h * resize_scale)), 1)
        new_w = max(int(round(orig_w * resize_scale)), 1)

        if new_h == orig_h and new_w == orig_w:
            return result

        # image: bilinear
        result["image"] = cv2.resize(
            result["image"], (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

        # depth: bilinear (continuous)
        if "depth" in result:
            result["depth"] = cv2.resize(
                result["depth"], (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )

        # masks: nearest (discrete)
        if "masks" in result:
            masks = result["masks"].astype(np.uint8)
            masks = cv2.resize(masks, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            if masks.ndim == 2:
                masks = masks[:, :, None]
            result["masks"] = masks.astype(bool)

        # boxes: scale coords
        if "boxes" in result:
            scale_x = float(new_w) / float(orig_w)
            scale_y = float(new_h) / float(orig_h)
            boxes = result["boxes"].copy()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            result["boxes"] = boxes

        # optional content mask: nearest
        if "content_mask" in result:
            cm = result["content_mask"].astype(np.uint8)
            cm = cv2.resize(cm, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            result["content_mask"] = cm.astype(bool)

        # optional depth noise mask: nearest
        if "noise_mask" in result:
            nm = result["noise_mask"].astype(np.uint8)
            nm = cv2.resize(nm, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            result["noise_mask"] = nm.astype(np.float32)

        return result


class FixedSizeCrop(Transform):
    """固定尺寸裁剪 (中心或随机)"""

    def __init__(self, crop_size: Tuple[int, int], random_crop: bool = True):
        """
        Args:
            crop_size: (height, width)
            random_crop: 是否随机裁剪 (False 则中心裁剪)
        """
        self.crop_size = crop_size
        self.random_crop = random_crop

    def __call__(self, result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if "image" not in result:
            return result

        h, w = result["image"].shape[:2]
        crop_h, crop_w = self.crop_size

        # 如果图像小于目标尺寸，先 padding
        if h < crop_h or w < crop_w:
            pad_h = max(crop_h - h, 0)
            pad_w = max(crop_w - w, 0)

            if "image" in result:
                result["image"] = np.pad(
                    result["image"], ((0, pad_h), (0, pad_w), (0, 0)), mode="constant"
                )
            if "depth" in result:
                if result["depth"].ndim == 2:
                    result["depth"] = np.pad(
                        result["depth"], ((0, pad_h), (0, pad_w)), mode="constant"
                    )
                else:
                    result["depth"] = np.pad(
                        result["depth"], ((0, pad_h), (0, pad_w), (0, 0)), mode="constant"
                    )
            if "masks" in result:
                result["masks"] = np.pad(
                    result["masks"], ((0, pad_h), (0, pad_w), (0, 0)), mode="constant"
                )
            if "content_mask" in result:
                result["content_mask"] = np.pad(
                    result["content_mask"], ((0, pad_h), (0, pad_w)), mode="constant"
                )
            if "noise_mask" in result:
                result["noise_mask"] = np.pad(
                    result["noise_mask"], ((0, pad_h), (0, pad_w)), mode="constant"
                )

            h, w = result["image"].shape[:2]

        # 计算裁剪位置
        if self.random_crop:
            top = random.randint(0, max(0, h - crop_h))
            left = random.randint(0, max(0, w - crop_w))
        else:
            top = (h - crop_h) // 2
            left = (w - crop_w) // 2

        # 裁剪
        if "image" in result:
            result["image"] = result["image"][
                top : top + crop_h, left : left + crop_w
            ].copy()

        if "depth" in result:
            result["depth"] = result["depth"][
                top : top + crop_h, left : left + crop_w
            ].copy()

        if "masks" in result:
            result["masks"] = result["masks"][
                top : top + crop_h, left : left + crop_w
            ].copy()

        if "content_mask" in result:
            result["content_mask"] = result["content_mask"][
                top : top + crop_h, left : left + crop_w
            ].copy()

        if "noise_mask" in result:
            result["noise_mask"] = result["noise_mask"][
                top : top + crop_h, left : left + crop_w
            ].copy()

        if "boxes" in result:
            boxes = result["boxes"].copy()
            boxes[:, [0, 2]] -= left
            boxes[:, [1, 3]] -= top
            boxes[:, 0] = np.clip(boxes[:, 0], 0, crop_w - 1)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, crop_w - 1)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, crop_h - 1)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, crop_h - 1)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            if "masks" in result:
                result["masks"] = result["masks"][..., valid]
            if "labels" in result:
                result["labels"] = result["labels"][valid]
            result["boxes"] = boxes[valid]

        return result


# =============================================================================
# 光度增强 (仅应用于 RGB)
# =============================================================================


class RGBPhotoAug(Transform):
    """RGB 光度增强"""

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
    ):
        """
        Args:
            brightness: 亮度调整范围 [0,1]
            contrast: 对比度调整范围 [0,1]
            saturation: 饱和度调整范围 [0,1]
            hue: 色调调整范围 [0,0.5]
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if "image" not in result:
            return result

        image = result["image"].astype(np.float32) / 255.0

        # 亮度
        if self.brightness > 0:
            factor = random.uniform(1.0 - self.brightness, 1.0 + self.brightness)
            image = image * factor

        # 对比度
        if self.contrast > 0:
            factor = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
            mean = image.mean()
            image = (image - mean) * factor + mean

        # 饱和度和色调 (需要 HSV 转换)
        if self.saturation > 0 or self.hue > 0:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            if self.saturation > 0:
                factor = random.uniform(1.0 - self.saturation, 1.0 + self.saturation)
                hsv[:, :, 1] = hsv[:, :, 1] * factor

            if self.hue > 0:
                shift = random.uniform(-self.hue, self.hue)
                hsv[:, :, 0] = (hsv[:, :, 0] + shift * 180) % 180

            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        image = np.clip(image, 0, 1)
        result["image"] = (image * 255).astype(np.uint8)

        return result


# =============================================================================
# 深度噪声增强
# =============================================================================


class DepthNoiseAug(Transform):
    """深度噪声增强"""

    def __init__(
        self,
        gaussian_std: float = 0.0,
        speckle_std: float = 0.0,
        drop_prob: float = 0.0,
        drop_val: float = 0.0,
    ):
        """
        Args:
            gaussian_std: 高斯噪声标准差
            speckle_std: 散斑噪声标准差
            drop_prob: 随机丢弃概率
            drop_val: 丢弃填充值
        """
        self.gaussian_std = gaussian_std
        self.speckle_std = speckle_std
        self.drop_prob = drop_prob
        self.drop_val = drop_val

    def __call__(self, result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if "depth" not in result:
            return result

        depth = result["depth"].astype(np.float32)

        # 高斯噪声
        if self.gaussian_std > 0:
            noise = np.random.normal(0, self.gaussian_std, depth.shape)
            depth = depth + noise

        # 散斑噪声
        if self.speckle_std > 0:
            noise = np.random.normal(0, self.speckle_std, depth.shape)
            depth = depth * (1 + noise)

        # 随机丢弃
        if self.drop_prob > 0:
            mask = np.random.random(depth.shape) < self.drop_prob
            depth[mask] = self.drop_val

        result["depth"] = depth

        return result


# =============================================================================
# 深度归一化
# =============================================================================


class DepthNormalize(Transform):
    """深度归一化"""

    def __init__(
        self,
        scale: float = 0.001,
        shift: float = 0.0,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        norm: Union[str, List[float]] = "minmax",
        per_sample_norm: bool = True,
    ):
        """
        Args:
            scale: 缩放因子
            shift: 平移量
            clip_min: 截断下限
            clip_max: 截断上限
            norm: 归一化方法 ('none', 'minmax', [min, max])
            per_sample_norm: 是否对每个样本进行独立的 min-max 归一化到 [0, 1]
                             当深度数据已经在一个很窄的范围内（如 [0.93, 0.96]）时，
                             需要开启此选项将其扩展到完整的 [0, 1] 范围
        """
        self.scale = scale
        self.shift = shift
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.norm = norm
        self.per_sample_norm = per_sample_norm

    def __call__(self, result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if "depth" not in result:
            return result

        depth = result["depth"].astype(np.float32)

        # 单位转换
        depth = depth * self.scale + self.shift

        # 截断
        if self.clip_max > self.clip_min:
            depth = np.clip(depth, self.clip_min, self.clip_max)

        # 归一化
        if self.norm == "none" or self.clip_max <= self.clip_min:
            pass
        elif self.per_sample_norm:
            # 逐样本 min-max 归一化到 [0, 1]
            # 这对于深度值已经在很窄范围内（如 [0.93, 0.96]）的数据至关重要
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 1e-6:
                depth = (depth - d_min) / (d_max - d_min)
            # 最终裁剪到 [0, 1]
            depth = np.clip(depth, 0.0, 1.0)

            # 可选的目标区间
            if isinstance(self.norm, (list, tuple)) and len(self.norm) == 2:
                a, b = self.norm
                depth = depth * (b - a) + a
                depth = np.clip(depth, min(a, b), max(a, b))
        elif self.clip_min == 0.0 and self.clip_max == 1.0:
            # 仅截断，不归一化（已弃用：当数据在窄范围内时会导致训练失败）
            # 保留此分支以向后兼容，但建议使用 per_sample_norm=True
            pass
        else:
            # Min-max 归一化（基于 clip 范围）
            depth = (depth - self.clip_min) / (self.clip_max - self.clip_min + 1e-6)
            depth = np.clip(depth, 0.0, 1.0)

            # 可选的目标区间
            if isinstance(self.norm, (list, tuple)) and len(self.norm) == 2:
                a, b = self.norm
                depth = depth * (b - a) + a
                depth = np.clip(depth, min(a, b), max(a, b))

        result["depth"] = depth

        return result


# =============================================================================
# 转换为 Tensor
# =============================================================================


class ToTensor(Transform):
    """将 NumPy 数组转换为 PyTorch Tensor"""

    def __call__(self, result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if "image" in result:
            # HWC -> CHW
            image = result["image"].transpose(2, 0, 1)
            result["image"] = torch.from_numpy(image).float()

        if "depth" in result:
            # HWC -> CHW (HW -> CHW)
            if result["depth"].ndim == 3:
                depth = result["depth"].transpose(2, 0, 1)
            else:
                depth = result["depth"][None, ...]
            result["depth"] = torch.from_numpy(depth).float()

        if "masks" in result:
            # NHW -> CNH (需要转置)
            masks = result["masks"].transpose(2, 0, 1)
            result["masks"] = torch.from_numpy(masks).bool()

        if "boxes" in result:
            result["boxes"] = torch.from_numpy(result["boxes"]).float()

        if "labels" in result:
            result["labels"] = torch.from_numpy(result["labels"]).long()

        if "content_mask" in result:
            cm = np.ascontiguousarray(result["content_mask"])
            result["content_mask"] = torch.from_numpy(cm).bool()

        if "noise_mask" in result:
            nm = np.ascontiguousarray(result["noise_mask"]).astype(np.float32)
            if nm.ndim == 3:
                if nm.shape[0] == 1:
                    nm = nm[0]
                elif nm.shape[-1] == 1:
                    nm = nm[..., 0]
                else:
                    nm = nm[..., 0]
            # Store as (1, H, W) float mask for MGM noise supervision.
            result["noise_mask"] = torch.from_numpy(nm).unsqueeze(0)

        return result


# =============================================================================
# RGBD 组合变换
# =============================================================================


class RGBDTransform:
    """
    RGB-D 组合数据变换。

    支持:
    - 几何变换 (同步应用于 RGB 和 Depth)
    - RGB 光度增强
    - 深度噪声和归一化
    """

    def __init__(
        self,
        image_size: int = 1024,
        min_scale: float = 0.1,
        max_scale: float = 2.0,
        random_flip: str = "horizontal",
        rgb_brightness: float = 0.0,
        rgb_contrast: float = 0.0,
        rgb_saturation: float = 0.0,
        rgb_hue: float = 0.0,
        depth_scale: float = 0.001,
        depth_shift: float = 0.0,
        depth_clip_min: float = 0.0,
        depth_clip_max: float = 1.0,
        depth_norm: Union[str, List[float]] = "minmax",
        depth_per_sample_norm: bool = True,
        depth_gaussian_std: float = 0.0,
        depth_speckle_std: float = 0.0,
        depth_drop_prob: float = 0.0,
        depth_drop_val: float = 0.0,
        is_train: bool = True,
    ):
        """
        Args:
            image_size: 目标图像尺寸
            min_scale: 最小缩放比例
            max_scale: 最大缩放比例
            random_flip: 随机翻转类型 ('horizontal', 'vertical', 'none')
            rgb_brightness: RGB 亮度增强
            rgb_contrast: RGB 对比度增强
            rgb_saturation: RGB 饱和度增强
            rgb_hue: RGB 色调增强
            depth_scale: 深度缩放因子
            depth_shift: 深度平移量
            depth_clip_min: 深度截断下限
            depth_clip_max: 深度截断上限
            depth_norm: 深度归一化方法
            depth_per_sample_norm: 是否对每个样本进行独立的 min-max 归一化到 [0, 1]
                                   （对于深度值已经在很窄范围内的数据至关重要）
            depth_gaussian_std: 深度高斯噪声
            depth_speckle_std: 深度散斑噪声
            depth_drop_prob: 深度丢弃概率
            depth_drop_val: 深度丢弃填充值
            is_train: 是否训练模式
        """
        transforms = []

        # 几何变换 (仅训练时)
        if is_train:
            transforms.append(InitContentMask())
            # 随机翻转
            if random_flip == "horizontal":
                transforms.append(RandomFlip(horizontal=True, prob=0.5))
            elif random_flip == "vertical":
                transforms.append(RandomFlip(vertical=True, prob=0.5))
            elif random_flip != "none":
                transforms.append(
                    RandomFlip(
                        horizontal=(random_flip == "both"),
                        vertical=(random_flip == "both"),
                        prob=0.5,
                    )
                )

            # 缩放
            transforms.append(
                ResizeScale(min_scale=min_scale, max_scale=max_scale, target_size=image_size)
            )

            # 固定尺寸裁剪
            transforms.append(FixedSizeCrop((image_size, image_size), random_crop=True))
        else:
            # 验证/测试时: 保持原始分辨率（与 COCO GT 严格对齐）
            # 任何 resize/crop 都会导致 mask 与 GT 尺寸不一致，从而 COCOeval IoU=0、AP=0。
            pass

        # 深度归一化
        transforms.append(
            DepthNormalize(
                scale=depth_scale,
                shift=depth_shift,
                clip_min=depth_clip_min,
                clip_max=depth_clip_max,
                norm=depth_norm,
                per_sample_norm=depth_per_sample_norm,
            )
        )

        # RGB 光度增强 (仅训练时)
        if is_train and (rgb_brightness > 0 or rgb_contrast > 0 or rgb_saturation > 0 or rgb_hue > 0):
            transforms.append(
                RGBPhotoAug(
                    brightness=rgb_brightness,
                    contrast=rgb_contrast,
                    saturation=rgb_saturation,
                    hue=rgb_hue,
                )
            )

        # 深度噪声 (仅训练时)
        if is_train and (depth_gaussian_std > 0 or depth_speckle_std > 0 or depth_drop_prob > 0):
            transforms.append(
                DepthNoiseAug(
                    gaussian_std=depth_gaussian_std,
                    speckle_std=depth_speckle_std,
                    drop_prob=depth_drop_prob,
                    drop_val=depth_drop_val,
                )
            )

        # 转换为 Tensor
        transforms.append(ToTensor())

        self.transform = Compose(transforms)

    def __call__(self, result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.transform(result)
