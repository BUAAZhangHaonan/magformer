# -*- coding: utf-8 -*-
"""
RGB-D Data Transformations

同步的 RGB 和深度图数据增强，支持几何变换和光度增强。
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import cv2
import torch
import logging

from magformer.data.dataset import _InstanceBank

logger = logging.getLogger(__name__)


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


_NON_INSTANCE_ARRAY_KEYS = frozenset(
    {
        "image",
        "depth",
        "content_mask",
        "noise_mask",
        "image_id",
        "height",
        "width",
    }
)
_DECLARED_INSTANCE_METADATA_KEYS = frozenset(
    {
        "labels",
        "boxes",
        "area",
        "areas",
        "iscrowd",
        "annotation_ids",
        "instance_ids",
        "track_ids",
        "weights",
    }
)


def _select_instance_values(value: Any, keep: np.ndarray, key: str) -> Any:
    """Select a per-instance value without changing its container type."""
    expected = int(keep.shape[0])
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.shape[0] != expected:
            raise ValueError(
                f"per-instance field {key!r} must have leading dimension {expected}, "
                f"got shape {value.shape}"
            )
        return value[keep]
    if isinstance(value, list):
        if len(value) != expected:
            raise ValueError(
                f"per-instance field {key!r} must have length {expected}, got {len(value)}"
            )
        return [item for item, selected in zip(value, keep) if selected]
    if isinstance(value, tuple):
        if len(value) != expected:
            raise ValueError(
                f"per-instance field {key!r} must have length {expected}, got {len(value)}"
            )
        return tuple(item for item, selected in zip(value, keep) if selected)
    raise TypeError(
        f"per-instance field {key!r} must be a numpy array, list, or tuple; "
        f"got {type(value).__name__}"
    )


def _axis_occupancy(masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-instance column/row occupancy without allocating per-instance index lists.

    Returns (cols, rows) where cols[i, n] is True when instance n has any
    foreground pixel in column i, and rows[j, n] likewise for row j.  Both are
    exactly ``masks.any(axis=0)`` / ``masks.any(axis=1)`` for 0/1-valued masks,
    computed with cv2's parallel reduce when possible (bit-identical values,
    lower allocation cost).
    """
    if masks.ndim != 3:
        raise ValueError(f"expected HxWxN instance masks, got shape {masks.shape}")
    if masks.shape[2] == 0:
        return (
            np.zeros((masks.shape[1], 0), dtype=bool),
            np.zeros((masks.shape[0], 0), dtype=bool),
        )
    view = masks.view(np.uint8) if masks.dtype == np.bool_ else masks
    if view.dtype == np.uint8:
        try:
            cols = cv2.reduce(view, 0, cv2.REDUCE_MAX)
            rows = cv2.reduce(view, 1, cv2.REDUCE_MAX)
            return (
                cols.reshape(masks.shape[1], -1) > 0,
                rows.reshape(masks.shape[0], -1) > 0,
            )
        except cv2.error:
            pass
    # Fallback identical to the naive reduction.
    return masks.any(axis=0), masks.any(axis=1)


def _boxes_from_axis_occupancy(
    cols: np.ndarray, rows: np.ndarray, height: int, width: int, instance_count: int
) -> np.ndarray:
    """Exclusive-xyxy boxes derived from per-instance axis occupancy.

    Produces exactly the same values as scanning each mask with np.nonzero:
    x1/x2 = first/last+1 occupied column, y1/y2 = first/last+1 occupied row.
    """
    boxes = np.zeros((instance_count, 4), dtype=np.float32)
    if instance_count == 0:
        return boxes
    x1 = np.argmax(cols, axis=0)
    x2 = width - np.argmax(cols[::-1], axis=0)
    y1 = np.argmax(rows, axis=0)
    y2 = height - np.argmax(rows[::-1], axis=0)
    boxes[:, 0] = x1
    boxes[:, 1] = y1
    boxes[:, 2] = x2
    boxes[:, 3] = y2
    return boxes


def _select_mask_channels(masks: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Select kept instance channels; returns the input array when all are kept.

    np.compress over the channel axis is materially faster than boolean fancy
    indexing for many-channel masks, and skipping entirely when nothing is
    dropped avoids a full-array copy.  Values are untouched, so outputs are
    bit-identical to ``masks[:, :, keep]``.
    """
    if keep.all():
        return masks
    return np.compress(keep, masks, axis=2)


def _boxes_from_instance_masks(masks: np.ndarray) -> np.ndarray:
    """Return exclusive-xyxy boxes from foreground instance masks."""
    count = masks.shape[2] if masks.ndim == 3 else 0
    if count == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cols, rows = _axis_occupancy(masks)
    keep = cols.any(axis=0)
    empty = np.flatnonzero(~keep)
    if empty.size > 0:
        raise ValueError(
            f"cannot build a box for empty transformed instance {int(empty[0])}"
        )
    return _boxes_from_axis_occupancy(cols, rows, masks.shape[0], masks.shape[1], count)


def synchronize_instances_after_geometry(result: Dict[str, Any]) -> Dict[str, Any]:
    """Make transformed instance targets a single mask-defined contract.

    Geometric operations can erase an instance mask while its old box remains
    positive.  The mask is the supervision truth, so this boundary removes
    every zero-foreground instance, applies the same selector to every
    per-instance field, and rebuilds boxes from the retained transformed masks.
    A crop is allowed to remove every instance; the resulting zero-instance
    target is valid and is consumed by the criterion as an empty target.
    """
    if "masks" not in result:
        return result

    masks = result["masks"]
    if not isinstance(masks, np.ndarray) or masks.ndim != 3:
        raise ValueError(
            "instance masks must be an HxWxN numpy array at the geometric "
            f"synchronization boundary, got {type(masks).__name__} with "
            f"shape {getattr(masks, 'shape', None)}"
        )
    if "image" in result and masks.shape[:2] != result["image"].shape[:2]:
        raise ValueError(
            "instance masks and image must have identical spatial dimensions at "
            f"the geometric synchronization boundary, got masks={masks.shape[:2]} "
            f"and image={result['image'].shape[:2]}"
        )

    # 0/1-valued uint8 masks (produced by the packed resize path) may keep
    # their dtype through the geometric chain: occupancy, selection and sums
    # are value-identical for bool and 0/1 uint8, and ToTensor emits bool
    # tensors either way.  Anything else still canonicalizes to bool.
    if result["masks"].dtype not in (np.bool_, np.uint8):
        result["masks"] = result["masks"].astype(bool, copy=False)
    masks = result["masks"]
    instance_count = masks.shape[2]

    # Occupancy along both axes gives the per-instance keep mask and the
    # rebuilt boxes in two vectorized passes instead of one np.nonzero scan
    # per instance.
    if instance_count == 0:
        keep = np.zeros((0,), dtype=bool)
        cols = np.zeros((masks.shape[1], 0), dtype=bool)
        rows = np.zeros((masks.shape[0], 0), dtype=bool)
    else:
        cols, rows = _axis_occupancy(masks)
        keep = cols.any(axis=0)

    # Every array/list/tuple whose leading dimension is the instance count is
    # target metadata and must follow the exact same selector.  Image-level
    # fields are excluded explicitly so valid metadata cannot silently drift.
    for key, value in list(result.items()):
        if key == "masks" or key in _NON_INSTANCE_ARRAY_KEYS:
            continue
        if key in _DECLARED_INSTANCE_METADATA_KEYS:
            result[key] = _select_instance_values(value, keep, key)
        elif isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == instance_count:
            result[key] = _select_instance_values(value, keep, key)
        elif isinstance(value, (list, tuple)) and len(value) == instance_count:
            result[key] = _select_instance_values(value, keep, key)

    retained_masks = _select_mask_channels(masks, keep)
    result["masks"] = retained_masks
    result["boxes"] = _boxes_from_axis_occupancy(
        cols[:, keep] if instance_count else cols,
        rows[:, keep] if instance_count else rows,
        masks.shape[0],
        masks.shape[1],
        int(keep.sum()),
    )

    # Areas are geometric metadata as well.  When present, make them agree
    # with the retained masks rather than leaving stale pre-transform values.
    if retained_masks.shape[2] == 0:
        retained_areas = np.zeros((0,), dtype=np.int64)
    else:
        retained_areas = retained_masks.sum(axis=(0, 1))
    for area_key in ("area", "areas"):
        if area_key not in result:
            continue
        value = result[area_key]
        if isinstance(value, np.ndarray):
            result[area_key] = retained_areas.astype(value.dtype, copy=False)
        elif isinstance(value, list):
            result[area_key] = retained_areas.astype(np.int64).tolist()
        elif isinstance(value, tuple):
            result[area_key] = tuple(retained_areas.astype(np.int64).tolist())
        else:
            raise TypeError(
                f"per-instance field {area_key!r} must be a numpy array, list, or tuple; "
                f"got {type(value).__name__}"
            )

    return result


class SynchronizeInstancesAfterGeometry(Transform):
    """Explicit transform wrapper for the shared instance geometry boundary."""

    def __call__(self, result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return synchronize_instances_after_geometry(result)


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

        # Flipping is a bijection on pixel positions: it can never empty an
        # instance mask, the flipped boxes computed above are exactly the
        # boxes of the flipped masks, and mask areas are unchanged.  The
        # instance sets that reach this transform are mask-derived (decode
        # rejects empty masks and every geometric boundary rebuilds boxes
        # from masks), so synchronize_instances_after_geometry would be a
        # value-identical no-op here and is skipped to avoid a redundant
        # full-mask occupancy pass per sample.
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
            return synchronize_instances_after_geometry(result)

        # image: bilinear
        result["image"] = cv2.resize(
            result["image"], (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

        # depth: bilinear (continuous)
        if "depth" in result:
            result["depth"] = cv2.resize(
                result["depth"], (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )

        # masks: nearest (discrete).  The resized result stays uint8 0/1
        # through the rest of the geometric chain: INTER_NEAREST preserves
        # values exactly and every downstream consumer (occupancy reduces,
        # channel selection, ToTensor's .bool()) is value-identical for bool
        # and 0/1 uint8, so the final bool tensors are bit-identical while
        # avoiding two full-array dtype copies per sample.
        if "masks" in result:
            masks = result["masks"]
            if masks.dtype != np.uint8:
                masks = masks.astype(np.uint8)
            masks = cv2.resize(masks, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            if masks.ndim == 2:
                masks = masks[:, :, None]
            result["masks"] = masks

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

        return synchronize_instances_after_geometry(result)


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

        # Compute crop position with minimum-objects guard.
        # When random_crop=True and there are objects, try up to 10 random
        # crop locations. If every attempt drops all objects, fall back to
        # a center crop which is more likely to contain objects.
        min_objects = (
            max(1, len(result.get("boxes", []))) if len(result.get("boxes", [])) > 0 else 0
        )
        best_top, best_left = None, None
        best_n_valid = -1

        if self.random_crop and min_objects > 0:
            for _attempt in range(10):
                _top = random.randint(0, max(0, h - crop_h))
                _left = random.randint(0, max(0, w - crop_w))
                # Check how many boxes survive this crop
                boxes_check = result["boxes"].copy()
                boxes_check[:, [0, 2]] -= _left
                boxes_check[:, [1, 3]] -= _top
                boxes_check[:, 0] = np.clip(boxes_check[:, 0], 0, crop_w - 1)
                boxes_check[:, 2] = np.clip(boxes_check[:, 2], 0, crop_w - 1)
                boxes_check[:, 1] = np.clip(boxes_check[:, 1], 0, crop_h - 1)
                boxes_check[:, 3] = np.clip(boxes_check[:, 3], 0, crop_h - 1)
                _valid = (boxes_check[:, 2] > boxes_check[:, 0]) & (
                    boxes_check[:, 3] > boxes_check[:, 1]
                )
                _n_valid = int(_valid.sum())
                if _n_valid > best_n_valid:
                    best_n_valid = _n_valid
                    best_top, best_left = _top, _left
                if _n_valid >= min_objects:
                    break
            # If no attempt kept enough objects, fall back to center crop
            if best_n_valid < min_objects:
                best_top = (h - crop_h) // 2
                best_left = (w - crop_w) // 2
        else:
            best_top = (h - crop_h) // 2
            best_left = (w - crop_w) // 2

        top, left = best_top, best_left

        # 裁剪
        if "image" in result:
            result["image"] = result["image"][top : top + crop_h, left : left + crop_w].copy()

        if "depth" in result:
            result["depth"] = result["depth"][top : top + crop_h, left : left + crop_w].copy()

        if "masks" in result:
            result["masks"] = result["masks"][top : top + crop_h, left : left + crop_w].copy()

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
                result["masks"] = _select_mask_channels(result["masks"], valid)
            if "labels" in result:
                result["labels"] = result["labels"][valid]
            result["boxes"] = boxes[valid]

        return synchronize_instances_after_geometry(result)


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
# GPU-deferred photometric augmentation
# =============================================================================
_HSV_FLOAT_EPS = float(np.finfo(np.float32).eps)


class DeferredRGBPhotoAug(Transform):
    """Draw RGBPhotoAug's random factors here, apply them on device later.

    Draws the exact same ``random.uniform`` values in the exact same order as
    :class:`RGBPhotoAug` (so every later RNG consumer sees an identical
    stream), stores them in ``result["photo_aug_params"]`` as
    ``[brightness, contrast, saturation, hue_shift]`` float32, and leaves the
    image as uint8.  The trainer applies
    :func:`apply_photo_aug_batch` on the device before the model forward; the
    device math replicates the cv2 float32 HSV path, matching the CPU
    reference within 1/255 per pixel (verified over randomized factor sets;
    residual differences are floor-boundary rounding of <=1 uint8 step caused
    only by reduction-order differences in the scalar mean).
    """

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if "image" not in result:
            return result

        factors = [1.0, 1.0, 1.0, 0.0]
        # Same draw order as RGBPhotoAug: brightness, contrast, saturation, hue.
        if self.brightness > 0:
            factors[0] = random.uniform(1.0 - self.brightness, 1.0 + self.brightness)
        if self.contrast > 0:
            factors[1] = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
        if self.saturation > 0:
            factors[2] = random.uniform(1.0 - self.saturation, 1.0 + self.saturation)
        if self.hue > 0:
            factors[3] = random.uniform(-self.hue, self.hue)

        result["photo_aug_params"] = np.asarray(factors, dtype=np.float32)
        return result


def apply_photo_aug_batch(
    images: "torch.Tensor", params: "torch.Tensor"
) -> "torch.Tensor":
    """Apply deferred RGBPhotoAug factors to a uint8 image batch on any device.

    Args:
        images: (B, 3, H, W) uint8 tensor in RGB order.
        params: (B, 4) float32 tensor of per-sample
            [brightness, contrast, saturation, hue_shift] factors as drawn by
            DeferredRGBPhotoAug.

    Returns:
        (B, 3, H, W) float32 tensor with values in [0, 255], equal to
        ``floor(clip(augmented_01, 0, 1) * 255)`` — the same values the CPU
        uint8 path produces before its float conversion.
    """
    if images.dtype != torch.uint8:
        raise ValueError(
            f"deferred photometric aug expects uint8 images, got {images.dtype}"
        )
    if params.shape[0] != images.shape[0] or params.shape[-1] != 4:
        raise ValueError(
            f"photo_aug_params must have shape (B, 4) matching the image batch, "
            f"got {tuple(params.shape)} for {tuple(images.shape)}"
        )

    x = images.float() / 255.0
    fb = params[:, 0].view(-1, 1, 1, 1).to(x.dtype)
    fc = params[:, 1].view(-1, 1, 1, 1).to(x.dtype)

    # brightness
    x = x * fb
    # contrast (per-sample scalar mean, like numpy's image.mean())
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * fc + mean

    # HSV round trip using OpenCV's float32 conversion semantics
    # (s = diff / (|v| + FLT_EPSILON), hue sector denominators diff + eps).
    r, g, b = x.unbind(dim=1)
    maxc = x.max(dim=1).values
    minc = x.min(dim=1).values
    v = maxc
    diff = maxc - minc
    s = diff / (v.abs() + _HSV_FLOAT_EPS)
    de = diff + _HSV_FLOAT_EPS
    dz = diff == 0
    rc = (maxc == r) & ~dz
    gc = (maxc == g) & ~dz & ~rc
    bc = (maxc == b) & ~dz & ~rc & ~gc
    h = torch.zeros_like(v)
    h = torch.where(rc, (60.0 * (g - b)) / de, h)
    h = torch.where(gc, 120.0 + (60.0 * (b - r)) / de, h)
    h = torch.where(bc, 240.0 + (60.0 * (r - g)) / de, h)
    h = torch.where(h < 0, h + 360.0, h)

    # saturation and hue shift (hue wrapped modulo 180 degrees, like the
    # reference implementation).  h/s are (B, H, W), so the per-sample factors
    # need three-dimensional views to broadcast without adding a leading axis.
    s = s * params[:, 2].view(-1, 1, 1).to(x.dtype)
    h = torch.remainder(h + params[:, 3].view(-1, 1, 1).to(x.dtype) * 180.0, 180.0)

    h6 = h * (1.0 / 60.0)
    sector = torch.floor(h6)
    f = h6 - sector
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - (s * (1.0 - f)))
    idx = sector.to(torch.int64) % 6
    r_out = torch.zeros_like(v)
    g_out = torch.zeros_like(v)
    b_out = torch.zeros_like(v)
    sector_rgb = ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))
    for k in range(6):
        selected = idx == k
        r_k, g_k, b_k = sector_rgb[k]
        r_out = torch.where(selected, r_k, r_out)
        g_out = torch.where(selected, g_k, g_out)
        b_out = torch.where(selected, b_k, b_out)

    x = torch.stack((r_out, g_out, b_out), dim=1)
    x = x.clamp(0.0, 1.0)
    # The CPU reference quantizes with astype(np.uint8) (truncation toward
    # zero, equal to floor for the non-negative clipped values).
    return (x * 255.0).floor()


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

        depth = result["depth"].astype(np.float32, copy=True)
        valid_mask = np.isfinite(depth) & (depth > 0.0)
        valid_count = int(valid_mask.sum())

        # Zero is the canonical invalid-depth value throughout the RGB-D path.
        # Draw perturbations only for finite positive measurements so missing,
        # padded, NaN, and infinite pixels cannot become synthetic observations.
        depth[~valid_mask] = 0.0

        # 高斯噪声
        if self.gaussian_std > 0 and valid_count > 0:
            noise = np.random.normal(0, self.gaussian_std, size=valid_count)
            depth[valid_mask] += noise.astype(np.float32, copy=False)

        # 散斑噪声
        if self.speckle_std > 0 and valid_count > 0:
            noise = np.random.normal(0, self.speckle_std, size=valid_count)
            depth[valid_mask] *= 1.0 + noise.astype(np.float32, copy=False)

        # 随机丢弃
        if self.drop_prob > 0 and valid_count > 0:
            drop_mask = np.random.random(size=valid_count) < self.drop_prob
            valid_depth = depth[valid_mask]
            valid_depth[drop_mask] = self.drop_val
            depth[valid_mask] = valid_depth

        depth[~valid_mask] = 0.0

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

        source_depth = result["depth"].astype(np.float32, copy=False)
        valid_mask = np.isfinite(source_depth) & (source_depth > 0.0)
        depth = np.zeros_like(source_depth, dtype=np.float32)
        valid_depth = source_depth[valid_mask].astype(np.float32, copy=True)

        # 单位转换
        valid_depth = valid_depth * self.scale + self.shift

        # 截断
        if self.clip_max > self.clip_min:
            valid_depth = np.clip(valid_depth, self.clip_min, self.clip_max)

        # 归一化
        if self.norm == "none" or self.clip_max <= self.clip_min:
            pass
        elif self.per_sample_norm:
            # Dataset-level normalization using fixed clip range.
            # Per-sample min-max (using each sample's own min/max) destroys
            # cross-sample consistency: the same physical depth gets different
            # normalized values across images.  Instead, normalize by the fixed
            # dataset clip range so depth values are consistent across all samples.
            depth_range = self.clip_max - self.clip_min
            if depth_range > 1e-6:
                valid_depth = (valid_depth - self.clip_min) / depth_range
            valid_depth = np.clip(valid_depth, 0.0, 1.0)

            # 可选的目标区间
            if isinstance(self.norm, (list, tuple)) and len(self.norm) == 2:
                a, b = self.norm
                valid_depth = valid_depth * (b - a) + a
                valid_depth = np.clip(valid_depth, min(a, b), max(a, b))
        elif self.clip_min == 0.0 and self.clip_max == 1.0:
            # 仅截断，不归一化（已弃用：当数据在窄范围内时会导致训练失败）
            # 保留此分支以向后兼容，但建议使用 per_sample_norm=True
            pass
        else:
            # Min-max 归一化（基于 clip 范围）
            valid_depth = (valid_depth - self.clip_min) / (
                self.clip_max - self.clip_min + 1e-6
            )
            valid_depth = np.clip(valid_depth, 0.0, 1.0)

            # 可选的目标区间
            if isinstance(self.norm, (list, tuple)) and len(self.norm) == 2:
                a, b = self.norm
                valid_depth = valid_depth * (b - a) + a
                valid_depth = np.clip(valid_depth, min(a, b), max(a, b))

        depth[valid_mask] = valid_depth

        result["depth"] = depth
        result["depth_valid_mask"] = np.ascontiguousarray(valid_mask).astype(
            np.bool_, copy=False
        )

        return result


# =============================================================================
# Copy-Paste Augmentation Transform
# =============================================================================


class CopyPasteTransform(Transform):
    """Copy-Paste augmentation as a pipeline transform.

    Inserts AFTER geometric transforms (FixedSizeCrop) but BEFORE photometric
    augmentation and ToTensor. Pastes instances from the shared instance bank
    onto the current image, then updates masks/boxes/labels accordingly.

    Designed for small-object oversampling: when prefer_small=True, the bank
    biases selection toward objects with area < small_threshold pixels.
    """

    def __init__(
        self,
        instance_bank: _InstanceBank,
        prob: float = 0.5,
        max_paste_instances: int = 5,
        min_instance_area: int = 16,
        max_instance_area_ratio: float = 0.3,
        prefer_small: bool = True,
        scale_jitter: tuple = (0.8, 1.2),
        iou_threshold: float = 0.7,
    ):
        self.instance_bank = instance_bank
        self.prob = prob
        self.max_paste_instances = max_paste_instances
        self.min_instance_area = min_instance_area
        self.max_instance_area_ratio = max_instance_area_ratio
        self.prefer_small = prefer_small
        self.scale_jitter = scale_jitter
        self.iou_threshold = iou_threshold

    def __call__(self, result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Only apply during training, when masks are present
        if "masks" not in result or "boxes" not in result or "labels" not in result:
            return result

        if random.random() > self.prob:
            return result

        bank = self.instance_bank
        if len(bank) == 0:
            return result

        image = result["image"]
        masks = result["masks"]  # [H, W, N] bool
        if masks.dtype != np.bool_:
            # The packed resize path keeps 0/1 uint8 masks; copy-paste needs
            # real boolean indexing, so canonicalize here.
            masks = masks.astype(bool)
            result["masks"] = masks
        boxes = result["boxes"]  # [N, 4] float32 (x1, y1, x2, y2)
        labels = result["labels"]  # [N] int64

        h_img, w_img = image.shape[:2]
        total_area = h_img * w_img
        n_existing = masks.shape[2] if masks.ndim == 3 else 0

        # Sample instances from the bank
        instances = bank.sample(
            self.max_paste_instances,
            prefer_small=self.prefer_small,
        )
        if not instances:
            return result

        # Build instance-ID canvas for overlap checking
        mask_canvas = np.zeros((h_img, w_img), dtype=np.int32)
        for i in range(n_existing):
            mask_canvas[masks[:, :, i]] = i + 1

        new_masks_list = []
        new_boxes_list = []
        new_labels_list = []

        for inst in instances:
            crop_img = inst["crop_img"]
            crop_mask = inst["crop_mask"]
            label = inst["label"]
            area = inst["area"]

            if area < self.min_instance_area:
                continue
            if area > total_area * self.max_instance_area_ratio:
                continue

            # Scale jitter
            scale = random.uniform(*self.scale_jitter)
            if abs(scale - 1.0) > 0.01:
                new_h = int(crop_img.shape[0] * scale)
                new_w = int(crop_img.shape[1] * scale)
                if new_h < 2 or new_w < 2:
                    continue
                crop_img = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                crop_mask = cv2.resize(
                    crop_mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            crop_h, crop_w = crop_img.shape[:2]

            # Random paste location (allow partial out-of-bounds)
            paste_y = random.randint(-crop_h // 2, max(0, h_img - crop_h // 2))
            paste_x = random.randint(-crop_w // 2, max(0, w_img - crop_w // 2))

            # Clip to image boundaries
            y1 = max(0, paste_y)
            x1 = max(0, paste_x)
            y2 = min(h_img, paste_y + crop_h)
            x2 = min(w_img, paste_x + crop_w)

            if y1 >= y2 or x1 >= x2:
                continue

            # Source crop offsets
            sy1 = y1 - paste_y
            sx1 = x1 - paste_x
            sy2 = sy1 + (y2 - y1)
            sx2 = sx1 + (x2 - x1)

            paste_region_mask = crop_mask[sy1:sy2, sx1:sx2]

            # Check IoU with existing instances
            existing_ids = mask_canvas[y1:y2, x1:x2]
            overlap_area = np.sum(paste_region_mask & (existing_ids > 0))
            paste_area = np.sum(paste_region_mask)
            if paste_area > 0 and overlap_area / max(paste_area, 1) > self.iou_threshold:
                continue

            # Paste onto image
            mask_3ch = np.stack([paste_region_mask] * 3, axis=-1)
            image[y1:y2, x1:x2] = np.where(
                mask_3ch,
                crop_img[sy1:sy2, sx1:sx2],
                image[y1:y2, x1:x2],
            )

            # Update mask canvas
            instance_id = n_existing + len(new_masks_list) + 1
            mask_canvas[y1:y2, x1:x2] = np.where(
                paste_region_mask,
                instance_id,
                mask_canvas[y1:y2, x1:x2],
            )

            # Extract pasted mask
            pasted_mask = mask_canvas == instance_id
            pasted_area = int(pasted_mask.sum())
            if pasted_area < self.min_instance_area:
                mask_canvas[mask_canvas == instance_id] = 0
                continue

            # Compute bbox from pasted mask
            ys, xs = np.where(pasted_mask)
            new_box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

            new_masks_list.append(pasted_mask)
            new_boxes_list.append(new_box)
            new_labels_list.append(label)

        if not new_masks_list:
            return result

        # Append pasted instances to existing annotations
        pasted_masks = np.stack(new_masks_list, axis=2)
        pasted_boxes = np.stack(new_boxes_list, axis=0)
        pasted_labels = np.array(new_labels_list, dtype=labels.dtype)

        if n_existing > 0:
            result["masks"] = np.concatenate([masks, pasted_masks], axis=2)
            result["boxes"] = np.concatenate([boxes, pasted_boxes], axis=0)
            result["labels"] = np.concatenate([labels, pasted_labels], axis=0)
        else:
            result["masks"] = pasted_masks
            result["boxes"] = pasted_boxes
            result["labels"] = pasted_labels

        result["image"] = image

        return synchronize_instances_after_geometry(result)


# =============================================================================
# 转换为 Tensor
# =============================================================================
class ToTensor(Transform):
    """将 NumPy 数组转换为 PyTorch Tensor"""

    def __call__(self, result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if "image" in result:
            # HWC -> CHW
            image = result["image"].transpose(2, 0, 1)
            if "photo_aug_params" in result:
                # Photometric aug is deferred to the device: keep the image
                # uint8 (4x smaller H2D transfer and worker-side allocation)
                # and carry the drawn factors alongside.
                result["image"] = torch.from_numpy(np.ascontiguousarray(image))
            else:
                result["image"] = torch.from_numpy(image).float()

        if "photo_aug_params" in result:
            result["photo_aug_params"] = torch.from_numpy(
                np.ascontiguousarray(result["photo_aug_params"])
            )

        if "depth" in result:
            # HWC -> CHW (HW -> CHW)
            if result["depth"].ndim == 3:
                depth = result["depth"].transpose(2, 0, 1)
            else:
                depth = result["depth"][None, ...]
            result["depth"] = torch.from_numpy(depth).float()

        if "depth_valid_mask" in result:
            valid = np.ascontiguousarray(result["depth_valid_mask"])
            if valid.ndim == 2:
                valid = valid[None, ...]
            elif valid.ndim == 3 and valid.shape[-1] == 1:
                valid = valid.transpose(2, 0, 1)
            if valid.ndim != 3 or valid.shape[0] != 1:
                raise ValueError(
                    "depth_valid_mask must convert to shape (1,H,W), "
                    f"got {valid.shape}"
                )
            result["depth_valid_mask"] = torch.from_numpy(valid).bool()

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
        copy_paste_enabled: bool = False,
        copy_paste_prob: float = 0.5,
        copy_paste_max_paste: int = 5,
        copy_paste_min_area: int = 16,
        copy_paste_max_area_ratio: float = 0.3,
        copy_paste_prefer_small: bool = True,
        copy_paste_scale_jitter: tuple = (0.8, 1.2),
        copy_paste_iou_threshold: float = 0.7,
        instance_bank: Optional[_InstanceBank] = None,
        sahi_crop_size: Optional[int] = None,
        photometric_on_gpu: bool = False,
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
            sahi_crop_size: SAHI crop size for training. When set (e.g. 768),
                crops to a smaller tile instead of image_size, making small objects
                occupy a larger fraction. None = disabled (backward compatible).
        """
        transforms = []
        valid_flip_modes = {"horizontal", "vertical", "both", "none"}
        if random_flip not in valid_flip_modes:
            raise ValueError(
                f"random_flip must be one of {sorted(valid_flip_modes)}, got {random_flip!r}"
            )
        if copy_paste_enabled and instance_bank is None:
            raise ValueError("copy_paste_enabled=True requires the dataset's instance_bank")

        # 几何变换 (仅训练时)
        if is_train:
            transforms.append(InitContentMask())
            # 随机翻转
            if random_flip == "horizontal":
                transforms.append(RandomFlip(horizontal=True, prob=0.5))
            elif random_flip == "vertical":
                transforms.append(RandomFlip(vertical=True, prob=0.5))
            elif random_flip == "both":
                transforms.append(
                    RandomFlip(
                        horizontal=True,
                        vertical=True,
                        prob=0.5,
                    )
                )

            # 缩放
            transforms.append(
                ResizeScale(min_scale=min_scale, max_scale=max_scale, target_size=image_size)
            )

            # SAHI-style fixed size crop: when sahi_crop_size is set, crop to a
            # smaller tile so small objects occupy a larger fraction. The ResizeScale
            # step still resizes relative to image_size; only the crop window shrinks.
            _sahi_crop = (
                (sahi_crop_size, sahi_crop_size)
                if sahi_crop_size is not None
                else (image_size, image_size)
            )
            transforms.append(FixedSizeCrop(_sahi_crop, random_crop=True))

            # Copy-Paste augmentation (after geometric, before photometric/ToTensor)
            if copy_paste_enabled:
                transforms.append(
                    CopyPasteTransform(
                        instance_bank=instance_bank,
                        prob=copy_paste_prob,
                        max_paste_instances=copy_paste_max_paste,
                        min_instance_area=copy_paste_min_area,
                        max_instance_area_ratio=copy_paste_max_area_ratio,
                        prefer_small=copy_paste_prefer_small,
                        scale_jitter=copy_paste_scale_jitter,
                        iou_threshold=copy_paste_iou_threshold,
                    )
                )
        else:
            # 验证/测试时: 保持原始分辨率（与 COCO GT 严格对齐）
            # 任何 resize/crop 都会导致 mask 与 GT 尺寸不一致，从而 COCOeval IoU=0、AP=0。
            pass

        # RGB 光度增强 (仅训练时)
        if is_train and (
            rgb_brightness > 0 or rgb_contrast > 0 or rgb_saturation > 0 or rgb_hue > 0
        ):
            if photometric_on_gpu:
                # Draw the same factors with the same RNG sequence but apply
                # them on the training device (see DeferredRGBPhotoAug).
                transforms.append(
                    DeferredRGBPhotoAug(
                        brightness=rgb_brightness,
                        contrast=rgb_contrast,
                        saturation=rgb_saturation,
                        hue=rgb_hue,
                    )
                )
            else:
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

        # Normalize after noise so validity is decided from raw finite positive
        # measurements rather than inferred from normalized values.
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

        # 转换为 Tensor
        transforms.append(ToTensor())

        self.transform = Compose(transforms)

    def __call__(self, result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.transform(result)


def get_weak_augmentation(config):
    """Build weak augmentation pipeline for teacher / unlabeled data.

    Only includes geometric resize/crop + depth normalize + ToTensor.
    No photometric augmentation, no depth noise.
    """
    image_size = getattr(config, "image_size", 1024)
    depth_cfg = getattr(config, "depth", None)

    depth_scale = 0.001
    depth_shift = 0.0
    depth_clip_min = 0.0
    depth_clip_max = 1.0
    depth_norm = "minmax"
    depth_per_sample_norm = True

    if depth_cfg is not None:
        depth_scale = getattr(depth_cfg, "scale", 0.001)
        depth_shift = getattr(depth_cfg, "shift", 0.0)
        depth_clip_min = getattr(depth_cfg, "clip_min", 0.0)
        depth_clip_max = getattr(depth_cfg, "clip_max", 1.0)
        depth_norm = getattr(depth_cfg, "norm", "minmax")
        depth_per_sample_norm = getattr(depth_cfg, "per_sample_norm", True)

    return Compose(
        [
            InitContentMask(),
            RandomFlip(horizontal=True, prob=0.5),
            ResizeScale(min_scale=1.0, max_scale=1.0, target_size=image_size),
            FixedSizeCrop((image_size, image_size), random_crop=False),
            DepthNormalize(
                scale=depth_scale,
                shift=depth_shift,
                clip_min=depth_clip_min,
                clip_max=depth_clip_max,
                norm=depth_norm,
                per_sample_norm=depth_per_sample_norm,
            ),
            ToTensor(),
        ]
    )
