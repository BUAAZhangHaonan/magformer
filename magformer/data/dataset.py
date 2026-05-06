# -*- coding: utf-8 -*-
"""
COCO RGB-D Instance Segmentation Dataset

纯 PyTorch 实现的 COCO 格式 RGB-D 数据集加载器。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import cv2


# =============================================================================
# COCO RGB-D Dataset
# =============================================================================
class CocoRgbdDataset(Dataset):
    """
    COCO 格式 RGB-D 实例分割数据集。

    目录结构:
        dataset_root/
        ├── annotations/
        │   ├── instances_train.json
        │   ├── instances_val.json
        │   └── instances_test.json
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── depth/
            └── depth_npy/
                ├── train/
                ├── val/
                └── test/

    支持:
        - RGB 图像读取
        - 深度 .npy 文件读取
        - 同步数据增强
        - COCO 格式标注
    """

    def __init__(
        self,
        dataset_root: str,
        ann_file: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        is_train: bool = True,
        has_annotations: bool = True,
    ):
        """
        Args:
            dataset_root: 数据集根目录
            ann_file: 标注文件名 (如 'instances_train.json')
            split: 数据分割名称 ('train', 'val', 'test')
            transform: 数据变换
            is_train: 是否训练模式
            has_annotations: 是否加载 COCO 标注 (False 用于无标签数据)
        """
        self.dataset_root = Path(dataset_root).resolve()
        self.split = split
        self.transform = transform
        self.is_train = is_train
        self.has_annotations = has_annotations

        # 图像目录
        self.image_dir = self.dataset_root / "images" / split
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}")

        if has_annotations:
            # 加载 COCO 标注
            ann_path = Path(ann_file)
            if not ann_path.is_absolute():
                candidate = (self.dataset_root / ann_file).resolve()
                if candidate.exists():
                    ann_path = candidate
                else:
                    # Avoid double "annotations/annotations/" nesting
                    if ann_file.startswith("annotations/"):
                        ann_path = (self.dataset_root / ann_file).resolve()
                    else:
                        ann_path = (self.dataset_root /
                                    "annotations" / ann_file).resolve()
            if not ann_path.exists():
                raise FileNotFoundError(f"Annotation file not found: {ann_path}")

            self.coco = COCO(str(ann_path))
            self.category_ids, self.class_names = self._resolve_single_class_metadata()
            self.category_id_to_label = {
                category_id: idx + 1 for idx, category_id in enumerate(self.category_ids)
            }
            self.label_to_category_id = {
                label: category_id for category_id, label in self.category_id_to_label.items()
            }

            # 获取所有图像
            self.image_ids = sorted(self.coco.imgs.keys())

            # 过滤空标注 (可选)
            # self.image_ids = self._filter_empty_annotations()
        else:
            # 无标注模式: 扫描图像目录构建文件列表
            import glob as glob_mod
            self.coco = None
            self.category_ids = [1]
            self.class_names = ["component"]
            self.category_id_to_label = {1: 1}
            self.label_to_category_id = {1: 1}
            exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
            files = []
            for ext in exts:
                files.extend(glob_mod.glob(str(self.image_dir / ext)))
            self._unlabeled_files = sorted([Path(f).name for f in files])
            self.image_ids = list(range(len(self._unlabeled_files)))

        # 深度目录: 兼容 depth/depth_npy/<split> 与 depth/<split>
        depth_candidates = [
            self.dataset_root / "depth" / "depth_npy" / split,
            self.dataset_root / "depth" / split,
        ]
        self.depth_dir = None
        for depth_candidate in depth_candidates:
            if depth_candidate.exists():
                self.depth_dir = depth_candidate
                break
        if self.depth_dir is None:
            raise FileNotFoundError(
                f"Depth directory not found. Tried: {depth_candidates[0]} and {depth_candidates[1]}"
            )

        # 噪声掩码目录 (可选)
        self.noise_mask_dir = self.dataset_root / "depth" / "depth_noise_mask" / split
        self.noise_mask_available = self.noise_mask_dir.exists()

        print(
            f"[CocoRgbdDataset] Loaded {len(self.image_ids)} images from {split} split"
        )

    def _resolve_single_class_metadata(self) -> Tuple[List[int], List[str]]:
        """Resolve and validate the shipped single-class dataset contract."""
        categories = self.coco.dataset.get("categories", []) or []
        categories_by_id = {
            int(category["id"]): str(category.get("name", f"class_{int(category['id'])}"))
            for category in categories
            if "id" in category
        }
        ann_category_ids = sorted(
            {
                int(ann["category_id"])
                for ann in self.coco.dataset.get("annotations", [])
                if ann.get("iscrowd", 0) == 0 and "category_id" in ann
            }
        )

        if ann_category_ids:
            if len(ann_category_ids) != 1:
                raise ValueError(
                    "MAGFormer ships a single-class dataset path and requires exactly one "
                    f"foreground category in annotations, got {ann_category_ids}."
                )
            category_id = ann_category_ids[0]
        else:
            category_ids = sorted(categories_by_id)
            if len(category_ids) != 1:
                raise ValueError(
                    "MAGFormer ships a single-class dataset path and requires exactly one "
                    "category when annotations are empty."
                )
            category_id = category_ids[0]

        category_name = categories_by_id.get(category_id, "component")
        return [category_id], [category_name]

    def _filter_empty_annotations(self) -> List[int]:
        """过滤掉没有标注的图像"""
        valid_ids = []
        for img_id in self.image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) > 0:
                valid_ids.append(img_id)
        return valid_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个样本。

        Returns:
            包含以下字段的字典:
            - image: RGB 图像 (HWC, uint8)
            - depth: 深度图 (H, float32) 或 (HWC, float32)
            - image_id: 图像 ID
            - annotations: 标注列表 (仅训练时)
        """
        img_id = self.image_ids[idx]

        # 读取图像信息
        if self.has_annotations and self.coco is not None:
            img_info = self.coco.loadImgs(img_id)[0]
            filename = img_info["file_name"]
            orig_h = img_info.get("height", None)
            orig_w = img_info.get("width", None)
        else:
            filename = self._unlabeled_files[img_id]
            orig_h = None
            orig_w = None

        # 读取 RGB 图像
        image_path = self.image_dir / filename
        if not image_path.exists():
            # 尝试直接使用 file_name 作为相对路径
            image_path = self.image_dir / Path(filename).name

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取深度图
        depth_path = self._get_depth_path(filename)
        depth = self._load_depth(depth_path)

        # 读取噪声掩码 (可选)
        noise_mask = None
        if self.noise_mask_available:
            noise_mask_path = self._get_noise_mask_path(filename)
            if noise_mask_path is not None:
                noise_mask = self._load_noise_mask(noise_mask_path)

        # 准备结果字典
        result = {
            "image": image,
            "depth": depth,
            "image_id": img_id,
            "height": orig_h if orig_h is not None else image.shape[0],
            "width": orig_w if orig_w is not None else image.shape[1],
        }

        if noise_mask is not None:
            result["noise_mask"] = noise_mask

        # 训练模式: 加载标注
        if self.is_train:
            annotations = self._load_annotations(img_id)
            masks, boxes, labels = self._build_instances(
                annotations, image.shape[:2])
            result["masks"] = masks
            result["boxes"] = boxes
            result["labels"] = labels

        # 应用变换
        if self.transform is not None:
            result = self.transform(result)

        return result

    def _get_depth_path(self, image_filename: str) -> Path:
        """获取深度文件路径"""
        # 替换扩展名为 .npy
        base_name = Path(image_filename).stem
        depth_path = self.depth_dir / f"{base_name}.npy"

        if not depth_path.exists():
            # 尝试 .npz
            depth_path = self.depth_dir / f"{base_name}.npz"

        return depth_path

    def _load_depth(self, path: Path) -> np.ndarray:
        """加载深度 .npy/.npz 文件"""
        if not path.exists():
            raise FileNotFoundError(f"Depth file not found: {path}")

        arr = np.load(path, allow_pickle=False)

        # 处理 npz 文件
        if isinstance(arr, np.lib.npyio.NpzFile):
            first_key = sorted(arr.files)[0]
            arr = arr[first_key]

        # 确保形状正确 (H, W) 或 (H, W, 1)
        if arr.ndim == 3:
            if arr.shape[2] == 1:
                arr = arr[:, :, 0]
            elif arr.shape[0] == 1:
                arr = arr[0]
            else:
                arr = arr[:, :, 0]

        return arr.astype(np.float32)

    def _get_noise_mask_path(self, image_filename: str) -> Optional[Path]:
        """获取噪声掩码文件路径"""
        if not self.noise_mask_available:
            return None

        base_name = Path(image_filename).stem
        extensions = [".png", ".jpg", ".jpeg", ".bmp", ".npy", ".npz"]

        for ext in extensions:
            mask_path = self.noise_mask_dir / f"{base_name}{ext}"
            if mask_path.exists():
                return mask_path

        return None

    def _load_noise_mask(self, path: Path) -> np.ndarray:
        """加载噪声掩码"""
        # npy/npz 文件
        if path.suffix in [".npy", ".npz"]:
            arr = np.load(path, allow_pickle=False)
            if isinstance(arr, np.lib.npyio.NpzFile):
                arr = arr[sorted(arr.files)[0]]

            if arr.ndim == 3:
                if arr.shape[2] == 1:
                    arr = arr[:, :, 0]
                elif arr.shape[0] == 1:
                    arr = arr[0]
                else:
                    arr = arr[:, :, 0]
        else:
            # 图像文件
            from PIL import Image

            arr = np.array(Image.open(path).convert("L"))

        # 归一化到 [0, 1]
        if arr.max() > 1:
            arr = arr / 255.0

        arr = (arr > 0.5).astype(np.float32)

        return arr

    def _load_annotations(self, img_id: int) -> List[Dict[str, Any]]:
        """
        加载 COCO 标注。

        Returns:
            标注列表，每个标注包含:
            - bbox: [x, y, w, h] (COCO 格式)
            - category_id: 类别 ID
            - segmentation: RLE 或 polygon
            - area: 面积
        """
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        # 过滤掉 iscrowd=1 的标注
        annotations = [
            ann for ann in annotations if ann.get("iscrowd", 0) == 0]

        return annotations

    def _build_instances(
        self,
        annotations: List[Dict[str, Any]],
        image_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """将 COCO 标注转换为 masks/boxes/labels。"""
        h, w = image_size
        masks = []
        boxes = []
        labels = []

        for ann in annotations:
            segmentation = ann.get("segmentation", None)
            if segmentation is None:
                continue

            if isinstance(segmentation, list):
                rles = coco_mask.frPyObjects(segmentation, h, w)
                mask = coco_mask.decode(rles)
                if mask.ndim == 3:
                    mask = mask.any(axis=2)
            else:
                mask = coco_mask.decode(segmentation)
                if mask.ndim == 3:
                    mask = mask[..., 0]

            mask = mask.astype(bool)
            if mask.sum() == 0:
                continue

            bbox = ann.get("bbox", None)
            if bbox is None:
                bbox = self._bbox_from_mask(mask)
            else:
                x, y, bw, bh = bbox
                bbox = [x, y, x + bw, y + bh]

            category_id = int(ann.get("category_id", self.category_ids[0]))
            if category_id not in self.category_id_to_label:
                raise ValueError(
                    "MAGFormer single-class dataset path received an unknown category_id "
                    f"{category_id}; expected one of {self.category_ids}."
                )

            masks.append(mask)
            boxes.append(bbox)
            labels.append(self.category_id_to_label[category_id])

        if len(masks) == 0:
            return (
                np.zeros((h, w, 0), dtype=bool),
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        masks = np.stack(masks, axis=2)
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        return masks, boxes, labels

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> List[int]:
        """从二值 mask 计算边界框 [x1, y1, x2, y2]。"""
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return [0, 0, 0, 0]
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        return [int(x1), int(y1), int(x2), int(y2)]

    def get_img_info(self, idx: int) -> Dict[str, Any]:
        """获取图像元信息"""
        img_id = self.image_ids[idx]
        return self.coco.loadImgs(img_id)[0]

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        批处理整理函数。

        Args:
            batch: 样本列表

        Returns:
            整理后的批次
        """
        # 提取各字段
        images = [item["image"] for item in batch]
        depths = [item["depth"] for item in batch]
        image_ids = [item["image_id"] for item in batch]

        # 堆叠为 Tensor
        images = torch.stack(images, dim=0)  # (B, C, H, W)
        depths = torch.stack(depths, dim=0)  # (B, 1, H, W)

        result = {
            "images": images,
            "depths": depths,
            "image_ids": torch.tensor(image_ids),
        }

        # 处理标注 (仅训练时)
        if "annotations" in batch[0]:
            result["annotations"] = [item["annotations"] for item in batch]

        # 处理噪声掩码
        if "noise_mask" in batch[0]:
            noise_masks = [item["noise_mask"] for item in batch]
            result["noise_masks"] = torch.stack(noise_masks, dim=0)

        return result
