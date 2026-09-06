# -*- coding: utf-8 -*-
"""
COCO RGB-D Instance Segmentation Dataset

纯 PyTorch 实现的 COCO 格式 RGB-D 数据集加载器。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections.abc import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import cv2
import threading
import random
import logging

logger = logging.getLogger(__name__)


_WORKER_STATE_VERSION = 1


def _encode_python_rng_state() -> Dict[str, Any]:
    version, internal_state, gaussian_cache = random.getstate()
    return {
        "version": int(version),
        "internal_state": torch.tensor(internal_state, dtype=torch.int64),
        "gaussian_cache": gaussian_cache,
    }


def _decode_python_rng_state(state: Any) -> tuple:
    if not isinstance(state, Mapping):
        raise TypeError("Python RNG state must be a mapping")
    version = state.get("version")
    internal_state = state.get("internal_state")
    gaussian_cache = state.get("gaussian_cache")
    if type(version) is not int:
        raise TypeError("Python RNG state version must have type int")
    if not torch.is_tensor(internal_state) or internal_state.ndim != 1:
        raise TypeError("Python RNG internal_state must be a one-dimensional tensor")
    if gaussian_cache is not None and type(gaussian_cache) is not float:
        raise TypeError("Python RNG gaussian_cache must be float or None")
    return (
        version,
        tuple(int(value) for value in internal_state.tolist()),
        gaussian_cache,
    )


def _encode_numpy_rng_state() -> Dict[str, Any]:
    bit_generator, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "bit_generator": str(bit_generator),
        "keys": torch.from_numpy(keys.astype(np.int64, copy=True)),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _decode_numpy_rng_state(state: Any) -> tuple:
    if not isinstance(state, Mapping):
        raise TypeError("NumPy RNG state must be a mapping")
    bit_generator = state.get("bit_generator")
    keys = state.get("keys")
    position = state.get("position")
    has_gauss = state.get("has_gauss")
    cached_gaussian = state.get("cached_gaussian")
    if type(bit_generator) is not str:
        raise TypeError("NumPy RNG bit_generator must have type str")
    if not torch.is_tensor(keys) or keys.ndim != 1:
        raise TypeError("NumPy RNG keys must be a one-dimensional tensor")
    if type(position) is not int or type(has_gauss) is not int:
        raise TypeError("NumPy RNG position and has_gauss must have type int")
    if type(cached_gaussian) is not float:
        raise TypeError("NumPy RNG cached_gaussian must have type float")
    return (
        bit_generator,
        keys.detach().cpu().numpy().astype(np.uint32, copy=True),
        position,
        has_gauss,
        cached_gaussian,
    )


def _decode_torch_rng_state(state: Any) -> torch.Tensor:
    if not torch.is_tensor(state) or state.dtype != torch.uint8 or state.ndim != 1:
        raise TypeError("Torch worker RNG state must be a one-dimensional uint8 tensor")
    return state.detach().cpu().clone()


# =============================================================================
# Instance Bank for Copy-Paste Augmentation
# =============================================================================
class _InstanceBank:
    """Thread-safe bank of cropped instances for copy-paste augmentation.

    Stores small object crops (image patch, mask, label) from recently
    loaded images so they can be pasted onto other images during training.
    """

    def __init__(self, capacity: int = 200, small_threshold: int = 1024):
        self.capacity = capacity
        self.small_threshold = small_threshold  # area in pixels (32x32=1024)
        self._lock = threading.Lock()
        self._small_bank: List[Dict[str, Any]] = []
        self._all_bank: List[Dict[str, Any]] = []

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    @staticmethod
    def _entry_to_state(entry: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "crop_img": torch.from_numpy(np.ascontiguousarray(entry["crop_img"])).clone(),
            "crop_mask": torch.from_numpy(np.ascontiguousarray(entry["crop_mask"])).clone(),
            "label": int(entry["label"]),
            "area": int(entry["area"]),
            "src_h": int(entry["src_h"]),
            "src_w": int(entry["src_w"]),
        }

    @staticmethod
    def _entry_from_state(entry: Any) -> Dict[str, Any]:
        if not isinstance(entry, Mapping):
            raise TypeError("Instance-bank entry must be a mapping")
        crop_img = entry.get("crop_img")
        crop_mask = entry.get("crop_mask")
        if not torch.is_tensor(crop_img) or not torch.is_tensor(crop_mask):
            raise TypeError("Instance-bank crops must be tensors")
        scalars = {}
        for field in ("label", "area", "src_h", "src_w"):
            value = entry.get(field)
            if type(value) is not int:
                raise TypeError(f"Instance-bank {field} must have type int")
            scalars[field] = value
        return {
            "crop_img": np.ascontiguousarray(crop_img.detach().cpu().numpy()).copy(),
            "crop_mask": np.ascontiguousarray(crop_mask.detach().cpu().numpy()).copy(),
            **scalars,
        }

    def state_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "version": 1,
                "capacity": int(self.capacity),
                "small_threshold": int(self.small_threshold),
                "small_bank": [self._entry_to_state(entry) for entry in self._small_bank],
                "all_bank": [self._entry_to_state(entry) for entry in self._all_bank],
            }

    def load_state_dict(self, state: Any) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("Instance-bank state must be a mapping")
        if state.get("version") != 1:
            raise ValueError(f"Unsupported instance-bank state version: {state.get('version')!r}")
        for field, current in (
            ("capacity", self.capacity),
            ("small_threshold", self.small_threshold),
        ):
            restored = state.get(field)
            if type(restored) is not int:
                raise TypeError(f"Instance-bank {field} must have type int")
            if restored != current:
                raise ValueError(
                    f"Instance-bank {field} mismatch: " f"checkpoint={restored}, current={current}"
                )
        small_state = state.get("small_bank")
        all_state = state.get("all_bank")
        if not isinstance(small_state, list) or not isinstance(all_state, list):
            raise TypeError("Instance-bank entries must be stored as lists")
        small_bank = [self._entry_from_state(entry) for entry in small_state]
        all_bank = [self._entry_from_state(entry) for entry in all_state]
        if len(small_bank) > self.capacity or len(all_bank) > self.capacity:
            raise ValueError("Instance-bank state exceeds configured capacity")
        with self._lock:
            self._small_bank = small_bank
            self._all_bank = all_bank

    def deposit(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Add instances from one image into the bank.

        Args:
            image: [H, W, 3] uint8 RGB image
            masks: [H, W, N] bool instance masks
            boxes: [N, 4] float32 boxes (x1, y1, x2, y2)
            labels: [N] int64 labels
        """
        if masks.ndim != 3 or masks.shape[2] == 0:
            return
        h, w = image.shape[:2]
        n_instances = masks.shape[2]

        new_small = []
        new_all = []

        for i in range(n_instances):
            mask_i = masks[:, :, i]
            area = int(mask_i.sum())
            if area < 4:
                continue

            box = boxes[i].astype(int)
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            # Clamp and add padding
            pad = 2
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad + 1)
            y2 = min(h, y2 + pad + 1)
            if x2 <= x1 or y2 <= y1:
                continue

            crop_img = image[y1:y2, x1:x2].copy()
            crop_mask = mask_i[y1:y2, x1:x2].copy()
            label = int(labels[i])

            entry = {
                "crop_img": crop_img,
                "crop_mask": crop_mask,
                "label": label,
                "area": area,
                "src_h": h,
                "src_w": w,
            }

            new_all.append(entry)
            if area < self.small_threshold:
                new_small.append(entry)

        if not new_small and not new_all:
            return

        with self._lock:
            self._small_bank.extend(new_small)
            self._all_bank.extend(new_all)
            # Trim to capacity, keeping most recent
            if len(self._small_bank) > self.capacity:
                self._small_bank = self._small_bank[-self.capacity :]
            if len(self._all_bank) > self.capacity:
                self._all_bank = self._all_bank[-self.capacity :]

    def sample(
        self,
        n: int,
        prefer_small: bool = True,
        small_weight: float = 3.0,
    ) -> List[Dict[str, Any]]:
        """Sample n instances from the bank.

        Args:
            n: Number of instances to sample.
            prefer_small: If True, bias toward small objects.
            small_weight: Relative weight for small objects when prefer_small=True.

        Returns:
            List of instance dicts.
        """
        with self._lock:
            small_available = list(self._small_bank)
            all_available = list(self._all_bank)

        if not all_available:
            return []

        results = []
        for _ in range(n):
            if not all_available:
                break

            if prefer_small and small_available:
                # Mix: sample from small with higher probability
                if random.random() < (small_weight / (small_weight + 1.0)):
                    idx = random.randint(0, len(small_available) - 1)
                    results.append(small_available[idx])
                    continue

            idx = random.randint(0, len(all_available) - 1)
            results.append(all_available[idx])

        return results

    def __len__(self) -> int:
        with self._lock:
            return len(self._all_bank)


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
        copy_paste_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            dataset_root: 数据集根目录
            ann_file: 标注文件名 (如 'instances_train.json')
            split: 数据分割名称 ('train', 'val', 'test')
            transform: 数据变换
            is_train: 是否训练模式
            has_annotations: 是否加载 COCO 标注 (False 用于无标签数据)
            copy_paste_config: Copy-paste augmentation config dict with keys:
                enabled, prob, max_paste_instances, min_instance_area,
                max_instance_area_ratio, prefer_small, small_threshold,
                bank_capacity, scale_jitter, iou_threshold
        """
        self.dataset_root = Path(dataset_root).resolve()
        self.split = split
        self.transform = transform
        self.is_train = is_train
        self.has_annotations = has_annotations

        # Copy-paste augmentation config
        self._copy_paste_enabled = False
        self._copy_paste_prob = 0.5
        self._copy_paste_max_paste = 5
        self._copy_paste_min_area = 16
        self._copy_paste_max_area_ratio = 0.3
        self._copy_paste_prefer_small = True
        self._copy_paste_scale_jitter = (0.8, 1.2)
        self._copy_paste_iou_threshold = 0.7
        self._instance_bank: Optional[_InstanceBank] = None

        if copy_paste_config and copy_paste_config.get("enabled", False):
            self._copy_paste_enabled = True
            self._copy_paste_prob = copy_paste_config.get("prob", 0.5)
            self._copy_paste_max_paste = copy_paste_config.get("max_paste_instances", 5)
            self._copy_paste_min_area = copy_paste_config.get("min_instance_area", 16)
            self._copy_paste_max_area_ratio = copy_paste_config.get("max_instance_area_ratio", 0.3)
            self._copy_paste_prefer_small = copy_paste_config.get("prefer_small", True)
            self._copy_paste_scale_jitter = tuple(copy_paste_config.get("scale_jitter", (0.8, 1.2)))
            self._copy_paste_iou_threshold = copy_paste_config.get("iou_threshold", 0.7)

            # Each worker owns the bank through its dataset instance. This makes
            # worker snapshots complete and also works with spawn-based workers.
            bank_capacity = copy_paste_config.get("bank_capacity", 200)
            small_threshold = copy_paste_config.get("small_threshold", 1024)
            self._instance_bank = _InstanceBank(
                capacity=bank_capacity,
                small_threshold=small_threshold,
            )

            logger.info(
                f"[CopyPaste] Enabled: prob={self._copy_paste_prob}, "
                f"max_paste={self._copy_paste_max_paste}, "
                f"prefer_small={self._copy_paste_prefer_small}"
            )

        # 图像目录
        self.image_dir = self.dataset_root / "images" / split
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        if has_annotations:
            # 加载 COCO 标注
            ann_path = Path(ann_file)
            if not ann_path.is_absolute():
                candidate = (self.dataset_root / ann_file).resolve()
                if candidate.exists():
                    ann_path = candidate
                else:
                    if ann_file.startswith("annotations/"):
                        ann_path = (self.dataset_root / ann_file).resolve()
                    else:
                        ann_path = (self.dataset_root / "annotations" / ann_file).resolve()
            if not ann_path.exists():
                raise FileNotFoundError(f"Annotation file not found: {ann_path}")

            self.coco = COCO(str(ann_path))
            self.category_ids, self.class_names = self._resolve_single_class_metadata()
            self.category_id_to_label = {
                category_id: idx for idx, category_id in enumerate(self.category_ids)
            }
            self.label_to_category_id = {
                label: category_id for category_id, label in self.category_id_to_label.items()
            }

            # 获取所有图像
            self.image_ids = sorted(self.coco.imgs.keys())
        else:
            import glob as glob_mod

            self.coco = None
            self.category_ids = [1]
            self.class_names = ["component"]
            self.category_id_to_label = {1: 1}
            self.label_to_category_id = {1: 1}
            exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
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

        print(f"[CocoRgbdDataset] Loaded {len(self.image_ids)} images from {split} split")

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

    @property
    def instance_bank(self) -> Optional[_InstanceBank]:
        return self._instance_bank

    def state_dict(self) -> Dict[str, Any]:
        """Return weights-only-safe state for one DataLoader worker replica."""
        return {
            "version": _WORKER_STATE_VERSION,
            "python_rng": _encode_python_rng_state(),
            "numpy_rng": _encode_numpy_rng_state(),
            "torch_rng": torch.get_rng_state().clone(),
            "instance_bank": (
                None if self._instance_bank is None else self._instance_bank.state_dict()
            ),
        }

    def load_state_dict(self, state: Any) -> None:
        """Restore one worker replica before it produces the next sample."""
        if not isinstance(state, Mapping):
            raise TypeError("Dataset worker state must be a mapping")
        if state.get("version") != _WORKER_STATE_VERSION:
            raise ValueError(f"Unsupported dataset worker state version: {state.get('version')!r}")

        python_rng = _decode_python_rng_state(state.get("python_rng"))
        numpy_rng = _decode_numpy_rng_state(state.get("numpy_rng"))
        torch_rng = _decode_torch_rng_state(state.get("torch_rng"))
        bank_state = state.get("instance_bank")
        if (self._instance_bank is None) != (bank_state is None):
            raise ValueError("Dataset copy-paste configuration does not match worker state")
        if self._instance_bank is not None:
            self._instance_bank.load_state_dict(bank_state)

        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)

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
        # Try up to 10 samples to handle missing/corrupt files
        for attempt in range(len(self.image_ids)):
            actual_idx = (idx + attempt) % len(self.image_ids)
            result = self._load_sample(actual_idx)
            if result is not None:
                return result
        # Final fallback: return the original sample attempt
        return self._load_sample(idx, allow_fail=True)

    def _load_sample(self, idx: int, allow_fail: bool = False) -> Dict[str, Any]:
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
            image_path = self.image_dir / Path(filename).name

        image = cv2.imread(str(image_path))
        if image is None:
            if allow_fail:
                raise FileNotFoundError(f"Failed to load image: {image_path}")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取深度图
        depth_path = self._get_depth_path(filename)
        depth = self._load_depth(
            depth_path,
            sample_id=img_id,
            image_filename=filename,
        )

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

        # Decode every non-crowd instance before any augmentation.  Training
        # keeps the targets; validation still validates the annotation contract
        # even though COCOeval owns its ground truth separately.
        if self.has_annotations and self.coco is not None:
            annotations = self._load_annotations(img_id)
            masks, boxes, labels = self._build_instances(
                annotations,
                image.shape[:2],
                sample_id=img_id,
                image_filename=filename,
            )

        # Preserve the canonical instance target fields for every annotated
        # split so collate_fn can build evaluation batches with targets too.
        if self.has_annotations and self.coco is not None:
            result["masks"] = masks
            result["boxes"] = boxes
            result["labels"] = labels

            # --- Copy-Paste: deposit raw instances into bank (before any transform) ---
            if self.is_train and self._copy_paste_enabled and self._instance_bank is not None:
                self._instance_bank.deposit(
                    image=image,
                    masks=masks,
                    boxes=boxes,
                    labels=labels,
                )

        # 应用变换
        if self.transform is not None:
            result = self.transform(result)

        # --- Copy-Paste: paste from bank (after geometric transforms, before ToTensor) ---
        # This is handled by CopyPasteTransform inserted in the pipeline.
        # See transforms.py CopyPasteTransform.

        return result

    def _get_depth_path(self, image_filename: str) -> Path:
        """获取深度文件路径"""
        base_name = Path(image_filename).stem
        depth_path = self.depth_dir / f"{base_name}.npy"

        if not depth_path.exists():
            depth_path = self.depth_dir / f"{base_name}.npz"

        return depth_path

    def _load_depth(
        self,
        path: Path,
        *,
        sample_id: Optional[Any] = None,
        image_filename: Optional[str] = None,
    ) -> np.ndarray:
        """加载深度 .npy/.npz 文件"""
        if not path.exists():
            raise FileNotFoundError(
                "Depth file not found for "
                f"sample_id={sample_id!r}, image={image_filename!r}, path={path}"
            )

        try:
            arr = np.load(path, allow_pickle=False)

            if isinstance(arr, np.lib.npyio.NpzFile):
                with arr:
                    if not arr.files:
                        raise ValueError("depth archive contains no arrays")
                    first_key = sorted(arr.files)[0]
                    arr = arr[first_key].copy()
        except (OSError, ValueError, EOFError) as exc:
            raise RuntimeError(
                "Failed to read depth for "
                f"sample_id={sample_id!r}, image={image_filename!r}, path={path}: {exc}"
            ) from exc

        if arr.ndim == 3:
            if arr.shape[2] == 1:
                arr = arr[:, :, 0]
            elif arr.shape[0] == 1:
                arr = arr[0]
            else:
                raise ValueError(
                    "Depth file must be 2D or single-channel 3D for "
                    f"sample_id={sample_id!r}, image={image_filename!r}, "
                    f"got shape {arr.shape} at {path}"
                )

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
            from PIL import Image

            arr = np.array(Image.open(path).convert("L"))

        if arr.max() > 1:
            arr = arr / 255.0

        arr = (arr > 0.5).astype(np.float32)
        return arr

    def _load_annotations(self, img_id: int) -> List[Dict[str, Any]]:
        """加载 COCO 标注。"""
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        annotations = [ann for ann in annotations if ann.get("iscrowd", 0) == 0]
        return annotations

    def _build_instances(
        self,
        annotations: List[Dict[str, Any]],
        image_size: Tuple[int, int],
        *,
        sample_id: Optional[Any] = None,
        image_filename: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """将 COCO 标注转换为 masks/boxes/labels。"""
        h, w = image_size
        masks = []
        boxes = []
        labels = []

        for ann in annotations:
            segmentation = ann.get("segmentation", None)
            if segmentation is None:
                raise ValueError(
                    "Instance annotation is missing segmentation for "
                    f"sample_id={sample_id!r}, image={image_filename!r}, "
                    f"annotation_id={ann.get('id')!r}"
                )

            annotation_id = ann.get("id", None)
            try:
                if isinstance(segmentation, list):
                    rles = coco_mask.frPyObjects(segmentation, h, w)
                    mask = coco_mask.decode(rles)
                    if mask.ndim == 3:
                        mask = mask.any(axis=2)
                else:
                    mask = coco_mask.decode(segmentation)
                    if mask.ndim == 3:
                        mask = mask[..., 0]
            except Exception as exc:
                raise ValueError(
                    "Failed to decode instance segmentation for "
                    f"sample_id={sample_id!r}, image={image_filename!r}, "
                    f"annotation_id={annotation_id!r}: {exc}"
                ) from exc

            mask = mask.astype(bool, copy=False)
            if mask.sum() == 0:
                raise ValueError(
                    "Decoded instance mask is empty for "
                    f"sample_id={sample_id!r}, image={image_filename!r}, "
                    f"annotation_id={annotation_id!r}"
                )

            # The segmentation is the instance supervision.  COCO bbox values
            # may be stale or inconsistent, so boxes always derive from masks.
            bbox = self._bbox_from_mask(mask)

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
        """从二值 mask 计算边界框 [x1, y1, x2, y2] (exclusive upper bounds)."""
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return [0, 0, 0, 0]
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        return [x1, y1, x2, y2]

    def get_img_info(self, idx: int) -> Dict[str, Any]:
        """获取图像元信息"""
        img_id = self.image_ids[idx]
        return self.coco.loadImgs(img_id)[0]

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批处理整理函数。"""
        images = [item["image"] for item in batch]
        depths = [item["depth"] for item in batch]
        image_ids = [item["image_id"] for item in batch]

        images = torch.stack(images, dim=0)
        depths = torch.stack(depths, dim=0)

        result = {
            "images": images,
            "depths": depths,
            "image_ids": torch.tensor(image_ids),
        }

        if "annotations" in batch[0]:
            result["annotations"] = [item["annotations"] for item in batch]

        if "noise_mask" in batch[0]:
            noise_masks = [item["noise_mask"] for item in batch]
            result["noise_masks"] = torch.stack(noise_masks, dim=0)

        return result
