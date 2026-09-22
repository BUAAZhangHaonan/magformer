# -*- coding: utf-8 -*-
"""
COCO Evaluator

纯 PyTorch 实现的 COCO 格式评估器。
使用 pycocotools 计算 mAP, AP50, AP75 等指标。
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .coco_json_eval import (
    STANDARD_IOU_THRESHOLDS,
    build_coco_eval_contract,
    evaluate_standard_coco_rows,
    extract_coco_metrics,
)


# =============================================================================
# COCO Evaluator
# =============================================================================
class COCOEvaluator:
    """
    COCO 格式实例分割评估器。

    支持指标:
    - AP (平均精度)
    - AP50 (IoU=0.5)
    - AP75 (IoU=0.75)
    - APs (小目标)
    - APm (中等目标)
    - APl (大目标)
    """

    def __init__(
        self,
        coco_gt: COCO,
        iou_types: List[str] = ["bbox"],
        max_dets: int = 100,
        iou_thresholds: Optional[List[float]] = None,
    ):
        """
        Args:
            coco_gt: COCO 格式真值对象
            iou_types: IoU 类型列表 ("bbox", "segm")
            max_dets: 每张图像最大检测数
            iou_thresholds: IoU 阈值列表
        """
        if isinstance(max_dets, bool) or not isinstance(max_dets, (int, np.integer)):
            raise TypeError(f"max_dets must be an integer greater than 10, got {max_dets!r}")
        if int(max_dets) <= 10:
            raise ValueError(f"max_dets must be greater than 10, got {max_dets}")

        configured_iou_thresholds: Optional[np.ndarray] = None
        if iou_thresholds is not None:
            configured_iou_thresholds = np.asarray(iou_thresholds, dtype=np.float64)
            if configured_iou_thresholds.ndim != 1 or configured_iou_thresholds.size == 0:
                raise ValueError("iou_thresholds must be a non-empty one-dimensional sequence")
            if not np.isfinite(configured_iou_thresholds).all():
                raise ValueError("iou_thresholds must contain only finite values")
            if np.any(configured_iou_thresholds < 0.0) or np.any(configured_iou_thresholds > 1.0):
                raise ValueError("iou_thresholds must be within [0, 1]")
            if np.any(np.diff(configured_iou_thresholds) <= 0.0):
                raise ValueError("iou_thresholds must be strictly increasing")

        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.max_dets = int(max_dets)
        self.iou_thresholds = configured_iou_thresholds

        self.results: List[Dict[str, Any]] = []
        self.image_ids: List[int] = []

    def update(
        self, predictions: List[Dict[str, Any]], *, image_ids: Iterable[int]
    ) -> None:
        """
        更新评估结果。

        Args:
            predictions: 预测结果列表，每个包含:
                - image_id: 图像 ID
                - category_id: 类别 ID
                - bbox: internal half-open XYXY [x1, y1, x2, y2]
                - score: 置信度分数
                - mask: RLE 或 polygon (可选)

            Standard COCO JSON rows use XYWH and must be evaluated through
            ``coco_json_eval.evaluate_standard_coco_rows`` instead.
        """
        evaluated_ids = [int(image_id) for image_id in image_ids]
        if not evaluated_ids:
            raise ValueError("COCOEvaluator.update requires at least one image_id")
        duplicate_batch_ids = self._find_duplicates(evaluated_ids)
        if duplicate_batch_ids:
            raise ValueError(
                f"Duplicate evaluated image IDs in batch: {duplicate_batch_ids}"
            )
        duplicate_registered_ids = sorted(set(evaluated_ids) & set(self.image_ids))
        if duplicate_registered_ids:
            raise ValueError(
                "Images were evaluated more than once: "
                f"{duplicate_registered_ids}"
            )
        prediction_ids = {int(prediction["image_id"]) for prediction in predictions}
        unknown_prediction_ids = sorted(prediction_ids - set(evaluated_ids))
        if unknown_prediction_ids:
            raise ValueError(
                "Predictions reference image IDs outside the evaluated batch: "
                f"{unknown_prediction_ids}"
            )
        self.results.extend(predictions)
        self.image_ids.extend(evaluated_ids)

    @staticmethod
    def _find_duplicates(image_ids: Iterable[int]) -> List[int]:
        seen = set()
        duplicates = set()
        for image_id in image_ids:
            image_id = int(image_id)
            if image_id in seen:
                duplicates.add(image_id)
            seen.add(image_id)
        return sorted(duplicates)

    def _validated_image_ids(self) -> List[int]:
        if not self.image_ids:
            raise ValueError("No evaluated image IDs were registered")
        duplicates = self._find_duplicates(self.image_ids)
        if duplicates:
            raise ValueError(f"Images were evaluated more than once: {duplicates}")
        evaluated_ids = set(self.image_ids)
        unknown_prediction_ids = sorted(
            {int(result["image_id"]) for result in self.results} - evaluated_ids
        )
        if unknown_prediction_ids:
            raise ValueError(
                "Predictions reference image IDs that were not evaluated: "
                f"{unknown_prediction_ids}"
            )
        return sorted(evaluated_ids)

    def synchronize_between_processes(self) -> None:
        """分布式训练时同步结果"""
        if not (dist.is_available() and dist.is_initialized()):
            return
        # Move results to CPU to reduce GPU memory during gather
        cpu_results = []
        for r in self.results:
            if isinstance(r, dict):
                cpu_r = {}
                for k, v in r.items():
                    if isinstance(v, torch.Tensor):
                        cpu_r[k] = v.cpu()
                    else:
                        cpu_r[k] = v
                cpu_results.append(cpu_r)
            else:
                if isinstance(r, torch.Tensor):
                    cpu_results.append(r.cpu())
                else:
                    cpu_results.append(r)

        world_size = dist.get_world_size()
        gathered_results = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_results, cpu_results)
        gathered_image_ids = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_image_ids, list(self.image_ids))

        merged_results = []
        for item in gathered_results:
            if item:
                merged_results.extend(item)
        merged_image_ids = []
        for item in gathered_image_ids:
            if item:
                merged_image_ids.extend(int(image_id) for image_id in item)

        self.results = merged_results
        self.image_ids = merged_image_ids
        self._validated_image_ids()

    def accumulate(self) -> None:
        """Compatibility no-op.

        All evaluator state is already stored in ``self.results`` and consumed
        directly inside ``summarize()``. This method intentionally keeps the
        COCO evaluator interface parity with external callers.
        """
        return

    def summarize(self) -> Dict[str, float]:
        """
        计算并返回评估指标。

        Returns:
            指标字典 {指标名: 值}
        """
        evaluated_image_ids = self._validated_image_ids()
        coco_results = self.to_coco_results()
        contract = build_coco_eval_contract(
            self.coco_gt,
            image_ids=evaluated_image_ids,
            max_dets=self.max_dets,
            iou_thresholds=(
                self.iou_thresholds
                if self.iou_thresholds is not None
                else STANDARD_IOU_THRESHOLDS
            ),
        )
        metrics = evaluate_standard_coco_rows(
            self.coco_gt,
            coco_results,
            contract=contract,
            iou_types=self.iou_types,
            cocoeval_cls=COCOeval,
        )

        # 打印结果
        self._print_results(metrics)

        return metrics

    def to_coco_results(self) -> List[Dict[str, Any]]:
        """Return current predictions as COCO result rows (bbox + segmentation RLE when available)."""
        if len(self.results) == 0:
            return []
        return self._convert_to_coco_format(self.results)

    def dump(self, path: str | Path) -> Path:
        """Write COCO results json to disk (UTF-8)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_coco_results(),
                       ensure_ascii=False) + "\n", encoding="utf-8")
        return out

    def _convert_to_coco_format(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        转换预测结果为 COCO 格式。

        Args:
            results: 预测结果

        Returns:
            COCO 格式结果列表
        """
        coco_results = []

        for result_index, result in enumerate(results):
            if "segmentation" in result:
                raise ValueError(
                    "COCOEvaluator.update accepts internal prediction rows with "
                    "half-open XYXY bbox and 'mask'. Standard COCO rows with "
                    "XYWH bbox/segmentation must use evaluate_standard_coco_rows; "
                    f"row={result_index}"
                )
            coco_result = {
                "image_id": int(result["image_id"]),
                "category_id": int(result.get("category_id", 1)),
                "score": float(result["score"]),
            }

            bbox = result.get("bbox", None)
            mask = result.get("mask", None)

            if bbox is None and mask is not None:
                bbox = self._bbox_from_mask(mask)

            if bbox is not None:
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    coco_result["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                else:
                    coco_result["bbox"] = bbox

            if mask is not None and "segm" in self.iou_types:
                coco_result["segmentation"] = self._mask_to_rle(mask)

            coco_results.append(coco_result)

        return coco_results

    def _mask_to_rle(self, mask: Any) -> Dict[str, Any]:
        """
        将掩码转换为 RLE 格式。

        Args:
            mask: 掩码 (tensor, numpy 或 RLE)

        Returns:
            RLE 格式字典
        """
        # 如果已经是 RLE
        if isinstance(mask, dict) and "counts" in mask:
            return mask

        # 转换为 numpy
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        # 如果是二值掩码，转换为 RLE
        from pycocotools import mask as coco_mask

        if isinstance(mask, np.ndarray):
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            rle = coco_mask.encode(np.asfortranarray(mask))
            if isinstance(rle["counts"], bytes):
                rle["counts"] = rle["counts"].decode("ascii")
            return {
                "size": rle["size"],
                "counts": rle["counts"],
            }

        return mask

    def _bbox_from_mask(self, mask: Any) -> List[float]:
        """Compute [x1, y1, x2, y2] from a binary mask."""
        if isinstance(mask, dict) and "counts" in mask:
            from pycocotools import mask as coco_mask

            rle = dict(mask)
            if isinstance(rle["counts"], str):
                rle["counts"] = rle["counts"].encode("ascii")
            mask = coco_mask.decode(rle)
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        mask_array = np.asarray(mask)
        if mask_array.ndim == 3:
            if mask_array.shape[0] == 1:
                mask_array = mask_array[0]
            elif mask_array.shape[-1] == 1:
                mask_array = mask_array[..., 0]
            else:
                raise ValueError(
                    "Mask bbox derivation requires one mask, got shape "
                    f"{mask_array.shape}"
                )
        if mask_array.ndim != 2:
            raise ValueError(
                "Mask bbox derivation requires a two-dimensional mask, got shape "
                f"{mask_array.shape}"
            )
        ys, xs = np.where(mask_array > 0)
        if len(xs) == 0 or len(ys) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        x1, x2 = xs.min(), xs.max() + 1
        y1, y2 = ys.min(), ys.max() + 1
        return [float(x1), float(y1), float(x2), float(y2)]

    def _extract_metrics(self, coco_eval: COCOeval, iou_type: str) -> Dict[str, float]:
        """
        从 COCO 评估器提取指标。

        Args:
            coco_eval: COCOeval 对象
            iou_type: IoU 类型

        Returns:
            指标字典
        """
        return extract_coco_metrics(coco_eval, iou_type, max_dets=self.max_dets)

    def _mean_valid_precision(
        self,
        coco_eval: COCOeval,
        iou_thresholds: Optional[Iterable[float]] = None,
        *,
        area_label: str = "all",
    ) -> float:
        if "precision" not in coco_eval.eval:
            raise RuntimeError("COCOeval precision tensor is unavailable after accumulate()")
        precision = np.asarray(coco_eval.eval["precision"])
        if precision.ndim != 5:
            raise RuntimeError(
                "COCOeval precision tensor must have shape [T,R,K,A,M], "
                f"got {precision.shape}"
            )
        self._validate_metric_tensor_axes(precision, coco_eval, tensor_name="precision")

        selected_iou_indices = self._select_iou_indices(coco_eval, iou_thresholds)
        if selected_iou_indices is None:
            return -1.0
        area_index = self._area_index(coco_eval, area_label)
        max_det_index = self._max_det_index(coco_eval, self.max_dets)
        values = precision[
            selected_iou_indices, :, :, area_index, max_det_index
        ]
        return self._mean_nonnegative(values)

    def _mean_valid_recall(
        self,
        coco_eval: COCOeval,
        *,
        max_dets: int,
        area_label: str = "all",
        iou_thresholds: Optional[Iterable[float]] = None,
    ) -> float:
        if "recall" not in coco_eval.eval:
            raise RuntimeError("COCOeval recall tensor is unavailable after accumulate()")
        recall = np.asarray(coco_eval.eval["recall"])
        if recall.ndim != 4:
            raise RuntimeError(
                "COCOeval recall tensor must have shape [T,K,A,M], "
                f"got {recall.shape}"
            )
        self._validate_metric_tensor_axes(recall, coco_eval, tensor_name="recall")

        selected_iou_indices = self._select_iou_indices(coco_eval, iou_thresholds)
        if selected_iou_indices is None:
            return -1.0
        area_index = self._area_index(coco_eval, area_label)
        max_det_index = self._max_det_index(coco_eval, max_dets)
        values = recall[selected_iou_indices, :, area_index, max_det_index]
        return self._mean_nonnegative(values)

    @staticmethod
    def _select_iou_indices(
        coco_eval: COCOeval,
        iou_thresholds: Optional[Iterable[float]],
    ) -> Optional[List[int]]:
        configured_iou_thresholds = np.asarray(coco_eval.params.iouThrs)
        if configured_iou_thresholds.ndim != 1 or configured_iou_thresholds.size == 0:
            raise RuntimeError("COCOeval params.iouThrs must be one-dimensional and non-empty")
        if iou_thresholds is None:
            return list(range(configured_iou_thresholds.size))

        selected_iou_indices: List[int] = []
        for threshold in iou_thresholds:
            matches = np.flatnonzero(
                np.isclose(configured_iou_thresholds, float(threshold), atol=1e-9)
            )
            if len(matches) == 0:
                return None
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one COCO IoU slice for {threshold:.2f}, "
                    f"found {len(matches)}"
                )
            selected_iou_indices.append(int(matches[0]))
        return selected_iou_indices

    @staticmethod
    def _area_index(coco_eval: COCOeval, area_label: str) -> int:
        area_labels = list(coco_eval.params.areaRngLbl)
        matches = [index for index, label in enumerate(area_labels) if label == area_label]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one COCO area slice for {area_label!r}, "
                f"found {len(matches)}"
            )
        return matches[0]

    @staticmethod
    def _max_det_index(coco_eval: COCOeval, max_dets: int) -> int:
        matches = [
            index
            for index, max_det in enumerate(coco_eval.params.maxDets)
            if int(max_det) == int(max_dets)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one COCO maxDets={max_dets} slice, "
                f"found {len(matches)}"
            )
        return matches[0]

    @staticmethod
    def _mean_nonnegative(values: np.ndarray) -> float:
        valid_values = values[values > -1]
        if valid_values.size and not np.isfinite(valid_values).all():
            raise RuntimeError("COCOeval metric tensor contains NaN or Inf values")
        return float(valid_values.mean()) if valid_values.size else -1.0

    @staticmethod
    def _validate_metric_tensor_axes(
        tensor: np.ndarray, coco_eval: COCOeval, *, tensor_name: str
    ) -> None:
        expected_iou = len(coco_eval.params.iouThrs)
        expected_area = len(coco_eval.params.areaRngLbl)
        expected_max_dets = len(coco_eval.params.maxDets)
        area_axis = 3 if tensor_name == "precision" else 2
        max_det_axis = 4 if tensor_name == "precision" else 3
        if (
            tensor.shape[0] != expected_iou
            or tensor.shape[area_axis] != expected_area
            or tensor.shape[max_det_axis] != expected_max_dets
        ):
            raise RuntimeError(
                f"COCOeval {tensor_name} axes do not match params: "
                f"shape={tensor.shape}, iou={expected_iou}, area={expected_area}, "
                f"maxDets={expected_max_dets}"
            )

    def _zero_metrics(self, iou_type: str) -> Dict[str, float]:
        prefix = "bbox" if iou_type == "bbox" else "segm"
        return {
            f"{prefix}_AP": 0.0,
            f"{prefix}_AP50": 0.0,
            f"{prefix}_AP75": 0.0,
            f"{prefix}_AP80": 0.0,
            f"{prefix}_AP85": 0.0,
            f"{prefix}_AP90": 0.0,
            f"{prefix}_AP95": 0.0,
            f"{prefix}_AP_H": 0.0,
            f"{prefix}_APs": 0.0,
            f"{prefix}_APm": 0.0,
            f"{prefix}_APl": 0.0,
            f"{prefix}_AP_small": 0.0,
            f"{prefix}_AP_medium": 0.0,
            f"{prefix}_AP_large": 0.0,
            f"{prefix}_AR1": 0.0,
            f"{prefix}_AR10": 0.0,
            f"{prefix}_AR{self.max_dets}": 0.0,
            f"{prefix}_ARs": 0.0,
            f"{prefix}_ARm": 0.0,
            f"{prefix}_ARl": 0.0,
        }

    def _print_results(self, metrics: Dict[str, float]) -> None:
        """打印评估结果"""
        print("\n" + "=" * 60)
        print("COCO Evaluation Results")
        print("=" * 60)

        # 按类型分组
        for iou_type in ["bbox", "segm"]:
            if iou_type not in ["_".join(k.split("_")[:-1]) for k in metrics.keys()]:
                continue

            prefix = "bbox" if iou_type == "bbox" else "segm"
            print(f"\n{prefix.upper()} Results:")

            # 基础指标
            for key in ["AP", "AP50", "AP75", "AP80", "AP85", "AP90", "AP95", "AP_H"]:
                metric_key = f"{prefix}_{key}"
                if metric_key in metrics:
                    print(f"  {key}: {metrics[metric_key]:.4f}")

            # 尺度指标
            for size in ["small", "medium", "large"]:
                metric_key = f"{prefix}_AP_{size}"
                if metric_key in metrics:
                    print(f"  AP_{size}: {metrics[metric_key]:.4f}")

            # Recall
            for key in ["AR1", "AR10", f"AR{self.max_dets}"]:
                metric_key = f"{prefix}_{key}"
                if metric_key in metrics:
                    print(f"  {key}: {metrics[metric_key]:.4f}")
            for size_short, size_long in [("ARs", "small"), ("ARm", "medium"), ("ARl", "large")]:
                metric_key = f"{prefix}_{size_short}"
                if metric_key in metrics:
                    print(f"  AR_{size_long}: {metrics[metric_key]:.4f}")

        print("=" * 60 + "\n")
