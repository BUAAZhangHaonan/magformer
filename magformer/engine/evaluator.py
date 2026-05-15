# -*- coding: utf-8 -*-
"""
COCO Evaluator

纯 PyTorch 实现的 COCO 格式评估器。
使用 pycocotools 计算 mAP, AP50, AP75 等指标。
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import torch.distributed as dist
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from magformer.engine.coco_export import internal_instances_to_coco_results


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
        image_ids: Optional[List[int]] = None,
    ):
        """
        Args:
            coco_gt: COCO 格式真值对象
            iou_types: IoU 类型列表 ("bbox", "segm")
            max_dets: 每张图像最大检测数
            iou_thresholds: IoU 阈值列表
        """
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.max_dets = max_dets
        self.iou_thresholds = iou_thresholds or [0.5, 0.75]
        self.image_ids = self._unique_image_ids(image_ids)

        # 评估结果缓存
        self.results = []

    def update(self, predictions: List[Dict[str, Any]]) -> None:
        """
        更新评估结果。

        Args:
            predictions: 预测结果列表，每个包含:
                - image_id: 图像 ID
                - category_id: 类别 ID
                - bbox: [x, y, w, h] 或 [x1, y1, x2, y2]
                - score: 置信度分数
                - mask: RLE 或 polygon (可选)
        """
        self.results.extend(predictions)

    def set_image_ids(self, image_ids: Optional[List[int]]) -> None:
        """Restrict COCOeval to the evaluated image ids."""
        self.image_ids = self._unique_image_ids(image_ids)

    @staticmethod
    def _unique_image_ids(image_ids: Optional[List[int]]) -> Optional[List[int]]:
        if image_ids is None:
            return None
        return list(dict.fromkeys(int(image_id) for image_id in image_ids))

    @staticmethod
    def _dedupe_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for result in results:
            bbox = result.get("bbox")
            if bbox is not None:
                bbox_key = tuple(round(float(value), 6) for value in bbox)
            else:
                bbox_key = None
            mask = result.get("mask")
            if isinstance(mask, dict):
                mask_key = (tuple(mask.get("size", [])), mask.get("counts"))
            else:
                mask_key = None
            key = (
                int(result.get("image_id", -1)),
                int(result.get("category_id", 1)),
                round(float(result.get("score", 0.0)), 6),
                bbox_key,
                mask_key,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(result)
        return deduped

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

        # Gather from all ranks
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, cpu_results)
        merged = []
        for item in gathered:
            if item:
                merged.extend(item)
        self.results = self._dedupe_results(merged)

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
        if len(self.results) == 0:
            print("[COCOEvaluator] No results to evaluate")
            metrics: Dict[str, float] = {}
            for iou_type in self.iou_types:
                metrics.update(self._zero_metrics(iou_type))
            return metrics

        self.results = self._dedupe_results(self.results)

        # 转换结果为 COCO 格式
        coco_results = self._convert_to_coco_format(self.results)

        # 为每个 IoU 类型评估
        metrics = {}

        for iou_type in self.iou_types:
            # 创建 COCO 评估器
            coco_dt = self.coco_gt.loadRes(coco_results)
            coco_eval = COCOeval(
                self.coco_gt, coco_dt, iouType=iou_type
            )

            # 评估
            coco_eval.params.maxDets = [1, 10, self.max_dets]
            if self.image_ids is not None:
                coco_eval.params.imgIds = self.image_ids
            coco_eval.evaluate()
            coco_eval.accumulate()

            # 打印统计
            coco_eval.summarize()

            # 提取指标
            iou_metrics = self._extract_metrics(coco_eval, iou_type)
            metrics.update(iou_metrics)

        # 打印结果
        self._print_results(metrics)

        return metrics

    def to_coco_results(self) -> List[Dict[str, Any]]:
        """Return current predictions as COCO result rows (bbox + segmentation RLE when available)."""
        if len(self.results) == 0:
            return []
        self.results = self._dedupe_results(self.results)
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
        return internal_instances_to_coco_results(results, iou_types=self.iou_types)

    def _extract_metrics(self, coco_eval: COCOeval, iou_type: str) -> Dict[str, float]:
        """
        从 COCO 评估器提取指标。

        Args:
            coco_eval: COCOeval 对象
            iou_type: IoU 类型

        Returns:
            指标字典
        """
        prefix = "bbox" if iou_type == "bbox" else "segm"

        metrics = {}

        def _summarize_ap(iou_thr: Optional[float] = None, area: str = "all") -> float:
            precision = coco_eval.eval.get("precision") if hasattr(coco_eval, "eval") else None
            if precision is None:
                return float("nan")
            precision = np.asarray(precision)
            params = coco_eval.params
            iou_thrs = np.asarray(params.iouThrs)
            area_labels = list(params.areaRngLbl)
            max_dets = list(params.maxDets)
            if area not in area_labels or self.max_dets not in max_dets:
                return float("nan")
            area_index = area_labels.index(area)
            max_det_index = max_dets.index(self.max_dets)
            values = precision[:, :, :, area_index, max_det_index]
            if iou_thr is not None:
                matches = np.where(np.isclose(iou_thrs, iou_thr))[0]
                if len(matches) == 0:
                    return float("nan")
                values = values[matches]
            values = values[values > -1]
            if values.size == 0:
                return -1.0
            return float(np.mean(values))

        # pycocotools.stats hardcodes AP to maxDets=100. Read precision
        # directly so custom final maxDets such as 200 report the intended AP.
        metrics[f"{prefix}_AP"] = _summarize_ap()
        metrics[f"{prefix}_AP50"] = _summarize_ap(iou_thr=0.5)
        metrics[f"{prefix}_AP75"] = _summarize_ap(iou_thr=0.75)
        metrics[f"{prefix}_APs"] = _summarize_ap(area="small")
        metrics[f"{prefix}_APm"] = _summarize_ap(area="medium")
        metrics[f"{prefix}_APl"] = _summarize_ap(area="large")

        fallback_keys = [
            f"{prefix}_AP",
            f"{prefix}_AP50",
            f"{prefix}_AP75",
            f"{prefix}_APs",
            f"{prefix}_APm",
            f"{prefix}_APl",
        ]
        for index, key in enumerate(fallback_keys):
            if np.isnan(metrics[key]):
                metrics[key] = float(coco_eval.stats[index])

        # Backward-compatible aliases
        metrics[f"{prefix}_AP_small"] = metrics[f"{prefix}_APs"]
        metrics[f"{prefix}_AP_medium"] = metrics[f"{prefix}_APm"]
        metrics[f"{prefix}_AP_large"] = metrics[f"{prefix}_APl"]

        return metrics

    def _zero_metrics(self, iou_type: str) -> Dict[str, float]:
        prefix = "bbox" if iou_type == "bbox" else "segm"
        return {
            f"{prefix}_AP": 0.0,
            f"{prefix}_AP50": 0.0,
            f"{prefix}_AP75": 0.0,
            f"{prefix}_APs": 0.0,
            f"{prefix}_APm": 0.0,
            f"{prefix}_APl": 0.0,
            f"{prefix}_AP_small": 0.0,
            f"{prefix}_AP_medium": 0.0,
            f"{prefix}_AP_large": 0.0,
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
            for key in ["AP", "AP50", "AP75"]:
                metric_key = f"{prefix}_{key}"
                if metric_key in metrics:
                    print(f"  {key}: {metrics[metric_key]:.4f}")

            # 尺度指标
            for size in ["small", "medium", "large"]:
                metric_key = f"{prefix}_AP_{size}"
                if metric_key in metrics:
                    print(f"  AP_{size}: {metrics[metric_key]:.4f}")

        print("=" * 60 + "\n")
