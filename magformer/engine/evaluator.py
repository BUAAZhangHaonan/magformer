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
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.max_dets = max_dets
        self.iou_thresholds = iou_thresholds or [0.5, 0.75]

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

    def synchronize_between_processes(self) -> None:
        """分布式训练时同步结果"""
        if not (dist.is_available() and dist.is_initialized()):
            return
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, self.results)
        merged = []
        for item in gathered:
            if item:
                merged.extend(item)
        self.results = merged

    def accumulate(self) -> None:
        """累积评估结果"""
        pass

    def summarize(self) -> Dict[str, float]:
        """
        计算并返回评估指标。

        Returns:
            指标字典 {指标名: 值}
        """
        if len(self.results) == 0:
            print("[COCOEvaluator] No results to evaluate")
            return {}

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

        for result in results:
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
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        if mask.ndim == 3:
            mask = mask[0]
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
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
        prefix = "bbox" if iou_type == "bbox" else "segm"

        metrics = {}

        # 标准 COCO 指标 (12 个值)
        # 0: AP@[0.50:0.95], 1: AP@0.50, 2: AP@0.75
        # 3: AP small, 4: AP medium, 5: AP large
        # 6: AR@1, 7: AR@10, 8: AR@100
        # 9: AR small, 10: AR medium, 11: AR large
        metrics[f"{prefix}_AP"] = coco_eval.stats[0]
        metrics[f"{prefix}_AP50"] = coco_eval.stats[1]
        metrics[f"{prefix}_AP75"] = coco_eval.stats[2]
        # Common COCO naming
        metrics[f"{prefix}_APs"] = coco_eval.stats[3]
        metrics[f"{prefix}_APm"] = coco_eval.stats[4]
        metrics[f"{prefix}_APl"] = coco_eval.stats[5]
        # Backward-compatible aliases
        metrics[f"{prefix}_AP_small"] = coco_eval.stats[3]
        metrics[f"{prefix}_AP_medium"] = coco_eval.stats[4]
        metrics[f"{prefix}_AP_large"] = coco_eval.stats[5]

        return metrics

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
