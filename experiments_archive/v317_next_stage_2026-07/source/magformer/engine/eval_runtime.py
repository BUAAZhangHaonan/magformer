# -*- coding: utf-8 -*-
"""Shared inference-time evaluation helpers for trainer and CLI entrypoints."""

from __future__ import annotations

import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.distributed as dist
from torch.amp import autocast
# import gc  # removed: gc.collect() disabled in eval loop

from .coco_export import outputs_to_coco_instances
from .evaluator import COCOEvaluator
from ..utils.visualization import save_evaluation_comparisons


@dataclass
class EvaluationResult:
    log_dict: Dict[str, float]
    coco_metrics: Dict[str, float]
    coco_results_path: Optional[Path]
    visualization_batch: Optional[Dict[str, Any]]
    visualization_outputs: Optional[Dict[str, Any]]


def _flatten_gathered_objects(gathered: Iterable[Any]) -> List[Any]:
    flattened: List[Any] = []
    for item in gathered:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def _gather_object(value: Any) -> List[Any]:
    if not (dist.is_available() and dist.is_initialized()):
        return [value]
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, value)
    return gathered


def _get_inference_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _select_global_eval_indices(
    total_images: int,
    max_images: Optional[int],
    *,
    seed: int = 42,
) -> List[int]:
    """Select one deterministic global evaluation set before rank sharding."""
    if total_images < 0:
        raise ValueError(f"total_images must be non-negative, got {total_images}")
    if max_images is None or max_images >= total_images:
        return list(range(total_images))
    if max_images <= 0:
        raise ValueError(f"max_images must be positive when set, got {max_images}")
    return random.Random(seed).sample(range(total_images), max_images)


def _rank_stride_eval_indices(
    global_indices: Iterable[int], rank: int, world_size: int
) -> List[int]:
    """Partition global indices without padding or duplication."""
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return [int(index) for index in list(global_indices)[rank::world_size]]


def _build_indexed_eval_loader(val_loader, indices: List[int]):
    """Rebuild a DataLoader over an exact, already rank-sharded index set."""
    from torch.utils.data import DataLoader, Subset

    dataset = getattr(val_loader, "dataset", None)
    if dataset is None:
        raise TypeError("Indexed evaluation requires val_loader.dataset")
    batch_size = getattr(val_loader, "batch_size", None)
    if batch_size is None:
        raise TypeError("Indexed evaluation requires an explicit val_loader.batch_size")

    num_workers = int(getattr(val_loader, "num_workers", 0))
    loader_kwargs: Dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": bool(getattr(val_loader, "pin_memory", False)),
        "drop_last": False,
        "collate_fn": val_loader.collate_fn,
        "timeout": getattr(val_loader, "timeout", 0),
        "worker_init_fn": getattr(val_loader, "worker_init_fn", None),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            getattr(val_loader, "persistent_workers", False)
        )
        prefetch_factor = getattr(val_loader, "prefetch_factor", None)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(Subset(dataset, indices), **loader_kwargs)


def _build_prefix_limited_eval_loader(val_loader, max_images: Optional[int]):
    """Build an exact first-N image loader without partial oversized batches."""
    dataset = getattr(val_loader, "dataset", None)
    if dataset is None:
        raise TypeError("Prefix-limited evaluation requires val_loader.dataset")
    dataset_size = len(dataset)
    if max_images is None:
        return val_loader, dataset_size
    if max_images <= 0:
        raise ValueError(f"max_images must be positive when set, got {max_images}")

    target_images = min(int(max_images), dataset_size)
    if target_images == dataset_size:
        return val_loader, target_images
    return (
        _build_indexed_eval_loader(val_loader, list(range(target_images))),
        target_images,
    )


def _require_unique_image_ids(image_ids: Iterable[int]) -> List[int]:
    normalized = [int(image_id) for image_id in image_ids]
    seen = set()
    duplicates = set()
    for image_id in normalized:
        if image_id in seen:
            duplicates.add(image_id)
        seen.add(image_id)
    if duplicates:
        raise RuntimeError(
            f"Distributed evaluation produced duplicate image IDs: {sorted(duplicates)}"
        )
    return normalized


def _raise_global_contract_error_on_all_ranks(
    error_message: Optional[str], *, rank: int
) -> None:
    """Broadcast a rank-zero contract failure before every rank raises it."""
    distributed = dist.is_available() and dist.is_initialized()
    if not distributed:
        if error_message is not None:
            raise RuntimeError(error_message)
        return

    payload = [error_message if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    received_message = payload[0]
    if received_message is not None:
        if not isinstance(received_message, str):
            raise TypeError("Distributed evaluation contract error must be a string")
        raise RuntimeError(received_message)


@torch.no_grad()
def run_inference_evaluation(
    model: torch.nn.Module,
    val_loader,
    *,
    coco_gt: Optional[Any],
    device: torch.device,
    output_dir: str | Path,
    optimizer_step: int,
    amp_enabled: bool,
    score_threshold: float = 0.0,
    mask_threshold: float = 0.5,
    category_offset: int = 1,
    category_ids: Optional[List[int]] = None,
    num_vis_images: int = 8,
    iou_types: Optional[List[str]] = None,
    max_images: Optional[int] = None,
    max_dets: int = 100,
    fail_on_empty: bool = False,
) -> EvaluationResult:
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    is_primary = rank == 0
    inference_model = _get_inference_model(model)

    if iou_types is None:
        iou_types = ["bbox", "segm"]

    dataset = getattr(val_loader, "dataset", None)
    expected_global_images: Optional[int] = None
    local_target_images: Optional[int] = None
    eval_loader = val_loader
    if dataset is not None:
        global_indices = _select_global_eval_indices(len(dataset), max_images)
        expected_global_images = len(global_indices)
        if distributed or max_images is not None:
            local_indices = _rank_stride_eval_indices(
                global_indices, rank=rank, world_size=world_size
            )
            eval_loader = _build_indexed_eval_loader(val_loader, local_indices)
            local_target_images = len(local_indices)
        else:
            local_target_images = len(dataset)
    else:
        if max_images is not None:
            raise TypeError(
                "max_images requires an indexable val_loader with a dataset"
            )
        try:
            local_target_images = len(val_loader)
        except TypeError:
            local_target_images = None
        if not distributed:
            expected_global_images = local_target_images

    target_description = (
        str(expected_global_images)
        if expected_global_images is not None
        else "caller-sharded iterable"
    )
    if is_primary:
        print(
            f"[Eval] Starting: iou_types={iou_types}, "
            f"max_images={max_images if max_images is not None else 'all'} "
            f"({target_description} images globally)"
        )

    evaluator = (
        COCOEvaluator(coco_gt=coco_gt, iou_types=iou_types, max_dets=max_dets)
        if coco_gt is not None
        else None
    )

    local_scores: List[float] = []
    local_bbox_area_ratios: List[float] = []
    local_total_masks = 0
    local_nonempty_masks = 0
    local_total_preds = 0
    local_image_ids: List[int] = []
    num_vis_images = max(1, num_vis_images)
    vis_batches: List[Any] = []
    vis_outputs: List[Any] = []

    if is_primary and dataset is not None and (distributed or max_images is not None):
        print(f"[Eval] Rank-strided exact sampling: rank0={local_target_images}, seed=42")
    eval_start = time.time()
    num_evaluated = 0
    num_images_evaluated = 0
    batch_idx = 0

    for batch in eval_loader:
        batch_idx += 1
        images = batch["images"].to(device)
        depths = batch["depths"].to(device)
        depth_valid_masks = batch.get("depth_valid_masks")
        if depth_valid_masks is not None:
            depth_valid_masks = depth_valid_masks.to(device)
        noise_masks = batch.get("noise_masks")
        if noise_masks is not None:
            noise_masks = noise_masks.to(device)
        padding_masks = batch.get("padding_masks")
        if padding_masks is not None:
            padding_masks = padding_masks.to(device)

        amp_context = autocast("cuda") if amp_enabled and device.type == "cuda" else nullcontext()
        with amp_context:
            inference_kwargs = {
                "padding_masks": padding_masks,
                "depth_noise_masks": noise_masks,
            }
            if depth_valid_masks is not None:
                inference_kwargs["depth_valid_masks"] = depth_valid_masks
            outputs = inference_model.forward_inference_raw(
                images,
                depths,
                **inference_kwargs,
            )

        if "image_ids" not in batch:
            raise KeyError("Evaluation batch is missing required image_ids")
        batch_image_ids = [int(image_id) for image_id in batch["image_ids"]]
        if len(batch_image_ids) != int(images.shape[0]):
            raise ValueError(
                "Evaluation image_ids length must match batch size: "
                f"image_ids={len(batch_image_ids)}, batch={int(images.shape[0])}"
            )
        predictions = outputs_to_coco_instances(
            outputs=outputs,
            image_ids=batch_image_ids,
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
            category_offset=category_offset,
            category_ids=category_ids,
        )
        if evaluator is not None:
            evaluator.update(predictions, image_ids=batch_image_ids)
        if is_primary and len(vis_batches) < num_vis_images:
            vis_batches.append(batch)
            vis_outputs.append(outputs)
        local_total_preds += len(predictions)
        local_image_ids.extend(batch_image_ids)

        for pred in predictions:
            score = pred.get("score")
            if score is not None:
                local_scores.append(float(score))
            mask = pred.get("mask")
            bbox = pred.get("bbox")
            if mask is None:
                continue
            local_total_masks += 1
            if isinstance(mask, dict) and "size" in mask:
                # RLE mask: check if area > 0 via bbox as proxy
                if bbox is not None and len(bbox) == 4:
                    box_area = (float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1]))
                    if box_area > 0:
                        local_nonempty_masks += 1
                else:
                    local_nonempty_masks += 1
            elif hasattr(mask, "sum"):
                if mask.sum() > 0:
                    local_nonempty_masks += 1
            else:
                local_nonempty_masks += 1
            if bbox is not None and len(bbox) == 4:
                if isinstance(mask, dict) and "size" in mask:
                    h, w = int(mask["size"][0]), int(mask["size"][1])
                else:
                    h, w = int(max(1, float(bbox[3]) - float(bbox[1]))), int(max(1, float(bbox[2]) - float(bbox[0])))
                denom = float(max(1, h * w))
                x1, y1, x2, y2 = [float(v) for v in bbox]
                box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                local_bbox_area_ratios.append(box_area / denom)
        del outputs
        if batch_idx % 50 == 0:
            # gc.collect()  # disabled: causes periodic stalls, forces CUDA memory reallocation
            # torch.cuda.empty_cache()  # disabled: same reason
            pass  # gc.collect() disabled

        num_evaluated += 1
        num_images_evaluated += images.shape[0]
        if num_evaluated % 50 == 0 and is_primary:
            elapsed = time.time() - eval_start
            rate = num_images_evaluated / elapsed if elapsed > 0 else 0
            target = local_target_images or num_images_evaluated
            remaining = max(0, target - num_images_evaluated) / rate if rate > 0 else 0
            print(f"[Eval] Inference: {num_images_evaluated}/{target} "
                  f"({rate:.1f} img/s, ETA {remaining:.0f}s, elapsed {elapsed:.0f}s)")

    inference_time = time.time() - eval_start
    if is_primary:
        print(f"[Eval] Inference done: {num_images_evaluated} images in {inference_time:.1f}s "
              f"({num_images_evaluated/inference_time:.1f} img/s)")

    if evaluator is not None:
        evaluator.synchronize_between_processes()

    total_preds = sum(int(v) for v in _gather_object(local_total_preds))
    eval_scores = [float(v) for v in _flatten_gathered_objects(_gather_object(local_scores))]
    eval_bbox_area_ratios = [
        float(v) for v in _flatten_gathered_objects(_gather_object(local_bbox_area_ratios))
    ]
    eval_total_masks = sum(int(v) for v in _gather_object(local_total_masks))
    eval_nonempty_masks = sum(int(v) for v in _gather_object(local_nonempty_masks))
    eval_image_ids = _require_unique_image_ids(
        _flatten_gathered_objects(_gather_object(local_image_ids))
    )
    global_contract_error = None
    if (
        is_primary
        and expected_global_images is not None
        and len(eval_image_ids) != expected_global_images
    ):
        global_contract_error = (
            "Evaluation image count does not match the global index plan: "
            f"expected={expected_global_images}, observed={len(eval_image_ids)}"
        )
    _raise_global_contract_error_on_all_ranks(global_contract_error, rank=rank)

    if not is_primary:
        return EvaluationResult(
            log_dict={},
            coco_metrics={},
            coco_results_path=None,
            visualization_batch=None,
            visualization_outputs=None,
        )

    if fail_on_empty and evaluator is not None and total_preds == 0:
        raise RuntimeError(
            "Evaluation exported zero COCO predictions. Check checkpoint "
            "compatibility, model output keys, and score/mask thresholds."
        )

    log_dict: Dict[str, float] = {}
    coco_metrics: Dict[str, float] = {}
    coco_results_path: Optional[Path] = None

    if evaluator is not None:
        if is_primary:
            print(f"[Eval] Running pycocotools evaluation (iou_types={iou_types})...")
        eval_coco_start = time.time()
        coco_metrics = evaluator.summarize()
        eval_coco_time = time.time() - eval_coco_start
        if is_primary:
            print(f"[Eval] pycocotools done in {eval_coco_time:.1f}s")
        for key, value in coco_metrics.items():
            log_dict[f"val/{key}"] = float(value)
        if "segm_AP" in coco_metrics:
            log_dict["val/mAP"] = float(coco_metrics["segm_AP"])
        elif "bbox_AP" in coco_metrics:
            log_dict["val/mAP"] = float(coco_metrics["bbox_AP"])
        if eval_scores:
            import numpy as np
            score_arr = np.asarray(eval_scores, dtype=np.float64)
            log_dict["val/diag_score_p50"] = float(np.percentile(score_arr, 50))
            log_dict["val/diag_score_p90"] = float(np.percentile(score_arr, 90))
        if eval_bbox_area_ratios:
            import numpy as np
            bbox_arr = np.asarray(eval_bbox_area_ratios, dtype=np.float64)
            log_dict["val/diag_bbox_area_ratio_p50"] = float(np.percentile(bbox_arr, 50))
            log_dict["val/diag_bbox_area_ratio_p90"] = float(np.percentile(bbox_arr, 90))
        if eval_total_masks > 0:
            log_dict["val/diag_mask_nonempty_ratio"] = float(
                eval_nonempty_masks / float(eval_total_masks)
            )
        log_dict["val/diag_num_eval_images"] = float(len(eval_image_ids))
        log_dict["val/diag_num_predictions"] = float(total_preds)
        coco_results_path = evaluator.dump(Path(output_dir) / "coco_instances_results.json")


    if not vis_batches or not vis_outputs:
        raise RuntimeError("Evaluation produced no batch for visualization")
    save_evaluation_comparisons(
        output_dir=output_dir,
        step=optimizer_step,
        batch=vis_batches[0],
        outputs=vis_outputs[0],
        max_samples=num_vis_images,
    )
    return EvaluationResult(
        log_dict=log_dict,
        coco_metrics=coco_metrics,
        coco_results_path=coco_results_path,
        visualization_batch=vis_batches[0] if vis_batches else None,
        visualization_outputs=vis_outputs[0] if vis_outputs else None,
    )
