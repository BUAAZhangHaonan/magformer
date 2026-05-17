#!/usr/bin/env python3
"""Build the R83 R80 4-view TTA COCO pseudo bank with segmentation RLE."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from magformer.config import load_config, set_seed  # noqa: E402
from magformer.config.loader import load_yaml_file, save_yaml_file  # noqa: E402
from magformer.config.schema import merge_configs  # noqa: E402
from magformer.data import CocoRgbdDataset  # noqa: E402
from magformer.data.collate import collate_fn  # noqa: E402
from magformer.models import build_model  # noqa: E402
from tools.diagnose_r82_tta_candidate_bank import (  # noqa: E402
    TTA_VIEWS,
    Candidate,
    _load_gt_buckets,
    _load_r78_stats,
    _require_finite_float,
    _resolve_project_path,
    _run_view,
    _view_candidates_to_gt,
    summarize_candidate_bank_buckets,
    union_cluster_candidates,
)
from tools.evaluate_1024_backmap import (  # noqa: E402
    Eval1024Transform,
    strict_load_with_evidence,
)


class R83BankError(RuntimeError):
    """Raised when the R83 pseudo-bank cannot be built or validated."""


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise R83BankError("distribution contains non-finite values")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _mask_area(candidate: Candidate) -> int:
    return int((candidate.mask > 0).sum())


def _bbox_area_xyxy(bbox: list[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _fill_ratio(candidate: Candidate) -> float:
    bbox_area = _bbox_area_xyxy(candidate.bbox)
    if bbox_area <= 0.0:
        return 0.0
    return float(_mask_area(candidate) / bbox_area)


def select_training_candidates(
    candidates: list[Candidate],
    *,
    min_score: float,
    min_mask_area: int,
    min_fill_ratio: float,
) -> tuple[list[Candidate], dict[str, int]]:
    stats = {
        "input": len(candidates),
        "kept": 0,
        "dropped_score": 0,
        "dropped_mask_area": 0,
        "dropped_fill_ratio": 0,
    }
    kept: list[Candidate] = []
    for candidate in candidates:
        if candidate.quality < float(min_score):
            stats["dropped_score"] += 1
            continue
        if _mask_area(candidate) < int(min_mask_area):
            stats["dropped_mask_area"] += 1
            continue
        if _fill_ratio(candidate) < float(min_fill_ratio):
            stats["dropped_fill_ratio"] += 1
            continue
        kept.append(candidate)
    stats["kept"] = len(kept)
    return kept, stats


def _encode_binary_mask(mask: np.ndarray) -> dict[str, Any]:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.ndim != 2:
        raise R83BankError(f"expected 2D mask, got shape={mask.shape}")
    rle = coco_mask.encode(np.asfortranarray(mask))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(v) for v in rle["size"]], "counts": counts}


def _bbox_xyxy_from_binary(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise R83BankError("cannot build annotation from empty mask")
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def candidate_to_coco_annotation(
    candidate: Candidate,
    *,
    annotation_id: int,
    image_id: int,
    image_height: int,
    image_width: int,
) -> dict[str, Any]:
    mask = (candidate.mask > 0).astype(np.uint8)
    if mask.shape != (int(image_height), int(image_width)):
        raise R83BankError(
            f"candidate mask shape {mask.shape} does not match image {(image_height, image_width)}"
        )
    x1, y1, x2, y2 = _bbox_xyxy_from_binary(mask)
    if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height or x2 <= x1 or y2 <= y1:
        raise R83BankError(
            f"candidate bbox out of bounds for image_id={image_id}: {[x1, y1, x2, y2]}"
        )
    area = int(mask.sum())
    if area <= 0:
        raise R83BankError(f"candidate area is zero for image_id={image_id}")
    score = _require_finite_float(candidate.quality, "candidate score")
    return {
        "id": int(annotation_id),
        "image_id": int(image_id),
        "category_id": int(candidate.category_id),
        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
        "area": area,
        "iscrowd": 0,
        "segmentation": _encode_binary_mask(mask),
        "score": score,
        "source_view": candidate.source_view,
        "fill_ratio": _fill_ratio(candidate),
    }


def _assert_json_finite(value: Any, context: str = "json") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_json_finite(item, f"{context}.{key}")
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            _assert_json_finite(item, f"{context}[{idx}]")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise R83BankError(f"non-finite value at {context}: {value!r}")


def _depth_path_for_image(dataset_root: Path, split: str, file_name: str) -> Path:
    stem = Path(file_name).stem
    for base in (
        dataset_root / "depth" / "depth_npy" / split,
        dataset_root / "depth" / split,
    ):
        for suffix in (".npy", ".npz"):
            path = base / f"{stem}{suffix}"
            if path.exists():
                return path
    raise R83BankError(f"depth path not found for image file {file_name}")


def _image_path_for_image(dataset_root: Path, split: str, file_name: str) -> Path:
    image_dir = dataset_root / "images" / split
    path = image_dir / file_name
    if path.exists():
        return path
    path = image_dir / Path(file_name).name
    if path.exists():
        return path
    raise R83BankError(f"image path not found for image file {file_name}")


def _bbox_iou_xywh(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, aw, ah = [float(v) for v in box_a]
    bx1, by1, bw, bh = [float(v) for v in box_b]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def duplicate_stats_by_image(kept_by_image: dict[int, list[Candidate]]) -> dict[str, int]:
    pairs_mask_iou_ge_095 = 0
    pairs_bbox_iou_ge_095 = 0
    for candidates in kept_by_image.values():
        for left_idx in range(len(candidates)):
            left = candidates[left_idx]
            for right in candidates[left_idx + 1 :]:
                if left.category_id != right.category_id:
                    continue
                left_mask = left.mask > 0
                right_mask = right.mask > 0
                inter = np.logical_and(left_mask, right_mask).sum(dtype=np.float64)
                union = np.logical_or(left_mask, right_mask).sum(dtype=np.float64)
                mask_iou = float(inter / union) if union > 0 else 0.0
                if mask_iou >= 0.95:
                    pairs_mask_iou_ge_095 += 1
                lx1, ly1, lx2, ly2 = left.bbox
                rx1, ry1, rx2, ry2 = right.bbox
                if _bbox_iou_xywh([lx1, ly1, lx2 - lx1, ly2 - ly1], [rx1, ry1, rx2 - rx1, ry2 - ry1]) >= 0.95:
                    pairs_bbox_iou_ge_095 += 1
    return {
        "same_class_pairs_mask_iou_ge_095": int(pairs_mask_iou_ge_095),
        "same_class_pairs_bbox_iou_ge_095": int(pairs_bbox_iou_ge_095),
    }


def validate_coco_bank(
    bank_path: Path,
    *,
    dataset_root: Path,
    split: str,
    expected_images: int | None,
    require_nonempty_images: bool,
) -> dict[str, Any]:
    coco = COCO(str(bank_path))
    images = list(coco.dataset.get("images", []))
    annotations = list(coco.dataset.get("annotations", []))
    categories = list(coco.dataset.get("categories", []))
    if expected_images is not None and len(images) != int(expected_images):
        raise R83BankError(f"expected {expected_images} images, got {len(images)}")
    if not categories:
        raise R83BankError("bank has no categories")
    category_ids = {int(category["id"]) for category in categories}
    anns_by_image: dict[int, list[dict[str, Any]]] = {int(image["id"]): [] for image in images}
    image_by_id = {int(image["id"]): image for image in images}
    for image in images:
        _image_path_for_image(dataset_root, split, str(image["file_name"]))
        _depth_path_for_image(dataset_root, split, str(image["file_name"]))
    mask_areas: list[float] = []
    bbox_areas: list[float] = []
    fill_ratios: list[float] = []
    scores: list[float] = []
    for ann in annotations:
        image_id = int(ann["image_id"])
        if image_id not in image_by_id:
            raise R83BankError(f"annotation references unknown image_id={image_id}")
        if int(ann["category_id"]) not in category_ids:
            raise R83BankError(f"annotation has illegal category_id={ann['category_id']}")
        image = image_by_id[image_id]
        h, w = int(image["height"]), int(image["width"])
        bbox = [float(v) for v in ann["bbox"]]
        if len(bbox) != 4:
            raise R83BankError(f"annotation {ann.get('id')} bbox is not length 4")
        x, y, bw, bh = bbox
        if x < 0 or y < 0 or bw <= 0 or bh <= 0 or x + bw > w or y + bh > h:
            raise R83BankError(f"annotation {ann.get('id')} bbox out of image bounds: {bbox}")
        segmentation = ann.get("segmentation")
        if not isinstance(segmentation, dict) or "counts" not in segmentation:
            raise R83BankError(f"annotation {ann.get('id')} lacks RLE segmentation")
        decoded = coco_mask.decode(segmentation)
        if decoded.shape != (h, w):
            raise R83BankError(
                f"annotation {ann.get('id')} decoded shape {decoded.shape} != {(h, w)}"
            )
        area = float(coco_mask.area(segmentation))
        json_area = float(ann["area"])
        if area <= 0 or abs(area - json_area) > 1e-6:
            raise R83BankError(
                f"annotation {ann.get('id')} area mismatch: rle={area} json={json_area}"
            )
        rle_bbox = [float(v) for v in coco_mask.toBbox(segmentation).tolist()]
        if any(abs(a - b) > 1e-6 for a, b in zip(rle_bbox, bbox)):
            raise R83BankError(
                f"annotation {ann.get('id')} bbox mismatch: rle={rle_bbox} json={bbox}"
            )
        score = _require_finite_float(ann.get("score", 1.0), f"score ann={ann.get('id')}")
        mask_areas.append(area)
        bbox_area = bw * bh
        bbox_areas.append(bbox_area)
        fill_ratios.append(area / bbox_area)
        scores.append(score)
        anns_by_image[image_id].append(ann)
    empty_images = [image_id for image_id, rows in anns_by_image.items() if not rows]
    if require_nonempty_images and empty_images:
        raise R83BankError(f"bank has empty images: {empty_images[:10]}")
    CocoRgbdDataset(
        dataset_root=str(dataset_root),
        ann_file=str(bank_path),
        split=split,
        transform=None,
        is_train=True,
    )
    return {
        "images": len(images),
        "annotations": len(annotations),
        "empty_images": len(empty_images),
        "categories": sorted(category_ids),
        "score": _distribution(scores),
        "mask_area": _distribution(mask_areas),
        "bbox_area": _distribution(bbox_areas),
        "fill_ratio": _distribution(fill_ratios),
    }


def _load_target_coco_subset(path: Path, image_ids: set[int]) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    images = [image for image in data["images"] if int(image["id"]) in image_ids]
    if len(images) != len(image_ids):
        raise R83BankError(f"target annotation missing images for subset: {len(images)} != {len(image_ids)}")
    return {
        "images": images,
        "annotations": [],
        "categories": data["categories"],
    }


def _r82_filter_count(path: Path, *, min_score: float, min_mask_area: int, min_fill_ratio: float) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    count = 0
    for record in rows:
        for item in record.get("candidates", []):
            score = float(item["score"])
            area = float(item["mask_area"])
            x1, y1, x2, y2 = [float(v) for v in item["bbox"]]
            bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            fill = area / bbox_area if bbox_area > 0.0 else 0.0
            if score >= min_score and area >= min_mask_area and fill >= min_fill_ratio:
                count += 1
    return count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/vc_suda_stage_c_r80_unsup_schedule_1024.yaml")
    parser.add_argument("--weights", default="output/vc_suda/stage_c_r80_unsup_schedule_1024/checkpoint_iter_0000750.pth")
    parser.add_argument("--dataset-root", default="magformer_datasets/pseudo_real_512")
    parser.add_argument("--target-ann", default="annotations/instances_target_unlabeled.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--r78-stats", default="output/diagnostics/r52_target_unlabeled_sampling_20260516/target_sampling_stats.json")
    parser.add_argument("--output-dir", default="output/diagnostics/r83_tta_coco_bank_20260517")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--inference-topk", type=int, default=100)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--train-score-threshold", type=float, default=0.90)
    parser.add_argument("--min-mask-area", type=int, default=20)
    parser.add_argument("--min-fill-ratio", type=float, default=0.1)
    parser.add_argument("--report-interval", type=int, default=20)
    parser.add_argument("--r82-candidate-json", default="output/diagnostics/r82_tta_candidate_bank_full200_20260517/r82_tta_candidate_predictions.json")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-empty-images", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.chdir(REPO_ROOT)
    out_dir = _resolve_project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bank_path = _resolve_project_path(args.output_json) if args.output_json else out_dir / "instances_tta_pseudo_score090.json"
    summary_path = out_dir / "r83_tta_coco_bank_summary.json"
    buckets_path = out_dir / "r83_tta_coco_bank_buckets.json"
    runtime_config = out_dir / "r83_tta_coco_bank_runtime.yaml"
    dataset_root = _resolve_project_path(args.dataset_root)
    target_ann_path = _resolve_project_path(dataset_root / args.target_ann)

    if args.validate_only:
        validation = validate_coco_bank(
            bank_path,
            dataset_root=dataset_root,
            split=args.split,
            expected_images=args.max_images,
            require_nonempty_images=not args.allow_empty_images,
        )
        print("[R83] validation_json=" + json.dumps(validation, sort_keys=True), flush=True)
        return {"validation": validation, "bank_json": str(bank_path)}

    if args.batch_size != 1:
        raise R83BankError("R83 TTA COCO bank export requires --batch-size 1")
    if args.max_images <= 0:
        raise R83BankError("--max-images must be positive")
    if len(TTA_VIEWS) != 4 or TTA_VIEWS != ((1.0, False), (1.0, True), (1.25, False), (1.25, True)):
        raise R83BankError(f"unexpected TTA views: {TTA_VIEWS}")

    overrides = {
        "data": {
            "dataset_root": args.dataset_root,
            "image_size": args.image_size,
            "min_scale": 1.0,
            "max_scale": 1.0,
        },
        "model": {
            "weights": args.weights,
            "finetune_weights": args.weights,
        },
        "runtime": {
            "device": "cuda" if str(args.device).startswith("cuda") else "cpu",
            "gpus": [0],
            "ddp_enabled": False,
            "num_workers": args.num_workers,
            "output_dir": str(out_dir),
        },
    }
    cfg_dict = merge_configs(load_yaml_file(args.base_config), overrides)
    save_yaml_file(cfg_dict, str(runtime_config))
    cfg = load_config(str(runtime_config))
    set_seed(int(cfg.runtime.seed))
    random.seed(int(cfg.runtime.seed))
    np.random.seed(int(cfg.runtime.seed))
    torch.manual_seed(int(cfg.runtime.seed))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise R83BankError(f"CUDA device requested but unavailable: {args.device}")
    weights = _resolve_project_path(args.weights)
    if not weights.exists():
        raise R83BankError(f"weights not found: {weights}")

    dataset = CocoRgbdDataset(
        dataset_root=args.dataset_root,
        ann_file=args.target_ann,
        split=args.split,
        transform=None,
        is_train=False,
    )
    if args.max_images > len(dataset):
        raise R83BankError(f"--max-images={args.max_images} exceeds dataset images={len(dataset)}")
    depth_cfg = cfg.data.depth
    dataset.transform = Eval1024Transform(
        image_size=cfg.data.image_size,
        depth_scale=depth_cfg.scale,
        depth_shift=depth_cfg.shift,
        depth_clip_min=depth_cfg.clip_min,
        depth_clip_max=depth_cfg.clip_max,
        depth_norm=depth_cfg.norm,
        depth_per_sample_norm=depth_cfg.per_sample_norm,
    )
    image_size_by_id = {
        int(img_id): (int(info["height"]), int(info["width"]))
        for img_id, info in dataset.coco.imgs.items()
    }
    category_ids = list(getattr(dataset, "category_ids", [])) or None
    legal_categories = set(category_ids or [1])
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    model = build_model(cfg)
    strict_load_with_evidence(model, str(weights))
    model = model.to(device)
    model.eval()

    gt_by_image, bottom20_threshold = _load_gt_buckets(target_ann_path)
    r78_by_image = _load_r78_stats(_resolve_project_path(args.r78_stats))

    print(
        "[R83] "
        + json.dumps(
            {
                "views": [(scale, hflip) for scale, hflip in TTA_VIEWS],
                "device": str(device),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "max_images": args.max_images,
                "inference_topk": args.inference_topk,
                "train_score_threshold": args.train_score_threshold,
                "min_mask_area": args.min_mask_area,
                "min_fill_ratio": args.min_fill_ratio,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    per_image: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    kept_by_image: dict[int, list[Candidate]] = {}
    sampled_image_ids: set[int] = set()
    raw_candidate_count = 0
    union_candidate_count = 0
    filter_stats = {
        "input": 0,
        "kept": 0,
        "dropped_score": 0,
        "dropped_mask_area": 0,
        "dropped_fill_ratio": 0,
    }
    score_values: list[float] = []
    mask_area_values: list[float] = []
    fill_ratio_values: list[float] = []
    annotation_id = 1
    start = time.time()

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.max_images:
                break
            image_id = int(batch["image_ids"][0].item())
            sampled_image_ids.add(image_id)
            out_h, out_w = image_size_by_id[image_id]
            images = batch["images"].to(device, non_blocking=True)
            depths = batch["depths"].to(device, non_blocking=True)
            padding_masks = batch.get("padding_masks")
            if torch.is_tensor(padding_masks):
                padding_masks = padding_masks.to(device, non_blocking=True)
            content_masks = batch.get("content_masks")
            if content_masks is None:
                raise R83BankError("batch lacks content_masks; refusing padded-coordinate export")

            view_candidates: list[Candidate] = []
            for scale, hflip in TTA_VIEWS:
                outputs = _run_view(
                    model,
                    images,
                    depths,
                    padding_masks,
                    scale=scale,
                    hflip=hflip,
                    device=device,
                    amp_enabled=bool(args.amp),
                    inference_topk=int(args.inference_topk),
                )
                pred = outputs["predictions"][0]
                view_candidates.extend(
                    _view_candidates_to_gt(
                        pred,
                        image_id=image_id,
                        content_mask=content_masks[0],
                        out_h=out_h,
                        out_w=out_w,
                        category_ids=category_ids,
                        scale=scale,
                        hflip=hflip,
                        score_threshold=float(args.score_threshold),
                        mask_threshold=float(args.mask_threshold),
                        base_size=tuple(int(v) for v in images.shape[-2:]),
                    )
                )
                del outputs
            if not view_candidates:
                raise R83BankError(f"zero raw view candidates for image_id={image_id}")
            raw_candidate_count += len(view_candidates)
            merged = union_cluster_candidates(view_candidates)
            if not merged:
                raise R83BankError(f"zero union candidates for image_id={image_id}")
            union_candidate_count += len(merged)
            kept, image_filter_stats = select_training_candidates(
                merged,
                min_score=float(args.train_score_threshold),
                min_mask_area=int(args.min_mask_area),
                min_fill_ratio=float(args.min_fill_ratio),
            )
            for key in filter_stats:
                filter_stats[key] += int(image_filter_stats[key])
            if not kept and not args.allow_empty_images:
                raise R83BankError(f"zero training-filtered candidates for image_id={image_id}")
            kept_by_image[image_id] = kept
            per_image.append(
                {
                    "image_id": image_id,
                    "target_size": (out_h, out_w),
                    "candidates": merged,
                    "kept": kept,
                }
            )
            for candidate in kept:
                if candidate.category_id not in legal_categories:
                    raise R83BankError(
                        f"illegal category_id={candidate.category_id} for image_id={image_id}"
                    )
                annotation = candidate_to_coco_annotation(
                    candidate,
                    annotation_id=annotation_id,
                    image_id=image_id,
                    image_height=out_h,
                    image_width=out_w,
                )
                annotations.append(annotation)
                annotation_id += 1
                score_values.append(float(candidate.quality))
                mask_area_values.append(float(annotation["area"]))
                fill_ratio_values.append(float(annotation["fill_ratio"]))
            if (batch_idx + 1) % int(args.report_interval) == 0 or (batch_idx + 1) == args.max_images:
                elapsed = time.time() - start
                print(
                    f"[R83] processed={batch_idx + 1}/{args.max_images} raw={raw_candidate_count} "
                    f"union={union_candidate_count} kept={len(annotations)} seconds={elapsed:.1f}",
                    flush=True,
                )
            del images, depths
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if len(sampled_image_ids) != int(args.max_images):
        raise R83BankError(f"sampled image count mismatch: {len(sampled_image_ids)}")
    coco_bank = _load_target_coco_subset(target_ann_path, sampled_image_ids)
    coco_bank["annotations"] = annotations
    _assert_json_finite(coco_bank)
    bank_path.write_text(json.dumps(coco_bank, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    bucket_summary = summarize_candidate_bank_buckets(
        per_image,
        gt_by_image=gt_by_image,
        r78_by_image=r78_by_image,
        bottom20_threshold=bottom20_threshold,
        target_ann_path=target_ann_path,
        r78_stats_path=_resolve_project_path(args.r78_stats),
    )
    validation = validate_coco_bank(
        bank_path,
        dataset_root=dataset_root,
        split=args.split,
        expected_images=args.max_images,
        require_nonempty_images=not args.allow_empty_images,
    )
    r82_expected = _r82_filter_count(
        _resolve_project_path(args.r82_candidate_json),
        min_score=float(args.train_score_threshold),
        min_mask_area=int(args.min_mask_area),
        min_fill_ratio=float(args.min_fill_ratio),
    )
    summary = {
        "bank_json": str(bank_path),
        "bucket_json": str(buckets_path),
        "summary_json": str(summary_path),
        "runtime_config": str(runtime_config),
        "config": str(_resolve_project_path(args.base_config)),
        "weights": str(weights),
        "target_ann": str(target_ann_path),
        "dataset_root": str(dataset_root),
        "r78_stats": str(_resolve_project_path(args.r78_stats)),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "max_images": int(args.max_images),
        "views": [{"scale": scale, "hflip": hflip} for scale, hflip in TTA_VIEWS],
        "raw_view_candidates": int(raw_candidate_count),
        "union_candidates": int(union_candidate_count),
        "annotations": len(annotations),
        "score_threshold": float(args.score_threshold),
        "mask_threshold": float(args.mask_threshold),
        "train_score_threshold": float(args.train_score_threshold),
        "min_mask_area": int(args.min_mask_area),
        "min_fill_ratio": float(args.min_fill_ratio),
        "inference_topk": int(args.inference_topk),
        "filter_stats": filter_stats,
        "score": _distribution(score_values),
        "mask_area": _distribution(mask_area_values),
        "fill_ratio": _distribution(fill_ratio_values),
        "duplicates": duplicate_stats_by_image(kept_by_image),
        "r82_candidate_json": str(_resolve_project_path(args.r82_candidate_json)),
        "r82_score090_area_fill_expected_annotations": r82_expected,
        "r82_count_delta": None if r82_expected is None else int(len(annotations) - r82_expected),
        "coverage_buckets": bucket_summary["buckets"],
        "validation": validation,
        "seconds": float(time.time() - start),
    }
    _assert_json_finite(summary)
    buckets_path.write_text(json.dumps(bucket_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[R83] bank_json=" + str(bank_path), flush=True)
    print("[R83] summary_json=" + str(summary_path), flush=True)
    print("[R83] bucket_json=" + str(buckets_path), flush=True)
    print("[R83] validation_json=" + json.dumps(validation, sort_keys=True), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        run(args)
    except R83BankError as exc:
        print(f"[R83][ERROR] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
