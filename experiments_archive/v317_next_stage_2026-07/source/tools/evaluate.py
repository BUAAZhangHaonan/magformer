#!/usr/bin/env python3
"""
MAGFormer Evaluation Script (Pure PyTorch)

Weight selection priority is explicit: --weights > config.model.weights > error.

Usage:
    python tools/evaluate.py --config-file configs/magformer.yaml --dataset-root /path/to/eccd
    python tools/evaluate.py --config-file configs/magformer.yaml --dataset-root /path/to/eccd --compile
"""

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from magformer.config import load_config, setup_device, set_seed
from magformer.data import CocoRgbdDataset
from magformer.data.transforms import RGBDTransform
from magformer.data.collate import collate_fn
from magformer.engine.eval_runtime import (
    _build_prefix_limited_eval_loader,
    run_inference_evaluation,
)
from magformer.models import build_model
from magformer.engine.utils import load_checkpoint, validate_best_model_artifact


def apply_ema_weights_from_checkpoint(model, checkpoint):
    """Apply EMA shadow weights from either model_best or training_state artifacts."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("--ema requested but checkpoint is not a mapping")
    validate_best_model_artifact(checkpoint)
    ema_state = checkpoint.get("ema_state_dict")
    if not isinstance(ema_state, Mapping):
        raise ValueError("--ema requested but checkpoint has no ema_state_dict mapping")
    shadow_params = ema_state.get("shadow")
    if not isinstance(shadow_params, Mapping):
        raise ValueError("--ema requested but checkpoint has no ema_state_dict.shadow")
    incompatible = model.load_state_dict(shadow_params, strict=False)
    print("[EMA] Loaded EMA shadow weights")
    return incompatible


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MAGFormer model")
    parser.add_argument("--config-file", required=True, help="Path to config yaml")
    parser.add_argument("--dataset-root", required=False, help="Override dataset root")
    parser.add_argument("--weights", required=False, help="Checkpoint path")
    parser.add_argument("--output", default="output/eval", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--max-images", type=int, default=None, help="Limit eval to first N images")
    parser.add_argument("--max-dets", type=int, default=100, help="Max detections per image for COCO eval")
    parser.add_argument("--iou-types", nargs="+", default=None, help="Override IoU types (e.g. bbox)")
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Enable torch.compile with reduce-overhead mode for faster inference. "
             "Adds a warmup phase (2 dummy forward passes) before timed evaluation.",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (default: reduce-overhead). Only used with --compile.",
    )
    parser.add_argument(
        "--compile-warmup-iters",
        type=int,
        default=2,
        help="Number of warmup forward passes for torch.compile JIT (default: 2). "
             "Only used with --compile.",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        default=False,
        help="Use EMA shadow weights for evaluation instead of training weights.",
    )
    parser.add_argument(
        "--gpu-export",
        action="store_true",
        default=False,
        help="Use GPU-native postprocess export path (gpu_export=True): top-k/"
             "sigmoid/threshold/mask-score fusion and bbox extraction stay on "
             "GPU, masks are bit-packed and moved to pinned host memory via one "
             "async copy. Produces bit-identical predictions to the default "
             "path; eliminates the synchronous DtoH mask copies.",
    )
    parser.add_argument(
        "--export-pipeline",
        choices=["serial", "procs", "threads"],
        default="serial",
        help="Export stage topology. 'serial' (default) keeps the historical "
             "synchronous per-image export in the eval loop. 'procs' overlaps "
             "the CPU export work (unpack bits -> fortran RLE encode -> COCO "
             "rows) with GPU compute via a transfer thread + multiprocessing "
             "pool (pycocotools encode holds the GIL, so processes are "
             "required for real parallelism). 'threads' uses consumer threads "
             "instead (kept for measurement; expect no speedup). Both "
             "pipelined modes imply --gpu-export and produce byte-identical "
             "COCO result rows.",
    )
    parser.add_argument(
        "--export-workers",
        type=int,
        default=4,
        help="Consumer count for --export-pipeline (processes or threads).",
    )
    parser.add_argument(
        "--export-depth",
        type=int,
        default=8,
        help="Bounded in-flight depth of the export pipeline (pinned ring "
             "slots; bounds pipeline RAM to roughly depth x ~13 MB of packed "
             "masks plus consumer-side transients).",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        default=False,
        help="Capture the fixed-shape inference region (model forward + GPU "
             "postprocess + export prep) into a manual CUDA graph "
             "(MagFormerArch.forward_inference_graphed). Bit-identical fp32 "
             "outputs; removes eager launch overhead. Implies --gpu-export. "
             "Mutually exclusive with --export-pipeline.",
    )
    return parser.parse_args()


def _compile_warmup(model, device, image_size, batch_size, warmup_iters=2):
    """Run dummy forward passes to trigger torch.compile JIT compilation.

    This ensures compilation overhead is NOT counted in the timed evaluation loop.
    Uses the model's forward_inference_raw path with random tensors matching eval shapes.
    """
    import time

    inference_model = model.module if hasattr(model, "module") else model

    # Create dummy inputs matching eval resolution
    dummy_images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    dummy_depths = torch.randn(batch_size, 1, image_size, image_size, device=device)
    dummy_padding_masks = torch.zeros(batch_size, image_size, image_size, dtype=torch.bool, device=device)
    dummy_depth_valid_masks = torch.ones_like(
        dummy_depths, dtype=torch.bool, device=device
    )
    dummy_noise_masks = torch.zeros(batch_size, 1, image_size, image_size, dtype=torch.bool, device=device)

    print(f"[Compile Warmup] Running {warmup_iters} warmup forward passes "
          f"(bs={batch_size}, img_size={image_size})...")
    print(f"[Compile Warmup] First pass will trigger JIT compilation -- this may take 1-3 minutes.")

    for i in range(warmup_iters):
        t0 = time.time()
        with torch.inference_mode():
            _ = inference_model.forward_inference_raw(
                dummy_images,
                dummy_depths,
                padding_masks=dummy_padding_masks,
                depth_valid_masks=dummy_depth_valid_masks,
                depth_noise_masks=dummy_noise_masks,
            )
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        print(f"[Compile Warmup] Pass {i+1}/{warmup_iters}: {elapsed:.2f}s")

    # Free dummy tensors
    del (
        dummy_images,
        dummy_depths,
        dummy_padding_masks,
        dummy_depth_valid_masks,
        dummy_noise_masks,
    )
    torch.cuda.empty_cache()
    print("[Compile Warmup] Done. Compilation overhead excluded from eval timing.")


def build_val_loader(config, dataset_root_override=None, num_workers=4, batch_size=1):
    data_cfg = config.data
    dataset_root = dataset_root_override or data_cfg.dataset_root
    val_split = getattr(data_cfg, "val_split", "val")

    dataset = CocoRgbdDataset(
        dataset_root=dataset_root,
        ann_file=data_cfg.val_ann,
        split=val_split,
        transform=None,
        is_train=False,
    )

    dataset.transform = RGBDTransform(
        image_size=data_cfg.image_size,
        min_scale=data_cfg.min_scale,
        max_scale=data_cfg.max_scale,
        random_flip="none",
        rgb_brightness=0.0,
        rgb_contrast=0.0,
        rgb_saturation=0.0,
        rgb_hue=0.0,
        depth_scale=data_cfg.depth.scale,
        depth_shift=data_cfg.depth.shift,
        depth_clip_min=data_cfg.depth.clip_min,
        depth_clip_max=data_cfg.depth.clip_max,
        depth_norm=data_cfg.depth.norm,
        depth_per_sample_norm=getattr(data_cfg.depth, "per_sample_norm", True),
        is_train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=4,
    )

    return dataset, loader


def rescale_predictions_to_original(predictions, original_sizes, target_size):
    """Rescale predictions without splitting mask and box geometry.

    Args:
        predictions: Internal rows use ``mask`` plus half-open XYXY ``bbox``.
            Standard COCO rows use ``segmentation`` plus XYWH ``bbox``. A
            bbox-only row is treated as internal half-open XYXY.
        original_sizes: mapping of image ID to (orig_h, orig_w)
        target_size: the (h, w) the images were resized to before inference

    For mask-bearing instances, the resized binary mask is the sole geometry
    source. Its RLE and bbox are regenerated together after nearest-neighbor
    resizing. Existing bbox coordinates are ignored.
    """
    from pycocotools import mask as coco_mask

    rescaled = []
    if not isinstance(target_size, (list, tuple)) or len(target_size) != 2:
        raise ValueError("target_size must be (height, width)")
    target_h, target_w = [int(value) for value in target_size]
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")

    for prediction_index, pred in enumerate(predictions):
        img_id = int(pred["image_id"])
        if img_id not in original_sizes:
            raise KeyError(f"missing original size for image_id={img_id}")
        orig_h, orig_w = [int(value) for value in original_sizes[img_id]]
        if orig_h <= 0 or orig_w <= 0:
            raise ValueError(
                f"original size for image_id={img_id} must be positive, "
                f"got {(orig_h, orig_w)}"
            )

        scale_x = orig_w / target_w
        scale_y = orig_h / target_h

        new_pred = {
            "image_id": img_id,
            "category_id": int(pred["category_id"]),
            "score": float(pred["score"]),
        }

        has_internal_mask = "mask" in pred
        has_standard_segmentation = "segmentation" in pred
        if has_internal_mask and has_standard_segmentation:
            raise ValueError(
                f"prediction row {prediction_index} cannot contain both "
                "'mask' and 'segmentation'"
            )

        if has_internal_mask or has_standard_segmentation:
            mask_key = "mask" if has_internal_mask else "segmentation"
            mask_data = pred[mask_key]
            if isinstance(mask_data, dict) and "counts" in mask_data:
                rle = {"size": mask_data["size"], "counts": mask_data["counts"]}
                if isinstance(rle["counts"], str):
                    rle["counts"] = rle["counts"].encode("ascii")
                binary = coco_mask.decode(rle)
            else:
                binary = np.asarray(mask_data)

            if binary.ndim == 3:
                if binary.shape[-1] != 1:
                    raise ValueError(
                        f"prediction row {prediction_index} must contain one mask, "
                        f"got shape {binary.shape}"
                    )
                binary = binary[..., 0]
            if binary.ndim != 2 or binary.shape != (target_h, target_w):
                raise ValueError(
                    f"prediction row {prediction_index} mask shape must be "
                    f"{(target_h, target_w)}, got {binary.shape}"
                )
            if not np.isfinite(binary).all():
                raise ValueError(f"prediction row {prediction_index} mask contains NaN or Inf")
            unique_values = np.unique(binary)
            if not np.isin(unique_values, [0, 1]).all():
                raise ValueError(
                    f"prediction row {prediction_index} mask must already be binary, "
                    f"got values {unique_values.tolist()}"
                )
            binary = (binary > 0).astype(np.uint8)

            resized = cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            resized = (resized > 0).astype(np.uint8)
            rle_new = coco_mask.encode(np.asfortranarray(resized))
            mask_bbox_xywh = [float(value) for value in coco_mask.toBbox(rle_new)]
            if isinstance(rle_new["counts"], bytes):
                rle_new["counts"] = rle_new["counts"].decode("ascii")
            final_rle = {"size": rle_new["size"], "counts": rle_new["counts"]}
            x, y, width, height = mask_bbox_xywh
            if has_internal_mask:
                new_pred["mask"] = final_rle
                new_pred["bbox"] = [x, y, x + width, y + height]
            else:
                new_pred["segmentation"] = final_rle
                new_pred["bbox"] = mask_bbox_xywh
        else:
            if "bbox" not in pred:
                raise ValueError(
                    f"prediction row {prediction_index} has neither mask geometry nor bbox"
                )
            bbox = pred["bbox"]
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError(
                    f"prediction row {prediction_index} bbox must be half-open XYXY"
                )
            x1, y1, x2, y2 = [float(value) for value in bbox]
            if not np.isfinite([x1, y1, x2, y2]).all():
                raise ValueError(f"prediction row {prediction_index} bbox contains NaN or Inf")
            new_pred["bbox"] = [
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y,
            ]

        rescaled.append(new_pred)

    return rescaled


def main() -> None:
    args = parse_args()

    if args.export_pipeline != "serial":
        args.gpu_export = True

    # Fork the export worker pool BEFORE any CUDA initialization / DataLoader
    # workers exist (safest fork point; children only ever run the numpy +
    # pycocotools export task).
    export_pool = None
    if args.export_pipeline == "procs":
        import multiprocessing as mp

        export_pool = mp.get_context("fork").Pool(processes=args.export_workers)

    overrides = {}
    if args.dataset_root is not None:
        overrides.setdefault("data", {})["dataset_root"] = args.dataset_root
    if args.weights is not None:
        overrides.setdefault("model", {})["weights"] = args.weights
    overrides.setdefault("runtime", {})["output_dir"] = args.output

    config = load_config(args.config_file, overrides=overrides)

    # Single-GPU eval: respect CUDA_VISIBLE_DEVICES
    config.runtime.gpus = [0]
    device = setup_device(config.runtime)
    set_seed(config.runtime.seed)

    output_dir = Path(config.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, loader = build_val_loader(
        config,
        dataset_root_override=args.dataset_root,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    target_image_size = config.data.image_size
    print(f"[Eval] Target image_size from config: {target_image_size}")

    # Build a map of image_id -> (orig_h, orig_w) from COCO GT
    original_sizes = {}
    for img_id in dataset.image_ids:
        img_info = dataset.coco.loadImgs(img_id)[0]
        orig_h = img_info.get("height", 0)
        orig_w = img_info.get("width", 0)
        if orig_h == 0 or orig_w == 0:
            # Load image to get actual size
            import cv2 as _cv2
            fname = img_info["file_name"]
            img_path = Path(dataset.image_dir) / fname
            if not img_path.exists():
                img_path = Path(dataset.image_dir) / Path(fname).name
            _img = _cv2.imread(str(img_path))
            if _img is not None:
                orig_h, orig_w = _img.shape[:2]
        original_sizes[img_id] = (orig_h, orig_w)

    # Check if any image differs from target_image_size
    needs_rescale = any(
        h != target_image_size or w != target_image_size
        for h, w in original_sizes.values()
    )
    print(f"[Eval] Original image sizes: sample={list(original_sizes.values())[:5]}")
    print(f"[Eval] Target eval size: {target_image_size}x{target_image_size}")
    print(f"[Eval] Needs rescale: {needs_rescale}")

    # Explicit priority: CLI --weights wins, then config.model.weights, then fail loudly.
    effective_weights = args.weights or getattr(config.model, "weights", None)
    if not effective_weights:
        raise ValueError(
            "Evaluation requires weights. Priority is --weights > config.model.weights > explicit error."
        )

    model = build_model(config)
    _ckpt, _incompat = load_checkpoint(effective_weights, model, strict=False)
    if _incompat is not None:
        _unexpected = sorted(_incompat.unexpected_keys)
        _missing = sorted(_incompat.missing_keys)
        if _unexpected:
            print(f"[Eval] Warning: {len(_unexpected)} unexpected keys in checkpoint (normal for ablation): {_unexpected[:5]}...")
        if _missing:
            print(f"[Eval] Warning: {len(_missing)} missing keys: {_missing[:5]}...")

    # Optionally load EMA shadow weights (subtly different from training weights)
    if args.ema:
        apply_ema_weights_from_checkpoint(model, _ckpt)

    model = model.to(device)
    model.eval()

    # --- torch.compile integration ---
    if args.compile:
        # Allow Triton custom autograd functions (fused_gate_blend) without graph breaks.
        # suppress_errors=True lets Dynamo fall back to eager for unsupported ops
        # instead of crashing the whole compile.
        import importlib; _dynamo = importlib.import_module("torch._dynamo")
        import torch as _torch_mod; _torch_mod._dynamo.config.suppress_errors = True

        print(f"[Eval] Compiling model with torch.compile(mode='{args.compile_mode}')...")
        compile_start = __import__("time").time()
        model = torch.compile(model, mode=args.compile_mode)
        compile_wall = __import__("time").time() - compile_start
        print(f"[Eval] torch.compile() wrapper created in {compile_wall:.2f}s "
              f"(actual JIT happens during warmup forward passes)")

    # --- Custom eval loop with resize and rescale ---
    from magformer.engine.coco_export import outputs_to_coco_instances
    from magformer.engine.evaluator import COCOEvaluator
    import time
    import gc
    from contextlib import nullcontext
    from torch.amp import autocast

    iou_types = args.iou_types or ["bbox", "segm"]

    # Determine what transforms the dataset currently applies
    # Check if images come out at target_image_size or original size
    sample = dataset[0]
    img_tensor = sample["image"]  # (C, H, W) after ToTensor
    actual_h, actual_w = img_tensor.shape[1], img_tensor.shape[2]
    print(f"[Eval] Sample image after transform: {actual_h}x{actual_w} (target: {target_image_size}x{target_image_size})")

    if actual_h != target_image_size or actual_w != target_image_size:
        # Transforms are NOT resizing -- need to apply resize manually
        print(f"[Eval] WARNING: Images are NOT being resized to {target_image_size}. Applying manual resize.")
        do_manual_resize = True
    else:
        print(f"[Eval] Images already at target size {target_image_size}x{target_image_size}. No manual resize needed.")
        do_manual_resize = False

    # Rebuild dataset with transforms that DO resize for eval
    if do_manual_resize:
        print("[Eval] Rebuilding dataset with resize-enabled transforms...")
        dataset.transform = RGBDTransform(
            image_size=target_image_size,
            min_scale=1.0,
            max_scale=1.0,
            random_flip="none",
            rgb_brightness=0.0,
            rgb_contrast=0.0,
            rgb_saturation=0.0,
            rgb_hue=0.0,
            depth_scale=config.data.depth.scale,
            depth_shift=config.data.depth.shift,
            depth_clip_min=config.data.depth.clip_min,
            depth_clip_max=config.data.depth.clip_max,
            depth_norm=config.data.depth.norm,
            depth_per_sample_norm=getattr(config.data.depth, "per_sample_norm", True),
            # Use is_train=True but with no augmentation to get resize+crop
            is_train=True,
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True,
            prefetch_factor=4,
        )

        # Verify resize is working
        sample2 = dataset[0]
        img2 = sample2["image"]
        print(f"[Eval] After rebuild: sample image size = {img2.shape[1]}x{img2.shape[2]}")

    loader, total_images = _build_prefix_limited_eval_loader(
        loader, args.max_images
    )

    # --- torch.compile warmup: trigger JIT before timed eval loop ---
    if args.compile:
        _compile_warmup(
            model, device,
            image_size=target_image_size,
            batch_size=args.batch_size,
            warmup_iters=args.compile_warmup_iters,
        )

    evaluator = COCOEvaluator(coco_gt=dataset.coco, iou_types=iou_types, max_dets=args.max_dets)

    export_pipeline = None
    if args.export_pipeline != "serial":
        if needs_rescale and do_manual_resize:
            raise ValueError(
                "--export-pipeline does not support the rescale path; use "
                "--export-pipeline serial when images need rescaling"
            )
        from magformer.engine.export_pipeline import ExportPipeline

        export_pipeline = ExportPipeline(
            workers=args.export_workers,
            depth=args.export_depth,
            backend=args.export_pipeline,
            category_id_list=list(getattr(dataset, "category_ids", [])) or None,
            category_offset=1,
            score_threshold=0.0,
            mask_threshold=0.5,
            allow_empty_fallback=False,
            empty_fallback_ratio=0.01,
            pool=export_pool,
        )
        export_pipeline.start()
        print(f"[Eval] Export pipeline ON: backend={args.export_pipeline}, "
              f"workers={args.export_workers}, depth={args.export_depth}")

    eval_start = time.time()
    num_evaluated = 0
    num_images_evaluated = 0
    seq = 0

    for batch in loader:
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

        if "image_ids" not in batch:
            raise KeyError("Evaluation batch is missing required image_ids")
        batch_image_ids = [int(image_id) for image_id in batch["image_ids"]]
        if len(batch_image_ids) != int(images.shape[0]):
            raise ValueError(
                "Evaluation image_ids length must match batch size: "
                f"image_ids={len(batch_image_ids)}, batch={int(images.shape[0])}"
            )

        inference_model = model.module if hasattr(model, "module") else model
        if export_pipeline is not None:
            with torch.inference_mode():
                outputs = inference_model.forward_inference_raw(
                    images,
                    depths,
                    padding_masks=padding_masks,
                    depth_valid_masks=depth_valid_masks,
                    depth_noise_masks=noise_masks,
                    export_ring=export_pipeline.ring,
                )
            if "slot" not in outputs:
                raise RuntimeError(
                    "Pipelined export requires the deferred GPU export slot; "
                    f"got output keys={sorted(outputs.keys())}"
                )
            # Non-blocking hand-off (bounded by ring depth + queue depth):
            # the GPU is immediately free to start the next forward while a
            # transfer thread syncs the payload event and worker processes
            # build the COCO rows.
            export_pipeline.submit(seq, batch_image_ids, outputs["slot"])
            seq += len(batch_image_ids)
            del outputs
        else:
            with torch.inference_mode():
                if getattr(args, "cuda_graph", False):
                    # Manual CUDA-graph path: capture on the first batch, then
                    # copy inputs -> replay per image. Bit-identical to the
                    # eager gpu_export path.
                    outputs = inference_model.forward_inference_graphed(
                        images,
                        depths,
                        padding_masks=padding_masks,
                        depth_valid_masks=depth_valid_masks,
                        depth_noise_masks=noise_masks,
                    )
                else:
                    outputs = inference_model.forward_inference_raw(
                        images,
                        depths,
                        padding_masks=padding_masks,
                        depth_valid_masks=depth_valid_masks,
                        depth_noise_masks=noise_masks,
                        gpu_export=args.gpu_export,
                    )

            predictions = outputs_to_coco_instances(
                outputs=outputs,
                image_ids=batch_image_ids,
                score_threshold=0.0,
                mask_threshold=0.5,
                category_offset=1,
                category_ids=list(getattr(dataset, "category_ids", [])) or None,
            )

            # Rescale predictions to original image size if needed
            if needs_rescale and do_manual_resize:
                predictions = rescale_predictions_to_original(
                    predictions, original_sizes,
                    target_size=(target_image_size, target_image_size),
                )

            evaluator.update(predictions, image_ids=batch_image_ids)
            del outputs

        num_evaluated += 1
        num_images_evaluated += len(batch_image_ids)
        if num_images_evaluated > total_images:
            raise RuntimeError(
                "Evaluation exceeded the exact image plan: "
                f"planned={total_images}, observed={num_images_evaluated}"
            )
        if num_evaluated % 50 == 0:
            elapsed = time.time() - eval_start
            rate = num_images_evaluated / elapsed if elapsed > 0 else 0
            remaining = (total_images - num_images_evaluated) / rate if rate > 0 else 0
            print(f"[Eval] {num_images_evaluated}/{total_images} "
                  f"({rate:.1f} img/s, ETA {remaining:.0f}s)")
            gc.collect()
            torch.cuda.empty_cache()

        if num_images_evaluated >= total_images:
            break

    if num_images_evaluated != total_images:
        raise RuntimeError(
            "Evaluation did not complete the exact image plan: "
            f"planned={total_images}, observed={num_images_evaluated}"
        )

    if export_pipeline is not None:
        # Drain strictly in submission order; rows are byte-identical to the
        # serial path, so evaluator.results keeps its historical order.
        ordered = export_pipeline.drain()
        if len(ordered) != num_images_evaluated:
            raise RuntimeError(
                "Export pipeline drained an unexpected number of images: "
                f"expected={num_images_evaluated}, got={len(ordered)}"
            )
        for image_id, rows in ordered:
            evaluator.update(rows, image_ids=[image_id])
        stats = export_pipeline.stats
        print(f"[Eval] Export pipeline stats: transfer_wait={stats['transfer_wait_s']:.2f}s "
              f"copy_out={stats['copy_out_s']:.2f}s submit={stats['submit_s']:.2f}s")
        export_pipeline.close()

    inference_time = time.time() - eval_start
    print(f"[Eval] Inference done: {num_images_evaluated} images in {inference_time:.1f}s "
          f"({num_images_evaluated/inference_time:.1f} img/s)")

    coco_metrics = evaluator.summarize()
    coco_results_path = evaluator.dump(output_dir / "coco_instances_results.json")
    coco_metrics_path = output_dir / "coco_metrics.json"
    coco_metrics_path.write_text(
        json.dumps(
            {
                "metric_scale": "fraction",
                "num_images": num_images_evaluated,
                "num_predictions": len(evaluator.results),
                "metrics": coco_metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[Eval] Results saved to {coco_results_path}")
    print(f"[Eval] Metrics saved to {coco_metrics_path}")
    print(coco_metrics)


if __name__ == "__main__":
    main()
