#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def _prepend_syspath(paths: Sequence[Path]):
    original = list(sys.path)
    for path in reversed([str(p) for p in paths]):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = original


def _sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _device_string_for_yolo(device: torch.device) -> str | int:
    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        return int(index)
    return "cpu"


def _resolve_image_path(dataset_root: Path, split: str, file_name: str) -> Path:
    candidates = [
        dataset_root / "images" / split / file_name,
        dataset_root / "images" / file_name,
        dataset_root / split / file_name,
        dataset_root / file_name,
        dataset_root / "images" / split / Path(file_name).name,
        dataset_root / "images" / Path(file_name).name,
        dataset_root / split / Path(file_name).name,
        dataset_root / Path(file_name).name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not resolve image path for {file_name} under {dataset_root}")


def _collect_val_image_paths(dataset_root: Path, ann_rel: str, limit: int) -> List[Path]:
    ann_path = dataset_root / ann_rel
    payload = _load_json(ann_path)
    image_rows = payload.get("images", [])
    out: List[Path] = []
    for row in image_rows[:limit]:
        out.append(_resolve_image_path(dataset_root, "val", str(row["file_name"])))
    return out


def _detect_model_id(out_dir: Path) -> str:
    metadata_path = out_dir / "metadata.json"
    if metadata_path.exists():
        return str(_load_json(metadata_path).get("model_id", out_dir.name))
    return out_dir.name


def _detect_register_id(out_dir: Path, dataset_root: Path) -> str:
    metadata_path = out_dir / "metadata.json"
    if metadata_path.exists():
        payload = _load_json(metadata_path)
        register = payload.get("register")
        if register:
            return str(register)
    return dataset_root.name


def _detect_family(out_dir: Path, model_id: str) -> str:
    model_key = model_id.lower()
    if (out_dir / "metrics_log.jsonl").exists():
        return "magformer"
    if (out_dir / "train" / "results.csv").exists():
        return "yolo"
    if model_key.startswith("mgm_mask2former"):
        return "mgm_mask2former"
    if model_key.startswith("official_mask2former"):
        return "official_mask2former"
    if model_key.startswith("maskrcnn"):
        return "detectron2"
    if model_key.startswith("uoais"):
        return "uoais"
    if model_key.startswith("msmformer"):
        return "msmformer"
    if model_key.startswith("ucn"):
        return "ucn"
    if model_key.startswith("cellpose"):
        return "cellpose"
    if model_key.startswith("stardist"):
        return "stardist"
    if model_key.startswith("iaunet"):
        return "iaunet"
    if model_key.startswith("unet"):
        return "unet"
    raise ValueError(f"Could not infer benchmark family for {model_id} under {out_dir}")


def _infer_image_size(metadata: Dict[str, Any], out_dir: Path, default: int = 1024) -> int:
    image_size = metadata.get("image_size")
    try:
        if image_size is not None:
            return int(image_size)
    except Exception:
        pass

    command = str(metadata.get("command", ""))
    match = re.search(r"--image-size(?:=|\s+)(\d+)", command)
    if match:
        return int(match.group(1))

    match = re.search(r"(?<!\d)(512|1024)(?!\d)", str(out_dir))
    if match:
        return int(match.group(1))

    return int(default)


def _find_magformer_ckpt(out_dir: Path) -> Path:
    if (out_dir / "model_best.pth").exists():
        return out_dir / "model_best.pth"
    ckpts = sorted(out_dir.glob("checkpoint_iter_*.pth"))
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError(f"No MagFormer checkpoint found under {out_dir}")


def _find_detectron2_ckpt(out_dir: Path) -> Path:
    if (out_dir / "model_final.pth").exists():
        return out_dir / "model_final.pth"
    last_checkpoint = out_dir / "last_checkpoint"
    if last_checkpoint.exists():
        ckpt_name = last_checkpoint.read_text(encoding="utf-8").strip()
        if ckpt_name:
            ckpt_path = out_dir / ckpt_name
            if ckpt_path.exists():
                return ckpt_path
    ckpts = sorted(out_dir.glob("model_*.pth"))
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError(f"No Detectron2 checkpoint found under {out_dir}")


def _find_unet_ckpt(out_dir: Path) -> Path:
    if (out_dir / "model_final.pth").exists():
        return out_dir / "model_final.pth"
    ckpts = sorted(out_dir.glob("model_*.pth"))
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError(f"No U-Net checkpoint found under {out_dir}")


def _find_yolo_weights(out_dir: Path) -> Path:
    weights = out_dir / "train" / "weights"
    for name in ["best.pt", "last.pt"]:
        candidate = weights / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No YOLO weights found under {weights}")


def _iter_with_limit(iterable: Iterable[Any], limit: int) -> List[Any]:
    items: List[Any] = []
    for item in iterable:
        items.append(item)
        if len(items) >= limit:
            break
    return items


def _collect_items_with_load_timing(
    iterable: Iterable[Any],
    *,
    warmup: int,
    timed_images: int,
) -> tuple[List[Any], Dict[str, Any]]:
    total_needed = warmup + timed_images
    items: List[Any] = []
    load_latencies_ms: List[float] = []
    iterator = iter(iterable)
    for idx in range(total_needed):
        start = time.perf_counter()
        try:
            item = next(iterator)
        except StopIteration:
            break
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        items.append(item)
        if idx >= warmup:
            load_latencies_ms.append(elapsed_ms)

    if not load_latencies_ms:
        return items, {}

    load_lat = np.asarray(load_latencies_ms, dtype=np.float64)
    return items, {
        "data_load_latency_ms_mean": float(load_lat.mean()),
        "data_load_latency_ms_p50": float(np.percentile(load_lat, 50)),
        "data_load_latency_ms_p90": float(np.percentile(load_lat, 90)),
    }


def _measure_latency(
    *,
    items: Sequence[Any],
    infer_fn: Callable[[Any], Any],
    device: torch.device,
    warmup: int,
    timed_images: int,
) -> Dict[str, Any]:
    total_needed = min(len(items), warmup + timed_images)
    if total_needed <= warmup:
        raise ValueError(
            f"Need more than warmup items to benchmark, got items={len(items)} warmup={warmup}"
        )

    latencies_ms: List[float] = []
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for idx, item in enumerate(items[:total_needed]):
        _sync(device)
        start = time.perf_counter()
        infer_fn(item)
        _sync(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if idx >= warmup:
            latencies_ms.append(elapsed_ms)

    lat = np.asarray(latencies_ms, dtype=np.float64)
    total_sec = float(lat.sum() / 1000.0)
    peak_memory_mb = None
    if device.type == "cuda" and torch.cuda.is_available():
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    return {
        "status": "ok",
        "warmup_images": warmup,
        "timed_images": int(lat.size),
        "latency_ms_mean": float(lat.mean()),
        "latency_ms_p50": float(np.percentile(lat, 50)),
        "latency_ms_p90": float(np.percentile(lat, 90)),
        "latency_ms_min": float(lat.min()),
        "latency_ms_max": float(lat.max()),
        "throughput_fps": float(lat.size / total_sec) if total_sec > 0 else None,
        "inference_peak_memory_mb": peak_memory_mb,
    }


def _measure_latency_phased(
    *,
    items: Sequence[Any],
    infer_fn: Callable[[Any], Any],
    postprocess_fn: Callable[[Any], Any],
    export_fn: Callable[[Any], Any],
    device: torch.device,
    warmup: int,
    timed_images: int,
) -> Dict[str, Any]:
    total_needed = min(len(items), warmup + timed_images)
    if total_needed <= warmup:
        raise ValueError(
            f"Need more than warmup items to benchmark, got items={len(items)} warmup={warmup}"
        )

    forward_latencies_ms: List[float] = []
    post_latencies_ms: List[float] = []
    export_latencies_ms: List[float] = []
    total_latencies_ms: List[float] = []
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for idx, item in enumerate(items[:total_needed]):
        _sync(device)
        total_start = time.perf_counter()

        forward_start = time.perf_counter()
        inferred = infer_fn(item)
        _sync(device)
        forward_elapsed_ms = (time.perf_counter() - forward_start) * 1000.0

        post_start = time.perf_counter()
        postprocessed = postprocess_fn(inferred)
        _sync(device)
        post_elapsed_ms = (time.perf_counter() - post_start) * 1000.0

        export_start = time.perf_counter()
        export_fn(postprocessed)
        _sync(device)
        export_elapsed_ms = (time.perf_counter() - export_start) * 1000.0
        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0

        if idx >= warmup:
            forward_latencies_ms.append(forward_elapsed_ms)
            post_latencies_ms.append(post_elapsed_ms)
            export_latencies_ms.append(export_elapsed_ms)
            total_latencies_ms.append(total_elapsed_ms)

    total_lat = np.asarray(total_latencies_ms, dtype=np.float64)
    forward_lat = np.asarray(forward_latencies_ms, dtype=np.float64)
    post_lat = np.asarray(post_latencies_ms, dtype=np.float64)
    export_lat = np.asarray(export_latencies_ms, dtype=np.float64)
    total_sec = float(total_lat.sum() / 1000.0)
    peak_memory_mb = None
    if device.type == "cuda" and torch.cuda.is_available():
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))

    return {
        "status": "ok",
        "warmup_images": warmup,
        "timed_images": int(total_lat.size),
        "latency_ms_mean": float(total_lat.mean()),
        "latency_ms_p50": float(np.percentile(total_lat, 50)),
        "latency_ms_p90": float(np.percentile(total_lat, 90)),
        "latency_ms_min": float(total_lat.min()),
        "latency_ms_max": float(total_lat.max()),
        "throughput_fps": float(total_lat.size / total_sec) if total_sec > 0 else None,
        "inference_peak_memory_mb": peak_memory_mb,
        "model_forward_latency_ms_mean": float(forward_lat.mean()),
        "postprocess_latency_ms_mean": float(post_lat.mean()),
        "export_latency_ms_mean": float(export_lat.mean()),
    }


def _benchmark_magformer(
    out_dir: Path,
    dataset_root: Path,
    device: torch.device,
    warmup: int,
    timed_images: int,
) -> Dict[str, Any]:
    from magformer.config import load_config, setup_device
    from magformer.models import build_model
    from magformer.engine.utils import load_checkpoint
    from tools.evaluate import build_val_loader

    config_path = out_dir / "magformer_runtime_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing runtime config: {config_path}")
    weights = _find_magformer_ckpt(out_dir)
    config = load_config(
        str(config_path),
        overrides={
            "runtime": {"device": str(device)},
            "data": {"dataset_root": str(dataset_root)},
        },
    )
    config.runtime.device = str(device)
    _, loader = build_val_loader(config, dataset_root_override=str(dataset_root), num_workers=0, batch_size=1)
    items, load_stats = _collect_items_with_load_timing(
        loader,
        warmup=warmup,
        timed_images=timed_images,
    )

    model = build_model(config)
    load_checkpoint(str(weights), model, strict=False)
    model = model.to(setup_device(config.runtime))
    model.eval()

    from magformer.models.ops.functions import ms_deform_attn_func

    getattr(ms_deform_attn_func, "reset_ms_deform_attn_runtime_fallback_error", lambda: None)()

    def infer_fn(batch: Dict[str, torch.Tensor]) -> Any:
        with torch.no_grad():
            outputs = model.forward_inference_decoder_outputs(
                batch["images"].to(device),
                batch["depths"].to(device),
            )
        return {
            "decoder_outputs": outputs,
            "image_shape": tuple(batch["images"].shape),
        }

    def postprocess_fn(payload: Dict[str, Any]) -> Any:
        return model._inference_raw(payload["decoder_outputs"], payload["image_shape"])

    def export_fn(raw_outputs: Dict[str, Any]) -> Any:
        return model._export_inference_predictions(
            raw_outputs,
            include_raw_tensors=False,
        )

    result = _measure_latency_phased(
        items=items,
        infer_fn=infer_fn,
        postprocess_fn=postprocess_fn,
        export_fn=export_fn,
        device=device,
        warmup=warmup,
        timed_images=timed_images,
    )
    result["amp_enabled"] = False
    result["msdeformattn_cuda_available"] = bool(
        getattr(ms_deform_attn_func, "ms_deform_attn_cuda_available", lambda: False)()
    )
    result["msdeformattn_import_error"] = str(
        getattr(ms_deform_attn_func, "ms_deform_attn_import_error", lambda: "")()
    )
    result["msdeformattn_runtime_fallback_error"] = str(
        getattr(ms_deform_attn_func, "ms_deform_attn_last_runtime_fallback_error", lambda: "")()
    )
    result["msdeformattn_runtime_fallback_count"] = int(
        getattr(ms_deform_attn_func, "ms_deform_attn_runtime_fallback_count", lambda: 0)()
    )
    result.update(load_stats)
    result.update({"framework": "magformer", "weights": str(weights), "config": str(config_path)})
    return result


def _benchmark_detectron2_like(
    *,
    out_dir: Path,
    dataset_root: Path,
    device: torch.device,
    warmup: int,
    timed_images: int,
    family: str,
) -> Dict[str, Any]:
    from detectron2.checkpoint import DetectionCheckpointer

    config_path = out_dir / "config.yaml"
    weights = _find_detectron2_ckpt(out_dir)
    register_id = _detect_register_id(out_dir, dataset_root)
    family_paths: List[Path] = [REPO_ROOT / "baselines", REPO_ROOT / "baselines" / "detectron2"]
    if family == "mgm_mask2former":
        family_paths.extend(
            [
                REPO_ROOT / "baselines" / "MGM_Mask2Former",
                REPO_ROOT / "baselines" / "Mask2Former",
            ]
        )
    elif family == "official_mask2former":
        family_paths.extend(
            [
                REPO_ROOT / "baselines" / "Mask2Former",
                REPO_ROOT / "baselines" / "MGM_Mask2Former",
            ]
        )
    else:
        family_paths.extend(
            [
                REPO_ROOT / "baselines" / "Mask2Former",
                REPO_ROOT / "baselines" / "MGM_Mask2Former",
            ]
        )
    family_paths.extend(
        [
            REPO_ROOT / "baselines" / "uoais",
            REPO_ROOT / "baselines" / "msmformer",
        ]
    )
    if family == "msmformer":
        family_paths.extend(
            [
                REPO_ROOT / "baselines" / "msmformer" / "MSMFormer",
                REPO_ROOT / "baselines" / "msmformer" / "tools",
            ]
        )

    with _prepend_syspath(family_paths):
        from baselines.ecc_datasets import register_ecc_coco, register_ecc_coco_rgbd

        if family == "official_mask2former":
            register_ecc_coco(register_id, str(dataset_root))
            module = _load_module(
                "official_mask2former_train",
                REPO_ROOT / "baselines" / "Mask2Former" / "train_net.py",
            )
        elif family == "mgm_mask2former":
            module = _load_module(
                "mgm_mask2former_train",
                REPO_ROOT / "baselines" / "MGM_Mask2Former" / "train_net_mgm_0831.py",
            )
        elif family == "detectron2":
            register_ecc_coco(register_id, str(dataset_root))
            module = _load_module(
                "detectron2_train",
                REPO_ROOT / "baselines" / "detectron2" / "tools" / "train_net.py",
            )
        elif family == "uoais":
            register_ecc_coco_rgbd(register_id, str(dataset_root))
            module = _load_module(
                "uoais_train",
                REPO_ROOT / "baselines" / "uoais" / "train_net.py",
            )
        elif family == "msmformer":
            module = _load_module(
                "msmformer_wrapper",
                REPO_ROOT / "baselines" / "run_msmformer_ecc.py",
            )
            register_ecc_coco_rgbd(register_id, str(dataset_root))
        else:
            raise ValueError(f"Unsupported detectron2-like family: {family}")

        args = SimpleNamespace(
            config_file=str(config_path),
            opts=["MODEL.WEIGHTS", str(weights), "OUTPUT_DIR", str(out_dir)],
            eval_only=True,
            resume=False,
            num_gpus=1,
            num_machines=1,
            machine_rank=0,
            dist_url="auto",
        )
        cfg = module.setup(args)
        cfg.defrost()
        cfg.MODEL.WEIGHTS = str(weights)
        cfg.MODEL.DEVICE = str(device)
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.freeze()
        if family == "mgm_mask2former":
            module.register_mgm_datasets(cfg)
        model = module.Trainer.build_model(cfg)
        checkpointer_cls = getattr(module, "AdetCheckpointer", None) or getattr(module, "DetectionCheckpointer", None) or DetectionCheckpointer
        checkpointer_cls(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
        loader = module.Trainer.build_test_loader(cfg, cfg.DATASETS.TEST[0])
        items = _iter_with_limit(loader, warmup + timed_images)
        model.eval()

        def infer_fn(batch: Any) -> Any:
            with torch.no_grad():
                return model(batch)

        result = _measure_latency(items=items, infer_fn=infer_fn, device=device, warmup=warmup, timed_images=timed_images)
    result.update(
        {
            "framework": family,
            "weights": str(weights),
            "config": str(config_path),
            "register": register_id,
        }
    )
    return result


def _benchmark_yolo(
    out_dir: Path,
    dataset_root: Path,
    device: torch.device,
    warmup: int,
    timed_images: int,
) -> Dict[str, Any]:
    from ultralytics import YOLO
    from baselines.normalization_stats import load_dataset_normalization_stats
    from baselines.yolo_stats_norm import StatsNormalizedSegmentationPredictor

    weights = _find_yolo_weights(out_dir)
    image_paths = _collect_val_image_paths(dataset_root, "annotations/instances_val.json", warmup + timed_images)
    images = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        images.append(image)
    model = YOLO(str(weights))
    yolo_device = _device_string_for_yolo(device)
    stats = load_dataset_normalization_stats(str(dataset_root))
    infer_fn = _make_yolo_infer_fn(
        model=model,
        predictor_cls=StatsNormalizedSegmentationPredictor,
        yolo_device=yolo_device,
        imgsz=1024,
        rgb_mean=list(stats.rgb_mean_rgb_255),
        rgb_std=list(stats.rgb_std_rgb_255),
    )

    result = _measure_latency(items=images, infer_fn=infer_fn, device=device, warmup=warmup, timed_images=timed_images)
    result.update({"framework": "yolo", "weights": str(weights), "config": None})
    return result


def _make_yolo_infer_fn(
    *,
    model: Any,
    predictor_cls: Any,
    yolo_device: str | int,
    imgsz: int,
    rgb_mean: Sequence[float],
    rgb_std: Sequence[float],
) -> Callable[[np.ndarray], Any]:
    overrides = {
        **getattr(model, "overrides", {}),
        "conf": 0.25,
        "batch": 1,
        "save": False,
        "mode": "predict",
        "rect": True,
        "device": yolo_device,
        "imgsz": int(imgsz),
        "verbose": False,
        "rgb_mean": [float(v) for v in rgb_mean],
        "rgb_std": [float(v) for v in rgb_std],
    }
    model.predictor = predictor_cls(overrides=overrides, _callbacks=getattr(model, "callbacks", None))
    model.predictor.setup_model(model=getattr(model, "model", None), verbose=False)

    def infer_fn(image: np.ndarray) -> Any:
        return model.predict(
            source=image,
            device=yolo_device,
            imgsz=int(imgsz),
            verbose=False,
            stream=False,
        )

    return infer_fn


def _benchmark_unet(
    out_dir: Path,
    dataset_root: Path,
    device: torch.device,
    warmup: int,
    timed_images: int,
    model_id: str,
) -> Dict[str, Any]:
    module = _load_module("unet_runner", REPO_ROOT / "baselines" / "run_unet_instance_ecc.py")
    weights = _find_unet_ckpt(out_dir)
    variant = model_id
    metadata = _read_metadata(out_dir)
    image_size = _infer_image_size(metadata, out_dir)
    reference_root = ""
    if metadata.get("model_id"):
        variant = str(metadata["model_id"])
    use_cuda = device.type == "cuda"
    module._configure_process_threads(module.recommend_main_process_threads(os.cpu_count(), 0))
    dataset = module.ECCUnetDataset(str(dataset_root), "val", image_size, False, variant)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        **module.build_loader_kwargs(0, use_cuda=use_cuda),
    )
    items = _iter_with_limit(loader, warmup + timed_images)
    model = module.build_instance_model(variant, in_channels=3, base_channels=16).to(device)
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    reference_cache = None
    if "reference" in variant and reference_root:
        reference_bank = module.load_reference_bank(reference_root, image_size=image_size)
        reference_cache = model.build_reference_cache(reference_bank, device)  # type: ignore[attr-defined]

    def infer_fn(batch: Dict[str, Any]) -> Any:
        images = batch["images"].to(device)
        depths = batch.get("depths")
        if depths is not None:
            depths = depths.to(device)
        with torch.no_grad():
            if "reference" in variant:
                outputs = model(images, query_depth=depths, reference_cache=reference_cache)
            else:
                outputs = model(images)
        return outputs

    result = _measure_latency(items=items, infer_fn=infer_fn, device=device, warmup=warmup, timed_images=timed_images)
    result.update({"framework": "unet", "weights": str(weights), "config": None, "image_size": image_size})
    return result


def _benchmark_cellpose(
    out_dir: Path,
    dataset_root: Path,
    device: torch.device,
    warmup: int,
    timed_images: int,
) -> Dict[str, Any]:
    module = _load_module("cellpose_model_runner", REPO_ROOT / "baselines" / "cellpose_instance_models.py")
    metadata = _read_metadata(out_dir)
    image_size = _infer_image_size(metadata, out_dir)
    weights = _find_unet_ckpt(out_dir)
    dataset = module.ECCCellPoseDataset(dataset_root, "val", image_size=image_size, train=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=module._collate,
        pin_memory=device.type == "cuda",
    )
    items = _iter_with_limit(loader, warmup + timed_images)
    model = module._load_model_from_checkpoint(weights, device=device)

    def infer_fn(batch: Dict[str, Any]) -> Any:
        images = [np.asarray(image, dtype=np.float32).transpose(2, 0, 1) for image in batch["image"]]
        eval_output = model.eval(
            images,
            batch_size=len(images),
            channel_axis=0,
            normalize=True,
            compute_masks=True,
            resample=True,
            flow_threshold=0.4,
            cellprob_threshold=0.0,
            min_size=20,
        )
        return {
            "image_id": int(batch["image_id"][0]),
            "eval_output": eval_output,
        }

    def postprocess_fn(payload: Dict[str, Any]) -> Any:
        label_maps, flows = module._normalize_eval_output(payload["eval_output"])
        cellprob = module._extract_cellprob(flows[0]) if flows else None
        masks, scores, category_ids = module.label_map_to_instance_predictions(
            label_maps[0],
            cellprob=cellprob,
            min_area=20,
        )
        return {
            **payload,
            "masks": masks,
            "scores": scores,
            "category_ids": category_ids,
        }

    def export_fn(payload: Dict[str, Any]) -> Any:
        return module.binary_masks_to_coco_rows(
            image_id=payload["image_id"],
            masks=payload["masks"],
            scores=payload["scores"],
            category_ids=payload["category_ids"],
            score_threshold=0.05,
            mask_threshold=0.5,
        )

    result = _measure_latency_phased(
        items=items,
        infer_fn=infer_fn,
        postprocess_fn=postprocess_fn,
        export_fn=export_fn,
        device=device,
        warmup=warmup,
        timed_images=timed_images,
    )
    result.update({"framework": "cellpose", "weights": str(weights), "config": None, "image_size": image_size})
    return result


def _benchmark_stardist(
    out_dir: Path,
    dataset_root: Path,
    device: torch.device,
    warmup: int,
    timed_images: int,
) -> Dict[str, Any]:
    runner = _load_module("stardist_runner_benchmark", REPO_ROOT / "baselines" / "run_stardist_instance_ecc.py")
    module = _load_module("stardist_utils_benchmark", REPO_ROOT / "baselines" / "stardist_instance_utils.py")
    metadata = _read_metadata(out_dir)
    image_size = _infer_image_size(metadata, out_dir)
    prob_thresh = float(metadata.get("prob_thresh", 0.5))
    nms_thresh = float(metadata.get("nms_thresh", 0.3))
    weights = out_dir / "stardist_model" / "model_final.weights.h5"
    if not weights.exists():
        weights = out_dir / "model_final.weights.h5"
    backend = runner._load_stardist_backend()
    model_name = str(metadata.get("model_id", _detect_model_id(out_dir)))
    model = runner._build_model(
        backend=backend,
        output_dir=out_dir,
        image_size=image_size,
        batch_size=1,
        model_name=model_name,
    )
    load_weights = getattr(model, "load_weights", None)
    if callable(load_weights):
        load_weights(str(weights))
    elif getattr(model, "keras_model", None) is not None and hasattr(model.keras_model, "load_weights"):
        model.keras_model.load_weights(str(weights))

    images, _labels, records = module.load_stardist_ecc_split(dataset_root, "val", image_size)
    items = _iter_with_limit(list(zip(records, images)), warmup + timed_images)

    def infer_fn(item: Any) -> Any:
        record, image = item
        labels, details = model.predict_instances(
            image,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
        )
        return {
            "image_id": int(record["image_id"]),
            "labels": np.asarray(labels),
            "details": details,
        }

    def postprocess_fn(payload: Dict[str, Any]) -> Any:
        return module.stardist_prediction_to_coco_rows(
            image_id=payload["image_id"],
            labels=np.asarray(payload["labels"]),
            details=payload["details"],
            score_threshold=prob_thresh,
        )

    def export_fn(rows: Any) -> Any:
        return json.dumps(rows, ensure_ascii=False)

    result = _measure_latency_phased(
        items=items,
        infer_fn=infer_fn,
        postprocess_fn=postprocess_fn,
        export_fn=export_fn,
        device=device,
        warmup=warmup,
        timed_images=timed_images,
    )
    result.update({"framework": "stardist", "weights": str(weights), "config": None, "image_size": image_size})
    return result


def _benchmark_iaunet(
    out_dir: Path,
    dataset_root: Path,
    device: torch.device,
    warmup: int,
    timed_images: int,
) -> Dict[str, Any]:
    runner = _load_module("iaunet_runner_benchmark", REPO_ROOT / "baselines" / "run_iaunet_instance_ecc.py")
    module = _load_module("iaunet_model_benchmark", REPO_ROOT / "baselines" / "iaunet_instance_models.py")
    metadata = _read_metadata(out_dir)
    image_size = _infer_image_size(metadata, out_dir)
    weights = _find_unet_ckpt(out_dir)
    model = module.IAUNetInstanceModel(
        in_channels=3,
        base_channels=int(metadata.get("base_channels", 32)),
        hidden_dim=int(metadata.get("hidden_dim", 256)),
        num_queries=int(metadata.get("num_queries", 100)),
        num_decoder_layers=int(metadata.get("num_decoder_layers", 4)),
        transformer_blocks_per_stage=int(metadata.get("transformer_blocks_per_stage", 3)),
        num_heads=int(metadata.get("num_heads", 8)),
    ).to(device)
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    dataset = runner.ECCIAUNetDataset(str(dataset_root), "val", image_size, train=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=runner._collate,
        pin_memory=device.type == "cuda",
    )
    items = _iter_with_limit(loader, warmup + timed_images)
    score_threshold = float(metadata.get("score_threshold", 0.4))
    mask_threshold = float(metadata.get("mask_threshold", 0.5))
    min_area = int(metadata.get("min_area", 20))

    def infer_fn(batch: Dict[str, Any]) -> Any:
        with torch.no_grad():
            outputs = model(batch["images"].to(device))
        return {
            "image_id": int(batch["image_ids"][0]),
            "orig_size": tuple(batch["orig_sizes"][0]),
            "outputs": outputs,
        }

    def postprocess_fn(payload: Dict[str, Any]) -> Any:
        predictions = module.iaunet_inference(
            payload["outputs"],
            original_sizes=[payload["orig_size"]],
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
            min_area=min_area,
        )
        return {
            "image_id": payload["image_id"],
            "prediction": predictions[0],
        }

    def export_fn(payload: Dict[str, Any]) -> Any:
        prediction = payload["prediction"]
        return runner.binary_masks_to_coco_rows(
            image_id=payload["image_id"],
            masks=prediction["masks"].numpy(),
            scores=prediction["scores"].numpy(),
            category_ids=prediction["category_ids"].numpy(),
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
        )

    result = _measure_latency_phased(
        items=items,
        infer_fn=infer_fn,
        postprocess_fn=postprocess_fn,
        export_fn=export_fn,
        device=device,
        warmup=warmup,
        timed_images=timed_images,
    )
    result.update({"framework": "iaunet", "weights": str(weights), "config": None, "image_size": image_size})
    return result


def _benchmark_ucn(
    out_dir: Path,
    dataset_root: Path,
    device: torch.device,
    warmup: int,
    timed_images: int,
) -> Dict[str, Any]:
    def _ucn_run_log_fallback() -> Dict[str, Any] | None:
        run_log = out_dir / "run.log"
        if not run_log.exists():
            return None
        pattern = re.compile(r"\[ucn-eval\]\s+(\d+)/(\d+)\s+images,\s+elapsed=([0-9.]+)s")
        last_match = None
        for line in run_log.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = pattern.search(line)
            if match:
                last_match = match
        if last_match is None:
            return None
        processed = int(last_match.group(1))
        total = int(last_match.group(2))
        elapsed_sec = float(last_match.group(3))
        images = min(processed, total)
        if images <= 0 or elapsed_sec <= 0:
            return None
        return {
            "status": "ok",
            "source": "ucn_eval_from_log",
            "warmup_images": None,
            "timed_images": images,
            "latency_ms_mean": float((elapsed_sec / images) * 1000.0),
            "latency_ms_p50": None,
            "latency_ms_p90": None,
            "latency_ms_min": None,
            "latency_ms_max": None,
            "throughput_fps": float(images / elapsed_sec),
            "inference_peak_memory_mb": None,
            "framework": "ucn",
            "weights": None,
            "config": None,
        }

    ann_path = dataset_root / "annotations" / "instances_val.json"
    fallback_result = _ucn_run_log_fallback()
    if not ann_path.exists() and fallback_result is not None:
        return fallback_result
    baselines_dir = REPO_ROOT / "baselines"
    ucn_repo = baselines_dir / "unseen_object_clustering"
    try:
        with _prepend_syspath([baselines_dir, ucn_repo, ucn_repo / "lib"]):
            module = _load_module("ucn_runner", baselines_dir / "run_ucn_0831_1k.py")
            from fcn.config import cfg  # type: ignore
            import networks  # type: ignore
            from utils.mean_shift import mean_shift_smart_init  # type: ignore

            dataset = module.ECC0831UCNDataset(
                dataset_root=str(dataset_root),
                split="val",
                img_size=1024,
                train=False,
                pixel_mean_bgr_255=[28.1363, 30.5413, 34.9731],
            )
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
            items = _iter_with_limit(loader, warmup + timed_images)
            cfg.TRAIN.NUM_UNITS = 64
            network = networks.seg_resnet34_8s_embedding(num_classes=2, num_units=cfg.TRAIN.NUM_UNITS, data=None).to(device)
            network.eval()

            def infer_fn(batch: Dict[str, Any]) -> Any:
                image = batch["image_color"].to(device)
                depth = batch["depth"].to(device)
                label = batch["label"].to(device)
                with torch.no_grad():
                    feat = network(image, label, depth)
                    feat_ds = F.interpolate(feat, size=(128, 128), mode="bilinear", align_corners=False)
                    embeddings = feat_ds[0].permute(1, 2, 0).reshape(-1, feat_ds.shape[1])
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    cluster_labels, _ = mean_shift_smart_init(
                        embeddings, kappa=20.0, num_seeds=20, max_iters=10, metric="cosine"
                    )
                    cluster_map = cluster_labels.view(128, 128).cpu().numpy().astype(np.int32)
                    orig_h, orig_w = int(batch["orig_size"][0][0].item()), int(batch["orig_size"][1][0].item())
                    cluster_map = cv2.resize(cluster_map, (dataset.img_size, dataset.img_size), interpolation=cv2.INTER_NEAREST)
                    cluster_map = cv2.resize(cluster_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    module._cluster_to_instances(cluster_map, int(batch["image_id"].item()), min_area=50, max_instances=50)

            result = _measure_latency(items=items, infer_fn=infer_fn, device=device, warmup=warmup, timed_images=timed_images)
            result.update({"framework": "ucn", "weights": None, "config": None, "source": "ucn_forward_clustering"})
            return result
    except Exception:
        if fallback_result is not None:
            return fallback_result
        raise


def benchmark_output_dir(
    *,
    out_dir: Path,
    dataset_root: Path,
    device: torch.device,
    warmup: int,
    timed_images: int,
) -> Dict[str, Any]:
    model_id = _detect_model_id(out_dir)
    family = _detect_family(out_dir, model_id)
    if family == "magformer":
        result = _benchmark_magformer(out_dir, dataset_root, device, warmup, timed_images)
    elif family in {"official_mask2former", "mgm_mask2former", "detectron2", "uoais", "msmformer"}:
        result = _benchmark_detectron2_like(
            out_dir=out_dir,
            dataset_root=dataset_root,
            device=device,
            warmup=warmup,
            timed_images=timed_images,
            family=family,
        )
    elif family == "yolo":
        result = _benchmark_yolo(out_dir, dataset_root, device, warmup, timed_images)
    elif family == "cellpose":
        result = _benchmark_cellpose(out_dir, dataset_root, device, warmup, timed_images)
    elif family == "stardist":
        result = _benchmark_stardist(out_dir, dataset_root, device, warmup, timed_images)
    elif family == "iaunet":
        result = _benchmark_iaunet(out_dir, dataset_root, device, warmup, timed_images)
    elif family == "unet":
        result = _benchmark_unet(out_dir, dataset_root, device, warmup, timed_images, model_id)
    elif family == "ucn":
        result = _benchmark_ucn(out_dir, dataset_root, device, warmup, timed_images)
    else:
        raise ValueError(f"Unsupported family: {family}")
    result.update({"model_id": model_id, "family": family})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark inference latency for experiment outputs")
    parser.add_argument("--out-dir", required=True, help="Experiment output directory")
    parser.add_argument("--dataset-root", required=True, help="Dataset root")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda or cpu")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup images")
    parser.add_argument("--timed-images", type=int, default=50, help="Timed images")
    parser.add_argument("--output-name", default="inference_speed.json", help="Output json filename")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    result = benchmark_output_dir(
        out_dir=out_dir,
        dataset_root=dataset_root,
        device=device,
        warmup=int(args.warmup),
        timed_images=int(args.timed_images),
    )
    output_path = out_dir / str(args.output_name)
    _write_json(output_path, result)
    print(f"[benchmark] wrote: {output_path}")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
