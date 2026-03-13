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
    if model_key.startswith("unet"):
        return "unet"
    raise ValueError(f"Could not infer benchmark family for {model_id} under {out_dir}")


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
    items = _iter_with_limit(loader, warmup + timed_images)

    model = build_model(config)
    load_checkpoint(str(weights), model, strict=False)
    model = model.to(setup_device(config.runtime))
    model.eval()

    def infer_fn(batch: Dict[str, torch.Tensor]) -> Any:
        with torch.no_grad():
            return model(batch["images"].to(device), batch["depths"].to(device))

    result = _measure_latency(items=items, infer_fn=infer_fn, device=device, warmup=warmup, timed_images=timed_images)
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

    with _prepend_syspath(family_paths):
        if family == "official_mask2former":
            register_mod = _load_module(
                "register_0831_coco",
                REPO_ROOT / "baselines" / "register_0831_1k_coco.py",
            )
            register_mod.register_0831_1k_coco(str(dataset_root))
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
            register_mod = _load_module(
                "register_0831_coco",
                REPO_ROOT / "baselines" / "register_0831_1k_coco.py",
            )
            register_mod.register_0831_1k_coco(str(dataset_root))
            module = _load_module(
                "detectron2_train",
                REPO_ROOT / "baselines" / "detectron2" / "tools" / "train_net.py",
            )
        elif family == "uoais":
            register_mod = _load_module(
                "register_0831_coco_rgbd",
                REPO_ROOT / "baselines" / "register_0831_1k_coco_rgbd.py",
            )
            register_mod.register_0831_1k_coco_rgbd(str(dataset_root))
            module = _load_module(
                "uoais_train",
                REPO_ROOT / "baselines" / "uoais" / "train_net.py",
            )
        elif family == "msmformer":
            module = _load_module(
                "msmformer_wrapper",
                REPO_ROOT / "baselines" / "run_msmformer_0831_1k.py",
            )
            register_mod = _load_module(
                "register_0831_coco_rgbd",
                REPO_ROOT / "baselines" / "register_0831_1k_coco_rgbd.py",
            )
            register_mod.register_0831_1k_coco_rgbd(str(dataset_root))
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
    result.update({"framework": family, "weights": str(weights), "config": str(config_path)})
    return result


def _benchmark_yolo(
    out_dir: Path,
    dataset_root: Path,
    device: torch.device,
    warmup: int,
    timed_images: int,
) -> Dict[str, Any]:
    from ultralytics import YOLO

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

    def infer_fn(image: np.ndarray) -> Any:
        return model.predict(
            source=image,
            device=yolo_device,
            imgsz=1024,
            verbose=False,
            stream=False,
        )

    result = _measure_latency(items=images, infer_fn=infer_fn, device=device, warmup=warmup, timed_images=timed_images)
    result.update({"framework": "yolo", "weights": str(weights), "config": None})
    return result


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
    reference_root = ""
    if (out_dir / "metadata.json").exists():
        meta = _load_json(out_dir / "metadata.json")
        if meta.get("model_id"):
            variant = str(meta["model_id"])
    use_cuda = device.type == "cuda"
    module._configure_process_threads(module.recommend_main_process_threads(os.cpu_count(), 0))
    dataset = module.ECCUnetDataset(str(dataset_root), "val", 1024, False, variant)
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
        reference_bank = module.load_reference_bank(reference_root, image_size=1024)
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
    result.update({"framework": "unet", "weights": str(weights), "config": None})
    return result


def _benchmark_ucn(
    out_dir: Path,
    dataset_root: Path,
    device: torch.device,
    warmup: int,
    timed_images: int,
) -> Dict[str, Any]:
    ann_path = dataset_root / "annotations" / "instances_val.json"
    if not ann_path.exists():
        run_log = out_dir / "run.log"
        if run_log.exists():
            pattern = re.compile(r"\[ucn-eval\]\s+(\d+)/(\d+)\s+images,\s+elapsed=([0-9.]+)s")
            last_match = None
            for line in run_log.read_text(encoding="utf-8", errors="ignore").splitlines():
                match = pattern.search(line)
                if match:
                    last_match = match
            if last_match is not None:
                processed = int(last_match.group(1))
                total = int(last_match.group(2))
                elapsed_sec = float(last_match.group(3))
                images = min(processed, total)
                if images > 0 and elapsed_sec > 0:
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
    baselines_dir = REPO_ROOT / "baselines"
    ucn_repo = baselines_dir / "unseen_object_clustering"
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
