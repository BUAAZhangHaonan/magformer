#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

BASELINES_DIR = Path(__file__).resolve().parent
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

try:
    from .coco_eval_results import evaluate_coco_results
    from .baseline_adapter_utils import write_baseline_run_artifacts
    from .stardist_instance_utils import (
        SUPPORTED_IMAGE_SIZES,
        load_stardist_ecc_split,
        stardist_prediction_to_coco_rows,
    )
except ImportError:  # pragma: no cover - file execution fallback
    from coco_eval_results import evaluate_coco_results
    from baseline_adapter_utils import write_baseline_run_artifacts
    from stardist_instance_utils import (
        SUPPORTED_IMAGE_SIZES,
        load_stardist_ecc_split,
        stardist_prediction_to_coco_rows,
    )


@dataclass(frozen=True)
class StarDistBackend:
    Config2D: Any
    StarDist2D: Any


def _load_stardist_backend() -> StarDistBackend:
    from stardist.models import Config2D, StarDist2D

    return StarDistBackend(Config2D=Config2D, StarDist2D=StarDist2D)


def _count_trainable_params(model: Any) -> int:
    keras_model = getattr(model, "keras_model", None)
    if keras_model is not None:
        trainable_weights = getattr(keras_model, "trainable_weights", None)
        if trainable_weights is not None:
            total = 0
            for weight in trainable_weights:
                shape = getattr(weight, "shape", None)
                if shape is None:
                    continue
                size = 1
                for dim in tuple(shape):
                    size *= int(dim)
                total += int(size)
            if total > 0:
                return total
        count_params = getattr(keras_model, "count_params", None)
        if callable(count_params):
            try:
                return int(count_params())
            except Exception:
                pass
    count_params = getattr(model, "count_params", None)
    if callable(count_params):
        try:
            return int(count_params())
        except Exception:
            pass
    return 0


def _save_checkpoint(model: Any, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model_final.weights.h5"
    keras_model = getattr(model, "keras_model", None)
    if keras_model is not None and hasattr(keras_model, "save_weights"):
        keras_model.save_weights(str(checkpoint_path))
    else:
        checkpoint_path.write_text("StarDist checkpoint unavailable in this environment.\n", encoding="utf-8")
    return checkpoint_path


def _stardist_steps_per_epoch(num_images: int, batch_size: int) -> int:
    num_images = max(1, int(num_images))
    batch_size = max(1, int(batch_size))
    return max(1, (num_images + batch_size - 1) // batch_size)


def _build_model(
    *,
    backend: StarDistBackend,
    output_dir: Path,
    image_size: int,
    batch_size: int,
    model_name: str,
) -> Any:
    config = backend.Config2D(
        n_rays=32,
        n_channel_in=3,
        grid=(1, 1),
        train_patch_size=(int(image_size), int(image_size)),
        train_batch_size=int(batch_size),
    )
    model_root = output_dir / "stardist_model"
    model_root.mkdir(parents=True, exist_ok=True)
    return backend.StarDist2D(config, name=model_name, basedir=str(model_root))


def _predict_split(
    *,
    model: Any,
    records: Sequence[Dict[str, Any]],
    images: Sequence[np.ndarray],
    prob_thresh: float,
    nms_thresh: float,
    max_images: int | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    limit = len(images) if max_images is None or int(max_images) <= 0 else min(len(images), int(max_images))
    for idx in range(limit):
        record = records[idx]
        image = images[idx]
        labels, details = model.predict_instances(
            image,
            prob_thresh=float(prob_thresh),
            nms_thresh=float(nms_thresh),
        )
        rows.extend(
            stardist_prediction_to_coco_rows(
                image_id=int(record["image_id"]),
                labels=np.asarray(labels),
                details=details,
                score_threshold=float(prob_thresh),
                output_size=(int(record["height"]), int(record["width"])),
            )
        )
    return rows


def train_and_eval(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = _load_stardist_backend()
    train_images, train_labels, _ = load_stardist_ecc_split(
        args.dataset_root,
        args.train_split,
        int(args.image_size),
        max_images=args.max_train_images,
    )
    val_images, val_labels, val_records = load_stardist_ecc_split(
        args.dataset_root,
        args.eval_split,
        int(args.image_size),
        max_images=args.max_val_images,
    )

    model = _build_model(
        backend=backend,
        output_dir=output_dir,
        image_size=int(args.image_size),
        batch_size=int(args.batch),
        model_name=args.model_name,
    )
    steps_per_epoch = _stardist_steps_per_epoch(len(train_images), int(args.batch))
    train_start = time.time()
    model.train(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        classes="auto",
        augmenter=None,
        seed=int(args.seed),
        epochs=int(args.epochs),
        steps_per_epoch=int(steps_per_epoch),
        workers=int(args.num_workers),
    )

    checkpoint_path = _save_checkpoint(model, output_dir / "stardist_model")
    final_rows = _predict_split(
        model=model,
        records=val_records,
        images=val_images,
        prob_thresh=float(args.prob_thresh),
        nms_thresh=float(args.nms_thresh),
        max_images=args.max_val_images,
    )

    artifact_paths = write_baseline_run_artifacts(output_dir, coco_rows=final_rows)
    results_path = artifact_paths["coco_instances_results"]
    ann_file = Path(args.dataset_root) / "annotations" / f"instances_{args.eval_split}.json"
    metrics = evaluate_coco_results(ann_file=ann_file, results_json=results_path, iteration=int(args.epochs))

    metadata = {
        "model_id": args.model_name,
        "backend": "stardist",
        "image_size": int(args.image_size),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "epochs": int(args.epochs),
        "batch": int(args.batch),
        "num_workers": int(args.num_workers),
        "num_train_images": len(train_images),
        "num_val_images": len(val_images),
        "prob_thresh": float(args.prob_thresh),
        "nms_thresh": float(args.nms_thresh),
        "checkpoint": str(checkpoint_path),
    }
    artifacts = write_baseline_run_artifacts(
        output_dir,
        metrics=metrics,
        metadata=metadata,
        last_checkpoint=checkpoint_path.name,
        wall_time_sec=int(time.time() - train_start),
        trainable_params=_count_trainable_params(model),
    )
    artifacts["checkpoint"] = checkpoint_path
    return {
        "artifacts": artifacts,
        "metrics": metrics,
        "checkpoint": checkpoint_path,
    }


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--image-size", type=int, default=512, choices=SUPPORTED_IMAGE_SIZES)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-split", type=str, default="train")
    ap.add_argument("--eval-split", type=str, default="val")
    ap.add_argument("--max-train-images", type=int, default=0)
    ap.add_argument("--max-val-images", type=int, default=0)
    ap.add_argument("--prob-thresh", type=float, default=0.5)
    ap.add_argument("--nms-thresh", type=float, default=0.3)
    ap.add_argument("--model-name", type=str, default="stardist")
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    train_and_eval(args)


if __name__ == "__main__":
    main()
