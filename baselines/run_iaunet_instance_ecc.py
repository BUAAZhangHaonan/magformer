#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

BASELINES_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASELINES_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.baseline_adapter_utils import (
    binary_masks_to_coco_rows,
    coco_rows_to_jsonable,
    write_baseline_run_artifacts,
)
from baselines.coco_eval_results import evaluate_coco_results
from baselines.ecc_data_utils import load_ecc_coco_rgb_image, load_ecc_coco_rgb_records
from baselines.iaunet_instance_models import (
    IAUNetCriterion,
    IAUNetHungarianMatcher,
    IAUNetInstanceModel,
    count_trainable_parameters,
    iaunet_inference,
)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resize_masks(masks: Sequence[np.ndarray], image_size: int) -> torch.Tensor:
    if len(masks) == 0:
        return torch.zeros((0, int(image_size), int(image_size)), dtype=torch.float32)
    resized = [
        cv2.resize(mask.astype(np.uint8), (int(image_size), int(image_size)), interpolation=cv2.INTER_NEAREST)
        for mask in masks
    ]
    return torch.from_numpy(np.stack(resized, axis=0).astype(np.float32))


class ECCIAUNetDataset(Dataset):
    def __init__(self, dataset_root: str, split: str, image_size: int, *, train: bool) -> None:
        self.records = load_ecc_coco_rgb_records(dataset_root, split)
        self.image_size = int(image_size)
        self.train = bool(train)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        image = load_ecc_coco_rgb_image(record["image_path"], image_size=self.image_size)
        masks = _resize_masks(record["annotation_targets"]["masks"], self.image_size)
        if self.train and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1, :])
            if masks.numel() > 0:
                masks = torch.flip(masks, dims=[2])

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        labels = torch.ones((masks.shape[0],), dtype=torch.int64)
        return {
            "image": image_tensor,
            "target": {"labels": labels, "masks": masks},
            "image_id": int(record["image_id"]),
            "orig_size": (int(record["height"]), int(record["width"])),
        }


def _collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": torch.stack([sample["image"] for sample in batch], dim=0),
        "targets": [sample["target"] for sample in batch],
        "image_ids": [int(sample["image_id"]) for sample in batch],
        "orig_sizes": [tuple(sample["orig_size"]) for sample in batch],
    }


@torch.no_grad()
def run_eval(
    *,
    model: IAUNetInstanceModel,
    loader: DataLoader,
    device: torch.device,
    ann_file: Path,
    results_json: Path,
    iteration: int,
    score_threshold: float,
    mask_threshold: float,
    min_area: int,
    max_images: int | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    seen = 0

    for batch in loader:
        images = batch["images"].to(device)
        outputs = model(images)
        predictions = iaunet_inference(
            outputs,
            original_sizes=batch["orig_sizes"],
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
            min_area=min_area,
        )
        for image_id, prediction in zip(batch["image_ids"], predictions):
            if max_images is not None and seen >= int(max_images):
                break
            rows.extend(
                binary_masks_to_coco_rows(
                    image_id=int(image_id),
                    masks=prediction["masks"].numpy(),
                    scores=prediction["scores"].numpy(),
                    category_ids=prediction["category_ids"].numpy(),
                    score_threshold=score_threshold,
                    mask_threshold=mask_threshold,
                )
            )
            seen += 1
        if max_images is not None and seen >= int(max_images):
            break

    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(coco_rows_to_jsonable(rows), ensure_ascii=False) + "\n", encoding="utf-8")
    metrics = evaluate_coco_results(ann_file=ann_file, results_json=results_json, iteration=iteration)
    return metrics, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--image-size", type=int, choices=[512, 1024], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.4)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-queries", type=int, default=64)
    parser.add_argument("--num-decoder-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    args = parser.parse_args()

    _seed_everything(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_dataset = ECCIAUNetDataset(args.dataset_root, args.train_split, args.image_size, train=True)
    val_dataset = ECCIAUNetDataset(args.dataset_root, args.val_split, args.image_size, train=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=_collate,
    )

    model = IAUNetInstanceModel(
        in_channels=3,
        base_channels=args.base_channels,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
    ).to(device)
    criterion = IAUNetCriterion(matcher=IAUNetHungarianMatcher())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    trainable_params = count_trainable_parameters(model)
    start_time = time.time()
    metrics_log_path = output_dir / "metrics.jsonl"
    if metrics_log_path.exists():
        metrics_log_path.unlink()

    ann_file = Path(args.dataset_root) / "annotations" / f"instances_{args.val_split}.json"
    best_ap = -1.0
    best_epoch = 0
    best_path = output_dir / "model_best.pth"
    total_steps = 0
    stop_after_epoch = False

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        for batch in train_loader:
            images = batch["images"].to(device)
            targets = [
                {
                    "labels": target["labels"].to(device),
                    "masks": target["masks"].to(device),
                }
                for target in batch["targets"]
            ]
            outputs = model(images)
            losses = criterion(outputs, targets)
            loss = sum(losses.values())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_steps += 1
            if int(args.max_train_steps) > 0 and total_steps >= int(args.max_train_steps):
                stop_after_epoch = True
                break

        epoch_results_path = output_dir / f"epoch_{epoch:04d}_results.json"
        metrics, _rows = run_eval(
            model=model,
            loader=val_loader,
            device=device,
            ann_file=ann_file,
            results_json=epoch_results_path,
            iteration=epoch,
            score_threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
            min_area=args.min_area,
            max_images=int(args.max_val_images) if int(args.max_val_images) > 0 else None,
        )
        with metrics_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, **metrics}, ensure_ascii=False) + "\n")

        segm_ap = float(metrics.get("segm/AP", 0.0))
        if segm_ap >= best_ap:
            best_ap = segm_ap
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
        if stop_after_epoch:
            break

    final_ckpt = output_dir / "model_final.pth"
    torch.save(model.state_dict(), final_ckpt)
    if not best_path.exists():
        torch.save(model.state_dict(), best_path)

    final_results_path = output_dir / "coco_instances_results.json"
    final_metrics, final_rows = run_eval(
        model=model,
        loader=val_loader,
        device=device,
        ann_file=ann_file,
        results_json=final_results_path,
        iteration=int(args.epochs),
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        min_area=args.min_area,
        max_images=int(args.max_val_images) if int(args.max_val_images) > 0 else None,
    )

    metadata = {
        "model_id": "iaunet",
        "model_name": "iaunet",
        "image_size": int(args.image_size),
        "train_split": args.train_split,
        "val_split": args.val_split,
        "num_queries": int(args.num_queries),
        "hidden_dim": int(args.hidden_dim),
        "base_channels": int(args.base_channels),
        "num_decoder_layers": int(args.num_decoder_layers),
        "best_epoch": int(best_epoch),
        "best_segm_ap": float(best_ap),
    }
    write_baseline_run_artifacts(
        output_dir,
        coco_rows=final_rows,
        metrics=final_metrics,
        metadata=metadata,
        last_checkpoint=final_ckpt.name,
        wall_time_sec=int(time.time() - start_time),
        trainable_params=trainable_params,
    )


if __name__ == "__main__":
    main()
