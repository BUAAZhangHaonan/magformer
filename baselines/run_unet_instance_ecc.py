#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Ensure sibling utilities are importable when running as a file.
BASELINES_DIR = Path(__file__).resolve().parent
import sys

if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

from unet_instance_models import (
    build_instance_model,
    instances_from_boundary_logits,
    instances_from_distance_logits,
)


def _ann_to_mask(ann: Dict[str, Any], h: int, w: int) -> np.ndarray:
    from pycocotools import mask as mask_utils

    segm = ann.get("segmentation")
    if isinstance(segm, list):
        rles = mask_utils.frPyObjects(segm, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(segm, dict):
        rle = segm
    else:
        raise TypeError(f"Unsupported segmentation type: {type(segm)}")
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 0).astype(np.uint8)


class ECCUnetDataset(Dataset):
    def __init__(self, dataset_root: str, split: str, image_size: int, train: bool, variant: str):
        from pycocotools.coco import COCO

        self.root = Path(dataset_root)
        self.split = split
        self.image_size = int(image_size)
        self.train = bool(train)
        self.variant = variant
        self.coco = COCO(str(self.root / "annotations" / f"instances_{split}.json"))
        self.image_ids = sorted(self.coco.getImgIds())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_id = int(self.image_ids[idx])
        info = self.coco.loadImgs([img_id])[0]
        image = cv2.imread(str(self.root / "images" / self.split / info["file_name"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(info["file_name"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        fg = np.zeros((h, w), dtype=np.uint8)
        boundary = np.zeros((h, w), dtype=np.uint8)
        distance = np.zeros((h, w), dtype=np.float32)
        for ann in anns:
            mask = _ann_to_mask(ann, h, w)
            fg = np.maximum(fg, mask)
            if "boundary" in self.variant:
                dilated = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
                eroded = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
                boundary = np.maximum(boundary, (dilated - eroded).clip(min=0))
            else:
                dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
                if dist.max() > 0:
                    dist = dist / dist.max()
                distance = np.maximum(distance, dist.astype(np.float32))

        if self.train and np.random.rand() < 0.5:
            image = image[:, ::-1].copy()
            fg = fg[:, ::-1].copy()
            boundary = boundary[:, ::-1].copy()
            distance = distance[:, ::-1].copy()

        if (h, w) != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            fg = cv2.resize(fg, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            boundary = cv2.resize(boundary, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            distance = cv2.resize(distance, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        fg_tensor = torch.from_numpy(fg[None, ...]).float()
        if "boundary" in self.variant:
            aux_tensor = torch.from_numpy(boundary[None, ...]).float()
        else:
            aux_tensor = torch.from_numpy(distance[None, ...]).float()

        return {
            "image_id": img_id,
            "file_name": info["file_name"],
            "orig_size": (h, w),
            "image": image_tensor,
            "fg_target": fg_tensor,
            "aux_target": aux_tensor,
        }


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "image_ids": [x["image_id"] for x in batch],
        "file_names": [x["file_name"] for x in batch],
        "orig_sizes": [x["orig_size"] for x in batch],
        "images": torch.stack([x["image"] for x in batch], dim=0),
        "fg_target": torch.stack([x["fg_target"] for x in batch], dim=0),
        "aux_target": torch.stack([x["aux_target"] for x in batch], dim=0),
    }


def _count_params(model: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def _encode_results(
    *,
    variant: str,
    image_id: int,
    fg_logits: np.ndarray,
    aux_logits: np.ndarray,
    orig_size: Tuple[int, int],
    min_area: int,
) -> List[Dict[str, Any]]:
    from pycocotools import mask as mask_utils

    orig_h, orig_w = orig_size
    fg_logits = cv2.resize(fg_logits, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    aux_logits = cv2.resize(aux_logits, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    if "boundary" in variant:
        masks = instances_from_boundary_logits(
            fg_logits=fg_logits,
            boundary_logits=aux_logits,
            min_area=min_area,
        )
    else:
        masks = instances_from_distance_logits(
            fg_logits=fg_logits,
            distance_logits=aux_logits,
            min_area=min_area,
        )

    rows: List[Dict[str, Any]] = []
    fg_prob = 1.0 / (1.0 + np.exp(-fg_logits))
    for mask in masks:
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        score = float(fg_prob[mask > 0].mean())
        rows.append(
            {
                "image_id": int(image_id),
                "category_id": 1,
                "score": score,
                "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
                "segmentation": rle,
            }
        )
    return rows


@torch.no_grad()
def run_eval(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    variant: str,
    ann_file: Path,
    results_json: Path,
    iteration: int,
    min_area: int,
) -> Dict[str, Any]:
    from coco_eval_results import evaluate_coco_results

    model.eval()
    rows: List[Dict[str, Any]] = []
    for batch in loader:
        images = batch["images"].to(device)
        fg_logits, aux_logits = model(images)
        fg_logits_np = fg_logits.squeeze(1).cpu().numpy()
        aux_logits_np = aux_logits.squeeze(1).cpu().numpy()
        for image_id, orig_size, fg_pred, aux_pred in zip(
            batch["image_ids"],
            batch["orig_sizes"],
            fg_logits_np,
            aux_logits_np,
        ):
            rows.extend(
                _encode_results(
                    variant=variant,
                    image_id=int(image_id),
                    fg_logits=fg_pred,
                    aux_logits=aux_pred,
                    orig_size=orig_size,
                    min_area=min_area,
                )
            )

    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(rows), encoding="utf-8")
    return evaluate_coco_results(ann_file=ann_file, results_json=results_json, iteration=iteration)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--variant", type=str, required=True)
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--min-area", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    train_ds = ECCUnetDataset(args.dataset_root, "train", args.image_size, True, args.variant)
    val_ds = ECCUnetDataset(args.dataset_root, "val", args.image_size, False, args.variant)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=_collate)

    model = build_instance_model(args.variant, in_channels=3, base_channels=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start = time.time()

    (output_dir / "params_trainable.txt").write_text(str(_count_params(model)) + "\n", encoding="utf-8")
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        metrics_path.unlink()

    ann_file = Path(args.dataset_root) / "annotations" / "instances_val.json"
    best_ap = -1.0
    best_epoch = 1
    best_ckpt = output_dir / "model_0000001.pth"

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        for batch in train_loader:
            images = batch["images"].to(device)
            fg_target = batch["fg_target"].to(device)
            aux_target = batch["aux_target"].to(device)
            fg_logits, aux_logits = model(images)
            loss_fg = F.binary_cross_entropy_with_logits(fg_logits, fg_target)
            if "boundary" in args.variant:
                loss_aux = F.binary_cross_entropy_with_logits(aux_logits, aux_target)
            else:
                loss_aux = F.l1_loss(torch.sigmoid(aux_logits), aux_target)
            loss = loss_fg + loss_aux
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        epoch_results_path = output_dir / f"epoch_{epoch:04d}_results.json"
        metrics = run_eval(
            model=model,
            loader=val_loader,
            device=device,
            variant=args.variant,
            ann_file=ann_file,
            results_json=epoch_results_path,
            iteration=epoch,
            min_area=args.min_area,
        )
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        segm_ap = float(metrics.get("segm/AP", 0.0))
        if segm_ap >= best_ap:
            best_ap = segm_ap
            best_epoch = epoch
            best_ckpt = output_dir / f"model_{epoch:07d}.pth"
            torch.save(model.state_dict(), best_ckpt)

    final_ckpt = output_dir / "model_final.pth"
    torch.save(model.state_dict(), final_ckpt)

    final_results_path = output_dir / "coco_instances_results.json"
    final_metrics = run_eval(
        model=model,
        loader=val_loader,
        device=device,
        variant=args.variant,
        ann_file=ann_file,
        results_json=final_results_path,
        iteration=args.epochs,
        min_area=args.min_area,
    )
    (output_dir / "metrics.cocoeval.json").write_text(json.dumps(final_metrics, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "last_checkpoint").write_text(final_ckpt.name + "\n", encoding="utf-8")
    (output_dir / "wall_time_sec.txt").write_text(str(int(time.time() - start)) + "\n", encoding="utf-8")
    print(f"[unet-instance] best_epoch={best_epoch} best_ap={best_ap:.4f}")


if __name__ == "__main__":
    main()
