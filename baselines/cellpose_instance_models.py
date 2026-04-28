from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, maximum_filter

import sys

BASELINES_DIR = Path(__file__).resolve().parent
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

from baseline_adapter_utils import binary_masks_to_coco_rows, coco_rows_to_jsonable, decode_coco_segmentation


def _load_lightweight_ecc_records(dataset_root: str | Path, split: str) -> List[Dict[str, Any]]:
    root = Path(dataset_root)
    ann_path = root / "annotations" / f"instances_{split}.json"
    img_dir = root / "images" / split
    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    annotations_by_image_id: Dict[int, List[Dict[str, Any]]] = {}
    for annotation in payload.get("annotations", []):
        annotations_by_image_id.setdefault(int(annotation.get("image_id", -1)), []).append(dict(annotation))

    records: List[Dict[str, Any]] = []
    for image_info in payload.get("images", []):
        image_id = int(image_info["id"])
        records.append(
            {
                "image_id": image_id,
                "file_name": str(image_info["file_name"]),
                "image_path": str(img_dir / image_info["file_name"]),
                "height": int(image_info["height"]),
                "width": int(image_info["width"]),
                "annotations": annotations_by_image_id.get(image_id, []),
            }
        )
    return records


def _annotations_to_instance_map(annotations: Sequence[Mapping[str, Any]], height: int, width: int) -> np.ndarray:
    instance_map = np.zeros((int(height), int(width)), dtype=np.int32)
    for instance_id, annotation in enumerate(annotations, start=1):
        mask = decode_coco_segmentation(annotation.get("segmentation"), int(height), int(width))
        instance_map[mask > 0] = int(instance_id)
    return instance_map


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.cpu().numpy()
    return np.asarray(value)


def _center_pixel(mask: np.ndarray) -> Tuple[int, int]:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return 0, 0
    cy = int(np.round(float(ys.mean())))
    cx = int(np.round(float(xs.mean())))
    if mask[cy, cx] > 0:
        return cy, cx
    idx = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
    return int(ys[idx]), int(xs[idx])


def instance_map_to_cellpose_targets(instance_map: np.ndarray) -> Dict[str, np.ndarray]:
    instance_map = np.asarray(instance_map, dtype=np.int32)
    cellprob = (instance_map > 0).astype(np.float32)
    flow = np.zeros((2, instance_map.shape[0], instance_map.shape[1]), dtype=np.float32)

    for instance_id in [int(v) for v in np.unique(instance_map).tolist() if int(v) > 0]:
        mask = instance_map == instance_id
        if not np.any(mask):
            continue
        dist = distance_transform_edt(mask.astype(np.uint8)).astype(np.float32)
        if float(dist.max()) <= 0.0:
            continue
        gy, gx = np.gradient(dist)
        dy = -gy.astype(np.float32)
        dx = -gx.astype(np.float32)
        norm = np.sqrt(dy**2 + dx**2)
        norm[norm == 0] = 1.0
        flow[0, mask] = dy[mask] / norm[mask]
        flow[1, mask] = dx[mask] / norm[mask]

    return {
        "instance_map": instance_map,
        "cellprob": cellprob,
        "flow": flow,
    }


def _bilinear_sample(field: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    h, w = field.shape
    ys = np.clip(ys, 0.0, max(0.0, float(h - 1)))
    xs = np.clip(xs, 0.0, max(0.0, float(w - 1)))
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wy = ys - y0
    wx = xs - x0
    top = field[y0, x0] * (1.0 - wx) + field[y0, x1] * wx
    bottom = field[y1, x0] * (1.0 - wx) + field[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def _integrate_flows(
    flow: np.ndarray,
    foreground: np.ndarray,
    *,
    niter: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(foreground > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)

    coords = np.stack([ys.astype(np.float32), xs.astype(np.float32)], axis=1)
    flow_y = flow[0].astype(np.float32, copy=False)
    flow_x = flow[1].astype(np.float32, copy=False)
    for _ in range(max(1, int(niter))):
        step_y = _bilinear_sample(flow_y, coords[:, 0], coords[:, 1])
        step_x = _bilinear_sample(flow_x, coords[:, 0], coords[:, 1])
        coords[:, 0] = np.clip(coords[:, 0] + step_y, 0.0, float(flow_y.shape[0] - 1))
        coords[:, 1] = np.clip(coords[:, 1] + step_x, 0.0, float(flow_x.shape[1] - 1))
    endpoints = np.rint(coords).astype(np.int32)
    return coords, endpoints


def _seed_mask_from_prob(prob: np.ndarray, threshold: float) -> np.ndarray:
    fg = prob >= float(threshold)
    if not np.any(fg):
        return np.zeros_like(prob, dtype=np.uint8)
    local_max = prob == maximum_filter(prob, size=3, mode="nearest")
    seeds = (fg & local_max).astype(np.uint8)
    if int(seeds.sum()) == 0:
        seeds = fg.astype(np.uint8)
    return seeds


def _connected_component_masks(binary_mask: np.ndarray, min_area: int) -> List[np.ndarray]:
    num, labels = cv2.connectedComponents(binary_mask.astype(np.uint8), connectivity=8)
    out: List[np.ndarray] = []
    for label_id in range(1, num):
        mask = (labels == label_id).astype(np.uint8)
        if int(mask.sum()) >= int(min_area):
            out.append(mask)
    return out


def _masks_from_flow_endpoints(
    *,
    foreground: np.ndarray,
    endpoints: np.ndarray,
    min_area: int,
) -> List[np.ndarray]:
    endpoint_mask = np.zeros_like(foreground, dtype=np.uint8)
    if endpoints.size == 0:
        return []
    endpoint_mask[endpoints[:, 0], endpoints[:, 1]] = 1
    endpoint_labels = cv2.connectedComponents(endpoint_mask, connectivity=8)[1]
    assigned_labels = endpoint_labels[endpoints[:, 0], endpoints[:, 1]]
    yx = np.column_stack(np.nonzero(foreground))
    masks: List[np.ndarray] = []
    for endpoint_label in [int(v) for v in np.unique(assigned_labels).tolist() if int(v) > 0]:
        chosen = assigned_labels == endpoint_label
        if not np.any(chosen):
            continue
        mask = np.zeros_like(foreground, dtype=np.uint8)
        mask[yx[chosen, 0], yx[chosen, 1]] = 1
        masks.extend(_connected_component_masks(mask, min_area=min_area))
    return masks


def predictions_from_logits(
    *,
    flow_logits: Any,
    cellprob_logits: Any,
    min_area: int = 20,
    score_threshold: float = 0.05,
    mask_threshold: float = 0.5,
    flow_niter: int = 64,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    flow = _as_numpy(flow_logits).astype(np.float32, copy=False)
    if flow.ndim == 4:
        flow = flow[0]
    cellprob_logits = _as_numpy(cellprob_logits).astype(np.float32, copy=False)
    if cellprob_logits.ndim == 3:
        cellprob_logits = cellprob_logits[0]

    cellprob = 1.0 / (1.0 + np.exp(-cellprob_logits))
    foreground = cellprob >= float(mask_threshold)
    if not np.any(foreground):
        return [], np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    _coords, endpoints = _integrate_flows(flow, foreground, niter=flow_niter)
    seeds = _seed_mask_from_prob(cellprob, float(mask_threshold))
    seed_labels = cv2.connectedComponents(seeds.astype(np.uint8), connectivity=8)[1]
    seed_ids = [int(v) for v in np.unique(seed_labels).tolist() if int(v) > 0]
    max_seed_distance_elements = 50_000_000
    if 0 < len(seed_ids) and int(endpoints.shape[0]) * len(seed_ids) <= max_seed_distance_elements:
        seed_centers = np.array([_center_pixel(seed_labels == seed_id) for seed_id in seed_ids], dtype=np.float32)
        yx = np.column_stack(np.nonzero(foreground))
        endpoint_coords = endpoints.astype(np.float32)
        dists = ((endpoint_coords[:, None, :] - seed_centers[None, :, :]) ** 2).sum(axis=2)
        assigned = dists.argmin(axis=1)
        masks = []
        for seed_idx, _seed_id in enumerate(seed_ids):
            mask = np.zeros_like(foreground, dtype=np.uint8)
            chosen = assigned == seed_idx
            if not np.any(chosen):
                continue
            mask[yx[chosen, 0], yx[chosen, 1]] = 1
            masks.extend(_connected_component_masks(mask, min_area=min_area))
    else:
        masks = _masks_from_flow_endpoints(foreground=foreground, endpoints=endpoints, min_area=min_area)
    if not masks and int(seeds.sum()) > 0:
        masks = _connected_component_masks(seeds.astype(np.uint8), min_area=min_area)
    if not masks:
        masks = _connected_component_masks(foreground.astype(np.uint8), min_area=min_area)

    masks = [mask.astype(np.uint8, copy=False) for mask in masks if int(mask.sum()) >= int(min_area)]
    scores = np.asarray(
        [float(cellprob[mask > 0].mean()) if int(mask.sum()) > 0 else 0.0 for mask in masks],
        dtype=np.float32,
    )
    keep = scores >= float(score_threshold)
    masks = [mask for mask, ok in zip(masks, keep.tolist()) if ok]
    scores = scores[keep]
    category_ids = np.zeros((len(masks),), dtype=np.int64)
    return masks, scores, category_ids


class _DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = _DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CellPoseFlowUNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 16):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        self.inc = _DoubleConv(in_channels, c1)
        self.down1 = _Down(c1, c2)
        self.down2 = _Down(c2, c3)
        self.down3 = _Down(c3, c4)
        self.up1 = _Up(c4, c3, c3)
        self.up2 = _Up(c3, c2, c2)
        self.up3 = _Up(c2, c1, c1)
        self.head = nn.Conv2d(c1, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.head(x)


def _cellpose_cache_key(record: Mapping[str, Any]) -> str:
    return f"{int(record['image_id']):012d}.npz"


def _read_cached_cellpose_targets(cache_path: Path) -> Dict[str, np.ndarray] | None:
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path) as payload:
            return {
                "instance_map": payload["instance_map"].astype(np.int32, copy=False),
                "cellprob": payload["cellprob"].astype(np.float32, copy=False),
                "flow": payload["flow"].astype(np.float32, copy=False),
            }
    except Exception:
        # A partially written cache entry should not poison the run.
        cache_path.unlink(missing_ok=True)
        return None


def _write_cached_cellpose_targets(cache_path: Path, targets: Mapping[str, np.ndarray]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.{time.time_ns()}.tmp")
    instance_map = np.asarray(targets["instance_map"], dtype=np.int32)
    max_instance_id = int(instance_map.max()) if instance_map.size else 0
    if max_instance_id <= np.iinfo(np.uint16).max:
        instance_map = instance_map.astype(np.uint16, copy=False)
    with tmp_path.open("wb") as handle:
        np.savez(
            handle,
            instance_map=instance_map,
            cellprob=np.asarray(targets["cellprob"] > 0, dtype=np.uint8),
            flow=np.asarray(targets["flow"], dtype=np.float16),
        )
    os.replace(tmp_path, cache_path)


def _flip_cellpose_targets(
    targets: Mapping[str, np.ndarray],
    *,
    horizontal: bool,
    vertical: bool,
) -> Dict[str, np.ndarray]:
    instance_map = np.asarray(targets["instance_map"], dtype=np.int32)
    cellprob = np.asarray(targets["cellprob"], dtype=np.float32)
    flow = np.asarray(targets["flow"], dtype=np.float32)
    if horizontal:
        instance_map = instance_map[:, ::-1].copy()
        cellprob = cellprob[:, ::-1].copy()
        flow = flow[:, :, ::-1].copy()
        flow[1] *= -1.0
    if vertical:
        instance_map = instance_map[::-1, :].copy()
        cellprob = cellprob[::-1, :].copy()
        flow = flow[:, ::-1, :].copy()
        flow[0] *= -1.0
    return {"instance_map": instance_map, "cellprob": cellprob, "flow": flow}


class ECCCellPoseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        image_size: int,
        train: bool,
        target_cache_dir: str | Path | None = None,
    ):
        from ecc_data_utils import load_ecc_coco_rgb_image

        self.dataset_root = Path(dataset_root)
        self.split = str(split)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.records = _load_lightweight_ecc_records(self.dataset_root, self.split)
        self._load_image = load_ecc_coco_rgb_image
        self.target_cache_dir = Path(target_cache_dir) if target_cache_dir else None

    def __len__(self) -> int:
        return len(self.records)

    def _targets_for_record(self, record: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        cache_path = None
        if self.target_cache_dir is not None:
            cache_path = self.target_cache_dir / _cellpose_cache_key(record)
            cached = _read_cached_cellpose_targets(cache_path)
            if cached is not None:
                return cached

        instance_map = _annotations_to_instance_map(
            record["annotations"],
            height=int(record["height"]),
            width=int(record["width"]),
        )
        if instance_map.shape[:2] != (self.image_size, self.image_size):
            instance_map = cv2.resize(instance_map, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        targets = instance_map_to_cellpose_targets(instance_map)
        if cache_path is not None:
            _write_cached_cellpose_targets(cache_path, targets)
        return targets

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        image = self._load_image(record["image_path"], image_size=self.image_size)
        targets = self._targets_for_record(record)
        horizontal = bool(self.train and np.random.rand() < 0.5)
        vertical = bool(self.train and np.random.rand() < 0.5)
        if horizontal:
            image = image[:, ::-1].copy()
        if vertical:
            image = image[::-1, :].copy()
        targets = _flip_cellpose_targets(targets, horizontal=horizontal, vertical=vertical)

        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        return {
            "image": image_t,
            "instance_map": torch.from_numpy(targets["instance_map"].astype(np.int32)),
            "cellprob": torch.from_numpy(targets["cellprob"][None, ...]),
            "flow": torch.from_numpy(targets["flow"]),
            "image_id": int(record["image_id"]),
            "file_name": str(record["file_name"]),
        }


def _collate(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values, dim=0)
        else:
            out[key] = values
    return out


def _count_params(model: nn.Module) -> int:
    if not hasattr(model, "parameters"):
        return 0
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def train_cellpose_model(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    image_size: int,
    epochs: int,
    batch: int,
    lr: float,
    num_workers: int,
    device: str,
    train_split: str = "train",
    max_train_steps: int = 0,
    target_cache_dir: str | Path | None = None,
    log_every: int = 50,
) -> Dict[str, Any]:
    if int(image_size) not in {512, 1024}:
        raise ValueError("--image-size must be one of {512, 1024}")

    from torch.utils.data import DataLoader

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(target_cache_dir) if target_cache_dir else out_dir / "target_cache" / f"{train_split}_{image_size}"
    train_ds = ECCCellPoseDataset(
        dataset_root,
        train_split,
        image_size=image_size,
        train=True,
        target_cache_dir=cache_dir,
    )
    loader_kwargs: Dict[str, Any] = {
        "num_workers": int(num_workers),
        "pin_memory": device.startswith("cuda"),
        "collate_fn": _collate,
    }
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 1
    train_loader = DataLoader(train_ds, batch_size=int(batch), shuffle=True, **loader_kwargs)

    model = CellPoseFlowUNet(in_channels=3, base_channels=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))

    total_steps = 0
    planned_steps = int(max_train_steps) if int(max_train_steps) > 0 else int(epochs) * len(train_loader)
    train_start = time.time()
    progress_path = out_dir / "train_progress.json"
    model.train()
    print(
        f"[cellpose-train] start image_size={image_size} epochs={epochs} "
        f"steps={planned_steps} batch={batch} num_workers={num_workers} target_cache_dir={cache_dir}",
        flush=True,
    )
    for epoch in range(int(epochs)):
        for batch_idx, batch_data in enumerate(train_loader, start=1):
            images = batch_data["image"].to(device)
            target_flow = batch_data["flow"].to(device)
            target_cellprob = batch_data["cellprob"].to(device)
            with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                logits = model(images)
                flow_logits = logits[:, :2]
                cellprob_logits = logits[:, 2:3]
                flow_loss = F.mse_loss(flow_logits, target_flow * 5.0)
                cellprob_loss = F.binary_cross_entropy_with_logits(cellprob_logits, target_cellprob)
                loss = flow_loss + cellprob_loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_steps += 1
            should_log = total_steps == 1 or (int(log_every) > 0 and total_steps % int(log_every) == 0)
            if should_log:
                elapsed_sec = max(0.0, time.time() - train_start)
                loss_value = float(loss.detach().cpu())
                progress = {
                    "epoch": int(epoch + 1),
                    "epochs": int(epochs),
                    "batch_in_epoch": int(batch_idx),
                    "batches_per_epoch": int(len(train_loader)),
                    "total_steps": int(total_steps),
                    "planned_steps": int(planned_steps),
                    "loss": loss_value,
                    "elapsed_sec": elapsed_sec,
                    "target_cache_dir": str(cache_dir),
                }
                progress_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
                print(
                    "[cellpose-train] "
                    f"epoch={epoch + 1}/{epochs} batch={batch_idx}/{len(train_loader)} "
                    f"step={total_steps}/{planned_steps} loss={loss_value:.6f} elapsed_sec={elapsed_sec:.1f}",
                    flush=True,
                )
            if int(max_train_steps) > 0 and total_steps >= int(max_train_steps):
                break
        if int(max_train_steps) > 0 and total_steps >= int(max_train_steps):
            break

    final_ckpt = out_dir / "model_final.pth"
    torch.save(model.state_dict(), final_ckpt)
    return {
        "model": model,
        "checkpoint": final_ckpt,
        "trainable_params": _count_params(model),
        "epochs": int(epochs),
    }


def _predict_rows_for_record(
    *,
    model: nn.Module,
    record: Mapping[str, Any],
    image_size: int,
    min_area: int,
    device: str,
    score_threshold: float,
    mask_threshold: float,
) -> List[Dict[str, Any]]:
    from ecc_data_utils import load_ecc_coco_rgb_image

    image = load_ecc_coco_rgb_image(record["image_path"], image_size=image_size)
    image_t = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        logits = model(image_t.to(device))
    flow_logits = logits[:, :2].detach().cpu().numpy()[0]
    cellprob_logits = logits[:, 2:3].detach().cpu().numpy()[0, 0]
    masks, scores, category_ids = predictions_from_logits(
        flow_logits=flow_logits,
        cellprob_logits=cellprob_logits,
        min_area=int(min_area),
        score_threshold=float(score_threshold),
        mask_threshold=float(mask_threshold),
    )
    rows = binary_masks_to_coco_rows(
        image_id=int(record["image_id"]),
        masks=masks,
        scores=scores,
        category_ids=category_ids,
        score_threshold=float(score_threshold),
        mask_threshold=float(mask_threshold),
    )
    return rows


def predict_records(
    *,
    model_bundle: Mapping[str, Any],
    dataset_root: str | Path,
    eval_split: str,
    image_size: int,
    min_area: int,
    device: str,
    max_val_images: int = 0,
    score_threshold: float = 0.05,
    mask_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    model = model_bundle["model"]
    records = _load_lightweight_ecc_records(dataset_root, eval_split)
    if int(max_val_images) > 0:
        records = records[: int(max_val_images)]
    rows: List[Dict[str, Any]] = []
    model.eval()
    for record in records:
        rows.extend(
            _predict_rows_for_record(
                model=model,
                record=record,
                image_size=image_size,
                min_area=min_area,
                device=device,
                score_threshold=score_threshold,
                mask_threshold=mask_threshold,
            )
        )
    return rows


def evaluate_results(
    *,
    dataset_root: str | Path,
    eval_split: str,
    rows: Sequence[Mapping[str, Any]],
    image_size: int,
    iteration: int,
) -> Dict[str, Any]:
    from coco_eval_results import evaluate_coco_results

    ann_file = Path(dataset_root) / "annotations" / f"instances_{eval_split}.json"
    tmp_results = Path(dataset_root) / f".cellpose_eval_{int(image_size)}_{int(iteration)}.json"
    tmp_results.write_text(
        json.dumps(coco_rows_to_jsonable(rows), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        return evaluate_coco_results(ann_file, tmp_results, iteration=int(iteration))
    finally:
        if tmp_results.exists():
            tmp_results.unlink()


def run_experiment(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    image_size: int,
    epochs: int,
    batch: int,
    lr: float,
    num_workers: int,
    min_area: int,
    device: str,
    max_train_steps: int = 0,
    max_val_images: int = 0,
    train_split: str = "train",
    val_split: str = "val",
    target_cache_dir: str | Path | None = None,
    log_every: int = 50,
    train_model_fn=train_cellpose_model,
    predict_records_fn=predict_records,
    evaluate_results_fn=evaluate_results,
) -> Dict[str, Any]:
    from baseline_adapter_utils import write_baseline_run_artifacts

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    bundle = train_model_fn(
        dataset_root=dataset_root,
        output_dir=output_dir,
        image_size=image_size,
        epochs=epochs,
        batch=batch,
        lr=lr,
        num_workers=num_workers,
        device=device,
        train_split=train_split,
        max_train_steps=max_train_steps,
        target_cache_dir=target_cache_dir,
        log_every=log_every,
    )
    if isinstance(bundle, nn.Module):
        bundle = {
            "model": bundle,
            "checkpoint": output_dir / "model_final.pth",
            "trainable_params": _count_params(bundle),
            "epochs": int(epochs),
        }
    model = bundle["model"]
    checkpoint = Path(bundle.get("checkpoint", output_dir / "model_final.pth"))
    if "trainable_params" in bundle:
        trainable_params = int(bundle["trainable_params"])
    else:
        trainable_params = _count_params(model)

    rows = predict_records_fn(
        model_bundle=bundle,
        dataset_root=dataset_root,
        eval_split=val_split,
        image_size=image_size,
        min_area=min_area,
        device=device,
        max_val_images=max_val_images,
    )
    metrics = evaluate_results_fn(
        dataset_root=dataset_root,
        eval_split=val_split,
        rows=rows,
        image_size=image_size,
        iteration=int(epochs),
    )

    artifacts = write_baseline_run_artifacts(
        output_dir,
        coco_rows=rows,
        metrics=metrics,
        metadata={
            "model_id": "cellpose",
            "image_size": int(image_size),
            "epochs": int(epochs),
            "batch": int(batch),
            "lr": float(lr),
            "num_workers": int(num_workers),
            "min_area": int(min_area),
            "train_split": str(train_split),
            "val_split": str(val_split),
            "max_train_steps": int(max_train_steps),
            "max_val_images": int(max_val_images),
            "target_cache_dir": str(target_cache_dir) if target_cache_dir else str(output_dir / "target_cache" / f"{train_split}_{image_size}"),
            "log_every": int(log_every),
        },
        last_checkpoint=checkpoint.name,
        wall_time_sec=max(0.0, time.time() - start_time),
        trainable_params=trainable_params,
    )
    return {"bundle": bundle, "metrics": metrics, "rows": rows, "artifacts": artifacts}
