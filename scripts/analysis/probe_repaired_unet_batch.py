#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_batches(text: str) -> List[int]:
    batches = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    return sorted({batch for batch in batches if batch > 0})


def _probe_iaunet(batch: int, image_size: int, device: torch.device) -> None:
    from baselines.iaunet_instance_models import IAUNetCriterion, IAUNetHungarianMatcher, IAUNetInstanceModel

    model = IAUNetInstanceModel().to(device)
    model.train()
    criterion = IAUNetCriterion(matcher=IAUNetHungarianMatcher())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    images = torch.randn(batch, 3, image_size, image_size, device=device)
    targets = []
    for _ in range(batch):
        mask = torch.zeros((1, image_size, image_size), dtype=torch.float32, device=device)
        y0 = max(1, image_size // 4)
        y1 = min(image_size - 1, y0 + max(4, image_size // 8))
        x0 = max(1, image_size // 4)
        x1 = min(image_size - 1, x0 + max(4, image_size // 8))
        mask[:, y0:y1, x0:x1] = 1.0
        targets.append({"labels": torch.ones((1,), dtype=torch.int64, device=device), "masks": mask})
    outputs = model(images)
    loss = sum(criterion(outputs, targets).values())
    loss.backward()
    optimizer.step()


def _probe_cellpose(batch: int, image_size: int, device: torch.device) -> None:
    from cellpose.models import CellposeModel
    from cellpose.train import train_seg

    model = CellposeModel(gpu=device.type == "cuda", pretrained_model=False, nchan=3, device=device, backbone="default")
    side = min(int(image_size), 128)
    train_data = []
    train_labels = []
    for _ in range(max(2, batch)):
        image = np.random.randint(0, 255, size=(3, side, side)).astype(np.float32)
        label = np.zeros((side, side), dtype=np.int32)
        label[side // 4 : side // 2, side // 4 : side // 2] = 1
        train_data.append(image)
        train_labels.append(label)
    with tempfile.TemporaryDirectory(prefix="cellpose-batch-probe-") as tmpdir:
        train_seg(
            model.net,
            train_data=train_data,
            train_labels=train_labels,
            test_data=train_data[:1],
            test_labels=train_labels[:1],
            load_files=False,
            batch_size=int(batch),
            learning_rate=1e-3,
            n_epochs=1,
            channels=None,
            channel_axis=None,
            rgb=True,
            normalize=True,
            compute_flows=True,
            save_path=tmpdir,
            save_every=2,
            save_each=False,
            min_train_masks=1,
            model_name="probe.pth",
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["iaunet", "cellpose"], required=True)
    ap.add_argument("--image-size", type=int, required=True)
    ap.add_argument("--candidate-batches", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output-json", type=str, default="")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    results: List[Dict[str, Any]] = []
    selected = 0
    for batch in _parse_batches(args.candidate_batches):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            if args.model == "iaunet":
                _probe_iaunet(batch, args.image_size, device)
            else:
                _probe_cellpose(batch, args.image_size, device)
            peak_mb = int(torch.cuda.max_memory_allocated() // (1024 * 1024)) if device.type == "cuda" else 0
            results.append({"batch": batch, "passed": True, "peak_allocated_mb": peak_mb})
            selected = batch
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            results.append({"batch": batch, "passed": False, "error": "cuda_out_of_memory"})
            if device.type == "cuda":
                torch.cuda.empty_cache()
            break

    payload = {"model": args.model, "image_size": int(args.image_size), "selected_batch": int(selected), "results": results}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
