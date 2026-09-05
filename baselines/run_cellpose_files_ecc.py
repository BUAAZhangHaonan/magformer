#!/usr/bin/env python3
"""Official cellpose on GISEC 32254 @1024 in DISK-BACKED files mode.

Why: the list-mode official path (train_seg with in-RAM data +
compute_flows=True) materializes images+labels (~128G) AND all flows
(~210G float32 at 1024) -> 240G+ RSS; two watchdog kills and one kernel
OOM on this 251G box (2026-09-04). This script stays on the official API
but uses its disk-backed lane:

  1. prepass: per image, dynamics.labels_to_flows([instance_map],
     files=[symlinked image]) writes <stem>_flows.tif (5xHxW float32,
     official format). Resume-safe (existing tifs skipped); the COCO json
     annotation tree is freed per record and malloc_trim'ed at the end.
  2. train: cellpose.train_seg(train_files=..., train_labels_files=...,
     load_files=False, compute_flows=False, channels=[1,2,3],
     channel_axis=2) -> lazy per-batch imread. channel params matter:
     _reshape_norm only transposes HWC->CHW when channels/channel_axis is
     non-None, but convert_image's nchan=2 default silently truncates RGB
     to 2 channels when channels=None - the explicit [1,2,3] keeps all of
     RGB (the 14:2x run trained 200k batches on R,G,0 and was killed).
  3. eval: unchanged full-val (3276) COCO segm AP at GISEC caliber
     (maxDets [1,10,100]) via the same _predict_rows_for_records_batch.

Run on 4029 (GPU4):
  python run_cellpose_files_ecc.py --dataset-root <DS> --output-dir <OUT> \
      --image-size 1024 --epochs 20 --batch 16 --lr 1e-3 --mode all
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

BASELINES_DIR = Path(__file__).resolve().parent
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

from cellpose import dynamics  # noqa: E402
from cellpose.models import CellposeModel  # noqa: E402
from cellpose.train import train_seg  # noqa: E402

from cellpose_instance_models import (  # noqa: E402
    _annotations_to_instance_map,
    _load_lightweight_ecc_records,
    _predict_rows_for_records_batch,
    _resize_instance_map,
)


def _malloc_trim() -> None:
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def prepass(args, split: str, max_images: int) -> dict:
    out_dir = Path(args.output_dir)
    links_dir = out_dir / f"flows_{split}"
    links_dir.mkdir(parents=True, exist_ok=True)
    records = _load_lightweight_ecc_records(args.dataset_root, split)
    if max_images > 0:
        records = records[:max_images]
    device = torch.device("cuda")
    t0 = time.time()
    created = existing = 0
    for i, rec in enumerate(records):
        stem = Path(rec["image_path"]).name
        flows_tif = links_dir / (Path(stem).stem + "_flows.tif")
        if flows_tif.exists():
            rec["annotations"] = []
            existing += 1
        else:
            link = links_dir / stem
            if not link.exists():
                os.symlink(rec["image_path"], link)
            instance_map = _annotations_to_instance_map(
                rec["annotations"],
                height=int(rec["height"]),
                width=int(rec["width"]),
            )
            instance_map = _resize_instance_map(instance_map, int(args.image_size))
            dynamics.labels_to_flows(
                [instance_map], files=[str(link)], device=device, return_flows=False
            )
            rec["annotations"] = []
            created += 1
        if (i + 1) % 200 == 0:
            print(
                f"[prepass:{split}] {i + 1}/{len(records)} created={created} "
                f"existing={existing} {time.time() - t0:.0f}s",
                flush=True,
            )
    _malloc_trim()
    summary = {
        "split": split,
        "n": len(records),
        "created": created,
        "existing": existing,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out_dir / f"prepass_{split}.json").write_text(json.dumps(summary, indent=1))
    print(f"[prepass:{split}] DONE {summary}", flush=True)
    return summary


def train(args) -> None:
    out_dir = Path(args.output_dir)
    train_links = out_dir / "flows_train"
    val_links = out_dir / "flows_val"
    train_files = sorted(str(p) for p in train_links.glob("*.png"))
    train_labels_files = [
        str(Path(f).with_name(Path(f).stem + "_flows.tif")) for f in train_files
    ]
    keep = [f for f, lf in zip(train_files, train_labels_files) if Path(lf).exists()]
    assert len(keep) == len(train_files), "missing flows tifs - run prepass first"
    val_files = sorted(str(p) for p in val_links.glob("*.png"))[: args.val_test_n]
    val_labels_files = [
        str(Path(f).with_name(Path(f).stem + "_flows.tif")) for f in val_files
    ]
    assert all(Path(p).exists() for p in val_labels_files), "missing val flows"

    device = torch.device("cuda")
    model = CellposeModel(
        gpu=True,
        pretrained_model=False,
        nchan=3,
        device=device,
        backbone=str(args.backbone),
    )
    t0 = time.time()
    filename, train_losses, test_losses = train_seg(
        model.net,
        train_files=train_files,
        train_labels_files=train_labels_files,
        test_files=val_files,
        test_labels_files=val_labels_files,
        load_files=False,
        batch_size=int(args.batch),
        learning_rate=float(args.lr),
        n_epochs=int(args.epochs),
        weight_decay=1e-5,
        channels=[1, 2, 3],
        channel_axis=2,
        rgb=True,
        normalize=True,
        compute_flows=False,
        save_path=str(out_dir),
        save_every=max(int(args.epochs) + 1, 2),
        save_each=False,
        min_train_masks=1,
        model_name="model_final",
    )
    final = out_dir / "model_final.pth"
    src = Path(filename)
    if src.exists() and src.resolve() != final.resolve():
        shutil.copy2(src, final)
    if not final.exists():
        model.net.save_model(str(final))
    meta = {
        "mode": "official-files-lane",
        "train_files": len(train_files),
        "epochs": int(args.epochs),
        "batch": int(args.batch),
        "lr": float(args.lr),
        "train_time_sec": round(time.time() - t0, 1),
        "final_train_loss": float(np.asarray(train_losses).reshape(-1)[-1]),
    }
    (out_dir / "train_files_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[train] DONE {meta}", flush=True)


def evaluate(args) -> None:
    out_dir = Path(args.output_dir)
    device = torch.device("cuda")
    model = CellposeModel(
        gpu=True,
        pretrained_model=str(out_dir / "model_final.pth"),
        nchan=3,
        device=device,
        backbone=str(args.backbone),
    )
    records = _load_lightweight_ecc_records(args.dataset_root, "val")
    if args.max_val_images > 0:
        records = records[: args.max_val_images]
    rows = []
    t0 = time.time()
    chunk = 200
    for s in range(0, len(records), chunk):
        rows.extend(
            _predict_rows_for_records_batch(
                model=model,
                records=records[s : s + chunk],
                image_size=int(args.image_size),
                min_area=int(args.min_area),
                device=device,
                score_threshold=0.05,
                mask_threshold=0.5,
                inference_batch_size=int(args.inference_batch),
            )
        )
        print(f"[eval] {min(s + chunk, len(records))}/{len(records)} imgs {time.time() - t0:.0f}s", flush=True)
    results_path = out_dir / "coco_instances_results.json"
    results_path.write_text(json.dumps(rows))
    ann = Path(args.dataset_root) / "annotations" / "instances_val.json"
    coco = COCO(str(ann))
    cocoDt = coco.loadRes(rows)
    ev = COCOeval(coco, cocoDt, "segm")
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    metrics = {
        "model": f"CellPose 3.1.1.1 (official files lane, backbone={args.backbone})",
        "n_images": len(records),
        "n_pred": len(rows),
        **{k: float(v) for k, v in zip("AP AP50 AP75 APs APm APl AR1 AR10 AR100 ARs ARm ARl".split(), ev.stats)},
    }
    (out_dir / "metrics_gisec.json").write_text(json.dumps(metrics, indent=1))
    print("METRICS", json.dumps(metrics), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-area", type=int, default=20)
    ap.add_argument("--inference-batch", type=int, default=4)
    ap.add_argument("--val-test-n", type=int, default=32)
    ap.add_argument("--max-val-images", type=int, default=0)
    ap.add_argument("--max-prepass-train", type=int, default=0)
    ap.add_argument(
        "--backbone",
        choices=["default", "transformer"],
        default="default",
        help="default = 6.6M res-unet; transformer = 92.2M segformer mit_b5+MAnet (MEDIAR)",
    )
    ap.add_argument("--mode", choices=["prepass", "train", "eval", "all"], default="all")
    args = ap.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.mode in ("prepass", "all"):
        prepass(args, "train", args.max_prepass_train)
        prepass(args, "val", max(args.val_test_n, 1))
    if args.mode in ("train", "all"):
        train(args)
    if args.mode in ("eval", "all"):
        evaluate(args)
    _malloc_trim()


if __name__ == "__main__":
    main()
