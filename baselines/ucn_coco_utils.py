from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils


def encode_binary_mask_rle(mask: np.ndarray) -> Dict[str, Any]:
    """
    Encode a 2D binary mask into COCO RLE (json-serializable).

    Args:
        mask: HxW uint8/bool array. Non-zero is treated as foreground.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask.shape}")

    m = (mask > 0).astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(m))
    # `counts` is bytes; convert to str for JSON.
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def coco_eval_stats(gt_json: str, dt_json: str, iou_type: str) -> Dict[str, float]:
    """
    Run COCOeval and return the standard 12 stats as a dict.
    """
    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]

    # Handle empty predictions gracefully
    with open(dt_json, "r") as f:
        dets = json.load(f)
    if len(dets) == 0:
        return {k: 0.0 for k in keys}

    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(dt_json)
    ev = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    s = [float(x) for x in list(ev.stats)]
    return dict(zip(keys, s))


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
