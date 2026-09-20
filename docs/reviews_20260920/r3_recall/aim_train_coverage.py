#!/usr/bin/env python3
"""AIM sel[:16] coverage on the TRAIN split, with scale augmentation.

matcher trigger: canvas area (1024^2, after ResizeScale s in [0.5,2]) < 4096.
sel keeps the FIRST 16 such GTs (annotation order); the rest fall back to the
uniform-point noisy column. Only per-annotation areas are needed.

Outputs aim_train_coverage.json.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = ("/home/hdd3/zhanghaonan/magformer/magformer_datasets/20260318_1K_32254/"
         "annotations/instances_train.validated.json")

print("loading train anns ...", flush=True)
data = json.load(open(TRAIN))
areas_by_img = {}
for a in data["annotations"]:
    areas_by_img.setdefault(a["image_id"], []).append(float(a["area"]))
print(f"{len(data['images'])} images, {len(data['annotations'])} anns", flush=True)


def coverage(thr_canvas):
    """thr_canvas: AIM threshold in ORIGINAL px^2 for a given aug scale."""
    total = capped = 0
    small_total = small_capped = 0
    n_img_cap = 0
    for _img, areas in areas_by_img.items():
        elig = [x for x in areas if 0 < x < thr_canvas]
        total += len(elig)
        if len(elig) > 16:
            n_img_cap += 1
            capped += len(elig) - 16
            small_capped += sum(1 for x in elig[16:] if x < 1024)
        small_total += sum(1 for x in areas if 0 < x < 1024)
    return {"n_img": len(areas_by_img), "n_img_cap": n_img_cap,
            "lt4096_total": total, "lt4096_capped": capped,
            "coco_small_total": small_total, "coco_small_capped": small_capped,
            "frac_small_no_aim": small_capped / max(small_total, 1)}


out = {}
# scale=1 exact
out["scale_1.0"] = coverage(4096)
# scale-augmented expectation: s ~ U(0.5, 2.0), area' = area * s^2
# AIM applies iff area * s^2 < 4096  <=>  area < 4096/s^2  (per-sample threshold)
import numpy as np
s_grid = np.linspace(0.5, 2.0, 16)
acc_small_capped = acc_small_total = 0.0
acc_cap_img = 0.0
for s in s_grid:
    c = coverage(4096 / s ** 2)
    acc_small_capped += c["coco_small_capped"]
    acc_small_total += c["coco_small_total"]
    acc_cap_img += c["n_img_cap"]
n = len(s_grid)
out["scale_aug_avg(0.5..2)"] = {
    "frac_small_no_aim": acc_small_capped / max(acc_small_total, 1),
    "n_img_cap_avg": acc_cap_img / n,
    "note": "average over 16-point s grid; per-sample crop may drop GTs",
}
# also: for reference, what original-area range can AIM ever cover at s<=2
out["note"] = ("canvas_area = orig_area * s^2; AIM iff canvas < 4096. "
               "COCO-small (orig<1024) is under threshold for all s<=2 "
               "(1024*4=4096 boundary), so the only coverage gaps are the "
               "sel[:16] cap and crop-dropped GTs.")
with open(os.path.join(HERE, "aim_train_coverage.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
