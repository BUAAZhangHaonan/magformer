import json, os, sys, time
import numpy as np
DET = "/home/hdd3/zhanghaonan/magformer/output/aps_20260913/p5_runs/f1_full_design_256k/eval_snapshots/2001_0920/dets.json"
t0=time.time()
dets = json.load(open(DET))
print("n_dets", len(dets), "load_s", round(time.time()-t0,1))
imgs = set(d["image_id"] for d in dets)
print("n_images_with_dets", len(imgs))
sc = np.array([d["score"] for d in dets])
print("score min/max/p25/p50/p75/p90:", sc.min(), sc.max(), *np.percentile(sc,[25,50,75,90]).round(4))
per_img = {}
for d in dets: per_img[d["image_id"]] = per_img.get(d["image_id"],0)+1
cnt = np.array(list(per_img.values()))
print("dets/img min/med/p90/max:", cnt.min(), np.median(cnt), np.percentile(cnt,90), cnt.max())
print("keys of det0:", sorted(dets[0].keys()))
print("cat ids:", set(d["category_id"] for d in dets))
# quick area sample from RLE without decode: use bbox? we have bbox
a = np.array([d["bbox"][2]*d["bbox"][3] for d in dets])
print("bbox-area<64:", (a<64).sum(), "64-256:", ((a>=64)&(a<256)).sum(), "256-1024:", ((a>=256)&(a<1024)).sum())
print("sample det0:", {k:dets[0][k] for k in ("image_id","category_id","score","bbox")}, "segm size", dets[0]["segmentation"]["size"])
