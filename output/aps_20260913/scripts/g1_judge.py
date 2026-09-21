#!/usr/bin/env python3.11
"""G1 preregistered judgement: B1(merged winners) vs A0(control) 16K arms.

Gates (preregistered 2026-09-21, docs/2026-09-21-p3p4-campaign.md):
  MAIN  : B1 - A0 >= +1.5pt val/segm_APs at the final (16K) eval
  G1    : <64 GT-bucket TP score median >= 0.40 (seal baseline 0.29),
          64-256 >= 0.68 (baseline 0.62)
  G2    : FP no-ride-along — <64 det-pool hard-FP score median <= 0.10 AND
          hard-FP(>0.5) count <= 150 AND neither worse than A0's own panel
Panel basis: first 1200 val images (same as subbucket_scores.py).
"""
import io
import json
import contextlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

RUNS = Path("/home/hdd3/zhanghaonan/magformer/output/aps_20260913/g1_runs")
GT = ("/home/hdd3/zhanghaonan/magformer/magformer_datasets/20260318_1K_32254/"
      "annotations/instances_val.validated.json")
KEYS = ("val/segm_AP", "val/segm_APs", "val/segm_APm", "val/segm_APl")
GATE_DELTA_APS = 0.015
N_PANEL = 1200


def val_entries(run_dir):
    out = {}
    p = run_dir / "metrics_log.jsonl"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or '"phase": "val"' not in line:
            continue
        e = json.loads(line)
        out[int(e["optimizer_step"])] = {k: e[k] for k in KEYS if k in e}
    return out


def paired_table():
    a0 = val_entries(RUNS / "g1_a0_control_16k")
    b1 = val_entries(RUNS / "g1_b1_merged_16k")
    rows = []
    for step in sorted(set(a0) & set(b1)):
        r = {"step": step}
        for k in KEYS:
            r[f"A0_{k.split('/')[-1]}"] = round(a0[step][k], 4)
            r[f"B1_{k.split('/')[-1]}"] = round(b1[step][k], 4)
        r["dAPs_pt"] = round((b1[step]["val/segm_APs"] - a0[step]["val/segm_APs"]) * 100, 2)
        r["dAP_pt"] = round((b1[step]["val/segm_AP"] - a0[step]["val/segm_AP"]) * 100, 2)
        rows.append(r)
    return rows, bool(a0) and bool(b1)


def panel(dets_path):
    """Sub-bucket TP scores + <64 det-pool hard-FP stats (score-order greedy)."""
    dets = json.loads(Path(dets_path).read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(GT)
    gt_by_img = defaultdict(list)
    for a in coco.dataset["annotations"]:
        if a["area"] < 1024:
            gt_by_img[a["image_id"]].append(a)
    dets_by_img = defaultdict(list)
    for d in dets:
        dets_by_img[d["image_id"]].append(d)

    buckets = defaultdict(lambda: [0, 0, []])       # GT-bucket -> n, matched, tp_scores
    fp_scores = []                                   # <64 det-pool unmatched scores
    n_img = 0
    for img_id, anns in gt_by_img.items():
        if n_img >= N_PANEL:
            break
        n_img += 1
        dds = sorted(dets_by_img.get(img_id, []), key=lambda d: -d["score"])
        dm = []
        for d in dds:
            s = d["segmentation"]
            rle = {"size": s["size"],
                   "counts": s["counts"].encode() if isinstance(s["counts"], str) else s["counts"]}
            m = maskUtils.decode(rle)
            dm.append((d["score"], m, maskUtils.area(rle), []))
        # greedy TP assignment by score order against all <1024 GTs
        gts = []
        for a in anns:
            g = maskUtils.decode(maskUtils.merge(
                maskUtils.frPyObjects(a["segmentation"], 1024, 1024)))
            gts.append((a["area"], g, False))
        for sc, m, area, taken in dm:
            if area >= 1024:
                continue  # panel tracks the small det pool only
            best, bi = 0.0, -1
            for i, (ga, g, used) in enumerate(gts):
                if used:
                    continue
                inter = (g & m).sum()
                union = (g | m).sum()
                iou = inter / union if union > 0 else 0
                if iou > best:
                    best, bi = iou, i
            if best >= 0.5:
                ga, g, _ = gts[bi]
                gts[bi] = (ga, g, True)
                b = "<64" if ga < 64 else "64-256" if ga < 256 else "256-1024"
                buckets[b][2].append(sc)
            elif area < 64:
                fp_scores.append(sc)
    # GT-bucket TP panels
    out_buckets = {}
    for b in ("<64", "64-256", "256-1024"):
        tps = buckets[b][2] if b in buckets else []
        out_buckets[b] = {"tp_n": len(tps),
                          "tp_score_median": float(np.median(tps)) if tps else None}
    fp_scores = np.array(sorted(fp_scores)) if fp_scores else np.array([])
    out_buckets["<64"]["fp_n"] = int(fp_scores.size)
    out_buckets["<64"]["fp_score_median"] = float(np.median(fp_scores)) if fp_scores.size else None
    out_buckets["<64"]["fp_gt05_count"] = int((fp_scores > 0.5).sum()) if fp_scores.size else 0
    return {"n_images": n_img, "buckets": out_buckets}


def main():
    report = {}
    rows, both = paired_table()
    report["paired"] = rows
    final = rows[-1] if rows else None
    if final:
        report["main_gate"] = {
            "final_step": final["step"],
            "delta_APs_pt": final["dAPs_pt"],
            "threshold_pt": GATE_DELTA_APS * 100,
            "PASS": bool(final["dAPs_pt"] >= GATE_DELTA_APS * 100),
        }
    panels = {}
    for arm, d in (("A0", RUNS / "g1_a0_control_16k" / "coco_instances_results.json"),
                   ("B1", RUNS / "g1_b1_merged_16k" / "coco_instances_results.json")):
        if d.exists():
            try:
                panels[arm] = panel(d)
            except Exception as exc:  # noqa: BLE001 - report, don't crash the table
                panels[arm] = {"error": repr(exc)}
    report["panels"] = panels
    if "A0" in panels and "B1" in panels and "buckets" in panels.get("B1", {}):
        b1s, a0s = panels["B1"]["buckets"], panels["A0"].get("buckets", {})
        report["gates"] = {
            "G1_tp_med_<64>=" + "0.40": b1s.get("<64", {}).get("tp_score_median"),
            "G1_tp_med_64-256>=0.68": b1s.get("64-256", {}).get("tp_score_median"),
            "G2_fp_med<=0.10": b1s.get("<64", {}).get("fp_score_median"),
            "G2_fp_gt05<=150": b1s.get("<64", {}).get("fp_gt05_count"),
            "G2_no_ridealong_vs_A0": {
                "fp_med_A0": a0s.get("<64", {}).get("fp_score_median"),
                "fp_gt05_A0": a0s.get("<64", {}).get("fp_gt05_count"),
            },
        }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
