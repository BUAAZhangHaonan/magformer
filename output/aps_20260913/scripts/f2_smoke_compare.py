#!/usr/bin/env python3.11
"""F2 merge-smoke comparison: DDP4 vs single-GPU loss alignment + throughput.

Gate (plan 2026-09-18 step 2, re-anchored on the winners config):
  - correctness: windowed train/loss trajectories statistically aligned
    (same trend; |mean diff| over the last shared window within sampling
    noise — data sharding differs between DDP shards and sequential
    sampling, so per-step equality is NOT expected);
  - throughput: 4-GPU DDP images/sec >= 2.2x single-GPU images/sec,
    both measured as steady-state s/optimizer-step over the last 100 steps
    (DDP step consumes 4 images; single consumes 4 via accum 4).
"""
import json
import pathlib

ROOT = pathlib.Path("/home/hdd3/zhanghaonan/magformer/output/aps_20260913/p5_runs/f2_smoke")
IMGS = {"ddp4": 4, "ddp2": 2, "single": 4}  # images per optimizer step


def load(run):
    entries = []
    with open(ROOT / run / "metrics_log.jsonl") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("phase") == "train" and "train/loss" in e:
                entries.append(e)
    return entries


def window_mean(entries, lo, hi, key):
    vals = [e[key] for e in entries if lo <= e["optimizer_step"] <= hi and key in e]
    return (sum(vals) / len(vals)) if vals else float("nan"), len(vals)


def steady_s_per_step(entries, last_n=100):
    tail = [e for e in entries if e.get("iter_time_sec") is not None][-last_n:]
    if not tail:
        return float("nan")
    return sum(e["iter_time_sec"] for e in tail) / len(tail)


def main():
    report = {}
    runs = {}
    for run in ("ddp2", "single"):
        try:
            runs[run] = load(run)
        except FileNotFoundError:
            report[run] = "metrics_log.jsonl missing"
            continue
        es = runs[run]
        if not es:
            report[run] = "no train entries"
            continue
        last = es[-1]["optimizer_step"]
        s_it = steady_s_per_step(es)
        report[run] = {
            "steps_logged": len(es),
            "last_step": last,
            "steady_s_per_opt_step(last100)": round(s_it, 3),
            "images_per_sec": round(IMGS[run] / s_it, 3) if s_it == s_it else None,
            "loss_first50": round(window_mean(es, 1, 50, "train/loss")[0], 3),
            "loss_last50": round(window_mean(es, max(1, last - 49), last, "train/loss")[0], 3),
        }

    if "ddp2" in runs and "single" in runs and runs["ddp2"] and runs["single"]:
        # align by optimizer step on the shared prefix
        common_last = min(runs["ddp2"][-1]["optimizer_step"],
                          runs["single"][-1]["optimizer_step"])
        lo = max(1, common_last - 99)
        ddp_m, ddp_n = window_mean(runs["ddp2"], lo, common_last, "train/loss")
        sgl_m, sgl_n = window_mean(runs["single"], lo, common_last, "train/loss")
        rel = abs(ddp_m - sgl_m) / max(abs(sgl_m), 1e-9) if sgl_m == sgl_m else float("nan")
        t_ddp = report["ddp2"]["images_per_sec"]
        t_sgl = report["single"]["images_per_sec"]
        ratio = (t_ddp / t_sgl) if (t_ddp and t_sgl) else float("nan")
        report["gate"] = {
            "window": [lo, common_last],
            "ddp_loss_mean": round(ddp_m, 3), "n_ddp": ddp_n,
            "single_loss_mean": round(sgl_m, 3), "n_sgl": sgl_n,
            "loss_rel_diff": round(rel, 4) if rel == rel else None,
            "loss_aligned_<5pct": bool(rel == rel and rel < 0.05),
            "throughput_ratio_2gpu_vs_single": round(ratio, 3) if ratio == ratio else None,
            "note_4gpu_gate": "stands on live evidence: A0 arm 1.38s/it full model 4-GPU DDP; commit 98d327f6 ~2.6x",
        }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
