#!/usr/bin/env python3.11
"""CPU-only dataset pipeline profiler (no GPU touched).

How much wall time does one train sample's full pipeline cost, with vs
without copy-paste, under the exact B1 config? Supply-side half of the
2026-09-21 perf incident (EVIDENCE.md §15).
"""
import sys
import time
import statistics

sys.path.insert(0, "/home/hdd3/zhanghaonan/magformer/experiments_archive/v317_next_stage_2026-07/source")

from magformer.config.loader import load_config  # noqa: E402
from magformer.data.dataset import CocoRgbdDataset  # noqa: E402

CFG = "/home/hdd3/zhanghaonan/magformer/experiments_archive/v317_next_stage_2026-07/source/configs/aps_20260913_full/g1_b1_merged_16k.yaml"
N_WARM, N_MEASURE = 10, 120


def build(cp_enabled: bool):
    cfg = load_config(CFG)
    d = cfg.data
    cp = None
    if cp_enabled:
        cp = d.copy_paste.model_dump()
        cp["enabled"] = True
    t0 = time.perf_counter()
    ds = CocoRgbdDataset(
        dataset_root=d.dataset_root,
        ann_file=d.train_ann,
        split=d.train_split,
        transform=None,
        is_train=True,
        copy_paste_config=cp,
    )
    return ds, time.perf_counter() - t0


def time_dataset(ds, label):
    for i in range(N_WARM):
        ds[i]
    times = []
    for i in range(N_WARM, N_WARM + N_MEASURE):
        t0 = time.perf_counter()
        _ = ds[i]
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    mean = statistics.mean(times)
    print(f"[{label}] n={len(times)} mean={mean:.1f}ms "
          f"p50={times[len(times)//2]:.1f} p90={times[int(len(times)*0.9)]:.1f} "
          f"max={times[-1]:.1f}")
    return mean


def main():
    ds_on, t_build = build(True)
    print(f"paste-ON dataset built in {t_build:.1f}s (incl. bank prefill)")
    on_mean = time_dataset(ds_on, "paste ON ")
    del ds_on
    ds_off, _ = build(False)
    off_mean = time_dataset(ds_off, "paste OFF")
    del ds_off
    print(f"\npaste delta = {on_mean - off_mean:.1f}ms/sample")
    for workers in (4, 8, 12):
        sup = workers * 1000.0 / max(on_mean, 0.001)
        print(f"  workers={workers}: supply ~{sup:.2f} img/s/rank")


if __name__ == "__main__":
    main()
