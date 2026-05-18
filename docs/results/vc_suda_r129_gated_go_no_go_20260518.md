# R129 Gated Go/No-Go Post-Eval Watcher

Date: 2026-05-18
Repo: `/home/hdd3/zhanghaonan/magformer`

R129 is a CPU-only post-eval watcher. It waits until R128 has produced complete R122 iter0099 remaining75 and val28 eval files. It does not start training. It does not start evaluation. It does not use CUDA.

## Active Watcher

Session:

```bash
tmux attach -t r129_gated_go_no_go
```

Log:

```bash
tail -f output/diagnostics/r129_gated_go_no_go_20260518.log
```

Script:

```bash
tools/run_r129_gated_go_no_go.sh
```

Stop command:

```bash
tmux kill-session -t r129_gated_go_no_go
```

## Gate

Every 300 seconds, R129 checks that all required eval files exist and are non-empty:

- `output/diagnostics/r122_depth_boundary_w001_iter0099_remaining75_1024_backmap_topk200_20260518/coco_instances_results.json`
- `output/diagnostics/r122_depth_boundary_w001_iter0099_remaining75_1024_backmap_topk200_20260518/metrics.cocoeval.json`
- `output/diagnostics/r122_depth_boundary_w001_iter0099_val28_1024_backmap_topk200_20260518/coco_instances_results.json`
- `output/diagnostics/r122_depth_boundary_w001_iter0099_val28_1024_backmap_topk200_20260518/metrics.cocoeval.json`

It also checks that no train/eval process is running and that `go_no_go.json` does not already exist.

If any input is missing, it logs the missing path and sleeps. It does not exit on missing eval files.

## Outputs

Candidate bucket compare:

```bash
output/diagnostics/r122_depth_boundary_w001_iter0099_bucket_compare_20260518/bucket_compare.csv
output/diagnostics/r122_depth_boundary_w001_iter0099_bucket_compare_20260518/summary.json
```

Go/no-go report:

```bash
output/diagnostics/r122_depth_boundary_w001_go_no_go_20260518/go_no_go.json
output/diagnostics/r122_depth_boundary_w001_go_no_go_20260518/go_no_go.md
```

R129 will not overwrite an existing candidate `bucket_compare.csv`. If it already exists, R129 runs only the comparator.

## Comparator Gates

R129 runs `tools/compare_r122_go_no_go.py` with:

- baseline CSV: `output/diagnostics/r120_error_atlas_20260518/bucket_compare.csv`
- candidate CSV: `output/diagnostics/r122_depth_boundary_w001_iter0099_bucket_compare_20260518/bucket_compare.csv`
- baseline run: `r114_remaining75`
- candidate run: `r122_remaining75`

The comparator gates are the existing R122 gates:

- tiny area bucket `tiny_area_le_256`: Boundary-F delta, best-IoU delta, and R75 floor.
- high-density bucket `high100`: Boundary-F delta, R75 floor, and FP75 non-increase.
- protection buckets `small_257_1024` and `mid50`: no negative delta on the protected metrics.

Comparator exit `0` means PASS. Exit `1` means a valid FAIL decision and still writes the JSON/Markdown report. Exit `2` means input/schema error.

## Safety

R129 uses:

```bash
CUDA_VISIBLE_DEVICES=""
```

It launches only:

```bash
tools/build_r122_bucket_compare.py
tools/compare_r122_go_no_go.py
```

It never launches `tools/train.py`, `torchrun`, `torch.distributed.run`, or `tools/evaluate_1024_backmap.py`.
