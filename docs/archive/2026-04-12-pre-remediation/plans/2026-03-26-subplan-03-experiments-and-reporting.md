# Experiments And Reporting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Verify the multires pipeline end to end, keep the current 1024 run alive, and launch new 512 work on GPU 0 with results ready for later table aggregation. The `256` branch was planned but cancelled.

**Architecture:** Treat experiments as a separate node after the builder and runner wiring are stable. Generate the derived datasets once, sanity-check the manifests, then launch new runs through the suite or selected baseline commands. Keep result aggregation compatible with the existing summary and extended-metrics scripts.

**Tech Stack:** Bash, Conda, Python experiment utilities, pytest, GPU scheduling helpers.

---

### Task 1: Derived dataset smoke validation

**Files:**
- Runtime only: `magformer_datasets/20260318_1K_1566_512`
- Runtime only: `magformer_datasets/20260318_1K_1566_256` (archival only; `256` was cancelled)

**Step 1: Validate dataset stats and manifests**

Run:
`conda run -n magformer python scripts/analysis/ensure_dataset_stats.py --dataset-root /home/team/zhanghaonan/magformer/magformer_datasets/20260318_1K_1566_512`

Expected:
- stats manifest points at the derived dataset cache

**Step 2: Validate benchmark/reporting compatibility**

Run:
`conda run -n magformer python scripts/analysis/benchmark_inference_suite.py --output-root <smoke-output> --dataset-root /home/team/zhanghaonan/magformer/magformer_datasets/20260318_1K_1566_512 --summary <smoke-output>/summary.json --continue-on-error`

Expected:
- script resolves the derived dataset root without annotation-path errors

### Task 2: GPU 0 experiment launch

**Files:**
- Runtime only: `output/experiments/20260326_repaired_pretrained_20260318_1k_1566_20ep_512_full19`
- Runtime only: `output/experiments/20260326_repaired_pretrained_20260318_1k_1566_20ep_256_full19` (historical; `256` was cancelled)

**Step 1: Launch a representative 512 smoke job on GPU 0**

Use a single model first, then expand to the suite after dry-run verification.

Success criteria:

- command uses GPU 0 only
- run metadata records the derived dataset root and image size
- training starts without missing-cache or missing-stats failures

**Step 2: Launch the suite or a staged subset**

Prefer starting with the heaviest/highest-risk families:

- `msmformer`
- `ucn`
- one Detectron2 family
- one U-Net family

Then expand to the full roster.

**Step 3: Keep the 1024 `msmformer` run intact**

Monitor only; do not terminate or reconfigure it.

### Task 3: Reporting integration

**Files:**
- Runtime only: `output/experiments/.../extended_metrics_table.{md,csv,json}`
- Create: `docs/experiments/2026-03-26-full19-multires-progress.md`

**Step 1: Update tables after the first successful multires runs**

Run:
`conda run -n magformer python scripts/analysis/write_extended_metrics_table.py --summary <summary-json> --out-json <out>.json --out-csv <out>.csv --out-md <out>.md`

Expected:
- tables include canonical model ids for the new track

**Step 2: Capture progress note**

Record:

- dataset roots
- stats cache ids
- launched commands
- completed models
- residual blockers

**Step 3: Final proof set**

Run:
`conda run -n magformer pytest -q tests/test_multires_dataset_builder.py tests/test_multires_roster_overlay.py tests/test_20260321_full19_suite_script.py tests/test_runner_dry_run_metadata_cmd_reproducible.py`

Expected:
- pass
