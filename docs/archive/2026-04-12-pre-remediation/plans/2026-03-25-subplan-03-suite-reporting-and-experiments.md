# UCN And MSMFormer Suite Reporting And Experiment Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make sure repaired `ucn` and `msmformer` flow cleanly through runners, rosters, suite summaries, and experiment reports so the canonical paper-facing outputs read from usable baselines.

**Architecture:** Once the two baselines are repaired, update the shell runners and roster/reporting code so canonical model ids map to the repaired outputs, then run short canaries and capture the new evidence in a fresh experiment note.

**Tech Stack:** Shell runners, JSON rosters, summarize/benchmark scripts, pytest, COCOeval output artifacts.

---

### Task 1: Runner Dry-Run Coverage

**Files:**
- Modify: `tests/test_runner_dry_run_metadata_cmd_reproducible.py`
- Modify: `scripts/experiments/run_ecc_20ep_trackp_ucn.sh`
- Modify: `scripts/experiments/run_ecc_20ep_trackp_msmformer.sh`

**Step 1: Write or extend failing tests**

Pin:
- canonical model ids remain `ucn` and `msmformer`
- dry-run output shows the intended config/command fragments
- dataset-root and output-root wiring still work

**Step 2: Run the targeted tests**

Run:
`conda run -n magformer pytest -q tests/test_runner_dry_run_metadata_cmd_reproducible.py -k \"ucn or msmformer\"`

**Step 3: Implement runner updates**

Adjust runner defaults and command strings until the tests pass.

### Task 2: Roster And Summary Integration

**Files:**
- Modify: `configs/experiments/full_20260318_1k_1566_roster.json`
- Modify: `configs/experiments/official_20260318_1k_1566_roster.json` if needed
- Modify: `scripts/experiments/full19_roster.py`
- Modify: `scripts/experiments/official_roster.py` if needed

**Step 1: Keep canonical rows pointing to repaired outputs**

Ensure canonical `ucn` and `msmformer` entries resolve to repaired runner outputs, not obsolete scratch-collapse artifacts.

**Step 2: Verify roster behavior**

Run the relevant roster or suite tests after edits.

### Task 3: Canary Experiment Recording

**Files:**
- Create: `docs/experiments/2026-03-25-ucn-msmformer-repair-canary.md`

**Step 1: Summarize repaired baseline evidence**

Include:
- commands
- output directories
- best metrics
- whether the zero-collapse condition is gone
- residual risks before full paper-scale reruns

### Task 4: Final Small Proof Set

**Files:**
- Test only

**Step 1: Run the focused regression suite**

Run:
`conda run -n magformer pytest -q tests/test_msmformer_imports.py tests/test_ucn_coco_export.py tests/test_runner_dry_run_metadata_cmd_reproducible.py tests/test_benchmark_inference_ucn_fallback.py`

Expected:
- all pass
