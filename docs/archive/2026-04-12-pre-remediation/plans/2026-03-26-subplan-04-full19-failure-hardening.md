# Full19 Failure Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the current MGM multires import failure and harden the `full19` suite so one broken baseline cannot silently kill the whole GPU 0 queue.

**Architecture:** Restore the missing `MGM_Mask2Former` dataset registration package so the current `mgm_mask2former_*` runners can import and train again. Then update the `run_20260321_20260318_1k_1566_full19.sh` suite runner to record per-model failures, continue with later models, and still emit summaries/artifacts for completed runs.

**Tech Stack:** Bash suite runners, Python baseline training entrypoints, pytest, Detectron2/Mask2Former-style dataset registration.

---

### Task 1: Document And Reproduce The MGM Import Failure

**Files:**
- Modify: `tests/test_mgm_repo_pathing.py`
- Reference: `baselines/MGM_Mask2Former/train_net_mgm_0831.py`
- Reference: `baselines/MGM_Mask2Former/mask2former/__init__.py`

**Step 1: Write the failing test**

Add a test that asserts the repo-local `MGM_Mask2Former` baseline contains `mask2former/data/datasets/__init__.py` and `mask2former/data/datasets/register_coco_rgbd_instance.py`.

**Step 2: Run test to verify it fails**

Run: `conda run -n magformer pytest -q tests/test_mgm_repo_pathing.py -k dataset_package`

Expected: FAIL because the worktree baseline currently lacks the `datasets` package.

**Step 3: Write minimal implementation**

Restore the missing package under `baselines/MGM_Mask2Former/mask2former/data/datasets/` with the repo-local RGB-D registration helper.

**Step 4: Run test to verify it passes**

Run: `conda run -n magformer pytest -q tests/test_mgm_repo_pathing.py -k dataset_package`

Expected: PASS.

### Task 2: Verify MGM Registration Imports Cleanly

**Files:**
- Modify: `tests/test_mgm_repo_pathing.py`
- Modify: `baselines/MGM_Mask2Former/mask2former/data/datasets/__init__.py`
- Modify: `baselines/MGM_Mask2Former/mask2former/data/datasets/register_coco_rgbd_instance.py`

**Step 1: Write the failing test**

Add a test that imports `mask2former.data.datasets.register_coco_rgbd_instance` from the repo-local baseline path without triggering the previous partial-initialization error.

**Step 2: Run test to verify it fails**

Run: `conda run -n magformer pytest -q tests/test_mgm_repo_pathing.py -k import_cleanly`

Expected: FAIL before the package restore is in place.

**Step 3: Write minimal implementation**

Keep the package import surface minimal and explicit so `mask2former.data` can import `datasets` and `train_net_mgm_0831.py` can import `register_all_coco_rgbd`.

**Step 4: Run test to verify it passes**

Run: `conda run -n magformer pytest -q tests/test_mgm_repo_pathing.py -k import_cleanly`

Expected: PASS.

### Task 3: Make Full19 Continue After A Single Model Failure

**Files:**
- Modify: `tests/test_20260321_full19_suite_script.py`
- Modify: `scripts/experiments/run_20260321_20260318_1k_1566_full19.sh`

**Step 1: Write the failing test**

Extend the fake suite fixture so one model command exits non-zero, then add a test asserting:
- the suite still runs the later model,
- the failed model is recorded to a failure manifest,
- completed models are archived normally,
- the overall script exits non-zero after finishing the loop.

**Step 2: Run test to verify it fails**

Run: `conda run -n magformer pytest -q tests/test_20260321_full19_suite_script.py -k continue_after_failure`

Expected: FAIL because the current suite exits immediately on the first broken model.

**Step 3: Write minimal implementation**

Update the suite loop to:
- capture per-model command failures without aborting the whole loop,
- append structured failure records to an output file,
- skip archive for failed models,
- keep going with later models,
- return non-zero after summaries/logging when any failures were seen.

**Step 4: Run test to verify it passes**

Run: `conda run -n magformer pytest -q tests/test_20260321_full19_suite_script.py -k continue_after_failure`

Expected: PASS.

### Task 4: Protect Summary Generation When Some Models Fail

**Files:**
- Modify: `tests/test_20260321_full19_suite_script.py`
- Modify: `scripts/experiments/run_20260321_20260318_1k_1566_full19.sh`

**Step 1: Write the failing test**

Add a suite test asserting that summary/extended-metrics generation still runs for completed outputs even when one model failed earlier in the loop.

**Step 2: Run test to verify it fails**

Run: `conda run -n magformer pytest -q tests/test_20260321_full19_suite_script.py -k writes_summary_after_failure`

Expected: FAIL until the suite is taught to continue through failures.

**Step 3: Write minimal implementation**

Keep the existing summary/benchmark/visualization calls, but ensure the failure status is delayed until after those shared reporting steps complete.

**Step 4: Run test to verify it passes**

Run: `conda run -n magformer pytest -q tests/test_20260321_full19_suite_script.py -k writes_summary_after_failure`

Expected: PASS.

### Task 5: Run The Targeted Regression Suite

**Files:**
- Test: `tests/test_mgm_repo_pathing.py`
- Test: `tests/test_20260321_full19_suite_script.py`
- Test: `tests/test_multires_roster_overlay.py`

**Step 1: Run focused verification**

Run:

```bash
conda run -n magformer pytest -q \
  tests/test_mgm_repo_pathing.py \
  tests/test_20260321_full19_suite_script.py \
  tests/test_multires_roster_overlay.py
```

Expected: PASS for the targeted queue and MGM regression coverage.

### Task 6: Resume GPU 0 Work Safely

**Files:**
- Modify if needed: `scripts/experiments/run_20260321_20260318_1k_1566_full19.sh`
- Operate on: `output/experiments/20260326_repaired_pretrained_20260318_1k_1566_20ep_256_full19`

**Step 1: Verify current failure point**

Run:

```bash
tail -n 40 output/experiments/20260326_repaired_pretrained_20260318_1k_1566_20ep_256_full19/run_all.log
```

Expected: the current `mgm_mask2former_nodpth_ref` import failure is still the latest blocker.

**Step 2: Relaunch the suite on GPU 0**

Run the existing `256 -> 512` queue command on `CUDA_VISIBLE_DEVICES=0` after the fixes land.

**Step 3: Verify it advances past the old failure**

Run:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
tail -n 60 output/experiments/20260326_repaired_pretrained_20260318_1k_1566_20ep_256_full19/run_all.log
```

Expected: `GPU 0` active again and the suite moves beyond `mgm_mask2former_nodpth_ref`.
