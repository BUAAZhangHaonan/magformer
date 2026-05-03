# Runner And Roster Wiring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the full19 experiment orchestration consume the derived 512/256 datasets and dataset-root-native stats without stale 1024 assumptions.

**Architecture:** Introduce a multires roster/track overlay rather than cloning a whole new suite by hand. Parameterize runner scripts and helpers where they still hard-code image size or resolution-specific paths, and keep custom-root stat lookup centralized in `ecc_common.sh` and Python helper modules.

**Tech Stack:** Bash, Python, pytest, JSON roster manifests.

---

### Task 1: Multires roster tests

**Files:**
- Create: `tests/test_multires_roster_overlay.py`
- Modify: `scripts/experiments/full19_roster.py`

**Step 1: Write the failing test**

Cover:

- `full19_roster.py` can render commands for a multires track
- output model ids stay canonical
- dataset root and image size flow through the rendered commands

**Step 2: Run test to verify it fails**

Run:
`conda run -n magformer pytest -q tests/test_multires_roster_overlay.py`

Expected:
- fail because multires overlay support does not exist yet

**Step 3: Write minimal implementation**

Add roster helpers so a track can override:

- dataset id/register
- dataset root
- output root
- default image size per model family where needed

**Step 4: Run test to verify it passes**

Run:
`conda run -n magformer pytest -q tests/test_multires_roster_overlay.py`

Expected:
- pass

### Task 2: Runner dry-run coverage for hard-coded families

**Files:**
- Modify: `scripts/experiments/ecc_common.sh`
- Modify: `scripts/experiments/run_0831_1k_20ep_1024_revisit_unet_semantic_inst.sh`
- Modify: `scripts/experiments/run_0831_1k_20ep_1024_revisit_unet_boundary_inst.sh`
- Modify: `scripts/experiments/run_0831_1k_20ep_1024_revisit_unetpp_boundary_inst.sh`
- Modify: `scripts/experiments/run_0831_1k_20ep_1024_revisit_magformer.sh`
- Modify: `scripts/experiments/run_0831_1k_20ep_1024_lightdepth_stage_a_magformer.sh`
- Modify: `scripts/experiments/run_0831_1k_20ep_1024_revisit_mgm_mask2former.sh`
- Modify: `scripts/experiments/run_0831_1k_20ep_scratch_maskrcnn.sh`
- Modify: `scripts/experiments/run_0831_1k_20ep_scratch_yolov8_seg.sh`
- Modify: `scripts/experiments/run_0831_1k_20ep_scratch_uoais.sh`
- Modify: `scripts/experiments/run_0831_1k_20ep_scratch_msmformer.sh`
- Modify: `scripts/experiments/run_0831_1k_20ep_scratch_ucn.sh`
- Test: `tests/test_runner_dry_run_metadata_cmd_reproducible.py`

**Step 1: Write the failing test**

Extend dry-run coverage for representative multires commands:

- one Detectron2 family runner
- one U-Net runner
- one MagFormer runner
- `ucn`
- `msmformer`

**Step 2: Run test to verify it fails**

Run:
`conda run -n magformer pytest -q tests/test_runner_dry_run_metadata_cmd_reproducible.py -k 'image_size or multires or 256 or 512'`

Expected:
- fail where image size or dataset-root-native stats are still hard-coded

**Step 3: Write minimal implementation**

Parameterize the runners so they:

- accept explicit image size where missing
- read stats from the provided dataset root
- avoid stale 1024-only strings in canonical commands

**Step 4: Run test to verify it passes**

Run:
`conda run -n magformer pytest -q tests/test_runner_dry_run_metadata_cmd_reproducible.py -k 'ucn or msmformer or multires or image_size'`

Expected:
- pass

### Task 3: Suite entrypoint

**Files:**
- Create: `scripts/experiments/run_20260326_20260318_1k_1566_multires_full19.sh`
- Modify: `scripts/experiments/full19_roster.py`
- Test: `tests/test_20260321_full19_suite_script.py`

**Step 1: Write the failing test**

Cover:

- new suite entrypoint can render 512 and 256 runs without renaming models
- staging and summary behavior still follow the full19 conventions

**Step 2: Run test to verify it fails**

Run:
`conda run -n magformer pytest -q tests/test_20260321_full19_suite_script.py -k 'full19 or roster'`

Expected:
- fail for the new multires suite path until implemented

**Step 3: Write minimal implementation**

Create a multires suite wrapper around the existing full19 orchestration with track-specific defaults.

**Step 4: Run test to verify it passes**

Run:
`conda run -n magformer pytest -q tests/test_20260321_full19_suite_script.py -k 'full19 or roster'`

Expected:
- pass
