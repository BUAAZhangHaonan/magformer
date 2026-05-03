# UCN Canonical Baseline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Repair the canonical `ucn` baseline so it reflects the original UCN instance-segmentation recipe while remaining compatible with ECC datasets and COCO evaluation.

**Architecture:** Keep the existing ECC wrapper entrypoint, but re-align its effective training configuration, preprocessing, and clustering defaults to the vendored UCN implementation. Validate this with tests that pin canonical model ids, dataset normalization, upstream-like hyperparameters, and non-broken COCO export behavior.

**Tech Stack:** PyTorch, vendored UnseenObjectClustering, ECC COCO RGB-D datasets, pytest, shell runners.

---

### Task 1: Lock Canonical UCN Behavior With Tests

**Files:**
- Create: `tests/test_ucn_recipe_alignment.py`
- Modify: `tests/test_runner_dry_run_metadata_cmd_reproducible.py`

**Step 1: Write the failing test**

Add tests that assert:
- canonical runner/model id remains `ucn`
- `run_ucn_ecc.py` defaults match the intended original-style recipe for optimizer/lr/num-units/cluster params
- dataset-specific RGB/depth normalization helpers are used

**Step 2: Run test to verify it fails**

Run:
`conda run -n magformer pytest -q tests/test_ucn_recipe_alignment.py tests/test_runner_dry_run_metadata_cmd_reproducible.py -k ucn`

Expected:
- fail because the current wrapper still encodes the simplified ECC recipe

**Step 3: Write minimal implementation**

Modify the UCN wrapper and runner defaults until the tests encode the canonical recipe.

**Step 4: Run test to verify it passes**

Run the same command again and confirm it passes.

### Task 2: Repair Training Recipe And Preprocessing

**Files:**
- Modify: `baselines/run_ucn_ecc.py`
- Modify: `baselines/run_ucn_0831_1k.py`
- Modify: `scripts/experiments/run_ecc_20ep_trackp_ucn.sh`
- Modify: `scripts/experiments/run_0831_1k_20ep_scratch_ucn.sh`

**Step 1: Align UCN config with the vendored upstream recipe**

Use the vendored config family under:
- `baselines/unseen_object_clustering/experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml`

Carry over the meaningful upstream choices:
- embedding metric / normalization
- optimizer family and LR scale
- weight decay
- RGBD fusion mode
- `NUM_UNITS`

**Step 2: Preserve ECC-correct normalization**

Ensure the wrapper still uses:
- dataset-specific BGR mean stats
- dataset-specific depth clipping stats

**Step 3: Keep the task as instance segmentation**

Do not replace clustering with a foreground-mask shortcut.
The output path must remain:
- embedding network
- mean-shift clustering
- instance mask export

### Task 3: Verify UCN Exports Real Instances

**Files:**
- Modify: `tests/test_ucn_coco_export.py`
- Create: `tests/test_ucn_instance_outputs.py`

**Step 1: Write a behavioral test**

Add a test that constructs a tiny synthetic cluster map or mock output and verifies the export path keeps distinct instances rather than collapsing to one foreground blob.

**Step 2: Run it to fail if needed**

Run:
`conda run -n magformer pytest -q tests/test_ucn_coco_export.py tests/test_ucn_instance_outputs.py`

**Step 3: Implement the minimal fix**

Adjust the cluster-to-instance conversion only if the test reveals an actual bug.

**Step 4: Re-run verification**

The same command must pass.

### Task 4: UCN Canary

**Files:**
- Runtime only: output dir under `output/experiments/.../ucn`

**Step 1: Run a short canary**

Use the canonical UCN runner on the active ECC dataset with a short budget.

**Step 2: Check artifacts**

Confirm:
- `coco_instances_results.json` exists
- instance count is non-zero
- AP is above the old collapsed baseline

**Step 3: Record findings**

Write the canary result into the experiment note referenced by the master plan.
