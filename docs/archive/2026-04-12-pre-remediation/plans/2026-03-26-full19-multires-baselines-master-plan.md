# Full19 Multiresolution Baselines Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add offline `512x512` dataset variants, per-dataset stats/cache generation, and full19 baseline runner support so every baseline can run fair multiresolution ECC instance-segmentation experiments efficiently. The `256x256` track was planned but later cancelled.

**Architecture:** Build the work in three layers. First, add a deterministic dataset derivation and cache builder that produces resized RGB/depth assets, rewritten COCO annotations, and per-dataset statistics. Second, wire full19 runners, roster generation, and shared shell helpers so every baseline consumes dataset-root-native stats and image geometry instead of stale 1024 assumptions. Third, verify the new path with targeted tests and launch new experiments on GPU 0 while leaving the ongoing 1024 `msmformer` run untouched.

**Tech Stack:** Python, Bash, PyTorch, Detectron2, vendored UCN/MSMFormer, Ultralytics YOLOv8, COCO annotations, NumPy, OpenCV, pytest.

---

## Node Order

1. Node A: planning docs and failing tests
2. Node B: derived dataset and shared cache builder
3. Node C: runner and roster integration for full19
4. Node D: verification, cleanup, and experiment launch

## Required Execution Rules

- No production code without a failing test first.
- Do not stop or modify the active 1024 `msmformer` training job.
- Keep multires work in the existing isolated worktree.
- All new multires paths must work from explicit dataset roots, not hidden register-only shortcuts.
- Avoid temporary junk files; if a test/helper script is only transitional, remove it before handoff.

## Subplans

- [2026-03-26 Derived Dataset And Cache Subplan](/home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/docs/plans/2026-03-26-subplan-01-derived-datasets-and-shared-cache.md)
- [2026-03-26 Runner And Roster Wiring Subplan](/home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/docs/plans/2026-03-26-subplan-02-runner-and-roster-wiring.md)
- [2026-03-26 Experiments And Reporting Subplan](/home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/docs/plans/2026-03-26-subplan-03-experiments-and-reporting.md)

## Node A: Planning And Safety Net

**Files:**
- Create: `docs/plans/2026-03-26-full19-multires-baselines-design.md`
- Create: `docs/plans/2026-03-26-full19-multires-baselines-master-plan.md`
- Create: `docs/plans/2026-03-26-subplan-01-derived-datasets-and-shared-cache.md`
- Create: `docs/plans/2026-03-26-subplan-02-runner-and-roster-wiring.md`
- Create: `docs/plans/2026-03-26-subplan-03-experiments-and-reporting.md`
- Test: `tests/test_multires_dataset_builder.py`
- Test: `tests/test_multires_roster_overlay.py`

**Step 1: Add failing test stubs for derived dataset generation and multires roster rendering**

Run:
`conda run -n magformer pytest -q tests/test_multires_dataset_builder.py tests/test_multires_roster_overlay.py`

Expected:
- fail because the new builder and roster helpers do not exist yet

**Step 2: Land all planning documents**

Write the design and subplans before touching implementation code.

**Step 3: Re-run the failing tests to confirm they still fail for the intended reason**

Run:
`conda run -n magformer pytest -q tests/test_multires_dataset_builder.py tests/test_multires_roster_overlay.py`

Expected:
- still fail

## Node B: Derived Dataset And Shared Cache

Execute [2026-03-26 Derived Dataset And Cache Subplan](/home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/docs/plans/2026-03-26-subplan-01-derived-datasets-and-shared-cache.md).

## Node C: Runner And Roster Integration

Execute [2026-03-26 Runner And Roster Wiring Subplan](/home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/docs/plans/2026-03-26-subplan-02-runner-and-roster-wiring.md).

## Node D: Verification And Experiments

Execute [2026-03-26 Experiments And Reporting Subplan](/home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/docs/plans/2026-03-26-subplan-03-experiments-and-reporting.md).

## Final Verification

Run the smallest complete proof set:

`conda run -n magformer pytest -q tests/test_multires_dataset_builder.py tests/test_multires_roster_overlay.py tests/test_runner_dry_run_metadata_cmd_reproducible.py tests/test_custom_dataset_stats_cache.py tests/test_ucn_recipe_alignment.py tests/test_msmformer_recipe_alignment.py`

Plan complete and saved to `docs/plans/2026-03-26-full19-multires-baselines-master-plan.md`. Execution will continue in this session with subagent-driven implementation.
