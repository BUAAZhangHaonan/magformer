# UCN And MSMFormer Instance Baselines Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore `ucn` and `msmformer` as canonical ECC instance-segmentation baselines that train, infer, and export valid COCO instances without zero-collapse.

**Architecture:** Repair the baselines in three tracks. First, re-anchor `ucn` to the original UCN embedding-clustering recipe while keeping ECC dataset adaptation and COCO export. Second, re-align `msmformer` to a vendor-faithful instance-segmentation recipe and fix the empty-mask collapse. Third, update runners, rosters, and reporting so the repaired canonical baselines are the ones seen by experiments and paper-facing summaries.

**Tech Stack:** Conda, PyTorch, vendored UCN, vendored MSMFormer, Detectron2, COCO RGB-D datasets, pytest, shell experiment runners.

---

## Node Order

1. Node A: planning docs and baseline safety tests
2. Node B: canonical `ucn` repair
3. Node C: canonical `msmformer` repair
4. Node D: runner / suite / reporting integration
5. Node E: canary experiments and result capture

## Required Execution Rules

- Every behavior change starts with a failing or incomplete test.
- `ucn` and `msmformer` keep their canonical names.
- No paper-facing table should read from known broken zero-collapse outputs once migration is complete.
- Every node ends with fresh verification commands.

## Subplans

- [2026-03-25 UCN Subplan](/home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/docs/plans/2026-03-25-ucn-subplan-01-canonical-baseline.md)
- [2026-03-25 MSMFormer Subplan](/home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/docs/plans/2026-03-25-msmformer-subplan-02-instance-repair.md)
- [2026-03-25 Suite And Experiment Subplan](/home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/docs/plans/2026-03-25-subplan-03-suite-reporting-and-experiments.md)

## Node A: Planning And Safety Net

**Files:**
- Create: `docs/plans/2026-03-25-ucn-msmformer-instance-baselines-design.md`
- Create: `docs/plans/2026-03-25-ucn-msmformer-instance-baselines-master-plan.md`
- Create: `docs/plans/2026-03-25-ucn-subplan-01-canonical-baseline.md`
- Create: `docs/plans/2026-03-25-msmformer-subplan-02-instance-repair.md`
- Create: `docs/plans/2026-03-25-subplan-03-suite-reporting-and-experiments.md`
- Test: `tests/test_msmformer_imports.py`
- Test: `tests/test_ucn_coco_export.py`

**Step 1: Verify baseline imports and existing export utilities**

Run:
`conda run -n magformer pytest -q tests/test_msmformer_imports.py tests/test_ucn_coco_export.py`

Expected:
- pass

**Step 2: Land all planning documents**

Write the design and subplans before touching the baseline code.

**Step 3: Re-run a minimal smoke check after docs exist**

Run:
`conda run -n magformer pytest -q tests/test_msmformer_imports.py tests/test_ucn_coco_export.py`

Expected:
- still pass

## Node B: Canonical UCN

Execute [2026-03-25 UCN Subplan](/home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/docs/plans/2026-03-25-ucn-subplan-01-canonical-baseline.md).

## Node C: Canonical MSMFormer

Execute [2026-03-25 MSMFormer Subplan](/home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/docs/plans/2026-03-25-msmformer-subplan-02-instance-repair.md).

## Node D: Suite Integration

Execute [2026-03-25 Suite And Experiment Subplan](/home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/docs/plans/2026-03-25-subplan-03-suite-reporting-and-experiments.md).

## Node E: Canary Experiments

**Files:**
- Modify: `output/experiments/...` runtime outputs only
- Create: `docs/experiments/2026-03-25-ucn-msmformer-repair-canary.md`

**Step 1: Run repaired `ucn` canary**

Run the canonical `ucn` runner on the active ECC dataset with a short budget.

Success criteria:
- non-empty `coco_instances_results.json`
- `segm/AP` strictly greater than the previous zero/near-zero collapsed run

**Step 2: Run repaired `msmformer` canary**

Run the canonical `msmformer` runner on the active ECC dataset with a short budget.

Success criteria:
- predicted masks are non-empty
- scores are non-zero
- evaluation is not `0.000`

**Step 3: Capture evidence**

Write one short experiment note with:
- command
- output dir
- best metrics
- residual risks

## Final Verification

Run the smallest complete proof set:

`conda run -n magformer pytest -q tests/test_msmformer_imports.py tests/test_ucn_coco_export.py tests/test_runner_dry_run_metadata_cmd_reproducible.py tests/test_benchmark_inference_ucn_fallback.py`

Plan complete and saved to `docs/plans/2026-03-25-ucn-msmformer-instance-baselines-master-plan.md`. Execution will continue in this session with subagent-driven implementation.
