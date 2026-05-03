# MAGFormer Codebase Review Master Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Review the in-repo MAGFormer stack end to end, validate the highest-risk findings against the current code and tests, and write a detailed review document under `docs/reviews/`.

**Architecture:** Split the audit into three independent tracks: runtime and entrypoints, model implementation, and repo integration. Run those tracks in parallel where possible, then merge the evidence into one review document with concrete file and line references, targeted validation, and clear follow-up recommendations.

**Tech Stack:** Python, PyTorch, pytest, shell inspection, repo-local markdown documentation.

---

### Task 1: Establish Review Scope And Evidence Sources

**Files:**
- Reference: `README.md`
- Reference: `docs/reviews/2026-03-20-repo-audit.md`
- Reference: `docs/plans/2026-03-13-project-status-and-next-steps.md`
- Reference: `docs/experiments/2026-03-12-rgbd-fusion-method-survey.md`
- Reference: `pyproject.toml`
- Reference: `requirements.txt`

**Step 1: Confirm the repo layout and current review conventions**

Run:

```bash
find magformer tools tests scripts configs docs -maxdepth 2 -type f | sort | sed -n '1,240p'
```

Expected: the active code, test, config, and documentation surfaces are visible and the audit scope can exclude vendored third-party code unless wrappers depend on it.

**Step 2: Record the current workflow claims**

Read:
- `README.md`
- `docs/plans/2026-03-13-project-status-and-next-steps.md`
- `docs/experiments/2026-03-12-rgbd-fusion-method-survey.md`

Expected: the review has a clear baseline for what the repo says it supports.

### Task 2: Execute Subplans In Parallel

**Files:**
- Reference: `docs/plans/2026-04-02-magformer-codebase-review-subplan-01-runtime.md`
- Reference: `docs/plans/2026-04-02-magformer-codebase-review-subplan-02-model.md`
- Reference: `docs/plans/2026-04-02-magformer-codebase-review-subplan-03-integration.md`

**Step 1: Run focused review on the runtime path**

Expected: concrete findings for `magformer/config`, `magformer/data`, `magformer/engine`, and `tools/*.py`.

**Step 2: Run focused review on the model path**

Expected: concrete findings for `magformer/models/**`.

**Step 3: Run focused review on repo integration**

Expected: concrete findings for `configs/`, `scripts/`, `tests/`, `README.md`, and workflow docs.

### Task 3: Validate The Highest-Risk Findings

**Files:**
- Reference: `tests/`
- Reference: `tools/train.py`
- Reference: `tools/evaluate.py`
- Reference: `tools/inference.py`
- Reference: `scripts/analysis/`
- Reference: `scripts/experiments/`

**Step 1: Re-open each flagged file around the cited lines**

Run:

```bash
nl -ba <file> | sed -n '<start>,<end>p'
```

Expected: every final finding is backed by current line-level evidence.

**Step 2: Run the narrowest meaningful validation for disputed behavior**

Run only targeted commands, for example:

```bash
pytest tests/test_train_distributed_context.py -q
pytest tests/test_magformer_raw_inference.py -q
pytest tests/test_runner_dry_run_metadata_cmd_reproducible.py -q
```

Expected: the review distinguishes between confirmed breakage, tested coverage, and untested risk.

### Task 4: Write And Sanity-Check The Final Review

**Files:**
- Create: `docs/reviews/2026-04-02-codebase-review.md`

**Step 1: Write the review**

Include:
- scope
- review method
- findings ordered by severity
- file and line references
- validation performed
- residual risks and missing tests

Expected: a collaborator can act on the review without repeating the same exploration.

**Step 2: Sanity-check the document**

Run:

```bash
sed -n '1,260p' docs/reviews/2026-04-02-codebase-review.md
```

Expected: the document is internally consistent, grounded in current files, and complete.

## Completion Criteria

- All three review tracks are completed.
- Every final finding has exact file and line references.
- At least one targeted validation command is run for each major risk area where validation is feasible.
- `docs/reviews/2026-04-02-codebase-review.md` exists and reflects the current repo state.
