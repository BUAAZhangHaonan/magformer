# MAGFormer Codebase Review Integration Subplan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Review the integration layer around configs, scripts, tests, and docs for contradictions, brittle workflows, and missing packaging or validation.

**Architecture:** Compare the repo's documented workflow against the actual config files, runner scripts, and test coverage, then isolate places where users can hit confusing or broken paths even if the core model code is sound.

**Tech Stack:** Python, shell scripts, pytest, markdown documentation.

---

### Task 1: Check Public Workflow Docs

**Files:**
- Reference: `README.md`
- Reference: `docs/plans/2026-03-13-project-status-and-next-steps.md`
- Reference: `docs/reviews/2026-03-20-repo-audit.md`

**Step 1: Compare the advertised workflow to the current repo**

Expected: concrete documentation mismatches and stale guidance are identified with line references.

### Task 2: Inspect Config And Script Entry Points

**Files:**
- Reference: `configs/`
- Reference: `scripts/analysis/`
- Reference: `scripts/experiments/`

**Step 1: Check for brittle command construction and hidden assumptions**

Check:
- hard-coded paths
- conflicting config names
- missing argument validation
- metadata or output assumptions
- coupling between scripts and specific filesystem layouts

Expected: concrete findings with file and line references.

### Task 3: Cross-Check Test Coverage

**Files:**
- Reference: `tests/`

**Step 1: Verify that public workflows are defended by tests**

Expected: the review identifies important gaps where docs or scripts promise behavior that no test actually covers.
