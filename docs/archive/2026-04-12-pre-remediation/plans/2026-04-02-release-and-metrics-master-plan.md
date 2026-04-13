# Release And Metrics Consolidation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Land the MAGFormer repair work on `master`, publish the branch state, retire the repair branch, and refresh the final cross-resolution metrics summary only if the repaired code invalidates any published numbers.

**Architecture:** Keep the release flow simple. First, confirm the branch graph and the experiment artifacts. Next, decide whether the repair changed model-selection or evaluation semantics for any published runs. Then build one canonical summary table from the existing final artifacts, commit the repair plus reporting updates on `master`, push, and delete the temporary repair branch.

**Tech Stack:** Git, PyTorch experiment artifacts, pytest, shell utilities, markdown docs

---

### Task 1: Lock The Release Scope

**Files:**
- Create: `docs/plans/2026-04-02-release-and-metrics-subplan-01-git-and-scope.md`
- Modify: none
- Test: `git status --short --branch`

**Step 1: Record the branch state**

Run: `git status --short --branch`
Expected: `master` is checked out and the repair files are visible in the working tree.

**Step 2: Record the branch graph**

Run: `git log --graph --decorate --oneline --all --max-count=40`
Expected: the repair branch history is visible and its merge state relative to `master` is clear.

**Step 3: Write the git-and-scope subplan**

Document the exact commands needed to commit, push, and retire the repair branch without rewriting history.

### Task 2: Decide Whether Results Need To Move

**Files:**
- Create: `docs/plans/2026-04-02-release-and-metrics-subplan-02-metrics-audit.md`
- Modify: none
- Test: experiment artifact inspection commands

**Step 1: Inspect summary and extended metrics artifacts**

Run targeted commands over the `1024`, `512`, and `256` experiment folders.
Expected: each resolution has machine-readable final metrics and runtime stats.

**Step 2: Inspect scripts, manifests, and logs for eval path evidence**

Check whether the published runs depended on the broken DDP validation path or the ambiguous offline `--weights` path.
Expected: a clear yes or no for retraining or reevaluation.

**Step 3: Write the metrics-audit subplan**

Record the decision rule for reusing existing numbers versus rerunning jobs.

### Task 3: Build The Canonical Final Table

**Files:**
- Create: `docs/plans/2026-04-02-release-and-metrics-subplan-03-reporting.md`
- Modify: `docs/experiments/2026-03-29-all-resolutions-instance-segmentation-results.md`
- Test: `python` table-generation command and doc review

**Step 1: Consolidate per-resolution metrics into one machine-readable view**

Use the existing `summary_*.json` and `extended_metrics_table.json` files.
Expected: one joined table with AP, F1, train runtime and memory, inference runtime and memory, and FPS columns.

**Step 2: Publish the final markdown table**

Update the all-resolution experiment doc so one file answers the user request directly.

**Step 3: Check the rendered values against the source artifacts**

Spot-check multiple rows across the three resolutions.

### Task 4: Verify, Publish, And Retire The Repair Branch

**Files:**
- Modify: repo files already changed by the repair work and reporting update
- Test: pytest, compileall, git diff checks, git push output

**Step 1: Run fresh validation**

Run the focused test suite plus syntax checks and `git diff --check`.

**Step 2: Commit the release work on `master`**

Use one commit that covers the repair changes and the final reporting update.

**Step 3: Push `master` and delete the repair branch**

Push the commit to `origin/master`, then delete `feature/ucn-msmformer-repair` locally and on origin.

### Task 5: Close With Evidence

**Files:**
- Modify: none
- Test: final checklist review

**Step 1: Re-read the user request and the checklist**

Confirm the release, the branch cleanup, the metrics decision, and the final summary table are all handled.

**Step 2: Report only what the evidence supports**

State clearly whether more training or evaluation was needed and whether the review was complete enough to trust, without overstating certainty.
