# English Metrics Summary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Publish one English metrics-only markdown document that covers all saved model results across `1024`, `512`, and `256`.

**Architecture:** Reuse the already verified experiment artifacts and the existing joined metrics table logic. Write one new markdown file that contains only an English table, with one row per model and resolution and no extra prose.

**Tech Stack:** Markdown, saved experiment JSON artifacts, git

---

### Task 1: Define The New Summary Target

**Files:**
- Create: `docs/experiments/2026-04-06-all-models-three-resolutions-metrics-table.md`
- Modify: none
- Test: source doc and artifact review

**Step 1: Reuse the latest verified metrics join**

Use the current saved summary artifacts and repaired `ucn` / `msmformer` rows.

**Step 2: Keep the output tables-only**

The file must contain only markdown table rows and an English header row.

### Task 2: Publish And Verify

**Files:**
- Modify: `docs/experiments/2026-04-06-all-models-three-resolutions-metrics-table.md`
- Test: file content review, `git diff --check`

**Step 1: Verify the file is English and table-only**

Check the rendered markdown source directly.

**Step 2: Commit and push**

Push the new document to `origin/master` so it is visible on GitHub.
