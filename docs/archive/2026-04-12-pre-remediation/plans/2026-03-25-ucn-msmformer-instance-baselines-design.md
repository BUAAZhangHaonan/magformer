# UCN And MSMFormer Instance Baseline Repair Design

**Date:** 2026-03-25

**Goal:** Repair the canonical `ucn` and `msmformer` baselines so both are real instance-segmentation models that can train, infer, and export COCO-evaluable instances on ECC datasets without collapsing to all-zero outputs.

## Problem

The current baseline family has two separate issues:

- `ucn` is implemented as an ECC-friendly wrapper, but its training recipe has drifted away from the vendored UCN project. That makes it hard to claim that the reported `ucn` row is the original instance-segmentation baseline.
- `msmformer` currently reaches the known degenerate state where classification stays non-trivial while mask logits collapse below zero, producing empty instances and `0.000` AP. That result is not fit for a paper comparison table.

The repair work must preserve two invariants:

1. `ucn` and `msmformer` remain instance-segmentation baselines, not foreground-mask models.
2. Each baseline uses the closest practical training recipe to its upstream implementation while still respecting the ECC dataset interface and the repo's canonical evaluation/export path.

## Canonical Naming

- Keep the paper-facing model ids as `ucn` and `msmformer`.
- Repair the existing runner/config paths in place where possible so downstream suite/reporting code continues to speak the canonical names.
- Only introduce `legacy` naming if an implementation detail must be preserved temporarily during migration. Legacy names must never be used in canonical rosters or paper tables.

## UCN Design

`ucn` should be the original embedding-clustering instance-segmentation baseline:

- Backbone: vendored UCN `seg_resnet34_8s_embedding`.
- Task: pixel embedding learning with mean-shift clustering for instance discovery.
- Dataset adaptation: ECC COCO masks are converted into dense instance-id labels per image.
- RGB normalization: dataset-specific BGR means from the ECC stats cache.
- Depth normalization: dataset-specific `p1/p99` clipping and normalization to `[0, 1]`.
- Hyperparameters: align with the vendored UCN config family for `rgbd_add` as closely as possible:
  - Adam optimizer
  - original embedding-loss-related config values
  - original weight decay / learning rate scale
  - original `NUM_UNITS`, fusion mode, and normalization choices

Because ECC is not the synthetic Tabletop training corpus, the "correct" recipe here means "closest faithful adaptation of upstream UCN to ECC", not blindly copying every epoch count from Tabletop scripts.

## MSMFormer Design

`msmformer` should remain a MaskFormer-style instance-segmentation model, but be restored to a vendor-aligned recipe:

- Preserve the vendored MSMFormer meta-architecture.
- Keep ECC dataset registration, RGB/depth loading, and COCO export in the repo wrapper.
- Re-align the effective config toward the vendored UCN/MSMFormer configs where the current wrapper drift is known to be harmful:
  - `MASK_DIM`, decoder settings, mean-shift attention settings, normalization flags, test thresholds
  - mask/class/dice loss balance
  - image sizing and augmentation behavior
  - dataset-specific RGB/depth normalization
- Validate not just end metrics but also intermediate model behavior:
  - non-empty predicted masks
  - non-zero bounding boxes
  - non-zero instance scores
  - one-image overfit no longer collapses to all-negative mask logits

## Shared Data And Evaluation Rules

Both baselines will share these ECC-specific rules:

- Input dataset root is the canonical ECC root or custom register root.
- RGB uses dataset-specific stats from the repo cache when available.
- Depth uses dataset-specific clipping stats from the repo cache when available.
- Final outputs are exported to `coco_instances_results.json` and evaluated with the existing COCOeval utilities so all baselines stay comparable.

## Verification Strategy

Verification will happen at three levels:

1. Static tests
   - config wiring
   - runner dry-run behavior
   - COCO export compatibility
   - canonical model ids and suite roster visibility

2. Behavioral smoke tests
   - import/build tests
   - one-batch or one-image checks
   - no empty-instance collapse for `msmformer`
   - non-empty cluster-to-instance export for `ucn`

3. Canary experiments
   - short training runs on ECC with canonical output dirs
   - fresh evaluation showing AP greater than the current zero-collapse baseline

## Risks

- UCN upstream hyperparameters were tuned on synthetic data; a faithful ECC adaptation may still need small budget-aware changes.
- MSMFormer may require more than config repair if its current wrapper still violates an upstream assumption at the target-preparation or inference stage.
- Runner and roster changes can accidentally break reporting unless covered by dry-run tests.

## Deliverables

- Canonical repaired `ucn` baseline
- Canonical repaired `msmformer` baseline
- Updated runner/config/test coverage
- Updated suite/reporting integration
- Canary experiment evidence that both baselines are usable in paper tables
