# MSMFormer Instance Repair Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Repair the canonical `msmformer` baseline so it behaves like a real instance-segmentation model on ECC datasets and no longer collapses to empty-mask, zero-score predictions.

**Architecture:** Keep the ECC Detectron2 wrapper, but re-align the effective config to the vendored MSMFormer recipe, add tests for the known collapse path, and fix the training/inference wiring until one-image overfit and canary evaluation produce non-empty instances.

**Tech Stack:** Detectron2, vendored MSMFormer, PyTorch, ECC COCO RGB-D datasets, pytest.

---

### Task 1: Lock The Known Failure Mode In Tests

**Files:**
- Create: `tests/test_msmformer_recipe_alignment.py`
- Create: `tests/test_msmformer_instance_inference.py`

**Step 1: Write the failing tests**

Add tests that pin the intended canonical config:
- vendor-aligned loss weights
- decoder settings
- attention-mask settings
- mask dimension
- instance-on inference path

Add an inference-side test that proves empty-mask predictions yield zero scores through `instance_inference`, so future fixes are targeted at preventing the collapse instead of hiding it.

**Step 2: Run tests to verify failure or incompleteness**

Run:
`conda run -n magformer pytest -q tests/test_msmformer_recipe_alignment.py tests/test_msmformer_instance_inference.py`

Expected:
- fail or reveal that the current config is still the collapsed one

**Step 3: Write minimal implementation**

Update the wrapper/config until the tests express the intended canonical recipe.

**Step 4: Re-run tests**

The same command must pass.

### Task 2: Re-align The ECC MSMFormer Config

**Files:**
- Modify: `configs/baselines/msmformer_0831_1k_tracks.yaml`
- Modify: `baselines/run_msmformer_ecc.py`
- Modify: `baselines/run_msmformer_0831_1k.py`
- Modify: `scripts/experiments/run_ecc_20ep_trackp_msmformer.sh`
- Modify: `scripts/experiments/run_0831_1k_20ep_scratch_msmformer.sh`

**Step 1: Bring the config closer to the vendored upstream**

Use the vendored configs under:
- `baselines/msmformer/MSMFormer/configs/mixture_UCN.yaml`
- `baselines/msmformer/MSMFormer/configs/UOAIS_UCN.yaml`

Carry over the impactful settings:
- decoder/loss balance
- self-attention and cross-attention flags
- mask dimension
- threshold policy
- decoder-block normalization

Adapt class count only where required by ECC's single foreground category.

**Step 2: Keep dataset-correct normalization**

Ensure the wrapper continues to inject:
- dataset-specific RGB mean/std
- dataset-specific depth normalization

**Step 3: Keep the model an instance-segmentation model**

Do not convert it into semantic foreground segmentation.
The output must still be Detectron2 `Instances` with masks, boxes, scores, and classes.

### Task 3: Add A Collapse Guardrail

**Files:**
- Modify: `baselines/run_msmformer_ecc.py`
- Create: `tests/test_msmformer_empty_mask_guardrail.py`

**Step 1: Write the failing test**

Capture the behavior we want to avoid:
- a training/inference setup that drives all mask logits below zero on a trivial batch or tiny synthetic example

If a full train-loop unit test is too heavy, encode the guardrail as a lightweight functional check on logits / thresholded masks.

**Step 2: Run it and confirm failure**

Run:
`conda run -n magformer pytest -q tests/test_msmformer_empty_mask_guardrail.py`

**Step 3: Implement the smallest real fix**

Fix the configuration or wrapper wiring causing the collapse. Do not patch the exporter to hide empty predictions.

**Step 4: Re-run the test**

The same command must pass.

### Task 4: MSMFormer Canary

**Files:**
- Runtime only: output dir under `output/experiments/.../msmformer`

**Step 1: Run a short canary**

Use the canonical MSMFormer runner on the active ECC dataset with a short budget.

**Step 2: Check intermediate behavior**

Confirm:
- non-empty masks in the output json or predictions pth
- non-zero scores
- non-zero bounding boxes
- AP no longer equals `0.000`

**Step 3: Record findings**

Add the canary result to the experiment note referenced by the master plan.
