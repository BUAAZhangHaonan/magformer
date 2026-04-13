# MAGFormer Codebase Review Runtime Subplan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Review the runtime path from config loading through data, training, evaluation, and inference entrypoints.

**Architecture:** Trace the public CLI entrypoints into the shared runtime code, then check whether the trainer, evaluator, and data path match the behavior promised by tests and docs.

**Tech Stack:** Python, PyTorch, pytest, YAML config loading.

---

### Task 1: Trace Runtime Entry Points

**Files:**
- Reference: `tools/train.py`
- Reference: `tools/evaluate.py`
- Reference: `tools/inference.py`
- Reference: `magformer/config/loader.py`
- Reference: `magformer/engine/trainer.py`
- Reference: `magformer/engine/evaluator.py`

**Step 1: Read the CLI entrypoints and shared config path**

Run:

```bash
nl -ba tools/train.py | sed -n '1,260p'
nl -ba tools/evaluate.py | sed -n '1,260p'
nl -ba tools/inference.py | sed -n '1,260p'
```

Expected: the review identifies the public contract and how arguments flow into runtime code.

### Task 2: Inspect The Runtime Stack

**Files:**
- Reference: `magformer/data/dataset.py`
- Reference: `magformer/data/transforms.py`
- Reference: `magformer/engine/trainer.py`
- Reference: `magformer/engine/evaluator.py`
- Reference: `magformer/engine/coco_export.py`

**Step 1: Look for correctness risks and hidden assumptions**

Check:
- dataset path handling
- split-specific behavior
- distributed setup and teardown
- checkpoint and resume flow
- evaluation/export contracts

Expected: concrete findings with file and line references.

### Task 3: Cross-Check Coverage

**Files:**
- Reference: `tests/test_train_distributed_context.py`
- Reference: `tests/test_train_imports_dataset.py`
- Reference: `tests/test_magformer_raw_inference.py`
- Reference: `tests/test_trainer_best_checkpoint_preservation.py`

**Step 1: Match risky runtime paths against tests**

Expected: the review notes where tests cover the behavior and where they stop short.
