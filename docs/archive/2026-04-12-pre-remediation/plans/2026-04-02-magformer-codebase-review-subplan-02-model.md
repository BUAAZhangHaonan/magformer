# MAGFormer Codebase Review Model Subplan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Review the MAGFormer model implementation for algorithmic correctness and train or infer mismatches.

**Architecture:** Follow model construction from `build.py` into the architecture, fusion, matcher, criterion, and pixel decoder paths, then compare those assumptions against the tests and the stated fusion design.

**Tech Stack:** Python, PyTorch, timm, deformable attention modules.

---

### Task 1: Trace The Build Path

**Files:**
- Reference: `magformer/models/build.py`
- Reference: `magformer/models/magformer/arch.py`
- Reference: `magformer/models/magformer/fusion.py`

**Step 1: Read the construction path from config to model**

Expected: the review identifies how fusion mode, backbone choice, and decoder setup are wired.

### Task 2: Inspect Core Model Components

**Files:**
- Reference: `magformer/models/common/criterion.py`
- Reference: `magformer/models/common/matcher.py`
- Reference: `magformer/models/common/pixel_decoder.py`
- Reference: `magformer/models/common/pixel_decoder_msdeformattn.py`
- Reference: `magformer/utils/depth_sanity.py`

**Step 1: Check for silent failure modes**

Check:
- shape assumptions
- device or dtype mismatches
- optional branch behavior
- training vs inference output contracts
- fallback paths that may hide real errors

Expected: concrete findings with file and line references.

### Task 3: Cross-Check Coverage

**Files:**
- Reference: `tests/test_cross_attn_fusion_mode.py`
- Reference: `tests/test_light_fusion_modes.py`
- Reference: `tests/test_magformer_ablation_switches.py`
- Reference: `tests/test_pixel_decoder_dpe_modulation.py`

**Step 1: Match risky model paths against tests**

Expected: the review notes what is defended by tests and what remains weak.
