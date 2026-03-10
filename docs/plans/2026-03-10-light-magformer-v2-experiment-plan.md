# LightMagFormer-v2 Experiment Plan

> **For Codex:** This document is the executable experiment plan for the lightweight RGB-D redesign line. It is intentionally constrained by compute-budget and fairness requirements.

**Goal:** Redesign the current MagFormer/MGM RGB-D branch into a lightweight, high-value fusion line that stays within a fixed memory/time budget while preserving or improving AP over the RGB-only control.

**Architecture:** Keep the RGB branch as the only full semantic backbone. Replace the current heavy depth branch with a lightweight standard backbone, restrict fusion to one or two scales, and compare low-cost fusion styles under a fixed 0831_1K / 1024 / 20-epoch protocol.

**Tech Stack:** PyTorch, Detectron2-style training utilities already in this repo, timm backbones where possible, existing MagFormer/MGM fusion + evaluation scripts.

---

## 1. Problem Statement

The current RGB-D branch is not failing because depth is useless. It is failing on **cost-efficiency**:

- AP gain exists, but the RGB-D line is too expensive in memory and training cost.
- The current design treats depth as a second semantic backbone instead of a geometric auxiliary signal.
- The current design mixes several expensive choices at once:
  - full depth backbone
  - full-resolution priors
  - multi-scale prior replication
  - multi-scale confidence prediction
  - DPE coupling to depth confidence

This plan treats the existing `magformer_depthnorm_on` result as the **upper-bound baseline**, not as the shape of the next design.

---

## 2. Hard Constraints

### 2.1 Fairness Constraints
All experiments in this line must use the same protocol unless explicitly marked otherwise:

- Dataset: `0831_1K`
- Resolution: `1024`
- Epochs: `20`
- Same training / validation split
- Same evaluation metric: COCO `segm AP`
- Same visualization style: YOLO-style mask overlay
- Same checkpoint retention policy: final + best only

### 2.2 Resource Constraints
Any candidate in this line is rejected if either of the following is violated relative to `magformer_nodpth_ref`:

- peak GPU memory > `1.5x`
- wall time / epoch > `1.5x`

### 2.3 Success Constraints
A candidate only remains in the shortlist if:

- `best segm AP > magformer_nodpth_ref`
- results are stable across the full 20-epoch run
- no pathological mask collapse is observed in overlay inspection

---

## 3. Fixed Baselines

These are the comparison anchors and must remain unchanged for this line:

- `magformer_nodpth_ref`
- `magformer_depthnorm_on`
- `mgm_mask2former_depthnorm_on`

Interpretation rule:

- `mgm_mask2former_nodpth_ref` is **not** a symmetric no-depth ablation and must **not** be used as the sole explanation of depth gain.
- It is a diagnostic reference only.

---

## 4. Design Principles

### 4.1 RGB remains the only full semantic backbone
Depth is treated as:

- geometry cue
- boundary cue
- occlusion / validity cue

Depth is **not** allowed to remain a second full semantic branch.

### 4.2 Fusion is single-scale first
The first batch of experiments must only fuse at:

- `res3`

Only if justified by results may a second scale be added later:

- `res4`

### 4.3 No full-resolution prior pyramid in v2
The default must change from:

- `prior.compute_on = full`

to:

- `prior.compute_on = res3`

The only reason to reintroduce `full` would be strong, measured AP gain under budget, which is unlikely.

### 4.4 Depth encoder must be a standard lightweight backbone
Do **not** use an ad-hoc pure convolution stem as the primary candidate line.

Allowed first-stage candidates:

- `MobileNetV3-Small`
- `MobileNetV3-Large`
- `ResNet18`
- `ConvNeXt-Tiny` truncated / reduced variant

Not allowed in stage 1:

- full `ConvNeXt-Tiny`
- `ResNet50`
- new transformer depth encoders

---

## 5. Candidate Components

## 5.1 Lightweight Depth Encoders

### D1. MobileNetV3-Small depth encoder
- Most efficient first candidate
- Output feature at `res3`
- Optional `res4` only in second-stage ablation

### D2. MobileNetV3-Large depth encoder
- Middle ground between MobileNet-Small and ResNet18

### D3. ResNet18 depth encoder
- Stronger baseline, still acceptable cost
- Good for measuring whether extra capacity matters

### D4. ConvNeXt-lite depth encoder
- Only if implemented as a reduced / truncated version
- Not the full current branch

---

## 5.2 Depth Priors to Compare

The first pass only compares these priors:

### P1. `edge`
- Depth gradient / edge magnitude
- Primary candidate for weak-texture boundaries

### P2. `valid-hole`
- Valid depth mask and missing depth region
- Primary candidate for occlusion and geometry discontinuity

### P3. `variance`
- Local depth variance
- Only on `res3`, never full-resolution replicated pyramid

### P4. `normal-proxy`
- Approximate surface-orientation cue derived from local gradients
- Only if implementation remains lightweight

First-stage recommended prior sets:

- `[edge]`
- `[edge, valid-hole]`
- `[edge, valid-hole, variance]`

---

## 5.3 Fusion Modes to Compare

### F1. Direct Add
- `rgb + depth_proj`
- Cheapest lower bound

### F2. Gated Add
- `rgb + gate(depth, prior) * depth_proj`
- Expected to be the most practical low-cost candidate

### F3. FiLM / Scale-Shift Modulation
- Use depth features / priors to produce `gamma / beta`
- Modulate RGB features at `res3`

### F4. Single-Scale Cross-Attention
- `res3` only
- RGB = query, depth = key/value
- 1 layer first, 2 layers only if justified

### F5. Prior-Guided Cross-Attention
- Optional second-stage extension after a winning backbone + F4 combination is found

---

## 6. Experiment Matrix

## Stage A — Lowest-Cost Feasibility

Purpose: determine whether a lightweight depth path can beat RGB-only.

Run exactly these candidates first:

1. `RGB-only`
   - existing `magformer_nodpth_ref`
2. `MobileNetV3-Small + Direct Add + [edge]`
3. `MobileNetV3-Small + Gated Add + [edge]`
4. `ResNet18 + Gated Add + [edge]`
5. `MobileNetV3-Small + FiLM + [edge, valid-hole]`

Decision rule:
- Keep the top 2 by AP under the memory/time budget.

## Stage B — Cross-Attention Validation

Using the best depth encoder from Stage A, run:

6. `best encoder + Cross-Attention(res3) + [edge]`
7. `best encoder + Cross-Attention(res3) + [edge, valid-hole]`
8. `best encoder + Cross-Attention(res3) + [edge, valid-hole, variance]`

Decision rule:
- Keep only candidates that beat the best Stage A model by a meaningful margin while staying within budget.

## Stage C — Final 20-Epoch Candidates

Only promote at most 2 candidates to the final 20-epoch set:

- one “best low-cost” candidate
- one “best accuracy” candidate if different

---

## 7. Required Code Surfaces

## 7.1 Core Model Files

### `magformer/models/magformer/arch.py`
Changes required:
- introduce a configurable `depth_mode`
- build a lightweight depth encoder based on config
- allow `res3`-only depth feature flow
- route new fusion modes cleanly

### `magformer/models/magformer/fusion.py`
Changes required:
- add fusion mode enum / dispatch
- restrict priors to configurable scale (`res3` by default)
- support single-scale depth feature injection
- support direct add / gated add / FiLM / cross-attn modes

### `magformer/models/common/pixel_decoder_msdeformattn.py`
Changes required:
- decouple lightweight geometric depth cues from the current heavy `confidence_maps` dependency
- preserve current behavior for legacy configs

## 7.2 Config Surfaces

Add new configs such as:
- `configs/magformer_0831_1k_20ep_1024_lightdepth_mobilenetv3.yaml`
- `configs/magformer_0831_1k_20ep_1024_lightdepth_resnet18.yaml`
- `configs/magformer_0831_1k_20ep_1024_lightdepth_crossattn.yaml`

Add/standardize config keys:
- `model.magformer.depth_mode`
- `model.magformer.depth_backbone.name`
- `model.magformer.depth_backbone.out_features`
- `model.magformer.modality_fusion.mode`
- `model.magformer.modality_fusion.priors`
- `model.magformer.modality_fusion.fuse_scales`
- `model.magformer.modality_fusion.prior.compute_on`

---

## 8. Logging and Evaluation Requirements

Every candidate must log:

- `best segm AP`
- `last segm AP`
- peak memory
- epoch time / total wall time
- final checkpoint paths
- best checkpoint path
- overlay outputs on the same 20 validation images used in the current experiment root

All candidate reports must be appended into a dedicated experiment summary, separate from the current fixed baseline report.

---

## 9. Acceptance Criteria

A candidate is considered successful if it satisfies all three:

1. `best segm AP > magformer_nodpth_ref`
2. peak memory within budget
3. wall time within budget

A candidate is considered strong if additionally:

4. AP gap to `magformer_depthnorm_on` is ≤ `2.0 AP`

A candidate is considered dominant if:

5. it matches or exceeds `magformer_depthnorm_on` while using ≤ `60%` of its memory footprint

---

## 10. Recommended Execution Order

1. Implement configuration plumbing only
2. Add lightweight depth encoder options
3. Force all priors to `res3`
4. Implement direct add + gated add first
5. Run Stage A
6. Only then implement cross-attention
7. Run Stage B
8. Select final candidates

---

## 11. Assumptions

- This line is not intended to replace the new reference-U-Net line as the primary research direction.
- This line exists to measure the best possible cost-efficient RGB-D gain under a fixed budget.
- Full-resolution prior computation is assumed to be removed from the default path.
- The current heavy MagFormer/MGM designs remain strong baselines, not future architecture templates.
