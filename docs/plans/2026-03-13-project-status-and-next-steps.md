# Project Status And Next Steps

> **For collaborators:** This document is the current handoff snapshot for the `magformer` repository as of 2026-03-13. It is intended to let a new contributor continue work without reconstructing context from commit history.

**Goal:** Summarize the current repository state, experiment results, infrastructure, technical debt, and the most valuable next steps.

**Architecture:** The repository now has a stable lightweight RGB-D experiment line built on top of the existing MagFormer codebase. The current best lightweight result uses a lighter depth backbone plus structured low-cost fusion, but still trails the original heavy RGB-D line.

**Tech Stack:** PyTorch, timm, custom MagFormer training stack, vendored Detectron2/Mask2Former/UOAIS/MSMFormer/UCN baselines, COCOeval-based evaluation, shell experiment runners.

---

## 1. What This Repo Contains

### 1.1 Main model families

- `MagFormer`:
  - in-repo implementation under `magformer/models/`
  - trained/evaluated by `tools/train.py`, `tools/evaluate.py`, `tools/export_results.py`
- External baselines:
  - official Detectron2 / Mask2Former wrappers live under `baselines/`
  - experiment runners live under `scripts/experiments/`

### 1.2 Important directories

- Core model code:
  - `magformer/models/magformer/`
  - `magformer/models/common/`
- Configs:
  - `configs/`
- Experiment runners:
  - `scripts/experiments/`
- Analysis / postprocess / reporting:
  - `scripts/analysis/`
- Result docs:
  - `docs/experiments/`
- Plans / collaborator docs:
  - `docs/plans/`

### 1.3 Current `build.py` behavior

- `magformer/models/build.py` now only constructs the in-repo `MagFormer`.
- External baselines are **not** built via `magformer.models.build`.
- For `ucn`, `msmformer`, `uoa_is`, etc., use the dedicated wrappers in `baselines/` or `scripts/experiments/`.
- This change was made because `magformer.models.baselines.*` did not exist and the previous branches were dead code.

---

## 2. Environment And Dependency Reality

### 2.1 Active environment

- The expected environment is the `magformer` conda environment.
- In that environment, the following key packages are available:
  - `detectron2`
  - `timm`
  - `torch`
  - `ultralytics`

### 2.2 Why Detectron2-based baselines still work

- The repo vendors Detectron2/Mask2Former/UOAIS/MSMFormer code under `baselines/`.
- The wrapper scripts register datasets and launch those codepaths directly.
- These baselines do **not** depend on `magformer.models.build`.

### 2.3 Warning cleanup status

- `magformer/models/build.py` dead baseline-import branches were removed and replaced with an explicit error.
- `magformer/config/schema.py` was migrated to `ConfigDict` and the deprecated `model_validator` pattern was updated.
- `magformer/models/common/__init__.py` was fixed to import `select_timm_out_indices` from the current location.
- `magformer/models/common/pixel_decoder_msdeformattn.py` was updated so DPE modulation no longer hard-depends on `confidence_maps`.

---

## 3. Lightweight RGB-D Line: Current Best Results

### 3.1 Fixed reference baselines

- `magformer_depthnorm_on`
  - `best segm AP = 77.4998`
- `mgm_mask2former_depthnorm_on`
  - `best segm AP = 77.3932`
- `magformer_nodpth_ref`
  - `best segm AP = 70.4891`

### 3.2 Best lightweight candidates

- `magformer_lightdepth_convnextlite_spatialgate_edge_validhole`
  - `best segm AP = 75.1968`
  - `last segm AP = 75.1351`
  - `peak memory = 29751.82 MB`
  - strongest current lightweight candidate

- `magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole`
  - `best segm AP = 74.8259`
  - `last segm AP = 74.8259`
  - lower-memory alternative

- `magformer_lightdepth_mobilenetv3_sagate_edge_validhole`
  - `best segm AP = 74.8344`
  - `last segm AP = 74.6657`
  - strongest peak AP among MobileNetV3 candidates, but slightly less stable

- `magformer_lightdepth_mobilenetv3_channelattn_edge`
  - `best segm AP = 74.8212`
  - `last segm AP = 74.8212`

- `magformer_lightdepth_mobilenetv3_esanetctx_edge_validhole`
  - `best segm AP = 73.9342`
  - `last segm AP = 73.5803`
  - fastest of the newly added literature-inspired follow-up methods

### 3.3 Cross-attention result

- Best Stage B result:
  - `magformer_lightdepth_convnextlite_crossattn_edge_validhole_variance`
  - `best segm AP = 71.1444`
- Conclusion:
  - `cross_attn(res3)` still does not beat the best Stage A low-cost fusion variants
  - keep it as a negative result / diagnostic reference, not the promoted direction

### 3.4 Gap to heavy RGB-D upper bound

- Current best lightweight:
  - `75.1968 AP`
- Heavy MagFormer RGB-D baseline:
  - `77.4998 AP`
- Remaining gap:
  - `2.3030 AP`

---

## 4. Metrics And Benchmark Infrastructure

### 4.1 What is now standardized

- `best segm AP`, `last segm AP`
- expanded AP columns:
  - `AP`, `AP50`, `AP75`, `APs`, `APm`, `APl`
- `params_trainable`
- `wall_time_sec`
- `peak_memory_mb` for new MagFormer/light-depth runs
- `inference_speed.json` with:
  - latency mean / p50 / p90
  - FPS
  - inference peak memory

### 4.2 Key scripts

- Benchmark one experiment directory:
  - `scripts/analysis/benchmark_inference.py`
- Benchmark all models under a suite root:
  - `scripts/analysis/benchmark_inference_suite.py`
- Build wide comparison table:
  - `scripts/analysis/write_extended_metrics_table.py`
- Summarize a suite root:
  - `scripts/experiments/summarize_suite.py`

### 4.3 Current wide table

- Wide comparison table:
  - `output/experiments/0831_1k_20ep_1024_combined/extended_metrics_table.md`
- Synced doc copy:
  - `docs/experiments/2026-03-13-extended-metrics-table.md`

### 4.4 Remaining metric gap

- Old baseline directories still do not all have standardized `peak_memory_mb`.
- This means the strict `1.5x` memory-budget check vs. `magformer_nodpth_ref` is still not fully closed from historical artifacts alone.

---

## 5. Implemented Fusion Modes

Current `magformer/models/magformer/fusion.py` supports:

- `legacy_gated`
- `direct_add`
- `gated_add`
- `film`
- `cross_attn`
- `channel_attn`
- `spatial_gate`
- `sa_gate`
- `esanet_ctx`

### Notes

- `direct_add` and `gated_add` remain useful low-disruption baselines.
- `channel_attn` and `spatial_gate` are currently the most practical literature-inspired MobileNetV3 variants.
- `sa_gate` improves peak AP but currently has a larger best-to-last drop.
- `esanet_ctx` is a speed-oriented residual context block, not the best AP variant.

---

## 6. Backbone Coverage

### Implemented depth backbones

- `ConvNeXtDepth`
- `MobileNetV3Depth`
  - small
  - large
- `ResNetDepth`
  - 18

### Key files

- `magformer/models/common/backbones/depth_base.py`
- `magformer/models/common/backbones/convnext.py`
- `magformer/models/common/backbones/mobilenet.py`
- `magformer/models/common/backbones/resnet.py`
- `magformer/models/common/backbones/depth.py`

### Current best backbone

- `ConvNeXt-lite` under the current Stage A setup is now the best-performing lightweight depth backbone.

---

## 7. What Is Still Not Done

### 7.1 Plan items still open

- Full implementation of `F5 prior-guided cross-attention` as a distinct fusion mode
- More complete budget closure on legacy baselines using standardized `peak_memory_mb`
- A dedicated final rerun suite (`Stage C`) instead of selecting purely from previously completed Stage A/B runs

### 7.2 Technical debt still remaining

- `magformer/models/common/pixel_decoder_msdeformattn.py` is only partially decoupled:
  - it now supports generic depth modulation maps
  - legacy DPE behavior is preserved
  - but the entire DPE stack has not been rethought or simplified yet

### 7.3 Things that should not be pursued first

- More generic transformer attention variants without a stronger geometric bias
- Adding heavier dual semantic branches before the current budget line is fully exhausted
- Broad refactors unrelated to experiment throughput or fusion quality

---

## 8. Recommended Next Work

### Priority 1

- Implement explicit `prior-guided cross-attention` as a separate mode
  - not just `cross_attn + more priors`
  - should have an explicit prior-conditioned key/value or prior-conditioned attention bias

### Priority 2

- Re-run a final candidate pack using only the top practical lightweight variants:
  - `convnextlite_spatialgate_edge_validhole`
  - `mobilenetv3_spatialgate_edge_validhole`
  - `mobilenetv3_channelattn_edge`
  - `mobilenetv3_directadd_edge`

### Priority 3

- Fill legacy baseline `peak_memory_mb` where possible, or document clearly that those values remain unavailable

### Priority 4

- If more fusion expansion is still needed, prefer:
  - residual prior-guided block
  - explicit depth-validity modulation
  - stronger boundary-aware low-cost fusion
  over:
  - generic global attention

---

## 9. Suggested Execution Order For The Next Collaborator

1. Read:
   - `docs/plans/2026-03-10-light-magformer-v2-experiment-plan.md`
   - `docs/experiments/2026-03-12-lightdepth-plan-status-and-metrics.md`
   - `docs/experiments/2026-03-13-lightdepth-followup-results.md`
2. Use:
   - `convnextlite_spatialgate_edge_validhole` as the working best candidate
3. Implement:
   - explicit `prior-guided cross-attention`
4. Run:
   - a focused Stage B successor on the `ConvNeXt-lite` line
5. Update:
   - the extended metrics table
   - final report
   - this handoff doc

---

## 10. Quick Commands

- Refresh suite summary:
  - `python scripts/experiments/summarize_suite.py --output-root <suite_root> --write`
- Refresh inference benchmark for a single model dir:
  - `conda run -n magformer python scripts/analysis/benchmark_inference.py --out-dir <out_dir> --dataset-root /home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K --warmup 5 --timed-images 20`
- Refresh wide comparison table:
  - `python scripts/analysis/write_extended_metrics_table.py --summary <summary1> --summary <summary2> --summary <summary3> --out-json <json> --out-csv <csv> --out-md <md>`
