# MAGFormer Repository Audit

Date: `2026-03-20`

## Scope

This audit was written during the post-migration bring-up of the `magformer` workspace. The goal is to let a new collaborator understand:

- what the repository currently contains
- which documentation files are authoritative
- how the main model and baselines are organized
- which dependency constraints matter for a unified runtime environment
- which commands should be used to verify the migrated workspace

## Executive Summary

- The repository is no longer just a standalone MAGFormer implementation. It is now a mixed workspace containing:
  - the in-repo MAGFormer model and training stack
  - experiment runners and reporting scripts
  - vendored or wrapped baseline code for Detectron2, Mask2Former, UOAIS, MSMFormer, UCN, UNet-style references, and Ultralytics YOLO
- The current documentation source of truth is **not** the root [README.md](/home/team/zhanghaonan/magformer/README.md). The most reliable high-level handoff is [docs/plans/2026-03-13-project-status-and-next-steps.md](/home/team/zhanghaonan/magformer/docs/plans/2026-03-13-project-status-and-next-steps.md), backed by the March experiment reports under [docs/experiments](/home/team/zhanghaonan/magformer/docs/experiments).
- The main project is a pure-PyTorch RGB-D instance segmentation stack with a YAML + Pydantic config system, custom dataset loader, custom trainer, Swin/ConvNeXt-style backbones, and deformable-attention operators under [magformer/models/ops](/home/team/zhanghaonan/magformer/magformer/models/ops).
- The biggest environment challenge is not PyTorch alone. It is the need to satisfy:
  - Detectron2 and its compiled extension
  - MAGFormer deformable-attention ops
  - UOAIS / AdelaiDet compiled extension
  - old MSMFormer / unseen-object-clustering code that still uses deprecated NumPy aliases
- The most important compatibility finding is: **do not use NumPy 1.24+ / 2.x in the unified environment unless we patch vendored baseline code**. Several baseline files still use `np.float`, `np.int`, and `np.bool`.

## Repository Layout

### Main code

- [magformer](/home/team/zhanghaonan/magformer/magformer): core package
- [tools](/home/team/zhanghaonan/magformer/tools): train/eval/inference and helper entrypoints
- [configs](/home/team/zhanghaonan/magformer/configs): YAML configs for MAGFormer and experiment variants
- [tests](/home/team/zhanghaonan/magformer/tests): automated tests, mostly import/config/runner/reporting coverage

### Supporting infrastructure

- [scripts/experiments](/home/team/zhanghaonan/magformer/scripts/experiments): shell runners for unified experiment execution
- [scripts/analysis](/home/team/zhanghaonan/magformer/scripts/analysis): reporting, conversion, benchmarking, summarization, and checkpoint utilities
- [docs](/home/team/zhanghaonan/magformer/docs): plans, experiment reports, and refactor rules
- [baselines](/home/team/zhanghaonan/magformer/baselines): wrapped and vendored baseline ecosystems

### Runtime state

- [magformer_datasets](/home/team/zhanghaonan/magformer/magformer_datasets): colocated datasets
- [output](/home/team/zhanghaonan/magformer/output): runtime outputs and pretrained assets

## Documentation Map

### Current source of truth

- [docs/plans/2026-03-13-project-status-and-next-steps.md](/home/team/zhanghaonan/magformer/docs/plans/2026-03-13-project-status-and-next-steps.md)
  - best current handoff snapshot
  - summarizes model families, environment assumptions, metrics, fusion modes, and next steps
- [docs/plans/2026-03-10-light-magformer-v2-experiment-plan.md](/home/team/zhanghaonan/magformer/docs/plans/2026-03-10-light-magformer-v2-experiment-plan.md)
  - primary design doc for the March light-depth direction
- [docs/plans/2026-03-10-reference-data-spec.md](/home/team/zhanghaonan/magformer/docs/plans/2026-03-10-reference-data-spec.md)
  - future-facing reference-conditioned data contract

### Experiment ledger

- [docs/experiments/track_a_2k_report.md](/home/team/zhanghaonan/magformer/docs/experiments/track_a_2k_report.md)
- [docs/experiments/track_b_2k_report.md](/home/team/zhanghaonan/magformer/docs/experiments/track_b_2k_report.md)
- [docs/experiments/2026-03-09-0831-1k-20ep-1024-depth-revisit-final.md](/home/team/zhanghaonan/magformer/docs/experiments/2026-03-09-0831-1k-20ep-1024-depth-revisit-final.md)
- [docs/experiments/2026-03-12-lightdepth-v2-final-report.md](/home/team/zhanghaonan/magformer/docs/experiments/2026-03-12-lightdepth-v2-final-report.md)
- [docs/experiments/2026-03-13-lightdepth-followup-results.md](/home/team/zhanghaonan/magformer/docs/experiments/2026-03-13-lightdepth-followup-results.md)
- [docs/experiments/2026-03-13-f5-prior-guided-cross-attention-final-summary.md](/home/team/zhanghaonan/magformer/docs/experiments/2026-03-13-f5-prior-guided-cross-attention-final-summary.md)
- [docs/experiments/2026-03-13-extended-metrics-table.md](/home/team/zhanghaonan/magformer/docs/experiments/2026-03-13-extended-metrics-table.md)

### Document issues to keep in mind

- [README.md](/home/team/zhanghaonan/magformer/README.md) is stale relative to the March plans and experiment reports.
- Some experiment docs are historical or partially templated:
  - [docs/experiments/final_20k_report.md](/home/team/zhanghaonan/magformer/docs/experiments/final_20k_report.md) is a paused branch, not the current roadmap.
  - [docs/experiments/baselines_0831_1k_5k_report.md](/home/team/zhanghaonan/magformer/docs/experiments/baselines_0831_1k_5k_report.md) still presents itself as a template and calls out metric-normalization gaps.
  - [docs/experiments/baselines_0831_1k_5k_scratch8_report.md](/home/team/zhanghaonan/magformer/docs/experiments/baselines_0831_1k_5k_scratch8_report.md) and [docs/experiments/baselines_0831_1k_20ep_scratch8_report.md](/home/team/zhanghaonan/magformer/docs/experiments/baselines_0831_1k_20ep_scratch8_report.md) contain `TBD` placeholders.
- Multiple runbooks still embed old absolute machine paths under `/home/k100/...`.

## Core Package Architecture

### Config system

- [magformer/config/loader.py](/home/team/zhanghaonan/magformer/magformer/config/loader.py)
- [magformer/config/schema.py](/home/team/zhanghaonan/magformer/magformer/config/schema.py)

The main project uses YAML + Pydantic. The shared parser accepts both `--config` and `--config-file`, even though some older docs only mention one spelling.

### Data path

- [magformer/data/dataset.py](/home/team/zhanghaonan/magformer/magformer/data/dataset.py)
- [magformer/data/transforms.py](/home/team/zhanghaonan/magformer/magformer/data/transforms.py)
- [magformer/data/collate.py](/home/team/zhanghaonan/magformer/magformer/data/collate.py)

The in-repo dataset loader expects a COCO-style RGB-D directory with:

- `images/<split>/`
- `depth/depth_npy/<split>/` or `depth/<split>/`
- `annotations/*.json`

Depth is loaded from `.npy` or `.npz`, and optional depth noise masks are supported.

### Model path

- [magformer/models/build.py](/home/team/zhanghaonan/magformer/magformer/models/build.py)
- [magformer/models/magformer/arch.py](/home/team/zhanghaonan/magformer/magformer/models/magformer/arch.py)
- [magformer/models/magformer/fusion.py](/home/team/zhanghaonan/magformer/magformer/models/magformer/fusion.py)
- [magformer/models/common](/home/team/zhanghaonan/magformer/magformer/models/common)

Key points:

- `magformer.models.build` now only builds the in-repo `MagFormer`.
- External baselines are intentionally rejected there and must be launched through baseline wrappers or experiment scripts.
- The architecture combines:
  - RGB backbone
  - depth backbone
  - modality fusion block
  - pixel decoder
  - transformer decoder
- The fusion module now covers multiple modes, including classic gated fusion and newer low-cost or diagnostic attention variants.

### Training path

- [tools/train.py](/home/team/zhanghaonan/magformer/tools/train.py)
- [tools/evaluate.py](/home/team/zhanghaonan/magformer/tools/evaluate.py)
- [tools/inference.py](/home/team/zhanghaonan/magformer/tools/inference.py)
- [magformer/engine/trainer.py](/home/team/zhanghaonan/magformer/magformer/engine/trainer.py)

The main training stack is pure PyTorch. It supports:

- AMP
- DDP
- checkpointing
- COCO export / evaluation
- TensorBoard and WandB logging

### Native extension

- [magformer/models/ops/setup.py](/home/team/zhanghaonan/magformer/magformer/models/ops/setup.py)
- [magformer/models/ops/modules/ms_deform_attn.py](/home/team/zhanghaonan/magformer/magformer/models/ops/modules/ms_deform_attn.py)

This is a local C++/CUDA extension and must match the PyTorch CUDA build used in the environment.

## Tests

- There are currently `73` top-level `tests/test_*.py` files.
- Coverage is strongest around:
  - config wiring
  - runner command construction
  - experiment metadata reproducibility
  - analysis utilities
  - selected model/fusion/backbone logic
- Coverage is weaker for:
  - full end-to-end multi-GPU training
  - full dataset-backed baseline training runs
  - compiled extension rebuild flows

## Baseline Taxonomy

### 1. Detectron2 official wrappers

- [baselines/run_detectron2_0831_1k.py](/home/team/zhanghaonan/magformer/baselines/run_detectron2_0831_1k.py)
- [baselines/run_detectron2_ecc.py](/home/team/zhanghaonan/magformer/baselines/run_detectron2_ecc.py)
- [baselines/detectron2](/home/team/zhanghaonan/magformer/baselines/detectron2)

These wrappers register datasets and then hand control to the vendored Detectron2 codebase.

### 2. Official Mask2Former / MGM Mask2Former

- [baselines/run_official_mask2former_0831_1k.py](/home/team/zhanghaonan/magformer/baselines/run_official_mask2former_0831_1k.py)
- [baselines/run_official_mask2former_ecc.py](/home/team/zhanghaonan/magformer/baselines/run_official_mask2former_ecc.py)
- [baselines/Mask2Former](/home/team/zhanghaonan/magformer/baselines/Mask2Former)
- [baselines/MGM_Mask2Former](/home/team/zhanghaonan/magformer/baselines/MGM_Mask2Former)

These depend on Detectron2 plus additional Mask2Former code and, for the MGM variant, another deformable-attention operator build.

### 3. UOAIS / AdelaiDet-based RGB-D baseline

- [baselines/run_uoais_0831_1k.py](/home/team/zhanghaonan/magformer/baselines/run_uoais_0831_1k.py)
- [baselines/run_uoais_ecc.py](/home/team/zhanghaonan/magformer/baselines/run_uoais_ecc.py)
- [baselines/uoais](/home/team/zhanghaonan/magformer/baselines/uoais)

This path depends on:

- Detectron2
- `adet` from the local UOAIS repo
- another compiled extension under the UOAIS package

The repo currently contains an old prebuilt binary:

- [baselines/uoais/adet/_C.cpython-37m-x86_64-linux-gnu.so](/home/team/zhanghaonan/magformer/baselines/uoais/adet/_C.cpython-37m-x86_64-linux-gnu.so)

That binary is not reusable in a modern Python environment and should be treated as stale.

### 4. MSMFormer / UCN-style stack

- [baselines/run_msmformer_0831_1k.py](/home/team/zhanghaonan/magformer/baselines/run_msmformer_0831_1k.py)
- [baselines/run_msmformer_ecc.py](/home/team/zhanghaonan/magformer/baselines/run_msmformer_ecc.py)
- [baselines/msmformer](/home/team/zhanghaonan/magformer/baselines/msmformer)
- [baselines/unseen_object_clustering](/home/team/zhanghaonan/magformer/baselines/unseen_object_clustering)

This family is older and carries the most legacy assumptions:

- `open3d`
- `transforms3d`
- `easydict`
- `imageio`
- older NumPy alias usage

### 5. UCN / non-Detectron2 baseline wrappers

- [baselines/run_ucn_0831_1k.py](/home/team/zhanghaonan/magformer/baselines/run_ucn_0831_1k.py)
- [baselines/run_ucn_ecc.py](/home/team/zhanghaonan/magformer/baselines/run_ucn_ecc.py)
- [baselines/unet_instance_models.py](/home/team/zhanghaonan/magformer/baselines/unet_instance_models.py)
- [baselines/run_unet_instance_ecc.py](/home/team/zhanghaonan/magformer/baselines/run_unet_instance_ecc.py)

These are lighter from a packaging perspective, but still rely on OpenCV, PyTorch, COCO tools, and dataset conventions.

### 6. Ultralytics / YOLO export path

- [baselines/yolo_export_coco.py](/home/team/zhanghaonan/magformer/baselines/yolo_export_coco.py)
- [baselines/ultralytics](/home/team/zhanghaonan/magformer/baselines/ultralytics)
- [baselines/ultralytics_tools/convert_coco_to_yolo_seg.py](/home/team/zhanghaonan/magformer/baselines/ultralytics_tools/convert_coco_to_yolo_seg.py)

This family is newer and generally compatible with modern Python, but is still part of the unified comparison workflow.

## Environment And Dependency Findings

### Machine capability

Observed locally:

- `conda 25.3.1`
- no global default `python`
- NVIDIA A100 GPUs
- driver CUDA `12.9`
- local CUDA toolkits present at `/usr/local/cuda-11.7`, `/usr/local/cuda-12.3`, `/usr/local/cuda-12.4`, `/usr/local/cuda-12.8`
- default `CUDA_HOME=/usr/local/cuda-12.8`
- `gcc/g++ 11.4.0`
- `ninja 1.10.1`

### Existing environment clues

Existing local conda envs suggest the machine has already been used with:

- `mask2former`: Python `3.11.13`, Torch `2.5.1+cu124`, TorchVision `0.20.1+cu124`, Detectron2 `0.6`
- `msmformer`: Python `3.8.20`, Torch `1.10.0+cu111`
- `uoais`: Python `3.10.18`, Torch `2.8.0+cu128`, but the recorded Detectron2 install points to an old remote path and `adet` is not importable, so this env should not be treated as healthy

### Critical compatibility risk: NumPy

The vendored baseline code still contains `np.float`, `np.int`, and `np.bool` usages, for example in:

- [baselines/Mask2Former/mask2former_video/data_video/datasets/ytvis_api/ytvoseval.py](/home/team/zhanghaonan/magformer/baselines/Mask2Former/mask2former_video/data_video/datasets/ytvis_api/ytvoseval.py)
- [baselines/uoais/eval/compute_PRF.py](/home/team/zhanghaonan/magformer/baselines/uoais/eval/compute_PRF.py)
- [baselines/msmformer/lib/networks/resnet_dilated.py](/home/team/zhanghaonan/magformer/baselines/msmformer/lib/networks/resnet_dilated.py)
- [baselines/unseen_object_clustering/lib/networks/resnet_dilated.py](/home/team/zhanghaonan/magformer/baselines/unseen_object_clustering/lib/networks/resnet_dilated.py)

Because those aliases were removed in NumPy 1.24, the unified environment should pin:

- `numpy<1.24`

The highest practical safe line without patching vendored code is `numpy==1.23.5`.

### Critical compatibility risk: CUDA build matching

If we select Torch `2.5.1+cu124`, native extensions should be compiled with CUDA `12.4`, not the default `12.8`, to avoid mismatched build/runtime combinations.

Recommended build-time environment variable for that stack:

```bash
export CUDA_HOME=/usr/local/cuda-12.4
```

### Critical compatibility risk: compiled extensions

The following must align with the chosen PyTorch + CUDA toolchain:

- Detectron2 extension under [baselines/detectron2](/home/team/zhanghaonan/magformer/baselines/detectron2)
- MAGFormer deformable attention under [magformer/models/ops](/home/team/zhanghaonan/magformer/magformer/models/ops)
- MGM Mask2Former deformable attention under [baselines/MGM_Mask2Former/mask2former/modeling/pixel_decoder/ops](/home/team/zhanghaonan/magformer/baselines/MGM_Mask2Former/mask2former/modeling/pixel_decoder/ops)
- UOAIS / AdelaiDet extension under [baselines/uoais](/home/team/zhanghaonan/magformer/baselines/uoais)

### Recommended unified environment baseline

For this machine, the most reasonable balance between “as new as possible” and “still compatible with all baselines” is:

- Python `3.11`
- Torch `2.5.1+cu124`
- TorchVision `0.20.1+cu124`
- NumPy `1.23.5`

Rationale:

- Python `3.11` and Torch `2.5.1` are already proven locally with a working Detectron2 install.
- Higher Torch lines exist on the machine, but they are not yet proven against the full Detectron2 + AdelaiDet + legacy baseline combination in this workspace.
- NumPy must stay below `1.24` unless we patch the vendored codebase.

## Pathing And Reproducibility Risks

- Experiment reports and some configs still reference old absolute paths.
- The workspace is currently dirty:
  - `.gitignore` modified
  - `baselines/detectron2` marked modified
- Some compiled artifacts and `__pycache__` files were migrated along with the repo. They are not reliable evidence that the current machine can rebuild the same stack.

## Recommended Verification Commands

### Core package

```bash
conda run -n magformer python -c "import torch, magformer; print(torch.__version__)"
conda run -n magformer pytest tests -q
```

### Main entrypoints

```bash
conda run -n magformer python tools/train.py --help
conda run -n magformer python tools/evaluate.py --help
conda run -n magformer python tools/inference.py --help
```

### Baseline wrappers

```bash
conda run -n magformer python baselines/run_detectron2_0831_1k.py --help
conda run -n magformer python baselines/run_official_mask2former_0831_1k.py --help
conda run -n magformer python baselines/run_msmformer_0831_1k.py --help
conda run -n magformer python baselines/run_uoais_0831_1k.py --help
conda run -n magformer python baselines/run_ucn_0831_1k.py --help
conda run -n magformer python baselines/yolo_export_coco.py --help
```

### Native extensions

```bash
conda run -n magformer python -c "import detectron2"
conda run -n magformer python -c "from magformer.models.ops.modules import MSDeformAttn"
```

## Actual Bring-Up Results On This Machine

As of `2026-03-20`, the following were verified in the new `magformer` conda environment:

- Python `3.11.15`
- Torch `2.5.1+cu124`
- TorchVision `0.20.1+cu124`
- NumPy `1.23.5`
- Detectron2 editable install succeeds with `--no-build-isolation`
- Ultralytics editable install succeeds
- Official Mask2Former deformable-attention op compiles successfully against CUDA `12.4`

Verified wrapper startup paths:

- `python baselines/run_detectron2_0831_1k.py -- --help`
- `python baselines/run_official_mask2former_0831_1k.py -- --help`
- `python baselines/run_msmformer_0831_1k.py -- --help`
- `python baselines/run_uoais_0831_1k.py -- --help`
- `python baselines/run_ucn_0831_1k.py --help`
- `python baselines/MGM_Mask2Former/train_net_mgm_0831.py --help`
- `yolo help`
- `python tools/train.py --help`
- `python -c "import torch; from magformer.models.ops.modules import MSDeformAttn"`

Verified tests:

- `tests/test_setup_package_discovery.py` passes after fixing top-level package discovery
- `tests/test_magformer_raw_inference.py` passes

Known remaining caveats:

- `tests/test_config_wiring.py` stalls because model construction triggers remote pretrained-weight resolution from Hugging Face; this is a test/data access issue, not an import failure in the environment itself.
- Building `baselines/uoais` as an editable package fails on modern Torch because its old CUDA extension still references `THC/THC.h`. However, the vendored ECC wrapper path starts successfully without that extension because local patches already make those ops optional for the active UOAIS baseline path.
- Directly importing `MultiScaleDeformableAttention` without importing `torch` first can fail to locate `libc10.so`; the validated code paths import `torch` before the extension and work correctly.

## Automation Added

To make this reproducible, the repository now includes:

- [environment.magformer.yml](/home/team/zhanghaonan/magformer/environment.magformer.yml)
- [scripts/dev/setup_magformer_env.sh](/home/team/zhanghaonan/magformer/scripts/dev/setup_magformer_env.sh)

The YAML file records the verified version baseline. The setup script records the validated installation order, including the Detectron2 editable install and official Mask2Former op compilation step.

## Immediate Next Actions

1. Create a unified `magformer` conda environment using the compatibility baseline above.
2. Install PyTorch first, then Detectron2 and local editable packages with a matching `CUDA_HOME`.
3. Verify imports for the main package and all wrapper entrypoints.
4. If UOAIS or an older baseline fails only because of NumPy alias removals, decide whether to:
   - keep the environment on `numpy==1.23.5`, or
   - patch the vendored baseline code and then upgrade NumPy later.
