# Project Summary Report

This report is based only on repository materials that were available in `/home/team/zhanghaonan/magformer` on 2026-04-13. Where the materials disagree, the report prefers the newest primary artifact and explicitly notes the conflict.

## 1. Project Overview and Background

The project is called `MAGFormer` in the top-level README and `MagFormer` in the Python package and configs. In both cases it refers to the same project: a pure PyTorch RGB-D instance segmentation system for dense clutter scenes, implemented around a transformer-style segmentation model with RGB and depth fusion. The public README defines the current shipped scope as single-class COCO-format RGB-D segmentation, not a general multi-class product. Sources: `README.md:1-10`, `magformer/models/build.py:13-49`, `magformer/config/validation.py:205-230`.

The project sits in computer vision, more specifically RGB-D instance segmentation and benchmarking. The repo is not just a single model implementation. It also contains dataset builders, experiment runners, metric-table writers, and vendored baseline trees for comparison against Mask2Former, MGM/Mask2Former, Mask R-CNN, YOLOv8 segmentation, UCN, MSMFormer, UOAIS, and U-Net variants. Sources: `README.md:12-26`, `setup.py:23-55`, `output/analysis/2026-04-12-all-models-metrics-1024-512-sorted-by-segm-ap.md:1-42`.

The available materials most strongly support research and engineering users rather than end-user application operators. That inference is grounded in the exposed CLIs (`magformer-train`, `magformer-eval`, `magformer-infer`, `magformer-export`), the experiment shell runners, and the result-publication scripts. Sources: `setup.py:23-55`, `scripts/experiments/common_runner.sh:1-132`, `scripts/analysis/write_extended_metrics_table.py:201-253`.

The motivation evolves over time. Early documents focus on fair baseline comparison and warm-start-vs-scratch tracking. The March 2026 planning documents then shift the project toward a lighter RGB-D branch that should preserve useful depth gains without the cost of the original heavy depth path. A later March spec extends the work toward reference-conditioned single-part workflows and ECC-style reference banks. Sources: `docs/experiments/2026-02-18-baseline-freeze.md:1-27`, `docs/plans/2026-03-10-light-magformer-v2-experiment-plan.md:13-27`, `docs/plans/2026-03-10-reference-data-spec.md:3-27`.

The project explicitly builds on or responds to existing systems rather than claiming a clean-sheet method. The fusion-method survey cites ESANet, CMX, SA-Gate, ACNet, RDFNet, and DFormer as design references, and the live benchmark tables compare MAGFormer against external baseline families. Sources: `docs/experiments/2026-03-12-rgbd-fusion-method-survey.md:5-130`, `output/analysis/2026-04-12-all-models-metrics-1024-512-sorted-by-segm-ap.csv:3-42`.

## 2. Application Scenarios

The clearest deployment scenario in the available materials is offline model development for RGB-D clutter segmentation. The repo expects RGB images, aligned depth maps, and COCO annotations; it supports full training, validation, export to COCO-style results, single-example inference, and visualization. Sources: `README.md:38-95`, `tools/train.py:31-174`, `tools/evaluate.py:28-127`, `tools/inference.py:29-125`, `magformer/engine/coco_export.py:19-159`.

Typical use cases are:

| Scenario | How the system is used | Evidence |
|---|---|---|
| Train a MAGFormer variant | Load a COCO-style RGB-D dataset, validate depth and class structure, train with AMP/checkpointing, and save best/final weights | `tools/train.py:499-729`, `magformer/engine/trainer.py:43-249`, `magformer/engine/trainer.py:772-847` |
| Evaluate a model on a validation split | Load a checkpoint, run the shared evaluation runtime, export COCO predictions, and compute COCO metrics | `tools/evaluate.py:28-127`, `magformer/engine/eval_runtime.py:51-199`, `magformer/engine/evaluator.py:23-145` |
| Run single-sample inference | Provide one RGB image and one depth file, run model inference, and optionally visualize masks | `tools/inference.py:29-125`, `magformer/utils/visualization.py:45-229` |
| Build derived datasets for multi-resolution studies | Resize RGB/depth/masks offline, regenerate annotation files, and keep per-root caches/stats | `scripts/analysis/build_multires_dataset.py:171-257`, `docs/plans/2026-03-26-full19-multires-baselines-design.md:51-175` |
| Run benchmark campaigns across many models | Use the experiment runners to normalize environment, wait for GPU memory, launch jobs, and later summarize suite metrics | `scripts/experiments/common_runner.sh:1-132`, `scripts/analysis/write_extended_metrics_table.py:201-253`, `output/analysis/2026-04-12-all-models-metrics-1024-512-sorted-by-segm-ap.csv:3-42` |

The main deployment constraints that are explicitly documented are technical rather than organizational. The project currently enforces a single foreground category, treats depth normalization as a hard requirement, and relies on GPU-driven training/evaluation workflows. One repair log shows an `NVIDIA A100 80GB PCIe` and CUDA 12.4 environment, and the common runner waits for available GPU memory before launching. Sources: `README.md:3-10`, `magformer/data/dataset.py:126-159`, `magformer/config/validation.py:52-80`, `scripts/experiments/common_runner.sh:1-132`, `output/experiments/20260318_1k_1566_20ep_1024_full19/repair_msm_unet_gpu1.log:19`.

The code and docs also show several unusual or edge scenarios that the project handles explicitly:

- No-depth control runs are supported, including a later “fair” rerun that tries to match the depth-enabled recipe more closely. Sources: `docs/experiments/2026-04-06-fair-nodpth-config-diff.md:1-45`, `output/experiments/20260406_1k_1566_20ep_1024_full19/magformer_nodpth_ref_fair/metadata.json:5`.
- Zero-prediction validation is treated as zero AP after the April 2026 sync-and-repair pass. Sources: `docs/experiments/2026-04-06-magformer-project-sync-and-repair-summary.md:3-11`.
- Derived 256 and 512 roots are built offline rather than resized on the fly, to keep preprocessing fair across the 19-model suite. Sources: `docs/plans/2026-03-26-full19-multires-baselines-design.md:5-20`, `magformer_datasets/20260318_1K_1566_256/dataset_info.json:95`, `magformer_datasets/20260318_1K_1566_512/dataset_info.json:95`.

## 3. Tasks and Objectives

The project has both core model tasks and auxiliary workflow tasks.

### Core Tasks

| Task | Input | Output | Success criteria found in repo materials | Evidence |
|---|---|---|---|---|
| RGB-D dataset loading and validation | COCO JSON, RGB files, depth files, optional noise masks | Batched tensors plus targets and padding masks | Dataset must contain exactly one foreground category; depth stats and depth clipping must pass validation; RGB/depth/mask alignment is preserved through transforms | `magformer/data/dataset.py:22-237`, `magformer/data/transforms.py:60-227`, `magformer/data/collate.py:17-187`, `magformer/config/validation.py:205-230` |
| MAGFormer training | Runtime config, dataset loaders, optional warm-start checkpoint | Best/final checkpoints, logs, metrics | Training startup passes class/depth sanity checks; optimizer/scheduler/checkpoint logic runs; experiment docs use `segm AP` as the main optimization target | `tools/train.py:499-729`, `magformer/engine/trainer.py:43-249`, `docs/plans/2026-03-10-light-magformer-v2-experiment-plan.md:43-55` |
| Evaluation and COCO export | Trained weights and validation data | COCO result JSON, COCOeval metrics, optional best-checkpoint updates | Shared evaluation path completes and writes metrics; zero-prediction runs are counted as zero AP after the April repair pass | `tools/evaluate.py:28-127`, `magformer/engine/eval_runtime.py:51-199`, `magformer/engine/evaluator.py:23-145`, `docs/experiments/2026-04-06-magformer-project-sync-and-repair-summary.md:3-11` |
| Inference and visualization | One RGB image and one depth map | Predicted masks and visualization overlays | Inference path runs outside the training loop and uses the same model factory/runtime contracts | `tools/inference.py:29-125`, `magformer/utils/visualization.py:45-229` |

### Auxiliary Tasks

| Task | Input | Output | Purpose | Evidence |
|---|---|---|---|---|
| Derived dataset construction | Existing RGB-D dataset root and target size | New `256`/`512` dataset roots, updated annotations, cached preprocessing artifacts | Keep multi-resolution experiments reproducible and fair across models | `scripts/analysis/build_multires_dataset.py:171-257`, `docs/plans/2026-03-26-full19-multires-baselines-design.md:51-175` |
| Runtime-config rendering | Base YAML plus dataset stats | Concrete runtime config with RGB/depth normalization injected | Avoid mismatched dataset statistics during runs | `scripts/analysis/render_magformer_runtime_config.py:67-132` |
| Experiment orchestration | Config path, output root, GPU selection, shell environment | Logged training/evaluation jobs | Provide reproducible large-scale campaigns across many models | `scripts/experiments/common_runner.sh:1-132`, `tests/test_20260321_ddp_smoke_script.py:7-87` |
| Metrics/report generation | Run outputs, metrics JSON/CSV, manifests | Consolidated markdown/CSV/JSON result tables | Publish comparable result summaries and spot stale/missing artifacts | `scripts/analysis/write_extended_metrics_table.py:201-253`, `output/analysis/2026-04-12-all-models-metrics-1024-512-sorted-by-segm-ap.csv:3-42` |
| Baseline integration and repair | External baseline trees plus project wrappers/configs | Comparable baseline runs and repaired exports | Keep non-MAGFormer baselines in the same evaluation/reporting workflow | `docs/plans/2026-03-25-ucn-msmformer-instance-baselines-design.md:25-67`, `docs/experiments/2026-03-25-ucn-msmformer-repair-canary.md:7-17` |

Across the codebase, the end-to-end pipeline is: config loading and validation, dataset/stat resolution, RGB-D sample loading and transforms, model build, train/eval/infer runtime, COCO export, metric aggregation, and suite-level reporting. Sources: `magformer/config/schema.py:15-260`, `magformer/data/dataset.py:22-237`, `tools/train.py:31-729`, `tools/evaluate.py:28-127`, `scripts/analysis/write_extended_metrics_table.py:201-253`.

## 4. Methods and Technical Approach

### 4.1 System Architecture

The project-owned implementation is centered on `magformer/`, while large comparison baselines are vendored under `baselines/` and kept separate from the main package and formatting/lint scope. Sources: `pyproject.toml:1-34`, `README.md:12-26`.

The main architecture is:

1. Configuration and validation  
   Typed schemas cover data, backbones, fusion, decoder, solver, and runtime. The runtime-config renderer injects dataset statistics such as `pixel_mean`, `pixel_std`, and depth clip bounds before training/evaluation. Sources: `magformer/config/schema.py:15-260`, `scripts/analysis/render_magformer_runtime_config.py:67-132`.

2. Data pipeline  
   `CocoRgbdDataset` loads RGB images, depth arrays (`.npy`/`.npz`), optional noise masks, and COCO annotations. `RGBDTransform` keeps RGB, depth, masks, boxes, and content masks aligned. `collate_fn` pads and stacks tensors while preserving target metadata. Sources: `magformer/data/dataset.py:22-237`, `magformer/data/transforms.py:60-227`, `magformer/data/collate.py:17-187`.

3. Model pipeline  
   `MagFormerArch` normalizes RGB input, runs the RGB backbone, optionally runs the depth backbone, fuses RGB/depth features, passes fused features through a pixel decoder, and finally predicts masks/classes with a transformer decoder. Sources: `magformer/models/magformer/arch.py:108-374`, `magformer/models/magformer/arch.py:450-630`.

4. Runtime layer  
   `tools/train.py` assembles datasets, loaders, model, optimizer, scheduler, warm-start handling, and checkpoint logic before handing off to `Trainer` or `DDPTrainer`. `tools/evaluate.py` and `tools/inference.py` call the shared evaluation/inference runtime instead of re-implementing it separately. Sources: `tools/train.py:177-251`, `tools/train.py:499-729`, `magformer/engine/trainer.py:43-249`, `magformer/engine/trainer.py:772-847`, `magformer/engine/eval_runtime.py:51-199`.

5. Export and reporting  
   Predictions are converted to COCO rows, evaluated with COCOeval-style logic, and then summarized into suite tables and markdown reports. Sources: `magformer/engine/coco_export.py:19-159`, `magformer/engine/evaluator.py:23-145`, `scripts/analysis/write_extended_metrics_table.py:201-253`.

### 4.2 Model and Algorithm Details

The project-owned model builder intentionally only instantiates `MagFormer`; baseline names are rejected and kept in external trees. That separation is a design decision: MAGFormer is the maintained in-repo model, while the rest of the repo serves as a benchmark harness around it. Source: `magformer/models/build.py:13-49`.

The core model is Mask2Former-like in structure. It uses:

- An RGB backbone built through `timm`, with Swin support that allows flexible image sizes (`strict_img_size=False`). Sources: `magformer/models/common/backbones/swin.py:55-68`.
- A pluggable depth backbone registry. The code and tests show support for ConvNeXt, ConvNeXt Lite, MobileNetV3, and ResNet18 depth branches. Sources: `magformer/models/common/backbones/depth.py:25-105`, `tests/test_depth_backbone_builder.py:33-65`.
- A fusion module with multiple experimental modes: `legacy_gated`, `direct_add`, `gated_add`, `film`, `cross_attn`, `prior_guided_cross_attn`, `channel_attn`, `spatial_gate`, `sa_gate`, and `esanet_ctx`. Priors can be derived from gradients, variance, valid/hole regions, and optional RGB edges. Sources: `magformer/models/magformer/fusion.py:278-399`, `magformer/models/magformer/fusion.py:458-978`, `magformer/config/schema.py:183-231`.
- Pixel decoders that can be either a simple decoder or an MSDeformAttn-based decoder. The deformable decoder can apply depth position encoding (DPE), implemented as a learned projection over `log1p(beta * depth)`. Sources: `magformer/models/common/pixel_decoder_msdeformattn.py:205-420`, `magformer/models/common/layers/depth_position_encoding.py:13-78`.
- Transformer decoders that can be simple or multi-scale masked decoders. The multi-scale version accepts separate `key_pos`, which is how depth-modulated positional information reaches cross-attention. Sources: `magformer/models/common/transformer/multiscale_decoder.py:51-119`, `magformer/models/common/transformer/multiscale_decoder.py:158-220`, `magformer/models/common/transformer/decoder.py:14-113`.
- Hungarian matching plus `SetCriterion` losses for class prediction, BCE mask loss, and Dice loss with deep supervision, which is a standard Mask2Former-style training objective. Sources: `magformer/models/common/matcher.py:48-117`, `magformer/models/common/criterion.py:109-198`.

### 4.3 Data Flow

Data enters the system as an RGB image, a depth array, and COCO annotations. The dataset loader checks class and path assumptions, transforms keep RGB/depth/label geometry aligned, the collate step creates batch tensors, and the model then consumes normalized RGB plus optional depth features. At the output side, decoder predictions become masks, scores, and optionally RLE-encoded COCO records for evaluation and export. Sources: `magformer/data/dataset.py:173-237`, `magformer/data/transforms.py:60-227`, `magformer/data/collate.py:17-187`, `magformer/models/magformer/arch.py:450-630`, `magformer/engine/coco_export.py:19-159`.

### 4.4 Dependencies and External Tools

The primary stack is PyTorch, torchvision, and timm for model construction; Pydantic, PyYAML, and OmegaConf for typed configuration; pycocotools and OpenCV for evaluation/visualization; and TensorBoard/WandB for experiment logging. The environment file also pins `fvcore`, `iopath`, `yacs`, `hydra-core`, `submitit`, and `segmentation-models-pytorch`, which appear to support the surrounding experiment and baseline ecosystem. Sources: `requirements.txt:4-42`, `environment.magformer.yml:10-47`, `magformer/models/common/backbones/depth_base.py:42-131`, `magformer/utils/visualization.py:45-229`.

### 4.5 Design Decisions and Trade-Offs

The strongest documented design trade-offs are:

- Lightweight depth experimentation is favored over bigger depth towers after March 2026. Sources: `docs/plans/2026-03-10-light-magformer-v2-experiment-plan.md:13-27`.
- Light-depth variants are restricted to `res3` fusion and reduced prior complexity during Stage A to control cost and isolate the fusion question. Sources: `docs/plans/2026-03-10-light-magformer-v2-experiment-plan.md:73-231`.
- Offline derived datasets are used for the 19-model multi-resolution suite instead of runtime resizing, to keep preprocessing and caching consistent. Sources: `docs/plans/2026-03-26-full19-multires-baselines-design.md:51-175`.
- Legacy root-level config keys are still accepted, but nested fields are preferred, which shows the config surface is in transition rather than fully stable. Sources: `magformer/config/validation.py:86-129`, `tests/test_config_wiring.py:13-74`.

## 5. Innovations and Contributions

The available materials do not support claiming that the repo introduces a wholly new segmentation paradigm. The project’s own documents mostly frame the work as an adaptation, simplification, or repair of known RGB-D segmentation ideas. The fusion survey is explicit about drawing from prior systems such as ESANet, CMX, SA-Gate, ACNet, RDFNet, and DFormer. Source: `docs/experiments/2026-03-12-rgbd-fusion-method-survey.md:5-130`.

Within that constraint, the main contributions are:

1. A large, source-backed lightweight RGB-D exploration space inside MAGFormer  
   The project adds multiple lightweight depth backbones and many fusion modes, then evaluates them under explicit memory and wall-time budgets. This is a real implementation and experiment contribution even if the component ideas are not individually novel. Sources: `magformer/models/magformer/fusion.py:458-978`, `tests/test_lightdepth_configs.py:8-25`, `docs/plans/2026-03-10-light-magformer-v2-experiment-plan.md:43-55`.

2. A specific `prior_guided_cross_attn` fusion mode  
   This is the clearest project-specific experimental mechanism in the March 13 F5 plan. However, the final F5 summary says it improved plain `cross_attn` by only `0.2061 AP` and still did not beat the earlier Stage A winners. Its contribution is therefore best described as a completed diagnostic branch, not a decisive algorithmic advance. Sources: `docs/plans/2026-03-13-f5-prior-guided-cross-attention-master-plan.md:5-30`, `docs/plans/2026-03-13-f5-subplan-01-fusion-core.md:5-46`, `docs/experiments/2026-03-13-f5-prior-guided-cross-attention-final-summary.md:25-60`.

3. A practical multiresolution experiment framework  
   The repo adds derived `256` and `512` dataset roots, per-root stats, shared caches, roster files, and reporting scripts so that 19 models can be compared under controlled preprocessing. This is a substantial engineering contribution. Sources: `docs/plans/2026-03-26-full19-multires-baselines-design.md:5-20`, `docs/plans/2026-03-26-full19-multires-baselines-design.md:107-217`, `magformer_datasets/20260318_1K_1566_256/dataset_info.json:95`, `magformer_datasets/20260318_1K_1566_512/dataset_info.json:95`.

4. Baseline repair and reporting hardening  
   The UCN/MSMFormer repair work, unified evaluation path, zero-prediction policy fix, and consolidated live metrics manifests are practical contributions that improve trustworthiness more than raw novelty. Sources: `docs/experiments/2026-03-25-ucn-msmformer-repair-canary.md:7-17`, `docs/experiments/2026-03-24-postsuite-audit.md:5-27`, `docs/experiments/2026-04-06-magformer-project-sync-and-repair-summary.md:3-31`.

## 6. Experimental Design

### 6.1 Datasets

The main dataset families that can be verified from metadata are summarized below.

| Dataset | Train images / anns | Val images / anns | Test images / anns | Categories | Scenes | Notes |
|---|---:|---:|---:|---:|---:|---|
| `0831_1K` | 886 / 47,814 | 110 / 6,189 | 110 / 5,843 | 1 | `[Not Found]` | Earlier baseline and light-depth line | 
| `20260318_1K_1566` | 1,261 / 77,125 | 149 / 9,494 | 156 / 9,276 | 1 | 2,264 | Main full-suite dataset |
| `20260318_1K_1566_256` | 1,261 / 76,744 | 149 / 9,444 | 156 / 9,239 | 1 | 2,264 | Historical offline derived 256 root kept in repo materials |
| `20260318_1K_1566_512` | 1,261 / 76,991 | 149 / 9,478 | 156 / 9,266 | 1 | 2,264 | Offline derived 512 root |

Sources: `magformer_datasets/0831_1K/annotations/instances_train.json:1`, `magformer_datasets/0831_1K/annotations/instances_val.json:1`, `magformer_datasets/0831_1K/annotations/instances_test.json:1`, `magformer_datasets/20260318_1K_1566/build_stats.json:2`, `magformer_datasets/20260318_1K_1566/dataset_info.json:5`, `magformer_datasets/20260318_1K_1566_256/dataset_info.json:95`, `magformer_datasets/20260318_1K_1566_512/dataset_info.json:95`.

The 20260318 build metadata describes a dataset generation setup with `counts = [25, 50, 100]`, `scenes_per_part_per_count = 20`, `num_views = 2`, binary mask export, `instance_visibility` GS mask mode, and normalized part scaling to a `0.045 m` target diagonal. Sources: `magformer_datasets/20260318_1K_1566/dataset_info.json:5`, `magformer_datasets/20260318_1K_1566/dataset_info.json:36`.

The same dataset family also includes explicit quality reports. Alignment shows `1566/1566` images passed with zero failures; mask parity stays right at the configured threshold edge; scene QC reports zero failed tasks. Sources: `magformer_datasets/20260318_1K_1566/alignment_report.json:10`, `magformer_datasets/20260318_1K_1566/mask_parity_report.json:1`, `magformer_datasets/20260318_1K_1566/scene_qc_report.json:1`.

### 6.2 Experiment Families

The repository materials show five main experiment families:

| Family | Goal | Main evidence |
|---|---|---|
| Baseline freeze and fairness tracks | Compare MAGFormer against baseline recipes under controlled warm-start/scratch conditions | `docs/experiments/2026-02-18-baseline-freeze.md:1-27`, `docs/experiments/track_a_2k_report.md:22-60`, `docs/experiments/track_b_2k_report.md:28-67` |
| Light-depth Stage A/B | Find cheaper RGB-D fusion/backbone choices that still beat RGB-only | `docs/plans/2026-03-10-light-magformer-v2-experiment-plan.md:30-55`, `docs/experiments/2026-03-11-lightdepth-stage-a-results.md:17-45`, `docs/experiments/2026-03-12-lightdepth-v2-final-report.md:17-60` |
| F5 prior-guided cross-attention | Test whether a prior-guided cross-attention variant helps the Stage B branch | `docs/plans/2026-03-13-f5-prior-guided-cross-attention-master-plan.md:5-30`, `docs/experiments/2026-03-13-f5-prior-guided-cross-attention-final-summary.md:25-60` |
| Full19 multiresolution suite | Designed for 19 models across 1024 plus derived 512/256 roots; the current live publication artifacts in this repo are 1024/512 | `configs/experiments/full_20260318_1k_1566_roster.json:1`, `docs/plans/2026-03-26-full19-multires-baselines-design.md:5-20`, `scripts/experiments/run_20260409_non256_completion_gpu1.sh:68-158`, `output/analysis/2026-04-12-all-models-metrics-1024-512-sorted-by-segm-ap.csv:1-41` |
| Repair and release sync | Fix stale metrics, repair broken baselines, and consolidate publication tables | `docs/experiments/2026-03-24-postsuite-audit.md:5-27`, `docs/experiments/2026-04-06-magformer-project-sync-and-repair-summary.md:3-31`, `docs/plans/2026-04-02-release-and-metrics-master-plan.md:34-105` |

### 6.3 Metrics, Baselines, and Conditions

The primary evaluation metric in the planning documents is segmentation AP (`segm AP`). Bounding-box AP is also reported widely, and the live 2026-04-12 rollup adds FPS for the 1024 runs. The light-depth plan also defines memory and wall-time acceptance criteria: lightweight candidates should stay within `1.5x` peak memory and `1.5x` wall time of `magformer_nodpth_ref` while outperforming the RGB-only control. Sources: `docs/plans/2026-03-10-light-magformer-v2-experiment-plan.md:43-55`, `output/analysis/2026-04-12-all-models-metrics-1024-512-sorted-by-segm-ap.csv:3-22`.

The comparison set includes MAGFormer variants, MGM/Mask2Former variants, Mask2Former, Mask R-CNN, YOLOv8 segmentation models, UCN, MSMFormer, UOAIS, and U-Net variants. Sources: `output/analysis/2026-04-12-all-models-metrics-1024-512-sorted-by-segm-ap.csv:3-42`, `output/experiments/20260318_1k_1566_20ep_1024_full19/models_manifest.json:2`.

### 6.4 Hyperparameters and Runtime Environment

The best-documented controlled setting is the light-depth line on `0831_1K / 1024 / 20 epochs`, with the same split, the same metric, and only best/final checkpoints retained. Stage A also limits fusion to `res3` and keeps prior computation lightweight. Sources: `docs/plans/2026-03-10-light-magformer-v2-experiment-plan.md:30-55`, `docs/plans/2026-03-10-light-magformer-v2-experiment-plan.md:73-231`.

For the later full-suite work, the rosters and configs show dedicated experiment manifests and per-dataset RGB/depth stats. The fair no-depth rerun metadata records a 20-epoch job with batch size 4, while the 512 Mask2Former pretrained metadata records batch size 8 and a `3850.0` second runtime. Sources: `configs/experiments/full_20260318_1k_1566_roster.json:1`, `configs/stats/20260318_1k_1566_rgb_stats.json:1`, `configs/stats/20260318_1k_1566_depth_stats.json:1`, `output/experiments/20260406_1k_1566_20ep_1024_full19/magformer_nodpth_ref_fair/metadata.json:5`, `output/experiments/20260406_1k_1566_20ep_512_full19/official_mask2former_pretrained/metadata.json:27`.

The software stack is Python plus PyTorch/timm/pycocotools/Pydantic/OmegaConf. A repair log provides the clearest hardware clue: `NVIDIA A100 80GB PCIe` with CUDA 12.4. Sources: `requirements.txt:4-42`, `environment.magformer.yml:10-47`, `output/experiments/20260318_1k_1566_20ep_1024_full19/repair_msm_unet_gpu1.log:19`.

The current operational record is also clear after the 2026-04-13 verification pass on `master`: the short verification run and the follow-on queue were launched under `tmux`, training was restricted to physical GPU 1, and the Python environment needed `PYTHONNOUSERSITE=1` so that the conda CUDA build of PyTorch stayed visible instead of a user-site CPU-only torch package. Sources: `output/experiments/20260413_master_verify_5ep/magformer_nodpth_ref/verify_session.log:1`, `output/experiments/20260413_master_gpu1_queue.log:1`, `environment.magformer.yml:10-47`.

## 7. Experimental Results

### 7.1 Canonical Live Result Source

The authoritative publication-ready narrative source in the repo is now `docs/2026-04-12-final-multi-resolution-results.md`, and its tables are grounded in the live `2026-04-10` CSV plus manifest artifacts. The older 2026-03-29 and 2026-04-06 prose summaries remain useful as historical snapshots only. Sources: `docs/2026-04-12-final-multi-resolution-results.md:1-86`, `output/analysis/2026-04-10-all-models-metrics-1024-512-sorted-by-segm-ap.csv:1-42`, `output/analysis/2026-04-10-live-metrics-manifest-fresh.json:1`.

### 7.2 Consolidated Full-Suite Results (Live 2026-04-12 Rollup)

| model_id | resolution | segm_ap | bbox_ap | fps |
|---|---:|---:|---:|---:|
| mgm_mask2former_depthnorm_on | 1024 | 72.7981 | 61.5294 | 6.5574 |
| magformer_depthnorm_on | 1024 | 68.4177 | 62.2810 | 2.1377 |
| magformer_lightdepth_convnextlite_spatialgate_edge_validhole | 1024 | 65.4526 | 57.6493 | 2.4335 |
| magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole | 1024 | 64.7399 | 57.6262 | 2.4549 |
| magformer_lightdepth_mobilenetv3_sagate_edge_validhole | 1024 | 64.5032 | 57.6543 | 2.5000 |
| magformer_nodpth_ref | 1024 | 59.5226 | 52.5796 |  |
| mask2former | 1024 | 58.7554 | 50.5996 | 15.0370 |
| maskrcnn | 1024 | 54.1012 | 59.4877 | 32.8938 |
| yolov8_seg_l | 1024 | 40.5626 | 61.7331 | 40.8188 |
| yolov8_seg_x | 1024 | 40.3242 | 61.9894 | 42.2827 |
| yolov8_seg_m | 1024 | 40.0791 | 60.8902 | 55.4547 |
| mgm_mask2former_nodpth_ref | 1024 | 39.6095 | 41.0790 | 12.3162 |
| yolov8_seg_s | 1024 | 36.4499 | 56.8499 | 62.2891 |
| yolov8_seg_n | 1024 | 32.2371 | 51.3223 | 52.1020 |
| uoais | 1024 | 17.6743 | 31.4972 | 21.3420 |
| unet_boundary_inst | 1024 | 13.5987 | 12.6285 |  |
| unetpp_boundary_inst | 1024 | 10.8825 | 10.0462 |  |
| unet_semantic_inst | 1024 | 3.5670 | 3.3375 |  |
| ucn | 1024 | 1.0332 | 2.0804 |  |
| msmformer | 1024 | 0.0000 | 0.0000 | 2.9246 |
| mgm_mask2former_depthnorm_on | 512 | 69.9039 | 53.1677 |  |
| magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole | 512 | 60.3642 | 58.4268 |  |
| magformer_depthnorm_on | 512 | 59.9099 | 49.4472 |  |
| magformer_lightdepth_mobilenetv3_sagate_edge_validhole | 512 | 59.7527 | 55.0087 |  |
| magformer_lightdepth_convnextlite_spatialgate_edge_validhole | 512 | 59.7362 | 54.9091 |  |
| magformer_nodpth_ref | 512 | 53.8752 | 49.2648 |  |
| mask2former | 512 | 43.0848 | 40.9818 |  |
| yolov8_seg_x | 512 | 42.9282 | 63.8584 |  |
| yolov8_seg_l | 512 | 42.6816 | 63.6713 |  |
| yolov8_seg_m | 512 | 42.1052 | 62.3341 |  |
| yolov8_seg_s | 512 | 40.2132 | 59.9811 |  |
| maskrcnn | 512 | 38.7801 | 45.6708 |  |
| yolov8_seg_n | 512 | 35.8836 | 54.2440 |  |
| mgm_mask2former_nodpth_ref | 512 | 32.2194 | 34.8756 |  |
| unetpp_boundary_inst | 512 | 10.1788 | 9.4918 |  |
| msmformer | 512 | 8.5705 | 8.3452 |  |
| uoais | 512 | 6.1641 | 15.4310 |  |
| unet_boundary_inst | 512 | 5.7868 | 4.5878 |  |
| unet_semantic_inst | 512 | 3.2537 | 3.1481 |  |
| ucn | 512 | 0.0493 | 0.3724 |  |

Source: `output/analysis/2026-04-12-all-models-metrics-1024-512-sorted-by-segm-ap.csv:3-42`.

### 7.3 High-Level Interpretation of the Live Suite

- Best 1024 segmentation AP is `mgm_mask2former_depthnorm_on` at `72.7981`, while best 1024 bounding-box AP is `magformer_depthnorm_on` at `62.2810`. Sources: `output/experiments/20260318_1k_1566_20ep_1024_full19/mgm_mask2former_depthnorm_on/metrics.cocoeval.json:1`, `output/experiments/20260318_1k_1566_20ep_1024_full19/magformer_depthnorm_on/metrics.cocoeval.json:1`.
- Best 512 segmentation AP is again `mgm_mask2former_depthnorm_on` at `69.9039`, while best 512 bounding-box AP is `yolov8_seg_x` at `63.8584`. Sources: `output/experiments/20260406_1k_1566_20ep_512_full19/mgm_mask2former_depthnorm_on/metrics.cocoeval.json:1`, `output/experiments/20260406_1k_1566_20ep_512_full19/yolov8_seg_x_pretrained/metrics.cocoeval.json:1`.
- Within the MAGFormer family, the 1024 depth-enabled model (`68.4177`) clearly outperforms the live 1024 no-depth reference (`59.5226`). At 512, the best light-depth variants cluster very tightly around `59.7362` to `60.3642`, while the depth-enabled main MAGFormer is `59.9099`. This suggests the light-depth branch preserved most of the 512 segmentation accuracy while changing the cost/performance shape. Source: `output/analysis/2026-04-12-all-models-metrics-1024-512-sorted-by-segm-ap.csv:3-42`.
- The speed trade-off is sharp at 1024. MAGFormer family runs are roughly `2.1-2.5 fps`, MGM/Mask2Former is `6.56 fps`, plain Mask2Former is `15.04 fps`, Mask R-CNN is `32.89 fps`, and YOLOv8 variants are about `40.82-62.29 fps`. Source: `output/analysis/2026-04-12-all-models-metrics-1024-512-sorted-by-segm-ap.csv:3-22`.

### 7.4 Earlier Experiment Reports and Their Meaning

The live 2026-04-12 table is the best source for the full 1024/512 suite, but earlier experiment documents add important context:

- The Stage A light-depth line reports that all lightweight RGB-D candidates beat the RGB-only control on the `0831_1K / 1024 / 20 epoch` setting. Across the March 11 to March 13 documents, the narrative shifts from an early MobileNetV3-Small + Direct Add leader to a later `ConvNeXt-lite + spatial_gate + valid-hole` recommendation. Sources: `docs/experiments/2026-03-11-lightdepth-stage-a-results.md:17-45`, `docs/experiments/2026-03-13-lightdepth-followup-results.md:15-64`.
- The best Stage B `cross_attn` result is reported as `71.1444 AP`, which the docs explicitly treat as worse than the strongest Stage A lightweight branch. Source: `docs/experiments/2026-03-12-lightdepth-v2-final-report.md:40-60`.
- The F5 prior-guided cross-attention line improves plain cross-attention by only `0.2061 AP` and still does not beat the earlier Stage A winners. Source: `docs/experiments/2026-03-13-f5-prior-guided-cross-attention-final-summary.md:25-60`.
- The UCN/MSMFormer repair canary shows that the wrapper and export bugs were real, that UCN recovered from “dead” to merely weak, and that MSMFormer scratch remained collapsed while official pretrained weights restored non-zero results. Source: `docs/experiments/2026-03-25-ucn-msmformer-repair-canary.md:106-167`.

These earlier reports are useful, but they should not be mixed directly with the 20260318 full-suite numbers without naming the dataset lineage and artifact root, because the repo contains multiple result lineages with different datasets, recipes, and repair states. Sources: `docs/experiments/2026-03-29-all-resolutions-instance-segmentation-results.md:5-23`, `docs/experiments/2026-04-08-all-models-all-metrics-all-resolutions.md:3-22`.

### 7.5 Negative Results, Anomalies, and Data Gaps

- `msmformer` remains the clearest negative result in the live 1024 suite, where it still records `0.0000` bbox and segm AP. The live manifest points to a repaired backup root, but the live metrics remain zero there. Source: `output/analysis/2026-04-12-live-metrics-manifest.json`, `output/analysis/2026-04-12-all-models-metrics-1024-512-sorted-by-segm-ap.csv:22`.
- The 512 pretrained Mask2Former run shows metric-computation warnings and `NaN`/zero-AP behavior in its log, which means not all 512 artifacts are equally clean. Sources: `output/experiments/20260406_1k_1566_20ep_512_full19/official_mask2former_pretrained/log.txt:931`, `output/experiments/20260406_1k_1566_20ep_512_full19/official_mask2former_pretrained/log.txt:1052`.
- The April 6 prose tables marked all 512 rows as missing, but the live rollup clearly contains 512 results. This is a documentation staleness issue, not a model result. Sources: `docs/experiments/2026-04-06-all-models-three-resolutions-metrics.md:23`, `output/analysis/2026-04-12-all-models-metrics-1024-512-sorted-by-segm-ap.csv:23-42`.
- Several older reports are partially templated or paused, including the `baselines_0831_1k_5k_*` reports and the paused `final_20k_report.md`, so they should not be treated as final quantitative evidence. Sources: `docs/experiments/baselines_0831_1k_5k_report.md:83-119`, `docs/experiments/baselines_0831_1k_5k_scratch8_report.md:61-106`, `docs/experiments/final_20k_report.md:53-59`.

### 7.6 2026-04-13 Master Verification Run and Queue State

A short verification run on `master` now provides an explicit end-to-end sanity check for the current MAGFormer training path. The run used the `magformer_nodpth_ref` recipe at `1024` resolution on the `20260318_1K_1566` dataset lineage for `5` epochs, saved checkpoints successfully, ran standalone evaluation, and exported COCO metrics. The verification output root contains `model_best.pth`, `coco_instances_results.json`, `metrics.cocoeval.json`, and the per-iteration training log. Sources: `output/experiments/20260413_master_verify_5ep/magformer_nodpth_ref/model_best.pth`, `output/experiments/20260413_master_verify_5ep/magformer_nodpth_ref/coco_instances_results.json`, `output/experiments/20260413_master_verify_5ep/magformer_nodpth_ref/metrics.cocoeval.json:1`, `output/experiments/20260413_master_verify_5ep/magformer_nodpth_ref/metrics_log.jsonl:1`.

The standalone post-training evaluation of `model_best.pth` produced finite, non-zero COCO metrics: `bbox/AP = 49.2911` and `segm/AP = 54.9992`. Those values are lower than the published 20-epoch live-suite numbers, as expected for a 5-epoch verification run, but they are well within a plausible range and confirm that the current `master` training and evaluation path is functioning. Source: `output/experiments/20260413_master_verify_5ep/magformer_nodpth_ref/metrics.cocoeval.json:1`.

After that verification pass, the remaining GPU 1 queue was launched from `master` in `tmux`. It did not need to resume or restart any unfinished run, because every queued job already had a `metrics.cocoeval.json` done marker. The queue therefore finished by skip-on-done-marker logic rather than by new training work or by interruption. Source: `output/experiments/20260413_master_gpu1_queue.log:1-49`.

## 8. Conclusion and Future Work

### 8.1 Key Takeaways

The repository supports three strong conclusions.

1. MAGFormer is a real, maintained RGB-D instance segmentation implementation with a complete research workflow around it, not just an isolated model file. The repo includes configuration, dataset handling, train/eval/infer entry points, experiment runners, output manifests, and test coverage for those pieces. Sources: `setup.py:23-55`, `tools/train.py:31-729`, `tools/evaluate.py:28-127`, `tests/test_eval_runtime_contract.py:121-305`.

2. The project’s clearest strength is practical engineering around experimentation: lightweight RGB-D fusion exploration, controlled multi-resolution benchmarking, baseline repair, and consolidated reporting. The available materials do not support overstating the method as a major new algorithmic breakthrough. Sources: `docs/plans/2026-03-10-light-magformer-v2-experiment-plan.md:13-27`, `docs/plans/2026-03-26-full19-multires-baselines-design.md:5-20`, `docs/experiments/2026-04-06-magformer-project-sync-and-repair-summary.md:3-31`.

3. The strongest live full-suite performer by segmentation AP is the MGM/Mask2Former depth-enabled baseline, not MAGFormer itself. MAGFormer remains competitive, especially on 1024 bbox AP, but the full-suite evidence is mixed rather than one-sided. Sources: `output/analysis/2026-04-12-all-models-metrics-1024-512-sorted-by-segm-ap.csv:3-42`.

### 8.2 Limitations

The main limitations that are directly supported by the available materials are:

- Single-class only. The README, dataset loader, and config validator all enforce a single foreground category. Sources: `README.md:3-10`, `magformer/data/dataset.py:126-159`, `magformer/config/validation.py:205-230`.
- Depth normalization is fragile enough to be a hard precondition. Sources: `magformer/config/validation.py:52-80`, `magformer/utils/depth_sanity.py:53-92`, `tools/train.py:621-665`.
- Some config and reporting surfaces are transitional or stale, including legacy DPE keys and older experiment summaries that are preserved as historical snapshots. Sources: `magformer/config/validation.py:86-129`, `docs/reviews/2026-03-20-repo-audit.md:21-22`, `docs/experiments/2026-04-06-all-models-three-resolutions-metrics.md:23`.
- The April 2 review is partly historical relative to the current tree. Current code and tests show that `tools/evaluate.py` now resolves `args.weights or config.model.weights`, both `Trainer` and `DDPTrainer` use the shared metric-based evaluation path, normal validation no longer logs `val/loss`, zero-mAP runs can still save `model_best.pth`, and the dataset loader now hard-fails on multi-category inputs. The repaired code paths and the current report now agree on those points. Sources: `tools/evaluate.py:107-112`, `magformer/engine/trainer.py:562-627`, `magformer/engine/trainer.py:810-847`, `magformer/data/dataset.py:126-159`, `tests/test_eval_runtime_contract.py:162-171`, `tests/test_eval_runtime_contract.py:250-257`, `tests/test_eval_runtime_contract.py:300-304`, `tests/test_eval_runtime_contract.py:347-351`, `tests/test_eval_runtime_real_smoke.py:40-65`.
- The project itself states that there is still no output-consistency loss or RGB-teacher distillation path in the current tree. Source: `docs/experiments/2026-04-06-magformer-project-sync-and-repair-summary.md:45-49`.

### 8.3 Future Work and Open Issues

The repository materials point to these next steps:

- Keep the paper-ready, lineage-clean report aligned with the current live `1024/512` suite as new artifacts land. The canonical report and manifest now agree on the publication scope. Sources: `docs/experiments/2026-03-29-all-resolutions-instance-segmentation-results.md:80-94`, `docs/experiments/2026-04-08-all-models-all-metrics-all-resolutions.md:23-62`, `output/analysis/2026-04-12-all-models-metrics-1024-512-sorted-by-segm-ap.md:1-42`.
- Keep operational discipline aligned with the repaired runtime: launch future training from `master`, mount long jobs in `tmux` or `nohup`, reserve physical GPU 1 for MAGFormer work on the shared server, and keep `PYTHONNOUSERSITE=1` set so the CUDA PyTorch build from the `magformer` environment is not shadowed by a user-site CPU-only install. Sources: `output/experiments/20260413_master_verify_5ep/magformer_nodpth_ref/verify_session.log:1`, `output/experiments/20260413_master_gpu1_queue.log:1`, `environment.magformer.yml:10-47`.

### 8.4 What Could Not Be Determined

Some required details could not be recovered confidently from the available materials:

- `[Not Found]` A definitive end-user deployment target outside research and benchmarking.

---

## Document Revision Log

| Date | Section | Change | Source |
|------|---------|--------|--------|
| 2026-04-12 | Sections 1-3, 7 | Updated live benchmark citations and the canonical full-suite result source from the April 10 rollup to the April 12 canonical `1024/512` publication set | `output/analysis/2026-04-12-live-metrics-manifest.json`, `output/analysis/2026-04-12-all-models-metrics-1024-512-sorted-by-segm-ap.md`, `output/analysis/2026-04-12-all-models-metrics-1024-512-sorted-by-segm-ap.csv` |
| 2026-04-12 | Section 6 | Replaced the unavailable `0831_1K` scene count with `[Not Found]` and clarified that the repo's `256` materials are historical background rather than the current live publication scope | `magformer_datasets/0831_1K/annotations/instances_train.json`, `magformer_datasets/0831_1K/annotations/instances_val.json`, `magformer_datasets/0831_1K/annotations/instances_test.json`, `scripts/experiments/run_20260409_non256_completion_gpu1.sh`, `output/analysis/2026-04-12-all-models-metrics-1024-512-sorted-by-segm-ap.csv` |
| 2026-04-12 | Section 8.2 | Replaced the stale open-risk summary with the current evaluation, checkpoint, and single-class behavior shown by code and tests | `tools/evaluate.py`, `magformer/engine/trainer.py`, `magformer/data/dataset.py`, `tests/test_eval_runtime_contract.py`, `tests/test_eval_runtime_real_smoke.py` |
| 2026-04-12 | Section 8.3 | Removed wording that implied `256` publication, MSMFormer retraining, or UCN retraining is currently required and kept only current-state wording | `output/analysis/2026-04-12-live-metrics-manifest.json`, `output/analysis/2026-04-12-all-models-metrics-1024-512-sorted-by-segm-ap.md`, `docs/experiments/2026-04-06-fair-nodpth-config-diff.md` |
| 2026-04-12 | Section 8.4 | Converted remaining unresolved items to `[Not Found]` and removed the unsupported privacy/annotation and 256-table unknowns | `README.md`, `setup.py`, `magformer_datasets/20260318_1K_1566/build_stats.json`, `scripts/experiments/run_20260409_non256_completion_gpu1.sh` |
| 2026-04-13 | Header, Sections 6.4, 7.1, 7.6, 8.3, 8.4 | Added the confirmed `master` verification run state, recorded the GPU1-only `tmux` queue completion, switched the canonical narrative source to the final multi-resolution report, and removed the outdated build-stats `[Not Found]` note now that the discrepancy is documented | `docs/2026-04-12-final-multi-resolution-results.md`, `docs/data/2026-04-12-build-stats-count-discrepancy-note.md`, `output/experiments/20260413_master_verify_5ep/magformer_nodpth_ref/metrics.cocoeval.json`, `output/experiments/20260413_master_gpu1_queue.log` |
