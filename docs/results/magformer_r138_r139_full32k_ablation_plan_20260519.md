# MagFormer R138/R139 Full 32K Depth Ablation Plan - 2026-05-19

## Conclusion Target

This round should validate the depth gain of MagFormer RGB-D itself. It should not continue the VC-SUDA path by stacking more modules.

The core comparison is:

- R138: current RGB-D MagFormer source checkpoint on the full 32K source validation split.
- R139: RGB-only target150 finetune from the same R98 checkpoint, with depth backbone and modality fusion disabled.
- R140: zero-depth or shuffled-depth ablation is deferred because there is no existing clean switch for it.

## R138: Full 32K Source Validation

Purpose: measure the R98 source checkpoint on the complete source validation protocol before changing training.

Required setup:

- Checkpoint: `output/baseline/r98_magformer_32k_source_short/checkpoint_iter_0000499.pth`
- Validation set: full 32K source val, `3276 images`
- Output directory: `output/diagnostics/r138_magformer_32k_fullval_20260519`
- No subset is allowed.
- The command must not contain `--max-images`, `topk subset`, `first300`, or any equivalent image limit.

Completion standard:

- The run logs show that the R98 checkpoint was loaded.
- The run uses the complete 32K source validation split, or at minimum has no subset/image-limit option in the actual command or logs.
- Inference starts on the validation set.
- Evaluation artifacts are written under the requested output directory.

## R139: RGB-only R98-warm Target150 Finetune

Purpose: compare the RGB-only target150 path against the RGB-D MagFormer path while keeping the initialization fixed.

Required setup:

- Base config: `configs/baseline_supervised_r114_magformer_r113warm_target150_balanced_2000.yaml`
- Initialization checkpoint: `output/baseline/r98_magformer_32k_source_short/checkpoint_iter_0000499.pth`
- Train split: `annotations/instances_target_labeled_r114_balanced_plus125.json`, `150 images`
- Val split: `annotations/instances_val.json`, `28 images`
- Output directory: `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519`
- RGB-only switches:
  - `model.magformer.depth_backbone.enabled=false`
  - `model.magformer.modality_fusion.enabled=false`

Execution order:

1. Run a smoke test first.
2. Start the formal target150 finetune only after the smoke test is valid.

Completion standard:

- The smoke run confirms that the R98 checkpoint can initialize the RGB-only model path.
- The formal run uses the R114 target150 protocol and the two RGB-only switches above.
- Training and validation artifacts are written under the requested output directory.

## R140: Zero-depth / Shuffled-depth Ablation

R140 should not start in this round.

Reason: there is no existing clean switch for zero-depth or shuffled-depth behavior. Adding a temporary fallback, heuristic, local post-processing path, or one-off data hack would not be a faithful ablation. This needs a proper implementation plan before launch.

## Risks

- R138 may be slower than prior diagnostics because it must evaluate all `3276 images` and must not use a subset.
- R139 may need a config override path that cleanly disables both depth backbone and modality fusion. If the existing launcher cannot express these switches cleanly, the run should stop instead of adding a local workaround.
- The R98 checkpoint may include RGB-D-specific weights. R139 must verify that RGB-only initialization is clean before formal finetuning.
- R140 remains blocked until a general, reviewable zero/shuffle depth mechanism exists.
