# VC-SUDA R67 Multisource Freeze Backbones

## Conclusion

R67 fails the gate. Freezing RGB backbone, depth backbone, fusion, and AGPE keeps target scale under the `2.0x` limit, but it does not recover source first50 AP and it weakens target AP.

## Setup

- Config: `configs/vc_suda_stage_b_r67_multisource_freeze_backbones_1024.yaml`
- Base: R65 multisource retention config.
- Warm start: `output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth`
- Source mix: `original:pseudo = 1:1`
- Stage: `B`
- `target_labeled_weight: 1.0`
- `unsupervised_weight: 0.0`
- Iterations: `500`
- Frozen prefixes: `rgb_backbone`, `depth_backbone`, `fusion`, `agpe`
- Train output: `output/vc_suda/vc_suda_stage_b_r67_multisource_freeze_backbones_1024`
- Train log: `output/vc_suda/vc_suda_stage_b_r67_multisource_freeze_backbones_1024/train_tmux_20260517.log`

## Commits

- Config/test milestone: `1de2a72a1225eb5b965f0c182d6c11b1bf407613`

Direct remote push from `4029` timed out. The commit was pushed by bundle bridge and verified with `git ls-remote`:

```text
1de2a72a1225eb5b965f0c182d6c11b1bf407613	refs/heads/feature/vc-suda-sim2real
```

## Freeze Check

The real R67 model check applied the configured prefixes before optimizer construction:

```text
matched_prefixes={'rgb_backbone': 177, 'depth_backbone': 138, 'fusion': 36, 'agpe': 14}
frozen_parameter_tensors=365
trainable_decoder_count=175
trainable_pixel_decoder_count=117
```

Training log confirmed the same freeze count on all ranks and reduced trainable parameters to `19,931,842`.

## Verification Before Training

- Freeze tests and real R67 model-prefix test: passed.
- Multi-source tests and entrypoint regression group: passed.
- Warm-start tests: passed.
- Teacher protocol tests: passed.
- Eval protocol tests: passed.
- Full targeted pytest command: `43 passed`.
- `ruff check tools/train.py tests/test_train_warm_start.py`: passed.
- `git diff --check`: passed.
- 1024 no-backward smoke: passed.

No-backward smoke evidence:

```text
source_dataset_names_first4=['original', 'pseudo', 'original', 'pseudo']
source_images=(4, 3, 1024, 1024)
source_depths=(4, 1, 1024, 1024)
target_labeled_images=(4, 3, 1024, 1024)
target_labeled_depths=(4, 1, 1024, 1024)
```

## Training

- tmux session: `r67_multisource_freeze_20260517`
- GPUs: physical `4,5,6,7`, exposed as local `0,1,2,3`
- Command used `CUDA_VISIBLE_DEVICES=4,5,6,7` and `torch.distributed.run --nproc_per_node=4`.
- Max sampled RAM during training: about `31.8 / 257.6 GB`, below the 90% cap.
- Max sampled GPU memory: about `13.8 / 24 GB`.
- Warm start: strict load matched `774/774`, missing `0`, unexpected `0`.
- Checkpoints:
  - `checkpoint_iter_0000499.pth`
  - `checkpoint_iter_0000500.pth`
- Internal val28 at iter 499: bbox AP `0.0440`, segm AP `0.0234`.
- Internal val28 at iter 500: bbox AP `0.0442`, segm AP `0.0236`.

## External Eval

All external evals used `checkpoint_iter_0000499.pth`.

| Split | Output | bbox AP/AP50/AP75 | segm AP/AP50/AP75 |
|---|---|---:|---:|
| original first50 | `output/diagnostics/r67_multisource_freeze_ckpt0499_original_first50_20260517` | `0.533748 / 0.791183 / 0.582487` | `0.494717 / 0.787543 / 0.536951` |
| target_unlabeled200 | `output/diagnostics/r67_multisource_freeze_ckpt0499_target_unlabeled200_20260517` | `0.147937 / 0.446801 / 0.058649` | `0.102135 / 0.357419 / 0.017971` |

Scale diagnostics:

- Target bbox area ratio: `1.926579`
- Target mask area ratio: `1.873443`
- GT bbox area p50 over images: `779.75`
- Pred bbox area p50 over images: `1502.25`
- GT mask area p50 over images: `501.75`
- Pred mask area p50 over images: `940.0`
- Prediction count p50 over images: `98.0`
- Ratio JSON: `output/diagnostics/r67_multisource_freeze_ckpt0499_target_unlabeled200_20260517/target_bbox_mask_area_ratios.json`
- Dense audit: `output/diagnostics/r67_multisource_freeze_ckpt0499_target_unlabeled200_20260517/r67_multisource_freeze_ckpt0499_target_unlabeled200_audit.md`

## Gate Decision

Fail.

| Gate | Required | R67 ckpt0499 | Result |
|---|---:|---:|---|
| Source first50 segm AP | `>=0.55` | `0.494717` | fail |
| Target bbox area ratio | `<=2.0` | `1.926579` | pass |
| Short-run target segm AP | `>=0.22` hopeful | `0.102135` | fail |
| Full target segm AP | `>=0.30` | `0.102135` | fail |

R67 answers the scale question only partly. The target masks stay near target scale, but freezing the representation path does not preserve enough source quality and sharply lowers target AP. There is no reason to extend this run.
