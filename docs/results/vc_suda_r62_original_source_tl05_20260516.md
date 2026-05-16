# VC-SUDA R62 original source + target_labeled 0.5 result - 2026-05-16

## Conclusion

R62 fails all hard gates. Original first50 source sanity is below the required `0.55`, target_unlabeled200 is near zero, and target prediction boxes remain much larger than target GT.

Because target AP is low and target bbox area ratio is above `3x`, the direct conclusion is: the stronger target_labeled signal is still insufficient under the original source anchor.

## Setup

- Config: `configs/vc_suda_stage_b_r62_original_source_tl05_1024.yaml`
- Base: R59 config, config-only experiment.
- Warm start: `output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth`
- Source root/ann: `magformer_datasets/20260318_1K_1566/annotations/instances_train.json`
- Target root: `magformer_datasets/pseudo_real_512`
- Stage: `B`
- `target_labeled_weight`: `0.5`
- Unsupervised, EMA teacher, and domain adaptation weights: `0.0` / disabled.
- No `freeze_modules`; Stage C not enabled.

## Config and Test Verification

Added a focused config/data protocol test:

- `tests/test_vc_suda_data_protocol.py::test_stage_b_r62_config_routes_original_source_and_pseudo_real_target`

It verifies:

- R62 loads `data.dataset_root=magformer_datasets/pseudo_real_512`.
- Source is routed through `vc_suda.source_root=magformer_datasets/20260318_1K_1566` and `vc_suda.source_ann=annotations/instances_train.json`.
- Target labeled stays on pseudo_real_512 with `annotations/instances_target_labeled.json`.
- `target_labeled_weight=0.5` is parsed.
- Stage B does not instantiate target_unlabeled training data.

Test command:

```bash
/home/hdd3/zhanghaonan/anaconda3/bin/conda run -n magformer pytest \
  tests/test_vc_suda_data_protocol.py \
  tests/test_vc_suda_trainer_runtime.py::test_stage_b_target_labeled_weight_scales_total_loss \
  tests/test_vc_suda_trainer_runtime.py::test_stage_b_target_labeled_weight_metrics_are_logged \
  tests/test_train_warm_start.py \
  tests/test_teacher_first50_protocol.py -q
```

Result: `44 passed`.

Config/test commit: `94c4a159a0931627ddd54eb52e7902cd023437ea`.

Push verification:

```text
94c4a159a0931627ddd54eb52e7902cd023437ea refs/heads/feature/vc-suda-sim2real
```

Direct `git push` from 4029 hung, so the config/test commit was pushed through a full git bundle and local bridge, then verified with `git ls-remote`.

## Training

Training ran in tmux on physical GPUs `4,5,6,7` as local `--gpus 0,1,2,3`.

- tmux session: `r62_original_source_tl05_20260516`
- log: `output/logs/r62_original_source_tl05_20260516.log`
- output: `output/vc_suda/vc_suda_stage_b_r62_original_source_tl05_1024`
- checkpoints: `checkpoint_iter_0000249.pth`, `checkpoint_iter_0000250.pth`
- trainer peak memory: `14863.919921875 MB`
- server RAM stayed below 90%.

Command:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
MAGFORMER_MS_DEFORM_ATTN_BACKEND=cuda \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=4 tools/train.py \
  --config configs/vc_suda_stage_b_r62_original_source_tl05_1024.yaml \
  --gpus 0,1,2,3
```

The first launch without `torchrun` failed before training because DDP requires a distributed environment. It produced no checkpoint. The rerun above completed and produced the final checkpoint.

Built-in val28 smoke at iter 250:

| Metric | AP | AP50 | AP75 |
|---|---:|---:|---:|
| bbox | `0.0013869778` | `0.0093467423` | `0.0` |
| segm | `0.0005123286` | `0.0036586700` | `0.0` |

This is only the built-in smoke signal, not the R62 decision metric.

## External Evals

Evaluated checkpoint:

- `output/vc_suda/vc_suda_stage_b_r62_original_source_tl05_1024/checkpoint_iter_0000250.pth`

### Original first50 source sanity

Output:

- `output/diagnostics/r62_original_source_tl05_ckpt0250_original_first50_20260516`

Protocol:

- base config: `configs/finetune_1k_full_1024.yaml`
- dataset root: `magformer_datasets/20260318_1K_1566`
- annotation: `annotations/instances_all.json`
- split: `all`
- max images: `50`
- image size: `1024`
- score/mask: `0.05/0.5`
- topk/maxDets: `100/100`
- IoU: `bbox,segm`
- PyTorch MSDA forced

Strict load: `774/774`, missing `0`, unexpected `0`, shape mismatch `0`.

| Metric | AP | AP50 | AP75 |
|---|---:|---:|---:|
| bbox | `0.5383536360` | `0.7892650425` | `0.5793874594` |
| segm | `0.4905393484` | `0.7868918221` | `0.5366512732` |

Hard gate: fail, because source first50 segm AP `0.4905393484 < 0.55`.

### Target_unlabeled200

Output:

- `output/diagnostics/r62_original_source_tl05_ckpt0250_target_unlabeled200_20260516`

Protocol:

- base config: `configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml`
- dataset root: `magformer_datasets/pseudo_real_512`
- annotation: `annotations/instances_target_unlabeled.json`
- split: `train`
- image size: `1024`
- score/mask: `0.05/0.5`
- topk/maxDets: `200/200`
- IoU: `bbox,segm`
- PyTorch MSDA forced

Strict load: `774/774`, missing `0`, unexpected `0`, shape mismatch `0`.

Prediction count: `25,265` over `200` images.

| Metric | AP | AP50 | AP75 |
|---|---:|---:|---:|
| bbox | `0.0013986618` | `0.0052943199` | `0.0009900990` |
| segm | `0.0012286272` | `0.0038254259` | `0.0009900990` |

Hard gate: fail, because target_unlabeled200 segm AP `0.0012286272 < 0.30`.

## Target Scale

Stats file:

- `output/diagnostics/r62_original_source_tl05_ckpt0250_target_unlabeled200_20260516/bbox_area_ratio_summary.json`

| Scope | GT bbox area p50 | Pred bbox area p50 | Ratio |
|---|---:|---:|---:|
| Over-image p50 | `779.75` | `4558.0` | `5.8454632895` |
| Global instance p50 | `744.0` | `4482.0` | `6.0241935484` |

Hard gate: fail, because target bbox area ratio `5.845x > 2.0`.

## Gate Decision

R62 hard metrics:

| Gate | Required | Actual | Result |
|---|---:|---:|---|
| original first50 source segm AP | `>=0.55` | `0.4905393484` | fail |
| target_unlabeled200 segm AP | `>=0.30` | `0.0012286272` | fail |
| target bbox area p50 / GT p50 | `<=2.0` | `5.8454632895` | fail |

R62 fails.

The target AP is low and the area ratio is above `3x`, so the target_labeled signal is still insufficient under the original source anchor. This stronger `target_labeled_weight=0.5` did not fix the R59 target-scale failure, and it also did not retain the source sanity threshold.
