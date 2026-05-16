# VC-SUDA R58 R8B/R12 early source sanity - 2026-05-16

## Conclusion

The source sanity collapse has two parts.

R8B `ckpt999` is already below the R44 Teacher source-sanity line, so part of the collapse happened before R12. R12 then drops much further within the first 249 iterations, from R8B segm AP `0.4427243041` to R12 `ckpt249` segm AP `0.2263511419`. By R12 `ckpt499`, the result is still collapsed at segm AP `0.2361521319`.

## Scope

No training was run.

Both checkpoints were evaluated on the original 1.5K first50 1024 protocol:

- base config: `configs/finetune_1k_full_1024.yaml`
- dataset root: `magformer_datasets/20260318_1K_1566`
- annotation: `annotations/instances_all.json`
- split: `all`
- image size: `1024`
- max images: `50`
- score/mask threshold: `0.05/0.5`
- IoU types: `bbox,segm`
- inference topk / COCO maxDets: `100/100`
- batch size / workers: `4/0`
- GPU: `4`
- MSDA: PyTorch backend with `--force-pytorch-msda`

`tools/check_eval_protocol.py original_first50_teacher` was run before each eval with `--allow-nondefault-weights`; both summaries report `checks.status=pass`.

The `magformer` conda environment needed the conda-packaged CUDA library path exported for `torch` import:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH
```

This only changed the shell environment for eval. No repository code was changed for this.

## Checkpoints

| Run | Checkpoint | Eval output |
|---|---|---|
| R8B ckpt999 | `output/vc_suda/stage_c_r8b_lsj10_low_lr_continue_1024_teacher8499/checkpoint_iter_0000999.pth` | `output/diagnostics/r58_r8b_ckpt0999_original_first50_20260516` |
| R12 ckpt249 | `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000249.pth` | `output/diagnostics/r58_r12_ckpt0249_original_first50_20260516` |

## Metrics

| Run | Source first50 bbox AP/AP50/AP75 | Source first50 segm AP/AP50/AP75 |
|---|---:|---:|
| R44 Teacher | `0.6519530084/0.8531437401/0.7175465372` | `0.6252930576/0.8679527609/0.7243386328` |
| R8B ckpt999 | `0.4991705518/0.7839142318/0.5401542953` | `0.4427243041/0.7562824128/0.4681354863` |
| R12 ckpt249 | `0.2874553120/0.6270917129/0.2413931660` | `0.2263511419/0.5506509459/0.1456172833` |
| R12 ckpt499 | `0.2921127104/0.6266512287/0.2366207989` | `0.2361521319/0.5620645370/0.1582073515` |

## Interpretation

The R44 Teacher reference is the existing corrected original first50 result from R44: segm AP `0.6252930576`.

R8B `ckpt999` is already down by `-0.1825687535` segm AP versus Teacher. That means the source sanity was not fully preserved before R12 started.

R12 `ckpt249` is down by another `-0.2163731623` segm AP versus R8B. This means the largest measured drop in this two-point check happens during R12 iterations `0-249`.

R12 `ckpt499` is close to R12 `ckpt249`, with segm AP `0.2361521319` versus `0.2263511419`. So by the first 249 R12 iterations, the source-sanity collapse has already reached the later R12 collapsed band.

## Decision

Both segments contributed:

- Before R12: R8B is already below the Teacher source-sanity line.
- R12 `0-249`: source sanity drops from partial damage to the collapsed R12 band.

The sharp collapse to the R12/R56 level happens within R12 `0-249` iterations, but the degradation begins before R12.

## Validation

- R8B protocol summary: `output/diagnostics/r58_r8b_ckpt0999_original_first50_20260516/protocol_check.summary.json`, `checks.status=pass`.
- R12 protocol summary: `output/diagnostics/r58_r12_ckpt0249_original_first50_20260516/protocol_check.summary.json`, `checks.status=pass`.
- R8B eval log has `[Weights] strict load OK`, `[MSDeformAttn] forced PyTorch core path`, and `evaluated_images=50 max_images=50`.
- R12 eval log has `[Weights] strict load OK`, `[MSDeformAttn] forced PyTorch core path`, and `evaluated_images=50 max_images=50`.
- Log scan found no `Traceback`, `OOM`, `out of memory`, or `CUDA out of memory` in the successful eval logs.
- GPU use stayed on GPU 4; GPUs 5-7 stayed idle.
- No training command was run for R58.
