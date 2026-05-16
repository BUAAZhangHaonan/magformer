# VC-SUDA R39 Checkpoint Inventory Eval - 2026-05-16

## Conclusion

No evaluated ready checkpoint beats the R12/R33 `segm AP 0.320048` teacher line.

The best R39 candidate is R11 `checkpoint_iter_0000749.pth` with segm AP `0.3199859441`, which is lower than `0.320048` by about `0.0000620559`. So the repository has no ready checkpoint that is a better teacher than R12/R33.

No training was run. No code was changed.

## Protocol

- Host: `4029` / `WS-4029GP-TRT`.
- Repo: `/home/hdd3/zhanghaonan/magformer`.
- Environment: clean `env -i` command shell with physical `CUDA_VISIBLE_DEVICES=4`.
- MSDA: `MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch` plus `--force-pytorch-msda`.
- Script: `tools/evaluate_1024_backmap.py`.
- Base config: `configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml`.
- Dataset: `magformer_datasets/pseudo_real_512`.
- Ann: `annotations/instances_target_unlabeled.json`.
- Split: `train`.
- Image size: `1024`.
- Batch size: `1`.
- Num workers: `0`.
- IoU types: `bbox,segm`.
- Score threshold: `0.05`.
- Mask threshold: `0.5`.
- Inference topk / maxDets: `200 / 200`.
- Inference stats: enabled for prediction count and topk truncation evidence.

## Smoke Checks

Each candidate first ran the same protocol with `--max-images 3`. All candidates passed strict load and smoke inference.

| Candidate | Smoke output | Strict load | Smoke status |
|---|---|---|---|
| R12 ckpt0749 | `output/diagnostics/r39_r12_ckpt0749_target_unlabeled200_1024_backmap_20260516_smoke3` | `774/774`, missing `0`, unexpected `0`, shape mismatch `0` | `EXIT_CODE=0` |
| R35 ckpt0750 | `output/diagnostics/r39_r35_ckpt0750_target_unlabeled200_1024_backmap_20260516_smoke3` | `774/774`, missing `0`, unexpected `0`, shape mismatch `0` | `EXIT_CODE=0` |
| R11 ckpt0749 | `output/diagnostics/r39_r11_ckpt0749_target_unlabeled200_1024_backmap_20260516_smoke3` | `774/774`, missing `0`, unexpected `0`, shape mismatch `0` | `EXIT_CODE=0` |

No candidate was skipped.

## Full Eval Results

| Candidate | Checkpoint | Output dir | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | Predictions | topk truncated |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R12 ckpt0749 | `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000749.pth` | `output/diagnostics/r39_r12_ckpt0749_target_unlabeled200_1024_backmap_20260516` | `0.3920480931` | `0.7329485443` | `0.3718585045` | `0.3184151431` | `0.6494638719` | `0.2818239164` | `13592` | `0/200` |
| R35 ckpt0750 | `output/vc_suda/stage_c_r35_original_split_true_resume_1024_teacher8499/checkpoint_iter_0000750.pth` | `output/diagnostics/r39_r35_ckpt0750_target_unlabeled200_1024_backmap_20260516` | `0.3942516678` | `0.7336598544` | `0.3799342310` | `0.3193067219` | `0.6494913282` | `0.2834525522` | `13562` | `0/200` |
| R11 ckpt0749 | `output/vc_suda/stage_c_r11_r8b_ckpt999_mask_loss75_continue_1024_teacher8499/checkpoint_iter_0000749.pth` | `output/diagnostics/r39_r11_ckpt0749_target_unlabeled200_1024_backmap_20260516` | `0.3943762547` | `0.7394137030` | `0.3732029907` | `0.3199859441` | `0.6491729062` | `0.2846548280` | `14495` | `0/200` |

## Readout

- Reference line: R12/R33 segm AP `0.320048`.
- Best R39 checkpoint: R11 ckpt0749 segm AP `0.3199859441`.
- Delta from reference: `-0.0000620559`.
- Decision: do not promote any of these checkpoints as a better teacher.

## Validation

- All full evals processed `200` images and ended with `EXIT_CODE=0`.
- All full eval logs confirm forced PyTorch MSDA and strict weight load with `0` missing, `0` unexpected, and `0` shape mismatches.
- Log scan found no `Traceback`, `CUDA error`, `OOM`, `out of memory`, `RuntimeError`, or `non-finite` hits in smoke or full eval logs.
- `inference_stats.json` summary reports `topk_truncated_images: 0` and `total_images: 200` for all full evals.
- Output directories are under `output/diagnostics/` and are ignored by git.
