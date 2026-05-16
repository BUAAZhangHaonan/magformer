# VC-SUDA R56 source sanity timeline - 2026-05-16

## Conclusion

Source sanity was already lost by R12 `ckpt499` on the original 1.5K first50 protocol.

R12 reaches only segm AP/AP50/AP75 `0.2361521319/0.5620645370/0.1582073515`, far below the R44 Teacher source-sanity line `0.6252930576/0.8679527609/0.7243386328`. R35 remains similarly low at segm AP `0.2265258169`. R46 and R52 partially recover source-domain AP to about `0.40`, but they still do not retain the Teacher `61+` source sanity.

This is not a target-domain gate. R46 remains the best target_unlabeled200 checkpoint in this table at segm AP `0.3232518088`, but that target gain sits on top of a source-sanity collapse that happened before R46.

## Scope

No training was run.

R56 evaluates source sanity on the corrected R54/R44 original first50 contract:

- base config: `configs/finetune_1k_full_1024.yaml`
- dataset root: `magformer_datasets/20260318_1K_1566`
- annotation: `annotations/instances_all.json`
- split: `all`
- image size: `1024`
- max images: `50`
- score/mask: `0.05/0.5`
- IoU types: `bbox,segm`
- inference topk / COCO maxDets: `100/100`
- batch size / workers: `4/0`
- GPU: `4`
- MSDA: PyTorch backend with `--force-pytorch-msda`

## Checkpoints

| Run | Checkpoint | Availability | Eval output |
|---|---|---|---|
| R12 ckpt499 | `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth` | available | `output/diagnostics/r56_source_sanity_r12_ckpt0499_original_first50_20260516` |
| R35 ckpt750 | `output/vc_suda/stage_c_r35_original_split_true_resume_1024_teacher8499/checkpoint_iter_0000750.pth` | available | `output/diagnostics/r56_source_sanity_r35_ckpt0750_original_first50_20260516` |
| R46 ckpt750 | `output/vc_suda/stage_c_r46_pseudoreal_source_true_resume_1024_teacher8499/checkpoint_iter_0000750.pth` | already evaluated in R55 | `output/diagnostics/r55_r46_ckpt0750_original_first50_1024_backmap_20260516` |
| R52 ckpt750 | `output/vc_suda/stage_c_r52_pseudoreal_source_target_sampling_true_resume_1024_teacher8499/checkpoint_iter_0000750.pth` | available | `output/diagnostics/r56_source_sanity_r52_ckpt0750_original_first50_20260516` |

## Command Summary

Each new eval first ran the static checker:

```bash
python tools/check_eval_protocol.py original_first50_teacher \
  --base-config configs/finetune_1k_full_1024.yaml \
  --dataset-root magformer_datasets/20260318_1K_1566 \
  --ann annotations/instances_all.json \
  --split all \
  --weights <checkpoint> \
  --image-size 1024 \
  --max-images 50 \
  --score-threshold 0.05 \
  --mask-threshold 0.5 \
  --iou-types bbox,segm \
  --inference-topk 100 \
  --max-dets 100 \
  --allow-nondefault-weights \
  --summary-json <output>/protocol_check.summary.json
```

Each new eval then ran:

```bash
CUDA_VISIBLE_DEVICES=4 \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
NUMEXPR_NUM_THREADS=4 \
MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch \
python tools/evaluate_1024_backmap.py \
  --base-config configs/finetune_1k_full_1024.yaml \
  --dataset-root magformer_datasets/20260318_1K_1566 \
  --ann annotations/instances_all.json \
  --split all \
  --weights <checkpoint> \
  --output-dir <output> \
  --image-size 1024 \
  --batch-size 4 \
  --num-workers 0 \
  --score-threshold 0.05 \
  --mask-threshold 0.5 \
  --iou-types bbox,segm \
  --inference-topk 100 \
  --max-dets 100 \
  --max-images 50 \
  --dump-inference-stats <output>/inference_stats.json \
  --force-pytorch-msda
```

R46 was not rerun. R56 reads the R55 output directly.

## Metrics

| Run | Source first50 bbox AP/AP50/AP75 | Source first50 segm AP/AP50/AP75 | Source segm AP delta vs R44 Teacher | target_unlabeled200 segm AP/AP50/AP75 |
|---|---:|---:|---:|---:|
| R44 Teacher | `0.6519530084/0.8531437401/0.7175465372` | `0.6252930576/0.8679527609/0.7243386328` | `0.0000000000` | n/a |
| R12 ckpt499 | `0.2921127104/0.6266512287/0.2366207989` | `0.2361521319/0.5620645370/0.1582073515` | `-0.3891409258` | `0.3200483457/0.6483631209/0.2841442923` |
| R35 ckpt750 | `0.2805064979/0.6142375111/0.2195981934` | `0.2265258169/0.5415158177/0.1478177865` | `-0.3987672407` | `0.3192995977/0.6495941396/0.2832123216` |
| R46 ckpt750 | `0.4657849322/0.7669137408/0.4908037106` | `0.4009417296/0.7265082520/0.3993570474` | `-0.2243513280` | `0.3232518088/0.6497278126/0.2874850698` |
| R52 ckpt750 | `0.4706981355/0.7760254840/0.4959568635` | `0.4049332088/0.7271635015/0.4016095950` | `-0.2203598488` | `0.3225963120/0.6494580497/0.2869152875` |

## Timeline Interpretation

The source sanity is not first lost at R46. It is already gone at R12 `ckpt499`.

The sequence is:

1. R44 Teacher original first50: healthy source sanity, segm AP `0.6252930576`.
2. R12 `ckpt499`: major source-sanity collapse to `0.2361521319` while target_unlabeled200 is `0.3200483457`.
3. R35 `ckpt750`: original-split true resume does not recover source sanity; it is slightly lower at `0.2265258169`.
4. R46/R52: pseudo-real source variants recover source first50 to about `0.40`, but still remain about `0.22` AP below Teacher.

Target AP improvement is therefore not explained as R46 trading away additional source sanity relative to R12. From R12 to R46, target segm AP improves by about `+0.003203`, and source first50 segm AP also improves by about `+0.164790`. The real cost happened earlier: the adapted Stage C path reached the target-domain `0.32` band only after the original-source Teacher sanity had already collapsed.

## Decision

Stop treating more target small-variable changes as the main path. The timeline points to a missing source-retention or objective-level constraint.

The next design should add a source-retention check/objective to the Stage C protocol, and source first50 should be evaluated as a companion sanity metric when a target-domain gate is promoted. R46 can stay the current target-domain best, but it should not be described as preserving the original-source `61+` line.

## Validation

- R12, R35, and R52 protocol checker summaries exist and report `checks.status=pass`.
- R46 R55 protocol checker summary exists and reports `checks.status=pass`.
- All four eval logs contain `[Weights] strict load OK` and `[MSDeformAttn] forced PyTorch core path`.
- Log scan count for `Traceback`, `OOM`, `out of memory`, and `CUDA out of memory` is `0` for R12, R35, R46, and R52.
- No training command was run for R56.
