# VC-SUDA R46 Pseudo-Real Source True-Resume Gate - 2026-05-16

## Conclusion

R46 passes the 250-iter gate.

The only training variable was changing the VC-SUDA source from the R35/R12 original 32K source split to `pseudo_real_512/annotations/instances_source.json`. The formal external `target_unlabeled200` 1024 backmap result at `ckpt0750` is segm AP/AP50/AP75 `0.3232518088/0.6497278126/0.2874850698`, above the R12/R33 baseline `0.3200483457/0.6483631209/0.2841442923`.

Do not start a follow-up experiment from this document. R46 is a single-variable pass record only.

## Single Variable

Base config:

- `configs/vc_suda_stage_c_r35_original_split_true_resume_1024_teacher8499.yaml`

R46 config:

- `configs/vc_suda_stage_c_r46_pseudoreal_source_true_resume_1024_teacher8499.yaml`

Allowed differences from R35:

- `name`, `runtime.output_dir`, `runtime.logger.log_dir`, and `runtime.logger.run_name` changed from R35 to R46 identity.
- `vc_suda.source_root: magformer_datasets/20260318_1K_32254 -> magformer_datasets/pseudo_real_512`.
- `vc_suda.source_ann: annotations/instances_train.json -> annotations/instances_source.json`.

Held fixed:

- Target split: `target_labeled=25`, `target_unlabeled=200`, `val=28`.
- True resume: `runtime.resume: output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth`.
- `model.finetune_weights: null`.
- `data.image_size: 1024`.
- `solver.max_iter: 750`, `solver.ims_per_batch: 4`, `solver.base_lr: 1.0e-05`.
- `model.magformer.mask_former.train_num_points: 12544`.
- `vc_suda.unsupervised_weight: 0.02`.
- Pseudo-label threshold `0.1` and curriculum `0.1 -> 0.1`.
- Depth minmax per-sample normalization and the existing eval protocol.

## Leak And Scale Evidence

| Split | Images | Instances | Mask area p50 |
|---|---:|---:|---:|
| source | `1008` | `61652` | `438` |
| target_labeled | `25` | `1697` | `401` |
| target_unlabeled | `200` | `11750` | `452` |
| val | `28` | `1892` | `399` |

Source basename overlap was `0` against target_labeled, target_unlabeled, and val. The source mask-area p50 `438` is close to target_unlabeled p50 `452`, unlike the ordinary 32K source p50 `7572` recorded in R43/R45.

## Preflight

Command:

```bash
python tools/verify_vc_suda_stage.py \
  --config configs/vc_suda_stage_c_r46_pseudoreal_source_true_resume_1024_teacher8499.yaml \
  --stage C \
  --emit-data-evidence \
  --evidence-max-samples 1
```

Result:

- `PASS config=configs/vc_suda_stage_c_r46_pseudoreal_source_true_resume_1024_teacher8499.yaml`.
- Dataset evidence: `source=1008`, `target_labeled=25`, `target_unlabeled=200`, `val=28`.
- Resume semantics: `resume_checkpoint_role: r12_ckpt499_true_resume`, `resume_iter: 499`.
- Checks included unlabeled label stripping, target_unlabeled/val overlap, checkpoint semantics, source/target evidence, and non-constant depth.

## Training Gate

Command contract:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
MAGFORMER_MS_DEFORM_ATTN_BACKEND=cuda \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
python -m torch.distributed.run --standalone --nproc_per_node=4 tools/train.py \
  --config configs/vc_suda_stage_c_r46_pseudoreal_source_true_resume_1024_teacher8499.yaml \
  --output-dir output/vc_suda/stage_c_r46_pseudoreal_source_true_resume_1024_teacher8499 \
  --gpus 0,1,2,3 \
  --num-workers 2
```

Run details:

- tmux session: `r46_gate`.
- Start: true-resumed from R12 `checkpoint_iter_0000499.pth` at iter `499`.
- Completed: `750/750`.
- Exit: `EXIT_CODE=0`.
- Wall time: `644` seconds by wrapper, `10:43.68` by `/usr/bin/time`.
- Final aggregate training log before final eval: iter `740/750`, loss `19.7936`, `loss_ce=0.0019`, `loss_dice=0.0464`, `loss_mask=0.0366`.
- Final visible rank losses at iter `750`: `37.1694`, `13.1760`, `65.3423`, `12.6049`.
- Peak RAM sample: `32 / 251 GiB`.
- Peak GPU samples: GPU4 `22967 MiB`, GPU5 `22615 MiB`, GPU6 `22627 MiB`, GPU7 `22961 MiB`.
- Log scan found no Traceback, OOM, CUDA error, non-finite loss, missing/unexpected keys, or shape mismatch. The only exit warning was the normal PyTorch NCCL process-group destroy warning after a clean status `0`.

Actual checkpoints:

- `output/vc_suda/stage_c_r46_pseudoreal_source_true_resume_1024_teacher8499/checkpoint_iter_0000749.pth`
- `output/vc_suda/stage_c_r46_pseudoreal_source_true_resume_1024_teacher8499/checkpoint_iter_0000750.pth`

The formal gate uses `checkpoint_iter_0000750.pth` because it is the actual final checkpoint.

## External Eval

Output directory:

- `output/diagnostics/r46_pseudoreal_source_true_resume_ckpt0750_target_unlabeled200_1024_backmap_20260516`

Command contract:

```bash
CUDA_VISIBLE_DEVICES=4 \
MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch \
python tools/evaluate_1024_backmap.py \
  --base-config configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml \
  --dataset-root magformer_datasets/pseudo_real_512 \
  --ann annotations/instances_target_unlabeled.json \
  --split train \
  --weights output/vc_suda/stage_c_r46_pseudoreal_source_true_resume_1024_teacher8499/checkpoint_iter_0000750.pth \
  --output-dir output/diagnostics/r46_pseudoreal_source_true_resume_ckpt0750_target_unlabeled200_1024_backmap_20260516 \
  --image-size 1024 \
  --batch-size 1 \
  --num-workers 0 \
  --score-threshold 0.05 \
  --mask-threshold 0.5 \
  --iou-types bbox,segm \
  --inference-topk 200 \
  --max-dets 200 \
  --dump-inference-stats output/diagnostics/r46_pseudoreal_source_true_resume_ckpt0750_target_unlabeled200_1024_backmap_20260516/inference_stats.json \
  --force-pytorch-msda
```

Validation:

- Forced PyTorch MSDA path: `[MSDeformAttn] forced PyTorch core path`.
- Strict load: `774/774`, `missing=0`, `unexpected=0`, `shape_mismatch=0`.
- Evaluated all `200` target_unlabeled images.
- Topk truncated images: `0/200`.
- Predictions: `13,758`.
- Exit: `EXIT_CODE=0`.
- Runtime: `2:10.89`.

Metrics:

| Metric | AP | AP50 | AP75 |
|---|---:|---:|---:|
| bbox | `0.3955750698` | `0.7337662078` | `0.3834772165` |
| segm | `0.3232518088` | `0.6497278126` | `0.2874850698` |

## Oracle And Miss Diagnostics

Existing R38/R40 scripts were reused from `/tmp`; no new repo logic was added.

R38-style oracle score output:

- Output dir: `output/diagnostics/r46_oracle_score_upper_bound_20260516`.
- Predictions: `13,758`.
- Max-IoU oracle R@50/R@75/R@90: `0.685787/0.348511/0.039489`.
- One-to-one oracle R@50/R@75: `0.685447/0.348511`.
- Dense `>90` oracle R@75: `0.170213`.
- Small `<=256` oracle R@75: `0.010566`.
- Matched-IoU score oracle segm AP/AP50/AP75: `0.362376/0.683168/0.346535`.

R40-style miss atlas output:

- Output dir: `output/diagnostics/r46_oracle_miss_atlas_20260516`.
- No-cover/low-quality/good/high-quality counts: `3692/3963/3631/464`.
- Rates: `31.42%/33.73%/30.90%/3.95%`.
- Bad GT instances: `7655`, all in images with prediction candidates.
- Bad subtypes: mask-alignment-shape `4189` (`54.72%`), mask-too-large `3023` (`39.49%`), mask-too-small `279` (`3.64%`), mask-offset `164` (`2.14%`).

Compared with R38/R40 baseline, R46 is a small but real improvement: segm AP improves `0.3200483457 -> 0.3232518088`, AP75 improves `0.2841442923 -> 0.2874850698`, oracle R@75 improves `0.347830 -> 0.348511`, and high-quality masks improve `434 -> 464`.

## Gate Decision

Baseline lines:

- R12/R33 segm AP/AP75: `0.3200483457/0.2841442923`.
- Historical pseudo-real source R8B segm AP: `0.319922`.

R46 result:

- Segm AP/AP75: `0.3232518088/0.2874850698`.

Decision: pass.

R46 clears the explicit pass gate because segm AP is above `0.320048` and AP75 is above `0.284144`. The result is still far from the final `61+` target, but it is an interpretable single-variable win for replacing the source split with scale-matched pseudo-real source under the R35/R12 true-resume protocol.

## Files

- Config: `configs/vc_suda_stage_c_r46_pseudoreal_source_true_resume_1024_teacher8499.yaml`.
- Result doc: `docs/results/vc_suda_r46_pseudoreal_source_true_resume_20260516.md`.
- Summary update: `docs/results/results_summary.md`.
