# VC-SUDA R41 Eval1536 Teacher R12 Check - 2026-05-16

## Conclusion

R41 fails the gate. Changing only eval image size from `1024` to `1536` lowered the R12 `ckpt499` `target_unlabeled200` teacher result instead of exposing hidden headroom.

Formal segm AP/AP50/AP75 is `0.284495/0.609380/0.236851`, below the R33/R15 `1024` baseline `0.320048/0.648363/0.284144`. Oracle diagnostics also declined: oracle-score matched segm AP/AP50/AP75 is only `0.332673/0.663366/0.306931`, global oracle R@75 is `0.300255`, dense `>90` R@75 is `0.142919`, and small `<=256` R@75 is `0.003408`.

Do not continue from R41. This is a protocol-valid stop, not a training result.

## Scope

- No training was run.
- No model code was changed.
- No branch was created and no `.worktree` was used.
- The single eval variable was `--image-size 1024 -> 1536`.
- R38/R40 diagnostics reused the existing temporary scripts `/tmp/r38_oracle_bound.py` and `/tmp/r40_oracle_miss_atlas.py` on `4029`.

## Inputs

- Repo: `/home/hdd3/zhanghaonan/magformer`
- Branch: `feature/vc-suda-sim2real`
- Checkpoint: `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth`
- Base config: `output/experiments/vc_suda_stage_c_r12_32ksource_iter0500_target_unlabeled200_1024_backmap_20260515_1606/eval_base_r12_ckpt499.yaml`
- Dataset root: `magformer_datasets/pseudo_real_512`
- Annotation: `annotations/instances_target_unlabeled.json`
- Split: `train`
- GT images / instances: `200 / 11,750`

## Eval Command

Ran on GPU4 in tmux session `r41_eval1536_teacher_r12_20260516`:

```bash
cd /home/hdd3/zhanghaonan/magformer
mkdir -p output/diagnostics/r41_eval1536_teacher_r12_20260516
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate magformer
unset LD_LIBRARY_PATH CUDA_HOME
export PATH="$CONDA_PREFIX/bin:$PATH"
export CUDA_VISIBLE_DEVICES=4
export MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/evaluate_1024_backmap.py \
  --base-config output/experiments/vc_suda_stage_c_r12_32ksource_iter0500_target_unlabeled200_1024_backmap_20260515_1606/eval_base_r12_ckpt499.yaml \
  --dataset-root magformer_datasets/pseudo_real_512 \
  --ann annotations/instances_target_unlabeled.json \
  --split train \
  --weights output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth \
  --output-dir output/diagnostics/r41_eval1536_teacher_r12_20260516 \
  --image-size 1536 \
  --batch-size 1 \
  --num-workers 0 \
  --score-threshold 0.05 \
  --mask-threshold 0.5 \
  --iou-types bbox,segm \
  --inference-topk 200 \
  --max-dets 200 \
  --dump-inference-stats output/diagnostics/r41_eval1536_teacher_r12_20260516/inference_stats.json \
  --force-pytorch-msda \
  2>&1 | tee output/diagnostics/r41_eval1536_teacher_r12_20260516/eval.log
```

Protocol evidence from log:

- `[MSDeformAttn] forced PyTorch core path`
- `[InputEvidence] images=[1, 3, 1536, 1536] depths=[1, 1, 1536, 1536]`
- `[CoordEvidence] model_input=1536x1536 prediction_masks_resized_to_gt=[(512, 512)]`
- strict load `matched=774/774 missing=0 unexpected=0 shape_mismatch=0`
- `evaluated_images=200 max_images=None`

## Formal Metrics

| Run | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | Predictions | topk truncated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R33/R15 1024 baseline | `0.394476` | `0.741115` | `0.379187` | `0.320048` | `0.648363` | `0.284144` | `13,806` | `0/200` |
| R41 1536 eval-only | `0.356160` | `0.717496` | `0.315430` | `0.284495` | `0.609380` | `0.236851` | `17,495` | `0/200` |
| Delta | `-0.038316` | `-0.023619` | `-0.063757` | `-0.035553` | `-0.038983` | `-0.047293` | `+3,689` | `0` |

Inference stats:

- pre-topk candidate sum: `40,000`
- post-topk sum: `40,000`
- post-score/post-mask/exported sum: `17,495`
- exported min/p50/max per image: `26 / 80 / 144`
- topk limit: `200`
- topk truncated: `0 / 200`

## R38 Oracle Bound

Command:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python /tmp/r38_oracle_bound.py \
  --ann magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json \
  --pred r41_eval1536 output/diagnostics/r41_eval1536_teacher_r12_20260516/coco_instances_results.json \
  --out-dir output/diagnostics/r41_eval1536_teacher_r12_20260516/oracle_bound
```

Output summary: `output/diagnostics/r41_eval1536_teacher_r12_20260516/oracle_bound/r41_eval1536_oracle_bound_summary.json`

| Metric | Value | Count |
|---|---:|---:|
| Oracle recall@50 | `0.660511` | `7,761 / 11,750` |
| Oracle recall@75 | `0.300255` | `3,528 / 11,750` |
| Oracle recall@90 | `0.022128` | `260 / 11,750` |
| One-to-one recall@50 | `0.660255` | `7,758 / 11,750` |
| One-to-one recall@75 | `0.300255` | `3,528 / 11,750` |

| Bucket | GT | R@50 | R@75 | R@90 | IoU p50 |
|---|---:|---:|---:|---:|---:|
| density `25-30` | `550` | `0.949091` | `0.700000` | `0.150909` | `0.820110` |
| density `46-60` | `6,547` | `0.767680` | `0.378494` | `0.025050` | `0.695335` |
| density `>90` | `4,653` | `0.475607` | `0.142919` | `0.002794` | `0.475884` |
| area `<=256` | `2,934` | `0.122359` | `0.003408` | `0.000000` | `0.242599` |
| area `257-452` | `2,943` | `0.709140` | `0.164798` | `0.001019` | `0.604520` |
| area `453-579` | `2,945` | `0.896774` | `0.464177` | `0.025127` | `0.737478` |
| area `>579` | `2,928` | `0.913251` | `0.568989` | `0.062500` | `0.772790` |

Oracle-score AP:

| Oracle score | segm AP | segm AP50 | segm AP75 | segm AR200 | bbox AP | bbox AP50 | bbox AP75 | bbox AR200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Max IoU score | `0.331950` | `0.660532` | `0.306655` | `0.328987` | `0.392408` | `0.747701` | `0.370066` | `0.422774` |
| Matched IoU score | `0.332673` | `0.663366` | `0.306931` | `0.328987` | `0.393974` | `0.754885` | `0.370137` | `0.422783` |

## R40 Miss Atlas

Command:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python /tmp/r40_oracle_miss_atlas.py \
  --ann magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json \
  --pred output/diagnostics/r41_eval1536_teacher_r12_20260516/coco_instances_results.json \
  --out-dir output/diagnostics/r41_eval1536_teacher_r12_20260516/miss_atlas
```

Output summary: `output/diagnostics/r41_eval1536_teacher_r12_20260516/miss_atlas/r40_oracle_miss_atlas_summary.json`

| Class | Definition | Count | Rate |
|---|---|---:|---:|
| no-cover | best IoU `<0.5` | `3,989` | `33.95%` |
| low-quality | `0.5<=IoU<0.75` | `4,233` | `36.03%` |
| good | `0.75<=IoU<0.9` | `3,268` | `27.81%` |
| high-quality | IoU `>=0.9` | `260` | `2.21%` |

Dense and small-object readout:

- dense `>90`: no-cover `52.44%`, low-quality `33.27%`, R@75 `0.142919`, R@90 `0.002794`.
- small `<=256`: no-cover `87.76%`, low-quality `11.90%`, R@75 `0.003408`, R@90 `0.000000`.
- all `8,222` no-cover or low-quality GT instances had at least one prediction candidate in the same image.
- bad modes: mask-alignment-shape `56.04%`, mask-too-large `39.09%`, mask-too-small `3.03%`, mask-offset `1.84%`.
- RGB/depth readable: `200/200` and `200/200`; all three contact sheets were generated.

## Gate

R41 fails both continue conditions and hits stop conditions.

Continue condition A fails:

- formal segm AP `0.284495 < 0.325`
- formal segm AP75 `0.236851 < 0.290`

Continue condition B fails:

- oracle-score matched segm AP `0.332673 < 0.385`
- global oracle R@75 `0.300255 < 0.38`
- dense R@75 `0.142919 < 0.23`
- small `<=256` R@75 `0.003408 < 0.04`

Stop conditions hit:

- segm AP `0.284495 < 0.322`
- oracle-score AP `0.332673 < 0.375`
- dense R@75 `0.142919 < 0.22`
- small R@75 `0.003408 < 0.03`

Protocol validity checks passed:

- evaluated all `200` images
- topk truncated `0/200`
- prediction count `17,495`, exported max per image `144`, below maxDets/topk `200`
- no prediction-empty protocol failure
- log scan found no Traceback, CUDA error, OOM, out-of-memory, RuntimeError, or non-finite hit

## Interpretation

The 1536 eval size increases the number of exported predictions but makes localization and masks worse. This is not a hidden-resolution fix for the R12 teacher pool.

The R38/R40 conclusion remains stronger after R41: the bottleneck is still mask-set coverage/localization, especially tiny masks and dense scenes. Larger eval resolution alone does not repair that pool and actually lowers the usable oracle bound.

## Output Files

Generated but not committed:

- `output/diagnostics/r41_eval1536_teacher_r12_20260516/eval.log`
- `output/diagnostics/r41_eval1536_teacher_r12_20260516/metrics.cocoeval.json`
- `output/diagnostics/r41_eval1536_teacher_r12_20260516/coco_instances_results.json`
- `output/diagnostics/r41_eval1536_teacher_r12_20260516/inference_stats.json`
- `output/diagnostics/r41_eval1536_teacher_r12_20260516/oracle_bound/r41_eval1536_oracle_bound_summary.json`
- `output/diagnostics/r41_eval1536_teacher_r12_20260516/miss_atlas/r40_oracle_miss_atlas_summary.json`
- `output/diagnostics/r41_eval1536_teacher_r12_20260516/miss_atlas/*.csv`
- `output/diagnostics/r41_eval1536_teacher_r12_20260516/miss_atlas/*.jpg`

## Validation

- `nvidia-smi` before launch showed GPU4 free except display memory.
- Server RAM stayed well below 90%; observed samples were around `6.84%` to `7.50%` used.
- GPU4 was released after eval.
- `metrics.cocoeval.json`, `inference_stats.json`, R38 summary, and R40 summary were read back after generation.
- `git status --porcelain` was clean before docs were edited.
