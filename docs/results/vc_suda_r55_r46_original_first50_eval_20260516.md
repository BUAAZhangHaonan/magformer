# VC-SUDA R55 R46 original first50 eval - 2026-05-16

## Conclusion

R46 does not retain the R44 Teacher source-sanity level on the original 1.5K first50 protocol.

Using the R54 protocol checker contract with nondefault R46 weights, `checkpoint_iter_0000750.pth` reaches segm AP/AP50/AP75 `0.4009417296/0.7265082520/0.3993570474`. The R44 Teacher same-protocol segm AP was `0.6252930576`, so R46 is lower by `0.2243513280` AP.

This is source sanity only. It does not replace the `target_unlabeled200` result, where R46 remains the current best at segm AP `0.323252`.

## Protocol Check

Output directory:

- `output/diagnostics/r55_r46_ckpt0750_original_first50_1024_backmap_20260516`

Command:

```bash
python tools/check_eval_protocol.py original_first50_teacher \
  --base-config configs/finetune_1k_full_1024.yaml \
  --dataset-root magformer_datasets/20260318_1K_1566 \
  --ann annotations/instances_all.json \
  --split all \
  --weights output/vc_suda/stage_c_r46_pseudoreal_source_true_resume_1024_teacher8499/checkpoint_iter_0000750.pth \
  --image-size 1024 \
  --max-images 50 \
  --score-threshold 0.05 \
  --mask-threshold 0.5 \
  --iou-types bbox,segm \
  --inference-topk 100 \
  --max-dets 100 \
  --allow-nondefault-weights \
  --summary-json output/diagnostics/r55_r46_ckpt0750_original_first50_1024_backmap_20260516/protocol_check.summary.json
```

Result: pass.

The summary confirms:

- protocol: `original_first50_teacher`
- base config: `configs/finetune_1k_full_1024.yaml`
- dataset root: `magformer_datasets/20260318_1K_1566`
- annotation: `annotations/instances_all.json`
- split: `all`
- image count: `1566`
- max images: `50`
- image size: `1024`
- score/mask thresholds: `0.05/0.5`
- IoU types: `bbox,segm`
- topk/maxDets: `100/100`
- original depth clip: `1.0015300512313843/2.095623016357422`
- nondefault weights allowed: `true`

## Eval

Command:

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
  --weights output/vc_suda/stage_c_r46_pseudoreal_source_true_resume_1024_teacher8499/checkpoint_iter_0000750.pth \
  --output-dir output/diagnostics/r55_r46_ckpt0750_original_first50_1024_backmap_20260516 \
  --image-size 1024 \
  --batch-size 4 \
  --num-workers 4 \
  --score-threshold 0.05 \
  --mask-threshold 0.5 \
  --inference-topk 100 \
  --max-dets 100 \
  --max-images 50 \
  --iou-types bbox,segm \
  --force-pytorch-msda
```

Output files:

- `output/diagnostics/r55_r46_ckpt0750_original_first50_1024_backmap_20260516/eval.log`
- `output/diagnostics/r55_r46_ckpt0750_original_first50_1024_backmap_20260516/eval_1024_runtime.yaml`
- `output/diagnostics/r55_r46_ckpt0750_original_first50_1024_backmap_20260516/metrics.cocoeval.json`
- `output/diagnostics/r55_r46_ckpt0750_original_first50_1024_backmap_20260516/coco_instances_results.json`
- `output/diagnostics/r55_r46_ckpt0750_original_first50_1024_backmap_20260516/protocol_check.summary.json`
- `output/diagnostics/r55_r46_ckpt0750_original_first50_1024_backmap_20260516/protocol_check.log`

## Metrics

| Metric | R46 ckpt0750 original first50 | R44 Teacher original first50 |
| --- | ---: | ---: |
| bbox AP | `0.4657849322` | `0.6519530084` |
| bbox AP50 | `0.7669137408` | `0.8531437401` |
| bbox AP75 | `0.4908037106` | `0.7175465372` |
| segm AP | `0.4009417296` | `0.6252930576` |
| segm AP50 | `0.7265082520` | `0.8679527609` |
| segm AP75 | `0.3993570474` | `0.7243386328` |

Segm AP delta vs R44 Teacher: `-0.2243513280`.

## Interpretation

This result says the adapted R46 checkpoint does not preserve the original-source `61+` sanity line. It is not a target-domain gate and should not be used to demote R46 on `target_unlabeled200`.

The earlier `61+` target and the target-domain goal were mixed. R44 Teacher `0.6252930576` belongs to original 1.5K first50 source sanity. R46 `0.323252` belongs to pseudo-real `target_unlabeled200`. Since R46 is low on original first50, domain adaptation hurt source sanity while improving or preserving the current target-domain best.

## Validation

- R54 checker pass summary exists at `protocol_check.summary.json`.
- Eval log contains `[Weights] strict load OK` for R46 `checkpoint_iter_0000750.pth`.
- Eval log has no `Traceback`, `OOM`, or CUDA OOM string.
- Eval used forced PyTorch MSDA: `[MSDeformAttn] forced PyTorch core path`.
- No training command was run.
