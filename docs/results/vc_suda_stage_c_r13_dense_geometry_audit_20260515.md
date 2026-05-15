# VC-SUDA Stage C R13 Dense Geometry Audit - 2026-05-15

Conclusion: do not start the next training run yet. The current evidence does not point to a gross 1024 backmap scale bug. It does show a dense-image candidate-budget risk and a need for pre-export instrumentation before another experiment.

## Scope

- No training was started.
- No threshold sweep was run.
- Audit inputs were existing COCO GT and prediction JSON files.
- Reports were written under `output/diagnostics/r13_dense_geometry_audit_20260515/` and must stay uncommitted.

## Read-Only Evidence

- Repository HEAD on 4029: `1e65874d3e1c4165c57c7e2c7e995fdab173e02a`.
- Branch: `feature/vc-suda-sim2real`.
- Existing untracked output dirs before R13: `output/diagnostics/`, `output/upper_bound/`.
- R12 ckpt499 prediction JSON: `output/experiments/vc_suda_stage_c_r12_32ksource_iter0500_target_unlabeled200_1024_backmap_20260515_1606/coco_instances_results.json`.
- R12 ckpt499 config: `output/experiments/vc_suda_stage_c_r12_32ksource_iter0500_target_unlabeled200_1024_backmap_20260515_1606/eval_1024_runtime.yaml`.
- R12 ckpt499 checkpoint: `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth`.
- R12 ckpt499 metrics file is complete: segm AP `0.32004639876247354`, bbox AP `0.3932659516433205`.

## Commands

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/audit_dense_geometry.py \
  --gt magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json \
  --predictions output/experiments/vc_suda_stage_c_r12_32ksource_iter0500_target_unlabeled200_1024_backmap_20260515_1606/coco_instances_results.json \
  --name r12_ckpt499_target_unlabeled200_1024_backmap \
  --output-dir output/diagnostics/r13_dense_geometry_audit_20260515 \
  --top-k 20

/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/verify_vc_suda_stage.py \
  --config configs/vc_suda_stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499.yaml \
  --skip-batch --emit-data-evidence --evidence-max-samples 3 \
  > output/diagnostics/r13_dense_geometry_audit_20260515/r12_source_target_evidence.txt
```

R8B ckpt999, R7 iter1000, and R7 iter2000 were audited with the same GT and comparison reports.

## Key Results

R12 ckpt499:

- Total GT/pred: `11750` / `13122`.
- Mask TP/FP/FN: `8017` / `5105` / `3733`.
- Mask FP split: low-IoU `5011`, duplicate `50`, background `44`.
- Top worst image ids: `99, 169, 132, 77, 14, 62, 164, 48, 155, 177`.
- Matched TP bbox area ratio: mean `1.034`, p50 `1.006`, p90 `1.278`.
- Matched TP mask area ratio: mean `1.142`, p50 `1.117`, p90 `1.402`.
- Low-IoU FP max-IoU: mean `0.260`, p50 `0.259`, p90 `0.453`.

Dense `90-100` GT bucket for R12:

- Images: `47`.
- GT/pred: `4653` / `4343`.
- Mask TP/FP/FN: `2303` / `2040` / `2350`.
- Mask FP split: low-IoU `2026`, duplicate `13`, background `1`.

R8B ckpt999:

- Mask TP/FP/FN: `8026` / `4886` / `3724`.
- Top worst image ids: `99, 169, 132, 14, 77, 177, 48, 164, 62, 101`.
- R8B top-20 overlaps R12 top-20 on 18 image ids.

R7 iter2000:

- Mask TP/FP/FN: `7988` / `5029` / `3762`.
- Top worst image ids: `99, 132, 169, 77, 14, 16, 164, 101, 155, 48`.
- R7 iter2000 top-20 overlaps R12 top-20 on 18 image ids.

## Geometry Read

Scale/backmap is not the leading suspect. R12 matched TP bbox area ratio is centered near 1.0, and bbox AP is higher than segm AP. The mask area ratio is biased high, but it is not a uniform resize/backmap blow-up.

The main failure mode remains low-IoU masks near objects. R12 has `5011` mask low-IoU FPs, and their max-IoU distribution sits below the TP threshold rather than at zero.

## Dense / maxDets Read

Candidate truncation is suspicious but not fully proven from exported JSON.

- R12 predictions per image: max `100`, p90 `97`, mean `65.61`.
- R12 has `10` images at exactly `100` predictions.
- R8B also has `10` images at `100`; R7 iter2000 has `12`.
- Code locations found by the audit:
  - `magformer/models/magformer/arch.py`: `forward_inference_raw topk=min(100, Nq*num_classes)`.
  - `magformer/engine/evaluator.py`: COCOeval `maxDets=[1,10,100]`.
  - `magformer/engine/eval_runtime.py`: eval export score threshold and `max_dets=100`.
  - `tools/evaluate_1024_backmap.py`: backmap export threshold and `max_dets=100`.

The exported prediction JSON cannot show pre-topk candidate count. The report therefore records `postprocess-pretopk unavailable`.

## Source / Target Evidence

Verifier evidence path: `output/diagnostics/r13_dense_geometry_audit_20260515/r12_source_target_evidence.txt`.

- Source root: `magformer_datasets/20260318_1K_32254`.
- Source ann/count: `annotations/instances_train.json`, `25654`.
- Target labeled ann/count: `annotations/instances_target_labeled.json`, `25`.
- Target unlabeled ann/count: `annotations/instances_target_unlabeled.json`, `200`.
- Val ann/count: `annotations/instances_val.json`, `28`.
- Computed Stage C dataset length: `25654`.
- Sampled source RGB/depth exist and have nonconstant stats.
- Sampled target RGB/depth exist and have nonconstant stats.

## Recommendation

Do not launch R14 training yet. Add instrumentation first:

- Log pre-topk query count, post-topk count, post-score count, post-mask-nonempty count, and exported count per image.
- Log the same counts by GT density bucket during eval.
- Keep the current 1024 backmap path, because R13 does not show a blocking scale/backmap bug.
