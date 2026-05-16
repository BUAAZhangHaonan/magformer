# VC-SUDA Stage C R35 Original Split True Resume Plan - 2026-05-16

## Conclusion

R35 isolates whether the R34 drop came from the `+25` split mixing or from true-resuming R12 `ckpt499` itself.

Run contract: true-resume from R12 `checkpoint_iter_0000499.pth`, keep the original R12 `target_labeled=25` / `target_unlabeled=200` split, and continue only to iter `750`. Do not start training until preflight passes.

## Context

R34 tested `+25` split true-resume from R12 `ckpt499`. It completed `750/750`, but external `target_unlabeled200` 1024 backmap eval reached only segm AP `0.301879`. This failed the hard line `0.319162`.

R33 did not train. It re-evaluated R12 `checkpoint_iter_0000499.pth` with current code and data, and reproduced the R12/R15 baseline: bbox AP/AP50/AP75 `0.394476/0.741115/0.379187`, segm AP/AP50/AP75 `0.320048/0.648363/0.284144`, with `13,806` predictions and `0/200` topk-truncated images.

So the benchmark is stable. The remaining question is whether the R34 damage comes from the `+25` data mix or from continuing R12 `ckpt499` by true resume.

## Config

- Config: `configs/vc_suda_stage_c_r35_original_split_true_resume_1024_teacher8499.yaml`.
- Base: `configs/vc_suda_stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499.yaml`.
- Output: `output/vc_suda/stage_c_r35_original_split_true_resume_1024_teacher8499`.
- True resume: `runtime.resume: output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth`.
- Model-only warm-start disabled: `model.finetune_weights: null`.
- Stop point: `solver.max_iter: 750`.
- Checkpoint/eval cadence: `runtime.checkpoint_period: 250`, `runtime.eval_period: 250`, so iter `750` can produce a gate checkpoint/eval signal.

## Fixed Variables

R35 keeps R12 original settings unchanged:

- Target split: `annotations/instances_target_labeled.json` and `annotations/instances_target_unlabeled.json`.
- Resolution: `data.image_size: 1024`.
- Source root: `magformer_datasets/20260318_1K_32254` with source ann `annotations/instances_train.json`.
- LR and optimizer: `base_lr=1e-5`, ADAMW, cosine schedule.
- Loss and pseudo settings: mask/dice/class, contrastive, pseudo threshold `0.10`, `max_instances=100`, and `unsupervised_weight=0.02` unchanged.
- Augmentation and depth settings unchanged.

## Preflight

Run only preflight before launch:

```bash
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate magformer
python tools/verify_vc_suda_stage.py \
  --config configs/vc_suda_stage_c_r35_original_split_true_resume_1024_teacher8499.yaml \
  --stage C \
  --emit-data-evidence \
  --evidence-max-samples 1
```

This is read-only and must not start training.

## Gate

After true-resume continuation to iter `750`, evaluate external `target_unlabeled200` at 1024 backmap with bbox+segm and topk200/maxDets200.

Decision:

- `segm AP >= 0.320048`: original-split continuation does not hurt; the `+25` data mix is the main suspect.
- `0.319162 <= segm AP < 0.320048`: record the result, but do not extend the run.
- `segm AP < 0.319162`: true-resume continuation itself also hurts and must be isolated before more split changes.

The gate metric is external `target_unlabeled200` 1024 backmap segm AP. Built-in 28-image eval is only a smoke signal.

## Result

R35 true-resumed from R12 `checkpoint_iter_0000499.pth`, started at resume iter `499`, ran to the final tail, and exited cleanly with `EXIT_CODE=0`. Both `checkpoint_iter_0000749.pth` and `checkpoint_iter_0000750.pth` are present in `output/vc_suda/stage_c_r35_original_split_true_resume_1024_teacher8499`.

External `target_unlabeled200` 1024 backmap eval used topk/maxDets=`200`.

| Metric | Value |
|---|---:|
| prediction count | `13557` |
| bbox AP/AP50/AP75 | `0.394235/0.733760/0.380052` |
| segm AP/AP50/AP75 | `0.319300/0.649594/0.283212` |

## Gate Decision

R35 falls in the record-only band because external `target_unlabeled200` 1024 backmap segm AP is `0.3192996`, which is within `0.319162-0.320048`.

Record R35. Do not extend the run.

## Review

R35 original-split true resume basically preserves the R12 level: R12 `0.320048` -> R35 `0.319300`, delta about `-0.00075`. R34 `+25` true resume dropped to `0.301879`.

So continued training itself is not the main cause. The main suspect is the `+25` split or data mixing. The next step should not directly expand labels. Check promoted-sample selection and target_labeled weighting/sampling strategy, or run a smaller and more balanced `+25` ablation.
