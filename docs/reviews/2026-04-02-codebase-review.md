# MAGFormer Codebase Review

Date: `2026-04-02`

## Scope

This review focuses on the in-repo MAGFormer stack:

- [magformer/](/home/team/zhanghaonan/magformer/magformer)
- [tools/](/home/team/zhanghaonan/magformer/tools)
- [configs/](/home/team/zhanghaonan/magformer/configs)
- [scripts/](/home/team/zhanghaonan/magformer/scripts)
- [tests/](/home/team/zhanghaonan/magformer/tests)

Vendored baseline internals under `baselines/` were not reviewed line by line here unless the in-repo wrappers or docs depend on them directly.

## Conclusion

The single-GPU MAGFormer path is the most coherent part of the repo. The weak spots are the public evaluation contract, the distributed training path, and the way the repo still mixes a single-class research setup with a more general COCO-style surface.

## Findings

### 1. High: DDP training never runs the same evaluation logic as single-GPU training

- [tools/train.py:670](/home/team/zhanghaonan/magformer/tools/train.py#L670)
- [magformer/engine/trainer.py:923](/home/team/zhanghaonan/magformer/magformer/engine/trainer.py#L923)
- [magformer/models/magformer/arch.py:534](/home/team/zhanghaonan/magformer/magformer/models/magformer/arch.py#L534)
- [magformer/engine/trainer.py:727](/home/team/zhanghaonan/magformer/magformer/engine/trainer.py#L727)

When `tools/train.py` switches to `DDPTrainer`, the evaluation path stops doing the things the single-GPU trainer relies on:

- `MagFormerArch.forward()` returns exported predictions in eval mode, not a loss dict.
- `DDPTrainer.evaluate()` only looks for `"total_loss"`, never constructs a `COCOEvaluator`, never updates `best_metric`, and never saves `model_best.pth`.
- The single-GPU trainer does all of that, so the repo currently has two incompatible meanings of “evaluation”.

Why this matters:

- A `torchrun` job can keep training and checkpointing while never producing meaningful model selection.
- The logged `val/loss` in DDP can stay at its default value even though the code prints that evaluation ran.
- This is exactly the kind of failure that looks healthy from the outside and wastes a full multi-GPU run.

Recommend:

- Move the single-GPU evaluation contract into a shared helper and make DDP gather predictions across ranks before metric computation.

### 2. Medium: `tools/evaluate.py` says `--weights` is optional, but it ignores `config.model.weights`

- [tools/evaluate.py:86](/home/team/zhanghaonan/magformer/tools/evaluate.py#L86)
- [tools/evaluate.py:108](/home/team/zhanghaonan/magformer/tools/evaluate.py#L108)

The script writes `args.weights` into `overrides["model"]["weights"]`, but later it only calls `load_checkpoint()` inside `if args.weights:`. `build_model(config)` does not load weights on its own here.

Why this matters:

- The CLI advertises one contract and implements another.
- A user can reasonably put weights in the config, omit `--weights`, and end up evaluating random initialization.
- This is easy to miss because the script still runs and still writes output files.

Recommend:

- Resolve an `effective_weights` value from `args.weights or config.model.weights`, then require it or fail fast with a clear error.

### 3. Medium: validation loss logging is not trustworthy because the validation dataset strips targets

- [tools/train.py:55](/home/team/zhanghaonan/magformer/tools/train.py#L55)
- [magformer/data/dataset.py:183](/home/team/zhanghaonan/magformer/magformer/data/dataset.py#L183)
- [magformer/engine/trainer.py:605](/home/team/zhanghaonan/magformer/magformer/engine/trainer.py#L605)
- [magformer/engine/trainer.py:670](/home/team/zhanghaonan/magformer/magformer/engine/trainer.py#L670)

`build_datasets()` creates the validation dataset with `is_train=False`. In `CocoRgbdDataset`, that means no `masks`, `boxes`, or `labels` are loaded for validation samples. The trainer still logs `val/loss`, but that meter only updates if a `"total_loss"` exists in model outputs.

In practice:

- single-GPU eval mostly works because it uses COCO metrics from predictions
- `val/loss` is not a real validation loss
- the fallback branch that uses loss when no mAP is available becomes misleading

Why this matters:

- logs imply a validation signal that the current data path does not supply
- downstream scripts or readers can treat `val/loss` as meaningful when it is just the meter default

Recommend:

- Either load validation targets explicitly for loss computation, or stop logging `val/loss` and make mAP the only validation signal for this path.

### 4. Medium: the dataset path is COCO-shaped, but category handling is hard-coded to one class

- [README.md:3](/home/team/zhanghaonan/magformer/README.md#L3)
- [magformer/data/dataset.py:333](/home/team/zhanghaonan/magformer/magformer/data/dataset.py#L333)
- [tools/inference.py:111](/home/team/zhanghaonan/magformer/tools/inference.py#L111)

The repo surface looks generic:

- README says “COCO RGB-D dataset support”
- config schema exposes `num_classes`

But the actual in-repo path does this:

- `CocoRgbdDataset._build_instances()` appends `labels.append(1)` for every annotation
- inference visualization hard-codes `class_names=["component"]`

Why this matters:

- a multi-class dataset will silently collapse into one foreground class
- this is a design contract, not just a visualization detail
- the current docs do not make that restriction clear enough

Recommend:

- Pick one story and enforce it in code.
  - If MAGFormer is intentionally single-class here, validate `num_classes == 1` and say so everywhere.
  - If multi-class support is intended, propagate `category_id` correctly from dataset to loss to export to visualization.

### 5. Medium: reproducibility is still tied to machine-local paths and prior outputs

- [configs/magformer_0831_1k_60ep_trackp.yaml:43](/home/team/zhanghaonan/magformer/configs/magformer_0831_1k_60ep_trackp.yaml#L43)
- [configs/magformer_0831_1k_mgm_finetune.yaml:43](/home/team/zhanghaonan/magformer/configs/magformer_0831_1k_mgm_finetune.yaml#L43)
- [configs/magformer_track_b_2k.yaml:45](/home/team/zhanghaonan/magformer/configs/magformer_track_b_2k.yaml#L45)
- [docs/experiments/track_b_2k_report.md:8](/home/team/zhanghaonan/magformer/docs/experiments/track_b_2k_report.md#L8)
- [docs/experiments/baselines_0831_1k_20ep_scratch8_report.md:33](/home/team/zhanghaonan/magformer/docs/experiments/baselines_0831_1k_20ep_scratch8_report.md#L33)

There are still configs and experiment docs that depend on:

- `/home/k100/...` absolute paths
- warm-start checkpoints from prior local runs
- `TBD` placeholders in published result tables

Why this matters:

- “known good” configs are not portable
- reproduction depends on private filesystem history, not only the repo
- readers cannot tell which experiment docs are finished and which are still templates

Recommend:

- replace machine-local config defaults with repo-relative or explicit CLI-required inputs
- mark historical docs as historical
- remove or quarantine docs that still publish `TBD` results as if they were final

## Test Gaps

The biggest review findings above are weakly covered or uncovered:

- [tests/test_train_distributed_context.py:6](/home/team/zhanghaonan/magformer/tests/test_train_distributed_context.py#L6) only checks launcher detection. It does not test DDP evaluation outputs, COCO metric flow, or best-checkpoint handling.
- [tests/test_train_imports_dataset.py:4](/home/team/zhanghaonan/magformer/tests/test_train_imports_dataset.py#L4) is only a string-match test against the source file. It does not exercise the train entrypoint behavior.
- [tests/test_magformer_raw_inference.py:9](/home/team/zhanghaonan/magformer/tests/test_magformer_raw_inference.py#L9) covers raw/export helpers, but there is no matching regression test for `tools/evaluate.py` honoring config-supplied weights.

Most useful additions:

1. A regression test that `tools/evaluate.py` loads `config.model.weights` when `--weights` is absent.
2. A trainer-level test that DDP evaluation computes the same metric contract as single-GPU evaluation.
3. A dataset-level test that fails loudly if a multi-class COCO annotation file is fed into the current single-class path.

## Validation Performed

- `python3 -m compileall magformer tools scripts/analysis`
  - passed

What I could not run in this shell:

- `python3 -m pytest ...`
  - blocked because `pytest` is not installed in the current interpreter
- a mocked runtime probe for `tools/evaluate.py`
  - blocked because `torch` is not installed in the current interpreter

So the findings above are based on direct code-path inspection plus syntax validation, not a full runtime test pass.

## Recommended Fix Order

1. Unify evaluation behavior across `Trainer` and `DDPTrainer`.
2. Fix `tools/evaluate.py` weight resolution and add a regression test.
3. Decide whether this repo is single-class by design or truly COCO-generic, then enforce that choice.
4. Clean path-bound configs and historical docs so “recommended” workflows are actually reproducible.
