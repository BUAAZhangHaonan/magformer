# MAGFormer Project Sync And Repair Summary

## Problem 1. DDP evaluation

The MAGFormer training path now has one real evaluation flow. `magformer/engine/eval_runtime.py`, `magformer/engine/trainer.py`, `tools/evaluate.py`, and `tools/export_results.py` all use the same inference-time evaluator. The real two-rank canary in `scripts/experiments/run_20260321_ddp_smoke_canary.sh --magformer-only --run` now reaches validation, writes COCO outputs once, updates `best_metric`, saves `model_best.pth` on rank 0, and proves that offline evaluation works without `--weights` when `config.model.weights` is set.

One small bug showed up while making that proof stable. When a validation pass produced zero predictions, `magformer/engine/evaluator.py` returned no metrics instead of valid zero AP, so the trainer could skip `model_best.pth` entirely. That is fixed now. Zero-prediction validation is treated as zero AP, which is the right behavior for checkpoint selection.

## Problem 2. `val/loss` is fake

The standard MAGFormer validation path is metric-only now. `val/loss` is not logged in normal validation, and `val/mAP` is the only model-selection signal in the repaired path. The real DDP canary and the focused test suite both confirm that normal validation rows now carry COCO metrics plus diagnostics, not a fake loss value.

## Problem 3. MGM `512` and `256` zero rows

The historical `512` and `256` MGM rows are not backed by live artifacts in the current tree. The older table depended on deleted `.worktrees/ucn-msmformer-repair/...` outputs, so I cannot prove the original root cause from the current workspace. What I can say is simpler and more solid: the current MGM runner works, a live `512` smoke run reaches training and evaluation, and the current tree does not contain finished `512` or `256` MGM exports. So those historical zero rows are not trustworthy anymore, and the refreshed table marks them as missing and rerun-required instead of repeating them.

## Problem 4. Missing `512` and `256` baselines

The old explanations were stale. `baselines/Mask2Former/train_net.py` exists, the official wrapper works, and a live `512` official Mask2Former smoke run now trains and evaluates. The only snag in the first smoke was a bad one-step scheduler setup: Detectron2 rejects `MAX_ITER=1` for that LR schedule. A two-step smoke works.

The YOLOv8 note was stale too. `ultralytics` imports cleanly in the `magformer` env, and a live `512` `yolov8n-seg` smoke run trains and validates on GPU. The real gap is not a dead runner. The real gap is missing finished `512` and `256` publication artifacts in `output/experiments`.

## Problem 5. The no-depth baseline is not fair

The 1024 no-depth comparison is still not a clean depth-only control. MAGFormer no-depth is a D2Swin MAGFormer run with a local MGM-derived warm start and the depth path disabled. Stock Mask2Former is the official detectron2 R50 recipe with converted official COCO-pretrained weights. The optimization budgets are also different: the live MAGFormer no-depth run uses batch `4` and `6320` steps, while the stock Mask2Former run uses batch `8` and `2220` steps. That means the current `48.78` vs `58.76` gap is not just “depth off vs depth on.”

There is one more trust caveat here. The live `magformer_nodpth_ref` artifact predates the DDP evaluation repair, so its best-checkpoint claim is weaker than the repaired path. The clean fix is straightforward: rerun the 1024 MAGFormer RGB-only control inside the repaired MAGFormer stack, keep the same backbone family and schedule as the depth run, and change only the depth switches.

## Problem 6. The metrics table

The table is rebuilt now from live artifacts only. The publication path uses `metrics.cocoeval.json` for AP metrics, `coco_instances_results.json` for `P@50 / R@50 / F1@50`, `inference_speed_clean.json` or `inference_speed.json` for runtime, and `metadata.json`, `wall_time_sec.txt`, `peak_memory_mb.txt`, and `params_trainable.txt` for run metadata. Old markdown tables and deleted worktree paths are no longer used as data sources.

This also fixes the detectron2-style bbox regression in the old summary pipeline. For example, the refreshed 1024 table now preserves `bbox AP = 50.5996` for stock Mask2Former and `bbox AP = 61.5294` for `mgm_mask2former_depthnorm_on` instead of dropping those numbers.

## Remaining human or GPU work

- Rebuild the missing `512` official Mask2Former rows with `bash scripts/experiments/run_0831_1k_20ep_scratch_official_mask2former.sh --register 20260318_1K_1566 --dataset-root /home/team/zhanghaonan/magformer/magformer_datasets/20260318_1K_1566 --output-root /home/team/zhanghaonan/magformer/output/experiments/20260318_1k_1566_20ep_512_full19 --candidate-id C1 --run-tag final --image-size 512 --pretrained --run`, then repeat with `256` and a matching output root.
- Rebuild the missing `512` and `256` YOLO rows with `bash scripts/experiments/run_0831_1k_20ep_scratch_yolov8_seg.sh --register 20260318_1K_1566 --dataset-root /home/team/zhanghaonan/magformer/magformer_datasets/20260318_1K_1566 --output-root /home/team/zhanghaonan/magformer/output/experiments/20260318_1k_1566_20ep_512_full19 --image-size 512 --model-size n --pretrained --run`, then repeat for `s`, `m`, `l`, `x`, and then repeat the full set at `256`.
- Rebuild the missing MGM multires rows with `bash scripts/experiments/run_0831_1k_20ep_1024_revisit_mgm_mask2former.sh --register 20260318_1K_1566 --dataset-root /home/team/zhanghaonan/magformer/magformer_datasets/20260318_1K_1566 --output-root /home/team/zhanghaonan/magformer/output/experiments/20260318_1k_1566_20ep_512_full19 --variant depthnorm_on --image-size 512 --run`, then repeat for `nodpth_ref` and for `256`.
- Rebuild MSMFormer and UCN multires rows with `bash scripts/experiments/run_0831_1k_20ep_scratch_msmformer.sh ... --image-size 512 --run` and `bash scripts/experiments/run_0831_1k_20ep_scratch_ucn.sh ... --image-size 512 --run`, then repeat at `256`. If you want the full command list rendered automatically, use `conda run -n magformer python scripts/experiments/full19_roster.py --format commands --register 20260318_1K_1566 --dataset-root /home/team/zhanghaonan/magformer/magformer_datasets/20260318_1K_1566 --output-root /home/team/zhanghaonan/magformer/output/experiments/20260318_1k_1566_20ep_512_full19 --mode run --image-size 512 --single-gpu`.
- Rerun the 1024 MAGFormer RGB-only control on the repaired evaluator before using it as a paper baseline. Keep the MAGFormer backbone family, warm-start source, LR schedule, batch, augmentation, and image size fixed, and only disable depth.

## Paper trust assessment

The repaired 1024 table is good enough for engineering comparison. The evaluator path is now unified, the detectron2 bbox metrics survive the summary pipeline, and the table is built from live artifacts only.

The three-resolution table is not paper-ready yet. Every `512` and `256` row is missing in the current tree, and the older multires rows depended on deleted worktree outputs. The right paper move is simple: either publish a verified 1024-only table now, or rerun the missing multires suite before making a three-resolution claim.

I also checked for the research-side concern about output consistency or RGB-teacher distillation. I did not find an output-consistency loss or RGB-teacher distillation path in the current tree.
