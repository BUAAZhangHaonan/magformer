# R128 Gated R122 Evaluator

Date: 2026-05-18

R128 adds a gated watcher for the R122 iter0099 evaluation. It does not replace R126 or R127. R126 still owns CUDA recovery and R121 smoke. R127 still owns the gated R122 300iter launch. R128 only watches the readonly R121/R122 state and may run the R122 evaluation after R122 training has finished.

## Active watcher

Session:

```bash
tmux attach -t r128_gated_r122_evaluator
```

Log:

```bash
tail -f output/diagnostics/r128_gated_r122_evaluator_20260518.log
```

Stop command:

```bash
tmux kill-session -t r128_gated_r122_evaluator
```

Watcher script:

```bash
tools/run_r128_gated_r122_evaluator.sh
```

## State gate

Every 300 seconds, the watcher runs:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python \
tools/check_r121_r122_resume_state.py \
--output-json /tmp/r128_state.json \
--output-md /tmp/r128_state.md
```

R128 can only consider evaluation when the JSON field `state` is exactly `NEED_R122_EVAL`.

For these states, R128 records the state and continues sleeping:

- `BLOCKED_CUDA`
- `NEED_R121_SMOKE`
- `NEED_R121_TRAIN`
- `R122_TRAINING`
- `NEED_R122_TRAIN`

For these states, R128 records the state and exits without evaluation:

- `R121_FAILED`
- `NEED_GO_NO_GO`
- `READY_TO_DECIDE`

R128 never starts R121 or R122. It does not run the go/no-go comparator automatically.

## Safety gates before eval

When the state is `NEED_R122_EVAL`, R128 must pass all of these checks before launching evaluation:

- No current-user process command contains `train.py`, `torchrun`, or `torch.distributed.run`.
- `output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300/checkpoint_iter_0000099.pth` exists.
- The checkpoint file size is identical across two `stat` checks separated by 30 seconds.
- GPU4 passes a single-GPU PyTorch CUDA probe.
- Each eval output directory is absent, or already complete with both `coco_instances_results.json` and `metrics.cocoeval.json`.
- If an eval output directory exists but either required file is missing, R128 records the partial directory and exits without overwriting it.

## Evaluation commands

R128 runs remaining75 first:

```bash
CUDA_VISIBLE_DEVICES=4 MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/evaluate_1024_backmap.py --base-config configs/baseline_vc_suda_r122_depth_boundary_w001_pseudo300.yaml --dataset-root magformer_datasets/pseudo_real_512 --ann annotations/instances_target_unlabeled_r114_balanced_minus125.json --split train --weights output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300/checkpoint_iter_0000099.pth --output-dir output/diagnostics/r122_depth_boundary_w001_iter0099_remaining75_1024_backmap_topk200_20260518 --image-size 1024 --batch-size 4 --num-workers 0 --score-threshold 0.05 --mask-threshold 0.5 --iou-types bbox,segm --inference-topk 200 --max-dets 200 --force-pytorch-msda
```

Then R128 runs val28:

```bash
CUDA_VISIBLE_DEVICES=4 MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/evaluate_1024_backmap.py --base-config configs/baseline_vc_suda_r122_depth_boundary_w001_pseudo300.yaml --dataset-root magformer_datasets/pseudo_real_512 --ann annotations/instances_val.json --split val --weights output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300/checkpoint_iter_0000099.pth --output-dir output/diagnostics/r122_depth_boundary_w001_iter0099_val28_1024_backmap_topk200_20260518 --image-size 1024 --batch-size 4 --num-workers 0 --score-threshold 0.05 --mask-threshold 0.5 --iou-types bbox,segm --inference-topk 200 --max-dets 200 --force-pytorch-msda
```

After both output directories contain `coco_instances_results.json` and `metrics.cocoeval.json`, R128 exits. It leaves `bucket_compare.csv` and go/no-go comparison to a separate manual step.

## Initial status

At startup on 2026-05-18, the watcher should write the first state check to `output/diagnostics/r128_gated_r122_evaluator_20260518.log`. If the current state is `BLOCKED_CUDA`, `NEED_R121_SMOKE`, `NEED_R121_TRAIN`, `R122_TRAINING`, or `NEED_R122_TRAIN`, it records the state and sleeps for 300 seconds without starting eval.
