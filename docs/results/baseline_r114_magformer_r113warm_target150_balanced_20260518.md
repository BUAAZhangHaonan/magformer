# R114 MagFormer R113-Warm Target150 Balanced

Date: 2026-05-18
Branch: feature/vc-suda-sim2real
Config: `configs/baseline_supervised_r114_magformer_r113warm_target150_balanced_2000.yaml`
Setup commit: `42033ad0`

## Purpose

R114 expands labeled target coverage from R113/R112 target100 to target150.

This is not a true resume. It uses the R113 iter2000 checkpoint as a model-only warm-start through `model.finetune_weights`, while optimizer, scheduler, and AMP scaler start fresh with `runtime.resume: null`.

No pseudo labels, Stage C, source replay, or new modules are introduced.

## Split Generation

Input files:

- Base train100: `annotations/instances_target_labeled_r112_balanced_plus75.json`.
- Candidate remaining125: `annotations/instances_target_unlabeled_r112_balanced_minus75.json`.

Generated files:

- Train150: `annotations/instances_target_labeled_r114_balanced_plus125.json`.
- Remaining75: `annotations/instances_target_unlabeled_r114_balanced_minus125.json`.

Selection rule:

- Treat hidden GT in remaining125 as newly promoted target annotations for this pseudo-real simulation.
- Group remaining125 images by annotation count: `low25 <=25`, `mid50 <=50`, `high100 >50`.
- Allocate 50 promoted slots using the R112 proportional largest-remainder recipe: `{low25: 5, mid50: 33, high100: 12}`.
- Within each group, sort by per-image median mask area, then annotation count, file name, and image id.
- Pick deterministic evenly spaced centered indices: `floor((i + 0.5) * n / k)`.

Promoted coverage:

- Promoted bands: `{low25: 5, mid50: 33, high100: 12}`.
- Candidate bands in remaining125: `{'mid50': 82, 'high100': 29, 'low25': 14}`.
- Promoted exact instance counts: `{25: 5, 50: 33, 92: 1, 97: 1, 98: 1, 99: 4, 100: 5}`.

## Non-Leakage Metrics

Primary non-leakage metrics:

- `val28`: `annotations/instances_val.json`, split `val`.
- `remaining75`: `annotations/instances_target_unlabeled_r114_balanced_minus125.json`, split `train`.

Reference-only metric:

- `full200`: `annotations/instances_target_unlabeled.json`, split `train`.

The `full200` row includes promoted training images, so it is only a continuity reference against R113 full200 `0.437321`.

## Remaining75 Risk

`remaining75` is smaller than R113 `remaining125`, so its AP can move more from individual-image composition. The fixed `val28` row is the stable anchor. Remaining75 is still required to stay above the requested threshold.

## Recipe Held Fixed

- Warm-start from `output/baseline/r113_magformer_r112_target100_resume2000/checkpoint_iter_0002000.pth` through `model.finetune_weights`.
- `runtime.resume: null`.
- `solver.max_iter: 2000`.
- `base_lr: 5e-5`, cosine schedule, batch size `4`.
- Image size `1024`.
- No RGB augmentation and no depth noise.
- `importance_sample_ratio: 0.0`.
- `dice_weight: 10.0`, `mask_weight: 5.0`.
- `runtime.depth_sanity.min_mask_fg_ratio: 0.0009`.

## Static Validation

Static validation passed before training.

- `validate_config(strict=True)`: valid. The only message was the existing DPE informational note.
- Warm-start checkpoint exists: `output/baseline/r113_magformer_r112_target100_resume2000/checkpoint_iter_0002000.pth`.
- Config checks: `model.finetune_weights` points to R113 iter2000, `runtime.resume=null`, `solver.max_iter=2000`, train annotation points to R114 train150, no RGB augmentation, no depth noise, `importance_sample_ratio=0.0`, `dice_weight=10.0`, `mask_weight=5.0`, and `runtime.depth_sanity.min_mask_fg_ratio=0.0009`.
- Annotation counts: train150 `150` images / `9083` anns, val28 `28` images / `1892` anns, remaining75 `75` images / `4364` anns, full200 reference `200` images / `11750` anns.
- Empty-annotation image ratio: train150 `0.000000`, val28 `0.000000`, remaining75 `0.000000`, full200 `0.000000`.
- File-name overlap: train150 vs remaining75 `0`; train150 vs val28 `0`; remaining75 vs val28 `0`.
- Loader smoke checked: raw dataset lengths train/val `150 / 28`; DataLoader dataset lengths train/val `150 / 25` because `runtime.eval_max_images=25`; first train batch images `[4, 3, 1024, 1024]`, depths `[4, 1, 1024, 1024]`; first val batch images `[4, 3, 512, 512]`, depths `[4, 1, 512, 512]`.

Current split counts from generation:

| split | images | annotations | empty images |
| --- | ---: | ---: | ---: |
| train150 | 150 | 9083 | 0 |
| remaining75 | 75 | 4364 | 0 |
| val28 | 28 | 1892 | 0 |
| full200 reference | 200 | 11750 | 0 |

## Training

Training ran in tmux session `r114_magformer_r113warm_target150` on GPUs `4,5,6,7`.

Command:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 tools/train.py --config configs/baseline_supervised_r114_magformer_r113warm_target150_balanced_2000.yaml --gpus 4,5,6,7 --num-workers 2
```

Evidence:

- Log: `output/baseline/r114_magformer_r113warm_target150_balanced_2000.tmux.log`.
- Start: `2026-05-18T11:53:57+08:00`.
- Completed: `2026-05-18 12:43:03 +0800`.
- Warm-start loaded R113 iter2000 through `model.finetune_weights` with missing keys `0` and unexpected keys `0`.
- Training started from iter `0/2000`; optimizer, scheduler, and scaler were fresh because `runtime.resume: null`.
- Final checkpoint: `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0002000.pth`.
- GPU memory stayed within available capacity during training; no OOM was observed.
- The train-loop internal final diagnostic eval uses the normal config eval subset and is not the metric used below. The rows below use the requested 1024 backmap external eval protocol.

## Gate Results

All values are COCO AP / AP50 / AP75. Gate decisions use segm AP.

| checkpoint | split | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| iter0999 | train150 | 0.560874 | 0.856788 | 0.653659 | 0.579775 | 0.878363 | 0.676064 |
| iter0999 | val28 | 0.356490 | 0.706913 | 0.325123 | 0.319329 | 0.637294 | 0.290806 |
| iter0999 | remaining75 | 0.430212 | 0.758313 | 0.432457 | 0.389598 | 0.704115 | 0.391594 |
| iter0999 | full200 reference | 0.515039 | 0.822101 | 0.573327 | 0.507673 | 0.816602 | 0.568348 |
| iter1499 | train150 | 0.567771 | 0.859448 | 0.663777 | 0.598508 | 0.888415 | 0.706024 |
| iter1499 | val28 | 0.353567 | 0.705338 | 0.319435 | 0.321155 | 0.637031 | 0.294728 |
| iter1499 | remaining75 | 0.428594 | 0.754434 | 0.435351 | 0.393437 | 0.703438 | 0.396390 |
| iter1499 | full200 reference | 0.517173 | 0.824411 | 0.580015 | 0.519065 | 0.819723 | 0.583536 |

Gate decisions:

- Iter0999 passed: val28 `0.319329 >= 0.300`, remaining75 `0.389598 >= 0.340`, and train150 was above `0.55`.
- Iter1499 passed: val28 `0.321155 >= 0.300`, remaining75 `0.393437 >= 0.345`, and train150 improved by `0.018733`, above the `0.005` stall guard.
- Training continued to iter2000.

## Final Best Eval

Best selected checkpoint: `checkpoint_iter_0002000.pth`. It has the highest external val28 segm AP among evaluated checkpoints.

| split | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train150 | 0.570574 | 0.859131 | 0.666607 | 0.599756 | 0.888771 | 0.708763 |
| val28 | 0.356494 | 0.704215 | 0.321469 | 0.321831 | 0.630752 | 0.295205 |
| remaining75 | 0.430296 | 0.756221 | 0.443207 | 0.392173 | 0.703252 | 0.397419 |
| full200 reference | 0.521151 | 0.825039 | 0.587714 | 0.520191 | 0.820041 | 0.589885 |
| original first50 source sanity | 0.421511 | 0.733598 | 0.442868 | 0.500775 | 0.791382 | 0.552180 |

## Success Check

| condition | result |
| --- | --- |
| train150 segm AP `>= 0.61` | no, `0.599756` |
| val28 segm AP `>= 0.305` | yes, `0.321831` |
| remaining75 segm AP `>= 0.350` | yes, `0.392173` |
| full200 reference segm AP `> 0.437321` | yes, `0.520191` |
| formal 61+ target | no |

Conclusion: R114 target150 improves full200 reference and keeps both non-leakage checks above threshold, but it does not reach the formal train150 `0.61+` target.

## Notes

- `remaining75` is small, so it should not be treated as the only quality signal. It passed the requested threshold.
- The final log includes the PyTorch 2.4 NCCL process-group destruction warning at normal exit. There was no training traceback or CUDA OOM in the observed run.
