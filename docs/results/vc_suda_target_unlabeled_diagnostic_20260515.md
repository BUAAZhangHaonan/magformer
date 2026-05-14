# VC-SUDA Target Unlabeled Diagnostic - 2026-05-15

## Conclusion

On the 200-image target_unlabeled set, R3-A10 is still better than Stage B under the same 1024 backmap evaluation protocol. The gain is small: bbox AP is higher by `+0.0051`, and segm AP is higher by `+0.0088`.

This means the current best checkpoint is not close to a 61+ result through small threshold or loss-weight changes. The next useful diagnostic should be an upper-bound run or a target_labeled sampling / supervision-ratio study.

## Data

- Annotation file: `magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json`
- Images: `200`
- Annotations: `11750`

## Evaluation Protocol

Entry point:

```bash
tools/evaluate_1024_backmap.py \
  --force-pytorch-msda \
  --image-size 1024 \
  --batch-size 1 \
  --num-workers 2
```

Stage B output:

- `output/experiments/diagnostic_stageb_vs_r3a10_target_unlabeled_20260515/stage_b/`

R3-A10 output:

- `output/experiments/diagnostic_stageb_vs_r3a10_target_unlabeled_20260515/r3_a10_best/`

## Results

| Run | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage B | `0.3784` | `0.7260` | `0.3537` | `0.2980` | `0.6392` | `0.2428` |
| R3-A10 | `0.3835` | `0.7330` | `0.3676` | `0.3068` | `0.6428` | `0.2552` |
| Delta | `+0.0051` | `+0.0070` | `+0.0139` | `+0.0088` | `+0.0036` | `+0.0124` |

## Readout

R3-A10 remains stable and slightly better than Stage B on the larger target set. The small delta suggests that more threshold or loss-weight micro-tuning is unlikely to explain the remaining gap.

Recommended next step:

- Run an upper-bound diagnostic, or sample target_labeled data to test supervision ratio sensitivity.
