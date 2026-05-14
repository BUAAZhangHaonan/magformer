# VC-SUDA Stage C Mid Eval - 2026-05-14

## Summary

- Host: WS-4029GP-TRT via `ssh 4029`.
- Repository: `/home/hdd3/zhanghaonan/magformer`.
- Branch: `feature/vc-suda-sim2real`.
- Conda env: `magformer`.
- Training run left untouched: `output/experiments/vc_suda_stage_c_1024_teacher8499_20260514_0734`.
- Evaluated checkpoint: `output/experiments/vc_suda_stage_c_1024_teacher8499_20260514_0734/checkpoint_iter_0003999.pth`.
- Eval output: `output/experiments/20260514_stage_c_mid_eval/checkpoint_3999_segm`.
- Eval dataset: `magformer_datasets/pseudo_real_512`, split `val`, annotation `annotations/instances_val.json`.
- Val images: `28`.
- COCO predictions exported: `1604`.
- Images with predictions: `28`.

## GPU Safety Check

Before eval, physical GPU 0-3 were idle and Stage C training was active on physical GPU 4-7:

```text
0, 38 MiB, 0%
1, 15 MiB, 0%
2, 15 MiB, 0%
3, 15 MiB, 0%
4, 21009 MiB, 100%
5, 21005 MiB, 100%
6, 21013 MiB, 100%
7, 20967 MiB, 0%
```

The active Stage C worker environment confirmed the physical GPU binding:

```text
CUDA_VISIBLE_DEVICES=4,5,6,7
CONDA_DEFAULT_ENV=magformer
```

The eval command used only physical GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate.py \
  --config-file configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml \
  --weights output/experiments/vc_suda_stage_c_1024_teacher8499_20260514_0734/checkpoint_iter_0003999.pth \
  --output output/experiments/20260514_stage_c_mid_eval/checkpoint_3999_segm \
  --batch-size 1 \
  --num-workers 2
```

During eval, GPU 0 held the eval process and GPU 4-7 continued holding the Stage C training workers. After eval, GPU 0 returned to idle and the Stage C workers were still active on GPU 4-7.

## Eval Config Safety

- Entry point: `tools/evaluate.py`.
- Eval IoU types: `['bbox', 'segm']`, confirmed in the eval log.
- Empty prediction guard: `fail_on_empty=True` is hardcoded in `tools/evaluate.py` for `run_inference_evaluation()`.
- Runtime GPU in the eval config is `runtime.gpus: [0]`; with `CUDA_VISIBLE_DEVICES=0`, this maps to physical GPU 0.
- No training process was stopped, started, or restarted.
- No branch or worktree was created.

## Metrics

| Type | AP | AP50 | AP75 |
| --- | ---: | ---: | ---: |
| bbox | 0.1576 | 0.4679 | 0.0710 |
| segm | 0.1082 | 0.3649 | 0.0356 |

Full COCO metric dict from `eval.log`:

```text
bbox_AP: 0.15764981616678725
bbox_AP50: 0.46790157823994705
bbox_AP75: 0.07103544780092662
segm_AP: 0.10821729105381235
segm_AP50: 0.36487699109126737
segm_AP75: 0.03557844887297123
```

## Artifacts

Kept under output paths and not committed:

- `output/experiments/20260514_stage_c_mid_eval/checkpoint_3999_segm/eval.log`
- `output/experiments/20260514_stage_c_mid_eval/checkpoint_3999_segm/coco_instances_results.json`
