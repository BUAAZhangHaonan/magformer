# VC-SUDA Stage C-R2 status - 2026-05-14

## Current status

Stage C-R2 has started and passed the first fast-eval continuation gate at iter1000. This is not the final target and does not prove Stage C-R2 is complete. The run should continue to iter2000, and the final judgment still needs a full bbox+segm evaluation.

## Launch record

- Remote: `4029:/home/hdd3/zhanghaonan/magformer`
- tmux session: `vc_suda_stage_c_r2_20260514`
- Output dir: `output/vc_suda/stage_c_r2_1024_teacher8499`
- Log: `output/vc_suda/stage_c_r2_1024_teacher8499/train_launch.log`
- Command:

```bash
torchrun --standalone --nproc_per_node=4 tools/train.py --config configs/vc_suda_stage_c_r2_1024_teacher8499.yaml --gpus 4,5,6,7 --num-workers 2
```

## Loaded weights and schedule

- Warm-start checkpoint: `output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth`
- Warm-start result: missing keys = 0, unexpected keys = 0
- `max_iter=4000`
- `eval_period=1000`
- `ckpt_period=1000`
- Fast eval mode: bbox-only
- `eval_max_images=28`
- `eval_batch_size=4`

## Iter1000 fast eval

The first eval row is recorded at iter999 in `metrics_log.csv`, which corresponds to the iter1000 scheduled eval.

| eval | bbox AP | bbox AP50 | bbox AP75 | segm metrics |
| --- | ---: | ---: | ---: | --- |
| Stage C-R2 iter1000 fast eval | 0.1713 | 0.5041 | 0.0806 | not run |
| Stage B baseline post eval | 0.16374048014700276 | 0.4824453284166137 | n/a | segm AP=0.11284750849899325, segm AP50=0.38104397305161036 |

Conclusion: iter1000 bbox AP is slightly above the Stage B bbox baseline, so this passes the first continue gate. This remains only a bbox-only quick check. Stage C-R2 should continue to iter2000, and the final decision still needs full bbox+segm evaluation.

## Pseudo branch metrics

`metrics_log.csv` shows that the pseudo branch has been writing metrics.

| iter | timestamp | pseudo_total | keep_rate | kept_count | empty_images | threshold |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 100 | 2026-05-14T18:27:11+08:00 | 25.149127960205078 | 0.07999999821186066 | 8.0 | 0.0 | 0.20000000298023224 |
| 1000 | 2026-05-14T19:04:11+08:00 | 40.911014556884766 | 0.019999999552965164 | 2.0 | 0.0 | 0.20000000298023224 |
| 1060 | 2026-05-14T19:06:29+08:00 | 33.62436294555664 | 0.03999999910593033 | 4.0 | 0.0 | 0.20000000298023224 |

The iter1060 row was the latest train row visible in `metrics_log.csv` during this documentation update.
