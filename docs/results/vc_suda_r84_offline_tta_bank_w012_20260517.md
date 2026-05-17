# VC-SUDA R84 offline TTA bank w0.12, 2026-05-17

R84 did not pass the target gate.

The low-weight offline bank finished training to iter750, but target AP moved below R80. Source first50 still passed its sanity gate. Hard buckets were mixed: tiny and bottom got small positive signals, but dense AP75 and overall oracle did not improve.

## Scope

- Host: `4029`.
- Project: `/home/hdd3/zhanghaonan/magformer`.
- Conda env: `/home/hdd3/zhanghaonan/anaconda3/envs/magformer`.
- Branch: `feature/vc-suda-sim2real`.
- Config: `configs/vc_suda_stage_c_r84_offline_tta_bank_w012_1024.yaml`.
- Output dir: `output/vc_suda/stage_c_r84_offline_tta_bank_w012_1024`.
- Checkpoint: `output/vc_suda/stage_c_r84_offline_tta_bank_w012_1024/checkpoint_iter_0000750.pth`.
- Local config commit: `87fb6a395d33e4d03ba8d353a77c26039bee4ad7`.
- GitHub connector config commit: `6e040c32948b23817317a9b8d89605b54c713b8a`.

GitHub SSH from `4029` timed out on `ssh.github.com:443`, so the config was pushed through the GitHub connector. The local repo still has the local config commit.

## Config Diff

Only the allowed fields changed from R83:

- `name`: `vc_suda_stage_c_r83_offline_tta_bank_1024` -> `vc_suda_stage_c_r84_offline_tta_bank_w012_1024`.
- `runtime.output_dir`: `output/vc_suda/stage_c_r83_offline_tta_bank_1024` -> `output/vc_suda/stage_c_r84_offline_tta_bank_w012_1024`.
- `runtime.logger.log_dir`: matching R84 output dir.
- `runtime.logger.run_name`: `vc_suda_stage_c_r84_offline_tta_bank_w012_1024`.
- `vc_suda.unsupervised_weight`: `0.5` -> `0.12`.

R83 semantics were otherwise kept: offline bank path, `min_score=0.9`, `max_instances=100`, R12 ckpt499 resume, `max_iter=750`, source/target splits, `unsupervised_warmup_epochs=1`, model/loss/depth/optimizer/balanced CE.

## Preflight

| check | result |
|---|---|
| R84 vs R83 diff | only allowed fields |
| `tools/verify_vc_suda_stage.py --stage C --skip-batch --emit-data-evidence` | pass |
| output dir | absent before launch |
| bank validate-only images / annotations | `200 / 13,386` |
| bank empty images | `0` |
| bank score mean / p50 | `0.953267 / 0.959388` |
| bank fill mean / p50 | `0.652340 / 0.670330` |

## Training

Tmux session: `r84_offline_tta_bank_w012_20260517`.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 MAGFORMER_MS_DEFORM_ATTN_BACKEND=cuda OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 tools/train.py \
  --config configs/vc_suda_stage_c_r84_offline_tta_bank_w012_1024.yaml \
  --output-dir output/vc_suda/stage_c_r84_offline_tta_bank_w012_1024 \
  --gpus 0,1,2,3 \
  --num-workers 2
```

The run resumed R12 `checkpoint_iter_0000499.pth`, reached iter750, and exited `0`.

Early pseudo gate:

| iter | loss | pseudo_total | weighted pseudo / total | offline count | offline empty | pseudo empty |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | `30.492641` | `24.625000` | `9.6909%` | `39` | `0` | `0` |
| 520 | `17.361153` | `15.000000` | `10.3680%` | `68` | `0` | `0` |

Later logged rows stayed finite. A late iter700 row reached `15.9742%`, but the formal stop gate was the iter500/520 check. No `Traceback`, OOM, `RuntimeError`, non-finite signal, RAM stop, online teacher marker, or scorer marker appeared in the training log. RAM peaked around `18.16%` in the resource log.

## External Eval

Both protocol checkers passed:

- Target: `pseudo_real_target_unlabeled200`, 200 images, 1024 backmap, `bbox,segm`, `inference_topk=200`, `max_dets=200`.
- Source: original first50, 1024 backmap, `bbox,segm`, `inference_topk=100`, `max_dets=100`, `--allow-nondefault-weights`.

| eval | metric | R80 baseline | R84 | gate | pass |
|---|---|---:|---:|---:|---|
| target_unlabeled200 | bbox AP | `0.389937` | `0.385125` | `>=0.3890` | no |
| target_unlabeled200 | segm AP | `0.336383` | `0.335082` | `>=0.3400` | no |
| target_unlabeled200 | segm AP75 | `0.317290` | `0.315377` | `>=0.3200` | no |
| original first50 | bbox AP | `0.474794` | `0.472308` | `>=0.4720` | yes |
| original first50 | segm AP | `0.439126` | `0.436850` | `>=0.4360` | yes |

## Hard Buckets

| bucket | R80 segm AP/AP75 | R84 segm AP/AP75 | delta AP/AP75 | R84 oracle R@50/R@75 |
|---|---:|---:|---:|---:|
| overall | `0.336383 / 0.317290` | `0.335082 / 0.315377` | `-0.001301 / -0.001913` | `0.696340 / 0.372085` |
| dense | `0.173411 / 0.115247` | `0.176744 / 0.113447` | `+0.003333 / -0.001800` | `0.511949 / 0.174689` |
| dense_tiny | `0.213297 / 0.158565` | `0.215045 / 0.159599` | `+0.001748 / +0.001034` | `0.552838 / 0.214847` |
| tiny_area_le_256 | `0.010156 / 0.000354` | `0.010866 / 0.000402` | `+0.000710 / +0.000048` | `0.190866 / 0.017382` |
| bottom20_area | `0.003265 / 0.000079` | `0.003318 / 0.000078` | `+0.000053 / -0.000002` | `0.127234 / 0.007660` |

Oracle:

- overall `oracle_r75`: `0.372085`.
- matched oracle-score segm AP: `0.376238`.
- dense `>90` oracle R@75: `0.191060`.
- tiny `<=256` oracle R@75: `0.017382`.

## Decision

R84 is not a pass. Lowering the offline pseudo weight fixed the early loss-ratio issue, but it did not improve the target metrics enough. The source sanity gate passed, so the main failure is target-side quality, not source collapse.

The hard-bucket result is not a clean overall win. Tiny and bottom have small positive signals, and dense_tiny improved a little, but target AP and overall oracle went down. This does not justify promoting R84.

## Artifacts

- Train log: `output/logs/r84_offline_tta_bank_w012_train_20260517.log`.
- Resource log: `output/logs/r84_offline_tta_bank_w012_resource_20260517.log`.
- Target metrics: `output/diagnostics/r84_offline_tta_bank_w012_target_unlabeled200_20260517/metrics.cocoeval.json`.
- Source metrics: `output/diagnostics/r84_offline_tta_bank_w012_original_first50_20260517/metrics.cocoeval.json`.
- Bucket AP: `output/diagnostics/r84_offline_tta_bank_w012_target_unlabeled200_20260517/r84_bucket_ap.md`.
- Oracle: `output/diagnostics/r84_offline_tta_bank_w012_target_unlabeled200_20260517/oracle/r53_prediction_union_oracle_summary.json`.
