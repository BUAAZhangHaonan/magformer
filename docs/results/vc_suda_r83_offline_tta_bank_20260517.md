# VC-SUDA R83 offline TTA bank formal run, 2026-05-17

R83 did not reach iter750. I stopped the formal tmux run because the early weighted pseudo loss ratio crossed the hard stop gate twice in a row.

No eval was run. There is no valid R83 iter750 checkpoint for target_unlabeled200, source first50, or hard-bucket diagnosis.

## Scope

- Host: `4029`.
- Project: `/home/hdd3/zhanghaonan/magformer`.
- Conda env: `/home/hdd3/zhanghaonan/anaconda3/envs/magformer`.
- Branch: `feature/vc-suda-sim2real`.
- Config: `configs/vc_suda_stage_c_r83_offline_tta_bank_1024.yaml`.
- Bank: `output/diagnostics/r83_tta_coco_bank_20260517/instances_tta_pseudo_score090.json`.
- Output dir: `output/vc_suda/stage_c_r83_offline_tta_bank_1024`.
- Train tmux session: `r83_offline_tta_bank_20260517`.
- Train log: `output/logs/r83_offline_tta_bank_train_20260517.log`.
- Resource log: `output/logs/r83_offline_tta_bank_resource_20260517.log`.

No training code, config semantics, or new module was changed for this run.

## Preflight

Preflight passed before training.

| check | result |
|---|---|
| output dir overwrite check | `output_dir_absent` |
| `tools/verify_vc_suda_stage.py --stage C` | pass |
| target_unlabeled count | `200` |
| offline bank validate-only images | `200` |
| offline bank validate-only annotations | `13,386` |
| offline bank empty images | `0` |
| offline bank score mean / p50 | `0.953267 / 0.959388` |
| offline bank fill mean / p50 | `0.652340 / 0.670330` |

Bank validate-only reported `empty_images: 0`, categories `[1]`, and finite score/fill/area summaries.

## Training Command

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 MAGFORMER_MS_DEFORM_ATTN_BACKEND=cuda OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python -m torch.distributed.run   --standalone --nproc_per_node=4 tools/train.py   --config configs/vc_suda_stage_c_r83_offline_tta_bank_1024.yaml   --output-dir output/vc_suda/stage_c_r83_offline_tta_bank_1024   --gpus 0,1,2,3   --num-workers 2
```

The wrapper also ran a RAM monitor and would stop the process at `>=90%` RAM.

## Stop Event

- Start: `2026-05-17T14:29:06+08:00`.
- Resume point: R12 `checkpoint_iter_0000499.pth`, resumed at iter `499`.
- Stop: `2026-05-17T14:33:29+08:00`.
- Exit code: `143`, from the intentional SIGTERM.
- Last progress in stdout: around iter `528`.
- Logged metric rows before stop: iter `500` and `520`.

The hard stop was: early weighted pseudo loss / total loss was continuously above `20%`.

I computed it as:

```text
weighted pseudo / total = vc_suda.unsupervised_weight * train/pseudo_total / train/loss
```

Here `vc_suda.unsupervised_weight` is `0.5`.

| iter | train loss | pseudo_total | weighted pseudo / total | offline mode | offline count | offline empty | offline score-weight ratio | pseudo loss present |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 500 | `39.829399` | `24.843750` | `31.1877%` | `1.0` | `39` | `0` | `0.959873` | yes |
| 520 | `23.607292` | `14.656250` | `31.0418%` | `1.0` | `68` | `0` | `0.952276` | yes |

This satisfied the stop rule: two consecutive early logged rows were `>20%`.

## Health Checks

- No `Traceback` was seen before the intentional stop.
- No OOM was seen.
- No `RuntimeError` was seen.
- No non-finite metric was seen in the logged rows.
- Pseudo loss was present in both logged rows.
- `train/pseudo_offline_empty_images` stayed `0.0`.
- No online pseudo/scorer marker was seen. The word `teacher` appeared only in checkpoint path/config text, not as an online teacher/scorer action.
- Max RAM in the resource log was `21.96%`, below the `90%` gate.
- Max GPU memory observed on GPUs 4-7 was `{4: 21365, 5: 21969, 6: 21633, 7: 20581}` MiB.

## Eval And Gate Status

Eval was not run because the formal run did not reach iter750 and no valid R83 iter750 checkpoint exists.

| item | status |
|---|---|
| target_unlabeled200 bbox+segm eval | blocked |
| original first50 source sanity | blocked |
| R78 hard bucket diagnosis | blocked |
| target AP gates | not evaluated |
| source first50 gates | not evaluated |
| hard bucket gates | not evaluated |

R83 does not pass the requested gate. The run stopped by the user-defined safety rule before the checkpoint needed for evaluation.

## Artifacts

These are run artifacts only and should not be committed:

- `output/logs/r83_offline_tta_bank_preflight_20260517.log`.
- `output/logs/r83_offline_tta_bank_train_20260517.log`.
- `output/logs/r83_offline_tta_bank_resource_20260517.log`.
- `output/vc_suda/stage_c_r83_offline_tta_bank_1024/metrics_log.jsonl`.
- `output/vc_suda/stage_c_r83_offline_tta_bank_1024/metrics_log.csv`.
- `output/vc_suda/stage_c_r83_offline_tta_bank_1024/config_resolved.yaml`.
- `output/vc_suda/stage_c_r83_offline_tta_bank_1024/depth_sanity.json`.

## Decision

Stop and do not evaluate this R83 attempt.

The offline bank itself loaded cleanly and gave non-empty pseudo labels, but the current R83 objective weight is too large at the start of the formal run. The next executable step would need a deliberate config-level objective change, not a continuation of this stopped run.
