# VC-SUDA R65 multi-source retention short diagnostic - 2026-05-16

## Conclusion

R65 fails the hard gate.

The explicit multi-source Stage B run learns target scale, but it does not keep source sanity and it does not reach target AP. The best checkpoint is `checkpoint_iter_0000999.pth`: source first50 segm AP `0.501977`, target_unlabeled200 segm AP `0.196696`, and target bbox area ratio `1.590x`.

This means the direction fixes the scale mismatch but not the AP objective. It is not a hard pass or partial useful by the stated thresholds, because source AP is below `0.55`.

## Configuration

- Config: `configs/vc_suda_stage_b_r65_multisource_retention_1024.yaml`
- Warm start: `output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth`
- Stage: `B`
- No pseudo / EMA / Stage C: `vc_suda.unsupervised_weight=0.0`, `vc_suda.ema_teacher.enabled=false`, `runtime.ema_enabled=false`
- Source datasets, 1:1:
  - `original`: `magformer_datasets/20260318_1K_1566`, `annotations/instances_train.json`, split `train`, weight `1`
  - `pseudo`: `magformer_datasets/pseudo_real_512`, `annotations/instances_source.json`, split `train`, weight `1`
- Target root: `magformer_datasets/pseudo_real_512`
- Target labeled: `annotations/instances_target_labeled.json`, `target_labeled_weight=1.0`
- Target unlabeled manifest kept configured as `annotations/instances_target_unlabeled.json`, but Stage B does not load a target-unlabeled branch.
- Solver: `max_iter=1000`, `checkpoint_period=500`, `eval_period=500`, `base_lr=1.0e-05`
- LR choice: `1e-5` follows the low-LR R8B/R46 family used for retention-sensitive target gates. The logged distributed LR was `2e-05`.

## Smoke

1024 no-backward real dataloader smoke passed.

- Source names first four samples: `original, pseudo, original, pseudo`
- Source image tensor: `(4, 3, 1024, 1024)`
- Source depth tensor: `(4, 1, 1024, 1024)`
- Target-labeled image tensor: `(4, 3, 1024, 1024)`
- Target-labeled depth tensor: `(4, 1, 1024, 1024)`

## Tests

Passed before the config milestone commit:

```bash
pytest tests/test_semi_supervised_multi_source_dataset.py   tests/test_vc_suda_train_entrypoint.py   tests/test_train_warm_start.py   tests/test_teacher_first50_protocol.py -q
```

Result: `35 passed`.

Also passed:

```bash
ruff check tests/test_vc_suda_train_entrypoint.py
git diff --check
```

The new regression test is `test_stage_b_r65_multisource_retention_config_routes_two_sources`.

## Config Milestone

- Commit: `e3256908bc07241fa6863c4f94155d1af3276392`
- Direct push from `4029` timed out with exit code `124`.
- Bundle bridge push succeeded.
- Verification:

```text
e3256908bc07241fa6863c4f94155d1af3276392 refs/heads/feature/vc-suda-sim2real
```

## Training

- tmux session: `r65_multisource_retention_20260516`
- Command log: `output/vc_suda/vc_suda_stage_b_r65_multisource_retention_1024/train_1000iter.log`
- GPUs: `4,5,6,7`
- Exit status: `0`
- Wall time: `47:21.86`
- Warm-start strict load: `missing keys: 0`, `unexpected keys: 0`
- Checkpoints:
  - `output/vc_suda/vc_suda_stage_b_r65_multisource_retention_1024/checkpoint_iter_0000499.pth`
  - `output/vc_suda/vc_suda_stage_b_r65_multisource_retention_1024/checkpoint_iter_0000999.pth`
  - `output/vc_suda/vc_suda_stage_b_r65_multisource_retention_1024/checkpoint_iter_0001000.pth`
- Resource observations:
  - Launch check before tmux: RAM about `15,953 / 257,582 MB`, GPUs 4-7 at `15 MiB` each.
  - During training sampled RAM stayed around `11.7%-12.8%`.
  - Sampled GPU memory stayed below about `17,655 MiB` on 24GB cards.
  - No OOM or collate failure appeared in the log.

The tmux memory guard line had a shell quoting bug and printed an empty `preflight_mem_pct`, but independent checks before and during the run showed memory was far below the 90% cap.

## External Eval

Both protocol checkers passed for both checkpoints:

- `original_first50_teacher` with `--allow-nondefault-weights`
- `pseudo_real_target_unlabeled200`

The first target eval attempt failed before inference because the direct command did not export `MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch`. The final target evals were rerun with the explicit PyTorch MSDA backend and exited `0`.

| Checkpoint | Source first50 bbox AP/AP50/AP75 | Source first50 segm AP/AP50/AP75 | Target200 bbox AP/AP50/AP75 | Target200 segm AP/AP50/AP75 | Target bbox area ratio |
|---|---:|---:|---:|---:|---:|
| `0000499` | `0.552316 / 0.803740 / 0.605977` | `0.501789 / 0.790285 / 0.551627` | `0.257596 / 0.596253 / 0.196361` | `0.186416 / 0.499178 / 0.084457` | `1.619x` |
| `0000999` | `0.556541 / 0.806486 / 0.608113` | `0.501977 / 0.791856 / 0.552810` | `0.270946 / 0.610781 / 0.214415` | `0.196696 / 0.513603 / 0.097482` | `1.590x` |

Area-ratio details:

- GT bbox area p50 over images: `779.75`
- `ckpt0000499` pred bbox area p50 over images: `1262.75`, ratio `1.619429`
- `ckpt0000999` pred bbox area p50 over images: `1240.0`, ratio `1.590253`
- Ratio JSON: `output/diagnostics/r65_multisource_retention_20260516/target_bbox_area_ratios.json`

## Gate Decision

Fail.

- Hard pass requires source first50 segm AP `>=0.55`, target_unlabeled200 segm AP `>=0.30`, and target bbox area ratio `<=2.0`.
- R65 has target area ratio `<=2.0`, but source segm AP is only about `0.502`, and target segm AP is only about `0.197`.
- Partial useful requires source AP `>=0.55` and target area ratio below `3`; R65 misses the source condition.

## Readout

Explicit multi-source mixing with strong target labels aligns target scale quickly. It does not recover the original source sanity line, and target AP remains far below the R46/R52 target-domain range. This points to an objective/optimization conflict rather than only a source-scale mix problem.
