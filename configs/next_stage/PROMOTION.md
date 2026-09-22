# Next-stage promotion seeds

These experiment configs are standalone because `load_config()` does not
support cross-file inheritance. A partial YAML file is therefore not a valid
promotion template.

The initial screening set is C0, P1, P2, and D0. P3 is conditional and must
not be launched until the screening decision promotes it.

After one seed-42 arm is selected, derive seeds 43 and 44 by copying that
arm's complete file:

```text
configs/next_stage/<arm>_seed42.yaml
  -> configs/next_stage/<arm>_seed43.yaml
  -> configs/next_stage/<arm>_seed44.yaml
```

Change exactly these three values in each copy:

```yaml
name: 20260712_next_stage_<arm>_seed43
runtime:
  output_dir: /home/g203-4028/magformer/output/experiments/next_stage/<arm>_seed43
  seed: 43
```

Use the corresponding `seed44` suffix and `seed: 44` for the second copy.
Keep `runtime.resume: null`.

Do not change the selected arm's point budgets, oversampling ratio,
`dccg_use_confidence`, model initialization, optimizer, schedule, data, or
evaluation settings. In particular, both promoted seeds must continue to
warm-start from:

```text
/home/g203-4028/magformer/output/experiments/v317_init_from_m2f/model_init.pth
```
