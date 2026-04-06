The fair no-depth control is cloned from the live `68.42 AP` depth recipe in `output/experiments/20260318_1k_1566_20ep_1024_full19/magformer_depthnorm_on/magformer_runtime_config.yaml`.

For this campaign, the model and solver recipe stay the same. The only model-side changes are:

```diff
- dpe_enabled: true
+ dpe_enabled: false

- model.magformer.depth_backbone.enabled: true
+ model.magformer.depth_backbone.enabled: false

- model.magformer.modality_fusion.enabled: true
+ model.magformer.modality_fusion.enabled: false
```

The remaining differences are bookkeeping and launch shape for this campaign:

```diff
- name: 20260318_1k_1566_20ep_1024_depthnorm_on
+ name: magformer_nodpth_ref_fair

- runtime.ddp_enabled: true
+ runtime.ddp_enabled: false

- runtime.find_unused_parameters: true
+ runtime.find_unused_parameters: false

- runtime.gpus: [0, 1]
+ runtime.gpus: [0]

- runtime.logger.log_dir: /home/team/zhanghaonan/magformer/output/experiments/20260318_1k_1566_20ep_1024_full19/_staging/magformer_depthnorm_on/logs
+ runtime.logger.log_dir: output/experiments/20260318_1k_1566_20ep_1024_full19/magformer_nodpth_ref_fair/logs

- runtime.logger.run_name: 20260318_1k_1566_20ep_1024_depthnorm_on
+ runtime.logger.run_name: 20260318_1k_1566_20ep_1024_nodpth_ref_fair

- runtime.output_dir: /home/team/zhanghaonan/magformer/output/experiments/20260318_1k_1566_20ep_1024_full19/_staging/magformer_depthnorm_on
+ runtime.output_dir: output/experiments/20260318_1k_1566_20ep_1024_full19/magformer_nodpth_ref_fair
```

The warm start stays the same in both runs:

```yaml
model.finetune_weights: output/pretrained/mgm_0831_full_to_magformer_d2swin_v2.pth
```
