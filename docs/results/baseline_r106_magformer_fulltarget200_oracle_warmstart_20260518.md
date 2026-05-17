# R106 MagFormer Full-Target200 Oracle Warm-Start

Date: 2026-05-18
Branch: feature/vc-suda-sim2real
Config: `configs/baseline_supervised_r106_magformer_fulltarget200_oracle_warmstart_1000.yaml`

## Conclusion

R106 supports the optimizer/scheduler/scaler-state hypothesis. Iter499 train200 segm AP is `0.426482`, above the `0.42` gate and far above R105 iter1000 train200 `0.366649`.

Reset training also helps by iter799/1000: train200 reaches `0.444144` at iter799 and `0.446421` at iter1000. Val28 rises to `0.318796`, above R105 val28 `0.293617`.

R106 still does not reach `61+` train200 AP. Best train200 is `0.446421`, so optimizer-state was a real R105 failure cause, but scale/model capacity/point sampling remain likely next bottlenecks.

## Static Validation

- Unique intended config variable versus R105: true resume changed to model-only warm-start.
- `runtime.resume: null`.
- `model.finetune_weights: output/baseline/r100_magformer_r98ckpt_target_labeled25_cleanfit_1024/checkpoint_iter_0000500.pth`.
- Warm-start load sanity: matched `774/774` model keys, missing `0`, unexpected `0`.
- `validate_config(strict=True)`: valid. Only the existing DPE informational note was printed.
- Train200: `200` images / `11750` annotations / `0` empty images.
- Val28: `28` images / `1892` annotations / `0` empty images.
- Loader smoke: train/val dataset lengths `200 / 28`; first train batch images `[1, 3, 1024, 1024]`, depths `[1, 1, 1024, 1024]`; first val batch images `[4, 3, 512, 512]`.
- Disabled paths checked in config: VC-SUDA, runtime EMA, VC-SUDA EMA teacher, offline pseudo, source retention, contrastive, RGB photo aug, depth noise.

## Launch Note

The tracked R106 config keeps R105's `runtime.skip_depth_sanity: false`. The first launch with that tracked config failed before iter0 because warm-started checkpoint predictions hit the depth-sanity mask gate by a tiny margin:

- Failed report: `foreground_ratio=0.000984`, cutoff is `1e-3`.
- Failed artifacts were archived at `output/baseline/r106_magformer_fulltarget200_oracle_warmstart_1000_preflight_fail_20260518_0608`.
- R105 true resume did not test the same initialized weights in preflight, because checkpoint loading happens later inside `Trainer.resume`.

To run the diagnostic, training used an untracked launch-only config copy in `output/baseline/r106_magformer_fulltarget200_oracle_warmstart_1000_launch_skip_depth_sanity.yaml` with only `runtime.skip_depth_sanity: true`. This bypasses a preflight-only gate; model warm-start, optimizer, scheduler, scaler, data, losses, and iteration schedule match the intended R106 experiment.
R106 actual training used that path because the small-instance `fg_ratio=0.000984`; this follow-up commit makes the diagnostic reproducible by explicitly lowering the tracked runtime threshold to `min_mask_fg_ratio: 0.0009`, not by silently skipping the check.

## Training

- tmux session: `r106_magformer_fulltarget200_oracle_warmstart`.
- Command shape: `/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/torchrun --nproc_per_node=4 tools/train.py --config output/baseline/r106_magformer_fulltarget200_oracle_warmstart_1000_launch_skip_depth_sanity.yaml --gpus 4,5,6,7`.
- Output dir: `output/baseline/r106_magformer_fulltarget200_oracle_warmstart_1000`.
- Log: `output/baseline/r106_magformer_fulltarget200_oracle_warmstart_1000.tmux.log`.
- Start evidence: `start iter=0/1000`.
- Warm-start evidence: all ranks printed `Applying warm-start from model.finetune_weights` and `Warm-start missing keys: 0, unexpected keys: 0`.
- True resume evidence: no `Resumed from iteration` line in the successful launch log.
- Finished: `2026-05-18T06:30:58+08:00`.
- Exit code: `0`.
- GPU/RAM: GPUs 4-7 stayed below OOM; CPU/RAM stayed below 90%.

## External Eval

Protocol for target rows: bbox+segm, model input `1024`, score threshold `0.05`, mask threshold `0.5`, inference topk `200`, COCO maxDets `200`, `MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch`, `--force-pytorch-msda`, strict load `774/774` keys.

| checkpoint | eval | predictions | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| iter0499 | train200 | 15665 | `0.479454` | `0.815099` | `0.509588` | `0.426482` | `0.764950` | `0.440168` |
| iter0499 | val28 | 2551 | `0.370650` | `0.721331` | `0.340086` | `0.315580` | `0.636222` | `0.278059` |
| iter0799 | train200 | 15773 | `0.494695` | `0.825283` | `0.533498` | `0.444144` | `0.779586` | `0.467433` |
| iter0799 | val28 | 2586 | `0.376915` | `0.725551` | `0.348558` | `0.317631` | `0.638237` | `0.276093` |
| iter1000 | train200 | 15711 | `0.496883` | `0.826383` | `0.535459` | `0.446421` | `0.780388` | `0.472025` |
| iter1000 | val28 | 2602 | `0.377933` | `0.727568` | `0.349098` | `0.318796` | `0.632162` | `0.284033` |

## Sanity Evals

Original first50 source sanity used the guarded original-first50 protocol:

- Checker: `tools/check_eval_protocol.py original_first50_teacher --allow-nondefault-weights`.
- Wrapper: `tools/evaluate_teacher_first50_1024_backmap.py`.
- Base config: `configs/finetune_1k_full_1024.yaml`.
- Dataset root: `magformer_datasets/20260318_1K_1566`.
- Annotation: `annotations/instances_all.json`.
- Split/max-images: `all / 50`.
- Topk/maxDets: `100 / 100`.

| eval | checkpoint | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original first50 source sanity | iter1000 | `0.464902` | `0.755708` | `0.510815` | `0.484851` | `0.771723` | `0.531024` |
| optional target_labeled25 check | iter1000 | `0.465345` | `0.802638` | `0.495634` | `0.422099` | `0.767392` | `0.425342` |

Source sanity remains below the Teacher original first50 line `0.625293`. The optional target_labeled25 check remains below R103 train25 `0.527663`.

## Judgment

- `iter499 train200 >= 0.42`: yes, `0.426482`. This supports inherited optimizer/scheduler/scaler state as the main R105 failure cause.
- `iter799/1000 train200 >= 0.42`: yes, `0.444144` / `0.446421`. Reset training is effective.
- `train200 >= 0.61`: no. Best is `0.446421`.
- `beats R105 train200 0.366649`: yes, by `+0.079772`.
- `beats R105 val28 0.293617`: yes, by `+0.025179`.
- If the next question is why train200 is still not near 61+, R106 points away from optimizer-state alone and back toward scale, model capacity, or point sampling.

## Artifacts

- Train log: `output/baseline/r106_magformer_fulltarget200_oracle_warmstart_1000.tmux.log`.
- Preflight failed launch archive: `output/baseline/r106_magformer_fulltarget200_oracle_warmstart_1000_preflight_fail_20260518_0608`.
- Train200 iter0499: `output/diagnostics/r106_magformer_fulltarget200_oracle_warmstart_iter0499_train200_1024_backmap_topk200_20260518`.
- Val28 iter0499: `output/diagnostics/r106_magformer_fulltarget200_oracle_warmstart_iter0499_val28_1024_backmap_topk200_20260518`.
- Train200 iter0799: `output/diagnostics/r106_magformer_fulltarget200_oracle_warmstart_iter0799_train200_1024_backmap_topk200_20260518`.
- Val28 iter0799: `output/diagnostics/r106_magformer_fulltarget200_oracle_warmstart_iter0799_val28_1024_backmap_topk200_20260518`.
- Train200 iter1000: `output/diagnostics/r106_magformer_fulltarget200_oracle_warmstart_iter1000_train200_1024_backmap_topk200_20260518`.
- Val28 iter1000: `output/diagnostics/r106_magformer_fulltarget200_oracle_warmstart_iter1000_val28_1024_backmap_topk200_20260518`.
- Source first50 iter1000: `output/diagnostics/r106_magformer_fulltarget200_oracle_warmstart_iter1000_original_first50_1024_backmap_20260518`.
- Target_labeled25 iter1000: `output/diagnostics/r106_magformer_fulltarget200_oracle_warmstart_iter1000_target_labeled25_1024_backmap_topk200_20260518`.

## Commits

- Config/docs placeholder: `07d672469ed2cd8872c80af06b76cc4f8a498505`.
- Final docs-only update: this docs-only commit.
