# R107 MagFormer Full-Target200 Oracle Random Points

Date: 2026-05-18
Branch: feature/vc-suda-sim2real
Config: `configs/baseline_supervised_r107_magformer_fulltarget200_oracle_randompoints_1000.yaml`

## Purpose

R107 tests one variable against R106 full-target200 oracle warm-start: `model.magformer.mask_former.importance_sample_ratio` changes from `0.75` to `0.0`.

The hypothesis is that random point sampling gives more foreground mask-loss points on the small full-target oracle set. If train-fit improves, the R106 point sampler was likely a bottleneck.

## Fixed Controls

- `train_num_points: 12544`.
- `oversample_ratio: 3.0`.
- `balanced_ce: true`.
- `balanced_ce_min_fg_ratio: 0.05`.
- Warm-start checkpoint, solver, data, and depth-sanity settings match R106.
- `runtime.skip_depth_sanity: false`.
- `runtime.depth_sanity.min_mask_fg_ratio: 0.0009`.
- VC-SUDA, EMA, pseudo bank, source replay, contrastive, and postprocess modules stay disabled.

## Gates

- Early stop failure: if iter499 train200 segm AP is below R106 iter499 `0.426482`.
- Success: iter1000 train200 segm AP must be at least `0.4565`.
- Train-fit target check: record whether train200 reaches `0.61+`.

## Static Validation

- Unique intended config variable versus R106: `model.magformer.mask_former.importance_sample_ratio: 0.75 -> 0.0`.
- Other config changes are only run naming/output paths.
- `runtime.resume: null`.
- `model.finetune_weights: output/baseline/r100_magformer_r98ckpt_target_labeled25_cleanfit_1024/checkpoint_iter_0000500.pth`.
- `train_num_points: 12544`, `oversample_ratio: 3.0`, `balanced_ce: true`, `balanced_ce_min_fg_ratio: 0.05`.
- `runtime.skip_depth_sanity: false`, `runtime.depth_sanity.min_mask_fg_ratio: 0.0009`.
- `validate_config(strict=True)`: valid. Only the existing DPE informational note was printed.
- Loader smoke: train/val dataset lengths `200 / 28`; first train batch images `[1, 3, 1024, 1024]`, depths `[1, 1, 1024, 1024]`; first val batch images `[4, 3, 512, 512]`.
- Disabled paths checked in config: VC-SUDA, runtime EMA, contrastive.

## Training

- tmux session: `r107_magformer_fulltarget200_oracle_randompoints`.
- Command shape: `/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/torchrun --nproc_per_node=4 tools/train.py --config configs/baseline_supervised_r107_magformer_fulltarget200_oracle_randompoints_1000.yaml --gpus 4,5,6,7`.
- Output dir: `output/baseline/r107_magformer_fulltarget200_oracle_randompoints_1000`.
- Log: `output/baseline/r107_magformer_fulltarget200_oracle_randompoints_1000.tmux.log`.
- Warm-start evidence: all ranks printed `Warm-start missing keys: 0, unexpected keys: 0`.
- Depth sanity evidence: `depth_sanity.json` was written and training continued.
- Start evidence: `start iter=0/1000`.
- Finished: `2026-05-18T07:15:38+08:00`.
- Exit state: tmux session exited; no OOM; GPUs 4-7 released.

## External Eval

Protocol for target rows: bbox+segm, model input `1024`, score threshold `0.05`, mask threshold `0.5`, inference topk `200`, COCO maxDets `200`, `MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch`, `--force-pytorch-msda`, strict load `774/774` keys.

| checkpoint | eval | predictions | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| iter0499 | train200 | 16540 | `0.463587` | `0.810875` | `0.483656` | `0.431359` | `0.764830` | `0.450030` |
| iter0499 | val28 | 2707 | `0.355406` | `0.712537` | `0.327299` | `0.316549` | `0.636277` | `0.283629` |
| iter0799 | train200 | 15772 | `0.483337` | `0.815157` | `0.519365` | `0.447053` | `0.779151` | `0.475640` |
| iter0799 | val28 | 2617 | `0.366151` | `0.718160` | `0.342733` | `0.320550` | `0.638155` | `0.284172` |
| iter1000 | train200 | 15885 | `0.483200` | `0.815235` | `0.519622` | `0.449679` | `0.779353` | `0.476159` |
| iter1000 | val28 | 2610 | `0.364906` | `0.717758` | `0.335791` | `0.321195` | `0.638089` | `0.284199` |

## Sanity Evals

Original first50 source sanity used the guarded original-first50 protocol:

- Checker: `tools/check_eval_protocol.py original_first50_teacher --allow-nondefault-weights`.
- Wrapper: `tools/evaluate_teacher_first50_1024_backmap.py`.
- Base config: `configs/finetune_1k_full_1024.yaml`.
- Dataset root: `magformer_datasets/20260318_1K_1566`.
- Annotation: `annotations/instances_all.json`.
- Split/max-images: `all / 50`.
- Topk/maxDets: `100 / 100`.

| eval | checkpoint | predictions | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original first50 source sanity | iter1000 | 3999 | `0.428962` | `0.733760` | `0.457476` | `0.495829` | `0.779668` | `0.545457` |
| optional target_labeled25 check | iter1000 | 2282 | `0.451798` | `0.790664` | `0.462424` | `0.428749` | `0.776710` | `0.428407` |

## Judgment

- Early stop gate: pass. Iter0499 train200 segm AP is `0.431359`, above R106 iter0499 `0.426482` by `+0.004877`.
- R107 success gate: fail. Iter1000 train200 segm AP is `0.449679`, below the required `0.4565` by `-0.006821`.
- Versus R106 iter1000 train200: R107 is higher by `+0.003258` (`0.449679` vs `0.446421`).
- Versus R106 iter1000 val28: R107 is higher by `+0.002399` (`0.321195` vs `0.318796`).
- Train200 `0.61+`: no. Best train200 is `0.449679`.
- Interpretation: random-only point sampling gives a small positive gain, but not enough to support point sampling as the main train-fit bottleneck under the R107 success rule.

## Artifacts

- Train log: `output/baseline/r107_magformer_fulltarget200_oracle_randompoints_1000.tmux.log`.
- Train200 iter0499: `output/diagnostics/r107_magformer_fulltarget200_oracle_randompoints_iter0499_train200_1024_backmap_topk200_20260518`.
- Val28 iter0499: `output/diagnostics/r107_magformer_fulltarget200_oracle_randompoints_iter0499_val28_1024_backmap_topk200_20260518`.
- Train200 iter0799: `output/diagnostics/r107_magformer_fulltarget200_oracle_randompoints_iter0799_train200_1024_backmap_topk200_20260518`.
- Val28 iter0799: `output/diagnostics/r107_magformer_fulltarget200_oracle_randompoints_iter0799_val28_1024_backmap_topk200_20260518`.
- Train200 iter1000: `output/diagnostics/r107_magformer_fulltarget200_oracle_randompoints_iter1000_train200_1024_backmap_topk200_20260518`.
- Val28 iter1000: `output/diagnostics/r107_magformer_fulltarget200_oracle_randompoints_iter1000_val28_1024_backmap_topk200_20260518`.
- Source first50 iter1000: `output/diagnostics/r107_magformer_fulltarget200_oracle_randompoints_iter1000_original_first50_1024_backmap_20260518`.
- Target_labeled25 iter1000: `output/diagnostics/r107_magformer_fulltarget200_oracle_randompoints_iter1000_target_labeled25_1024_backmap_topk200_20260518`.

## Commits

- Config/docs placeholder: `258248866c3000f51d59fd6d90c5ca1d9ec772c6`.
- Final docs-only update: this docs-only commit.
