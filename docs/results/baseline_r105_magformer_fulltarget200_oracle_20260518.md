# R105 MagFormer Full-Target200 Oracle

Date: 2026-05-18
Branch: feature/vc-suda-sim2real
Config: `configs/baseline_supervised_r105_magformer_fulltarget200_oracle_1000.yaml`

## Conclusion

R105 does not reach the `61+` train200 oracle gate. The best train200 checkpoint is iter1000 with segm AP `0.366649`, far below `0.61` and still below `0.50`.

Held-out val28 does improve: best val28 segm AP is iter1000 `0.293617`, above R104 val28 `0.283886` by `+0.009731`.

The result points to an optimization / scale bottleneck under this MagFormer/RGB-D cleanfit resume recipe, not a pure label-coverage ceiling. Full hidden-GT target supervision gives a small val28 gain, but it cannot fit train200 anywhere near the known clean target25 overfit line.

R105 is an oracle upper-bound diagnosis. It directly uses hidden GT from `target_unlabeled200` for supervised training. Do not report it as a formal UDA result or paper metric.

## Static Validation

- `validate_config(strict=True)`: valid. Only the existing DPE informational note was printed.
- Train annotation: `annotations/instances_target_unlabeled.json`.
- Val annotation: `annotations/instances_val.json`, split `val`.
- Resume: `output/baseline/r100_magformer_r98ckpt_target_labeled25_cleanfit_1024/checkpoint_iter_0000500.pth`.
- Resume checkpoint evidence: `iter=500`, `774` model state keys.
- Train200: `200` images / `11750` annotations / `0` empty images.
- Val28: `28` images / `1892` annotations / `0` empty images.
- Loader smoke: train/val dataset lengths `200 / 28`; first train batch images `[4, 3, 1024, 1024]`, depths `[4, 1, 1024, 1024]`, target mask counts `[50, 50, 98, 50]`.
- Disabled paths checked: VC-SUDA, runtime EMA, VC-SUDA EMA teacher, offline pseudo, source retention, contrastive, RGB photo aug, depth noise.

## Training

- tmux session: `r105_magformer_fulltarget200_oracle`
- Command shape: `/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/torchrun --nproc_per_node=4 tools/train.py --config configs/baseline_supervised_r105_magformer_fulltarget200_oracle_1000.yaml --gpus 4,5,6,7`
- Output dir: `output/baseline/r105_magformer_fulltarget200_oracle_1000`
- Log: `output/baseline/r105_magformer_fulltarget200_oracle_1000.tmux.log`
- Resume evidence: all ranks loaded the R100 checkpoint and reported `Resumed from iteration 500`.
- Started: `2026-05-18 05:32:45 +0800`.
- Training completed: `2026-05-18 05:44:51 +0800`.
- Exit code: `0`.
- Checkpoints used for external eval: `checkpoint_iter_0000599.pth`, `checkpoint_iter_0000799.pth`, `checkpoint_iter_0001000.pth`.
- GPU/RAM: GPU 4-7 stayed below OOM; CPU/RAM stayed below 90%.

Built-in trainer eval is diagnostic only. The decision metrics below use external 1024 backmap.

## External Eval

Protocol for target rows: bbox+segm, model input `1024`, score threshold `0.05`, mask threshold `0.5`, inference topk `200`, COCO maxDets `200`, `MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch`, `--force-pytorch-msda`, strict load `774/774` keys.

| checkpoint | eval | predictions | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| iter0599 | train200 | 19518 | `0.415522` | `0.774806` | `0.397941` | `0.345247` | `0.678987` | `0.316866` |
| iter0599 | val28 | 3097 | `0.350395` | `0.714759` | `0.296926` | `0.284032` | `0.606515` | `0.232852` |
| iter0799 | train200 | 17731 | `0.432747` | `0.784826` | `0.431115` | `0.364164` | `0.703659` | `0.341519` |
| iter0799 | val28 | 2804 | `0.359384` | `0.730155` | `0.320569` | `0.292934` | `0.618905` | `0.246614` |
| iter1000 | train200 | 17263 | `0.434572` | `0.785899` | `0.433105` | `0.366649` | `0.704939` | `0.349413` |
| iter1000 | val28 | 2730 | `0.358867` | `0.723538` | `0.317621` | `0.293617` | `0.619907` | `0.244717` |

Best checkpoint by train200 and val28 is iter1000.

## Sanity Evals

Original first50 source sanity used the guarded original-first50 protocol:

- Checker: `tools/check_eval_protocol.py original_first50_teacher --allow-nondefault-weights`
- Wrapper: `tools/evaluate_teacher_first50_1024_backmap.py`
- Base config: `configs/finetune_1k_full_1024.yaml`
- Dataset root: `magformer_datasets/20260318_1K_1566`
- Annotation: `annotations/instances_all.json`
- Split/max-images: `all / 50`
- Topk/maxDets: `100 / 100`

| eval | checkpoint | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original first50 source sanity | iter1000 | `0.460450` | `0.752204` | `0.493870` | `0.498462` | `0.778922` | `0.542730` |
| optional target_labeled25 check | iter1000 | `0.489797` | `0.816002` | `0.526436` | `0.439798` | `0.778930` | `0.458687` |

Source sanity is below the Teacher original first50 line `0.625293`, so R105 does not preserve the source-domain teacher level. The optional target_labeled25 check also drops below R103 train25 `0.527663`.

## Judgment

- `train200 >= 0.61`: no. Best is `0.366649`.
- `train200 >= 0.50`: no. Best is still below `0.50`.
- `val28 improves over R104 0.283886`: yes. Best is `0.293617`.
- `label coverage / UDA bottleneck`: partially supported only for val28, because hidden-GT full target supervision improves val28 by about one AP point.
- `optimization / scale bottleneck`: primary diagnosis. Even direct full-target GT supervision does not fit train200 beyond `0.366649`.
- `overfitting`: not the main observed failure at 1000 iter, because held-out val28 improves with train200; the issue is low fit capacity/optimization under this recipe, not high train AP with bad val AP.

Do not extend this to 2000 in R105 without a new explicit task. Iter1000 is far from `0.61`, so this is not the "near 61 but undertrained" case.

## Artifacts

- Train log: `output/baseline/r105_magformer_fulltarget200_oracle_1000.tmux.log`
- Train200 iter0599: `output/diagnostics/r105_magformer_fulltarget200_oracle_iter0599_train200_1024_backmap_topk200_20260518`
- Val28 iter0599: `output/diagnostics/r105_magformer_fulltarget200_oracle_iter0599_val28_1024_backmap_topk200_20260518`
- Train200 iter0799: `output/diagnostics/r105_magformer_fulltarget200_oracle_iter0799_train200_1024_backmap_topk200_20260518`
- Val28 iter0799: `output/diagnostics/r105_magformer_fulltarget200_oracle_iter0799_val28_1024_backmap_topk200_20260518`
- Train200 iter1000: `output/diagnostics/r105_magformer_fulltarget200_oracle_iter1000_train200_1024_backmap_topk200_20260518`
- Val28 iter1000: `output/diagnostics/r105_magformer_fulltarget200_oracle_iter1000_val28_1024_backmap_topk200_20260518`
- Source first50 iter1000: `output/diagnostics/r105_magformer_fulltarget200_oracle_iter1000_original_first50_1024_backmap_20260518`
- Target_labeled25 iter1000: `output/diagnostics/r105_magformer_fulltarget200_oracle_iter1000_target_labeled25_1024_backmap_topk200_20260518`

## Commits

- Config/docs placeholder: `46c05ab0`
- Final docs-only update: this docs-only commit.
