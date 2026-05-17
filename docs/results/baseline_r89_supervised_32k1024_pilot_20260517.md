# R89 supervised-only 32K 1024 cache pilot

This is a minimal supervised-only baseline pilot for R89. It is not a final convergence run.

## Scope

- Config: `configs/baseline_supervised_r89_32k1024_cache_pilot.yaml`
- Data: 32K synthetic source at `magformer_datasets/20260318_1K_32254`
- Train annotations: `cache/coco_loader/instances_train.sqlite`
- Resolution: 1024
- GPUs: 4, 5, 6, 7
- Pilot length: `solver.max_iter: 20`
- Disabled: VC-SUDA, EMA, pseudo labels, offline pseudo bank, source retention, contrastive loss

## Validation commands

Static config load and protocol check:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python - <<'PY'
from magformer.config import load_config
from tools.train import is_vc_suda_enabled, validate_vc_suda_config
cfg = load_config("configs/baseline_supervised_r89_32k1024_cache_pilot.yaml")
validate_vc_suda_config(cfg)
assert not is_vc_suda_enabled(cfg)
assert cfg.data.train_ann.endswith(".sqlite")
assert cfg.solver.max_iter == 20
assert cfg.runtime.gpus == [4, 5, 6, 7]
PY
```

Loader smoke:

```bash
timeout 180s env CUDA_VISIBLE_DEVICES=4,5,6,7 /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python - <<'PY'
from torch.utils.data import DataLoader
from magformer.config import load_config
from magformer.data import CocoRgbdDataset
from magformer.data.transforms import RGBDTransform
from magformer.data.collate import collate_fn
cfg = load_config("configs/baseline_supervised_r89_32k1024_cache_pilot.yaml")
dataset = CocoRgbdDataset(
    dataset_root=cfg.data.dataset_root,
    ann_file=cfg.data.train_ann,
    split=cfg.data.train_split,
    transform=None,
    is_train=True,
)
dataset.transform = RGBDTransform(
    image_size=cfg.data.image_size,
    min_scale=cfg.data.min_scale,
    max_scale=cfg.data.max_scale,
    random_flip="none",
    rgb_brightness=0.0,
    rgb_contrast=0.0,
    rgb_saturation=0.0,
    rgb_hue=0.0,
    depth_scale=cfg.data.depth.scale,
    depth_shift=cfg.data.depth.shift,
    depth_clip_min=cfg.data.depth.clip_min,
    depth_clip_max=cfg.data.depth.clip_max,
    depth_norm=cfg.data.depth.norm,
    depth_per_sample_norm=cfg.data.depth.per_sample_norm,
    depth_gaussian_std=0.0,
    is_train=True,
)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
batch = next(iter(loader))
assert len(dataset) == 25654
assert batch["images"].shape[-2:] == (1024, 1024)
assert batch["depths"].shape[-2:] == (1024, 1024)
assert "source_images" not in batch
PY
```

Optional bounded training smoke, if needed later:

```bash
tmux new-session -d -s r89_supervised_smoke 'cd /home/hdd3/zhanghaonan/magformer && CUDA_VISIBLE_DEVICES=4,5,6,7 /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/torchrun --nproc_per_node=4 tools/train.py --config configs/baseline_supervised_r89_32k1024_cache_pilot.yaml'
```

The config is capped at 20 iterations. `eval_period` and `checkpoint_period` are set to 21, so the pilot does not run validation or write numbered checkpoints during the capped run.

## Stop standard

Stop after the loader smoke passes or after the bounded pilot reaches 20 iterations. Do not extend this config into a convergence run.

A long supervised baseline should be a separate R90 run and a separate commit. R90 should choose its own max iteration, evaluation cadence, checkpoint policy, and reporting target instead of reusing this smoke pilot as a training recipe.
