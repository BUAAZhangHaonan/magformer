# VC-SUDA Stage C Preflight - 2026-05-14

## Scope

- Host: WS-4029GP-TRT via `ssh 4029`
- Repository: `/home/hdd3/zhanghaonan/magformer`
- Branch: `feature/vc-suda-sim2real`
- Conda env: `magformer`
- Config: `configs/vc_suda_stage_c_1024_teacher8499.yaml`
- This was a preflight only. No formal Stage C training was launched.

## Gate Summary

- Stage C config exists and loads through the normal Pydantic config path.
- `vc_suda.stage: C` and `vc_suda.ema_teacher.enabled: true`.
- `vc_suda.target_unlabeled_ann: annotations/instances_target_unlabeled.json`.
- Effective target-unlabeled split is `train`.
- `vc_suda.unsupervised_weight: 0.1` with `unsupervised_warmup_epochs: 10`.
- `runtime.eval_iou_types: [bbox, segm]`.
- `runtime.resume: null`.
- `model.finetune_weights` is a placeholder for the Stage B final model checkpoint: `output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/model_final.pth`.
- Depth normalization keeps the Stage B values: `norm: minmax`, `per_sample_norm: true`, `clip_min: 0.0`, `clip_max: 2.095623016357422`.
- `runtime.skip_depth_sanity: false` for Stage C.

## Validation Commands

Static preflight, allowing the Stage B final checkpoint placeholder because Stage B is still running:

```bash
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate magformer
python tools/verify_vc_suda_stage.py \
  --config configs/vc_suda_stage_c_1024_teacher8499.yaml \
  --skip-batch \
  --allow-missing-finetune
```

Result:

```text
PASS config=configs/vc_suda_stage_c_1024_teacher8499.yaml
checks=stage,unlabeled_split,eval_iou_types,unsupervised_weight,ema_teacher,depth_norm,target_unlabeled_val_overlap,checkpoint_semantics
WARNING model.finetune_weights does not exist yet; allowed because Stage B final checkpoint is pending.
details={"finetune_weights": "/home/hdd3/zhanghaonan/magformer/output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/model_final.pth", "target_unlabeled_images": 200, "target_unlabeled_split": "train", "unsupervised_weight": 0.1, "val_images": 28}
```

One-batch CPU/data preflight:

```bash
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate magformer
python tools/verify_vc_suda_stage.py \
  --config configs/vc_suda_stage_c_1024_teacher8499.yaml \
  --allow-missing-finetune
```

Result:

```text
PASS config=configs/vc_suda_stage_c_1024_teacher8499.yaml
checks=stage,unlabeled_split,eval_iou_types,unsupervised_weight,ema_teacher,depth_norm,target_unlabeled_val_overlap,checkpoint_semantics,unlabeled_batch_strip,depth_nonconstant
WARNING model.finetune_weights does not exist yet; allowed because Stage B final checkpoint is pending.
details={"finetune_weights": "/home/hdd3/zhanghaonan/magformer/output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/model_final.pth", "target_strong_depth": {"max": 1.0467190742492676, "min": -0.04866107553243637, "shape": [1, 1, 1024, 1024], "std": 0.4546217620372772, "unique": 970057}, "target_unlabeled_images": 200, "target_unlabeled_split": "train", "target_weak_depth": {"max": 1.0505133867263794, "min": -0.049213748425245285, "shape": [1, 1, 1024, 1024], "std": 0.4546493589878082, "unique": 969998}, "unsupervised_weight": 0.1, "val_images": 28}
```

Targeted pytest coverage:

```bash
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate magformer
python -m pytest \
  tests/test_vc_suda_stage_c_preflight.py \
  tests/test_vc_suda_data_protocol.py \
  tests/test_vc_suda_train_entrypoint.py \
  -q
```

Result: passed. Existing Pydantic deprecation warnings for legacy `dpe_enabled` and `dpe_beta` were still present.

## Gate Status

- Config gate: pass.
- Static preflight: pass with one expected warning because the Stage B final checkpoint does not exist yet.
- One-batch unlabeled data gate: pass.
- Formal Stage C launch gate: blocked until Stage B produces the final `model_final.pth` path used by `model.finetune_weights`.
