# VC-SUDA Fast Eval Protocol - 2026-05-14

## Conclusion

Use bbox-only subset eval for daily diagnosis. It should not run segmentation COCO eval and it should not replace the final full bbox+segm eval.

For daily Stage C training checks, use `runtime.eval_iou_types: [bbox]`, `runtime.eval_max_images: 200`, and `runtime.eval_batch_size: 4`. Reserve `segm` and full-data evaluation for final checkpoint evaluation.

Two committed fast-eval configs are available now:

- `configs/eval_full_1566_fast_bbox_200.yaml`
- `configs/eval_full_1566_fast_bbox.yaml` (`eval_max_images: 300`)

## Pseudo-Real Val Smoke

This checks the 28-image pseudo-real validation path with a small subset.

```bash
cd /home/hdd3/zhanghaonan/magformer
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate magformer
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
python tools/evaluate.py \
  --config-file configs/vc_suda_stage_c_r2_1024_teacher8499.yaml \
  --weights output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth \
  --output output/experiments/eval_smoke_stage_b_bbox4 \
  --batch-size 4 \
  --num-workers 2
```

The R2 config sets `runtime.eval_iou_types: [bbox]`, `runtime.eval_max_images: 28`, and `runtime.eval_batch_size: 4`. The main Stage C config now uses bbox-only 200-image quick eval every 1000 iterations. For an even smaller local smoke, temporarily pass a copied config with `runtime.eval_max_images: 4`; do not edit committed configs just for a smoke.

## Original 1.5K Fast Eval

Use this for Teacher or daily diagnosis on the original 1.5K data. The committed configs limit eval to 200 or 300 images, bbox-only.

CLI eval is the cleanest daily path. Pick a single visible GPU with `CUDA_VISIBLE_DEVICES`. `tools/evaluate.py` now uses `runtime.eval_batch_size` from the config when `--batch-size` is omitted, and `--batch-size` remains an explicit override.

### 200 images, batch size 4

```bash
cd /home/hdd3/zhanghaonan/magformer
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate magformer
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
python tools/evaluate.py \
  --config-file configs/eval_full_1566_fast_bbox_200.yaml \
  --weights output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth \
  --output output/experiments/eval_full_1566_fast_bbox_200_bs4 \
  --num-workers 4
```

### 300 images, batch size 8

```bash
cd /home/hdd3/zhanghaonan/magformer
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate magformer
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
python tools/evaluate.py \
  --config-file configs/eval_full_1566_fast_bbox.yaml \
  --weights output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth \
  --output output/experiments/eval_full_1566_fast_bbox_300_bs8 \
  --batch-size 8 \
  --num-workers 4
```

To run on a different single card, keep the same command and only change `CUDA_VISIBLE_DEVICES`, for example `export CUDA_VISIBLE_DEVICES=1`.

### Training built-in DDP eval

Built-in eval now applies `runtime.eval_max_images` as a dataset-level global subset before `DistributedSampler`. Set these keys in the training config before launching DDP:

```yaml
runtime:
  eval_iou_types: ["bbox"]
  eval_max_images: 200
  eval_batch_size: 4
```

On 4 GPUs, 200 divides evenly across ranks, so `DistributedSampler` does not need padding duplicates for this fast-eval size.

```bash
cd /home/hdd3/zhanghaonan/magformer
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate magformer
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --standalone --nproc_per_node=4 tools/train.py \
  --config configs/vc_suda_stage_b_1024_teacher8499.yaml \
  --output-dir output/experiments/vc_suda_stage_b_1024_teacher8499_fast_eval \
  --gpus 0,1,2,3 \
  --num-workers 2
```

Expected contract:

- Output includes `bbox_AP` metrics only.
- Output does not include `segm_AP` metrics.
- `coco_instances_results.json` does not include `segmentation` fields for bbox-only eval.
- `val/diag_num_eval_images` should be the number of unique evaluated image IDs, not the number of batches.
- Subset eval passes the actual evaluated `image_ids` into `COCOeval.params.imgIds`, so missing non-evaluated images do not create false low AP.
- For DDP subset sizes that do not divide evenly by world size, `DistributedSampler` can still pad duplicate samples. We do not silently collapse flat COCO rows by `image_id`, because that can also drop valid multi-instance predictions for a real image. Prefer CLI single-process eval or a subset size divisible by world size when you need to avoid that padding path.

## 1024 Backmap Teacher 1.5K Reproduction

`tools/evaluate_1024_backmap.py` defaults to the VC-SUDA pseudo-real smoke path. It does not reproduce Teacher 8499 on the original 1.5K all split unless the original 1.5K data arguments are passed explicitly. Use this shape for that run:

```bash
python tools/evaluate_1024_backmap.py \
  --iou-types bbox,segm \
  --base-config configs/finetune_1k_full_1024.yaml \
  --dataset-root /home/hdd3/zhanghaonan/magformer/magformer_datasets/20260318_1K_1566 \
  --ann annotations/instances_all.json \
  --split all \
  --weights output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth \
  --output-dir output/experiments/eval_1024_backmap_teacher8499_full1566
```

The backmap script writes `coco_instances_results.json` through `COCOEvaluator.dump()`. With `--iou-types bbox`, exported rows omit `segmentation`. With `--iou-types bbox,segm`, exported rows include COCO xywh `bbox` and `segmentation` for final full eval.

## Final Eval Boundary

Final checkpoint selection still needs full bbox+segm eval. Use the formal eval config and do not use `configs/eval_full_1566_fast_bbox.yaml` for final segmentation claims.
