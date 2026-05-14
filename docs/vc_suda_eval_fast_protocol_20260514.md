# VC-SUDA Fast Eval Protocol - 2026-05-14

## Conclusion

Use bbox-only subset eval for daily diagnosis. It should not run segmentation COCO eval and it should not replace the final full bbox+segm eval.

For daily checks, use `runtime.eval_iou_types: [bbox]`, `runtime.eval_max_images: 200` or `300`, and `runtime.eval_batch_size: 4` or `8` when memory allows. Reserve `segm` for final checkpoint evaluation.

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

The config sets `runtime.eval_iou_types: [bbox]`, `runtime.eval_max_images: 28`, and `runtime.eval_batch_size: 4`. For an even smaller local smoke, temporarily pass a copied config with `runtime.eval_max_images: 4`; do not edit the committed R2 config just for a smoke.

## Original 1.5K Fast Eval

Use this for Teacher or daily diagnosis on the original 1.5K data. The config limits eval to 300 images, bbox-only.

```bash
cd /home/hdd3/zhanghaonan/magformer
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate magformer
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
python tools/evaluate.py \
  --config-file configs/eval_full_1566_fast_bbox.yaml \
  --weights output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth \
  --output output/experiments/eval_full_1566_fast_bbox_teacher8499 \
  --batch-size 4 \
  --num-workers 4
```

Expected contract:

- Output includes `bbox_AP` metrics only.
- Output does not include `segm_AP` metrics.
- `coco_instances_results.json` does not include `segmentation` fields for bbox-only eval.
- `val/diag_num_eval_images` should be the number of unique evaluated image IDs, not the number of batches.
- Subset eval passes the actual evaluated `image_ids` into `COCOeval.params.imgIds`, so missing non-evaluated images do not create false low AP.

## Final Eval Boundary

Final checkpoint selection still needs full bbox+segm eval. Use the formal eval config and do not use `configs/eval_full_1566_fast_bbox.yaml` for final segmentation claims.
