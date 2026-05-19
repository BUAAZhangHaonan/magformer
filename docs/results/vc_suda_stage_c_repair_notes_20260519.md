# VC-SUDA Stage C Repair Notes - 2026-05-19

## R142 source split

R142 uses `magformer_datasets/20260318_1K_32254/cache/coco_loader/instances_train.sqlite`.
That cache is the 25,654-image source train split from the 32,254 total source dataset.
The run name and output directory use `32254_train25654` so the split is explicit.

## Historical invalid configs

The R118/R121/R122 historical configs are retained only as fail-fast negative examples.
They collapse `source_ann` into `target_labeled_ann` and set `target_labeled_weight=0`.
`tools/start_vc_suda_watchers.sh` no longer starts those historical configs.
