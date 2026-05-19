# Checkpoint Cleanup Manifest - 2026-05-19

This is a dry-run manifest. It does not delete files.

## Summary

- Checkpoints scanned: 502
- Kept checkpoints: 77
- Delete candidates: 425
- Estimated reclaim: 233.9 GiB (251164906065 bytes)
- First batch recommendation: only R139/R141 intermediate checkpoints after manual review.
- First batch R139/R141 reclaim: 109.5 GiB (117615522442 bytes)

## Safety Rules

- The planner never deletes files.
- It only lists checkpoint files, not run directories.
- Logs, metrics, configs, and JSON files are never listed as delete candidates.
- Checkpoints referenced from docs/results, configs, scripts, or tools are kept.
- Final/gate and warm-start checkpoints from the manual keep rules are kept.

## Delete Candidates

| Path | Size | Reason |
| --- | ---: | --- |
| `output/baseline/r100_magformer_r98ckpt_target_labeled25_cleanfit_1024/checkpoint_iter_0000499.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r101_magformer_cleanfit_r99solver_1024/checkpoint_iter_0000499.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r102_magformer_augnoise_strongsolver_1024/checkpoint_iter_0000499.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r103_magformer_r100_resume_target_labeled25_1000/checkpoint_iter_0000599.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r103_magformer_r100_resume_target_labeled25_1000/checkpoint_iter_0000699.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r103_magformer_r100_resume_target_labeled25_1000/checkpoint_iter_0000799.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r103_magformer_r100_resume_target_labeled25_1000/checkpoint_iter_0000899.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r103_magformer_r100_resume_target_labeled25_1000/checkpoint_iter_0000999.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r104_magformer_r100_resume_target_labeled50_balanced_1000/checkpoint_iter_0000599.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r104_magformer_r100_resume_target_labeled50_balanced_1000/checkpoint_iter_0000699.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r104_magformer_r100_resume_target_labeled50_balanced_1000/checkpoint_iter_0000799.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r104_magformer_r100_resume_target_labeled50_balanced_1000/checkpoint_iter_0000899.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r104_magformer_r100_resume_target_labeled50_balanced_1000/checkpoint_iter_0000999.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r105_magformer_fulltarget200_oracle_1000/checkpoint_iter_0000599.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r105_magformer_fulltarget200_oracle_1000/checkpoint_iter_0000699.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r105_magformer_fulltarget200_oracle_1000/checkpoint_iter_0000799.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r105_magformer_fulltarget200_oracle_1000/checkpoint_iter_0000899.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r105_magformer_fulltarget200_oracle_1000/checkpoint_iter_0000999.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r105_magformer_fulltarget200_oracle_1000/checkpoint_iter_0001000.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r106_magformer_fulltarget200_oracle_warmstart_1000/checkpoint_iter_0000099.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r106_magformer_fulltarget200_oracle_warmstart_1000/checkpoint_iter_0000199.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r106_magformer_fulltarget200_oracle_warmstart_1000/checkpoint_iter_0000299.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r106_magformer_fulltarget200_oracle_warmstart_1000/checkpoint_iter_0000399.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r106_magformer_fulltarget200_oracle_warmstart_1000/checkpoint_iter_0000499.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r106_magformer_fulltarget200_oracle_warmstart_1000/checkpoint_iter_0000599.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r106_magformer_fulltarget200_oracle_warmstart_1000/checkpoint_iter_0000699.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r106_magformer_fulltarget200_oracle_warmstart_1000/checkpoint_iter_0000799.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r106_magformer_fulltarget200_oracle_warmstart_1000/checkpoint_iter_0000899.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r106_magformer_fulltarget200_oracle_warmstart_1000/checkpoint_iter_0000999.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r106_magformer_fulltarget200_oracle_warmstart_1000/checkpoint_iter_0001000.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r107_magformer_fulltarget200_oracle_randompoints_1000/checkpoint_iter_0000099.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r107_magformer_fulltarget200_oracle_randompoints_1000/checkpoint_iter_0000199.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r107_magformer_fulltarget200_oracle_randompoints_1000/checkpoint_iter_0000299.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r107_magformer_fulltarget200_oracle_randompoints_1000/checkpoint_iter_0000399.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r107_magformer_fulltarget200_oracle_randompoints_1000/checkpoint_iter_0000499.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r107_magformer_fulltarget200_oracle_randompoints_1000/checkpoint_iter_0000599.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r107_magformer_fulltarget200_oracle_randompoints_1000/checkpoint_iter_0000699.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r107_magformer_fulltarget200_oracle_randompoints_1000/checkpoint_iter_0000799.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r107_magformer_fulltarget200_oracle_randompoints_1000/checkpoint_iter_0000899.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r107_magformer_fulltarget200_oracle_randompoints_1000/checkpoint_iter_0000999.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r107_magformer_fulltarget200_oracle_randompoints_1000/checkpoint_iter_0001000.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r108_magformer_fulltarget200_oracle_dice75_1000/checkpoint_iter_0000099.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r108_magformer_fulltarget200_oracle_dice75_1000/checkpoint_iter_0000199.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r108_magformer_fulltarget200_oracle_dice75_1000/checkpoint_iter_0000299.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r108_magformer_fulltarget200_oracle_dice75_1000/checkpoint_iter_0000399.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r108_magformer_fulltarget200_oracle_dice75_1000/checkpoint_iter_0000499.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r108_magformer_fulltarget200_oracle_dice75_1000/checkpoint_iter_0000599.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r108_magformer_fulltarget200_oracle_dice75_1000/checkpoint_iter_0000699.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r108_magformer_fulltarget200_oracle_dice75_1000/checkpoint_iter_0000799.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r108_magformer_fulltarget200_oracle_dice75_1000/checkpoint_iter_0000899.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r108_magformer_fulltarget200_oracle_dice75_1000/checkpoint_iter_0000999.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r108_magformer_fulltarget200_oracle_dice75_1000/checkpoint_iter_0001000.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r109_magformer_fulltarget200_oracle_dice100_1000/checkpoint_iter_0000099.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r109_magformer_fulltarget200_oracle_dice100_1000/checkpoint_iter_0000199.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r109_magformer_fulltarget200_oracle_dice100_1000/checkpoint_iter_0000299.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r109_magformer_fulltarget200_oracle_dice100_1000/checkpoint_iter_0000399.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r109_magformer_fulltarget200_oracle_dice100_1000/checkpoint_iter_0000499.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r109_magformer_fulltarget200_oracle_dice100_1000/checkpoint_iter_0000599.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r109_magformer_fulltarget200_oracle_dice100_1000/checkpoint_iter_0000699.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r109_magformer_fulltarget200_oracle_dice100_1000/checkpoint_iter_0000799.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r109_magformer_fulltarget200_oracle_dice100_1000/checkpoint_iter_0000899.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r109_magformer_fulltarget200_oracle_dice100_1000/checkpoint_iter_0000999.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r109_magformer_fulltarget200_oracle_dice100_1000/checkpoint_iter_0001000.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r110_magformer_fulltarget200_oracle_dice125_1000/checkpoint_iter_0000099.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r110_magformer_fulltarget200_oracle_dice125_1000/checkpoint_iter_0000199.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r110_magformer_fulltarget200_oracle_dice125_1000/checkpoint_iter_0000299.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r110_magformer_fulltarget200_oracle_dice125_1000/checkpoint_iter_0000399.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r110_magformer_fulltarget200_oracle_dice125_1000/checkpoint_iter_0000499.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r110_magformer_fulltarget200_oracle_dice125_1000/checkpoint_iter_0000599.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r110_magformer_fulltarget200_oracle_dice125_1000/checkpoint_iter_0000699.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r111_magformer_r109recipe_target_labeled50_balanced_1000/checkpoint_iter_0000099.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r111_magformer_r109recipe_target_labeled50_balanced_1000/checkpoint_iter_0000199.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r111_magformer_r109recipe_target_labeled50_balanced_1000/checkpoint_iter_0000299.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r111_magformer_r109recipe_target_labeled50_balanced_1000/checkpoint_iter_0000399.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r111_magformer_r109recipe_target_labeled50_balanced_1000/checkpoint_iter_0000599.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r111_magformer_r109recipe_target_labeled50_balanced_1000/checkpoint_iter_0000699.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r111_magformer_r109recipe_target_labeled50_balanced_1000/checkpoint_iter_0000899.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r111_magformer_r109recipe_target_labeled50_balanced_1000/checkpoint_iter_0000999.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r112_magformer_r111recipe_target_labeled100_balanced_1000/checkpoint_iter_0000099.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r112_magformer_r111recipe_target_labeled100_balanced_1000/checkpoint_iter_0000199.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r112_magformer_r111recipe_target_labeled100_balanced_1000/checkpoint_iter_0000299.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r112_magformer_r111recipe_target_labeled100_balanced_1000/checkpoint_iter_0000399.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r112_magformer_r111recipe_target_labeled100_balanced_1000/checkpoint_iter_0000599.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r112_magformer_r111recipe_target_labeled100_balanced_1000/checkpoint_iter_0000699.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r112_magformer_r111recipe_target_labeled100_balanced_1000/checkpoint_iter_0000899.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r112_magformer_r111recipe_target_labeled100_balanced_1000/checkpoint_iter_0000999.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r113_magformer_r112_target100_resume2000/checkpoint_iter_0001099.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r113_magformer_r112_target100_resume2000/checkpoint_iter_0001199.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r113_magformer_r112_target100_resume2000/checkpoint_iter_0001299.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r113_magformer_r112_target100_resume2000/checkpoint_iter_0001399.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r113_magformer_r112_target100_resume2000/checkpoint_iter_0001599.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r113_magformer_r112_target100_resume2000/checkpoint_iter_0001699.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r113_magformer_r112_target100_resume2000/checkpoint_iter_0001799.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r113_magformer_r112_target100_resume2000/checkpoint_iter_0001899.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r113_magformer_r112_target100_resume2000/checkpoint_iter_0001999.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0000099.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0000199.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0000299.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0000399.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0000499.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0000599.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0000699.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0000799.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0000899.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0001099.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0001199.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0001299.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0001399.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0001599.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0001699.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0001799.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0001899.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0001999.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r115b_magformer_r114warm_fulltarget200_oracle_2000/checkpoint_iter_0000099.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r115b_magformer_r114warm_fulltarget200_oracle_2000/checkpoint_iter_0000199.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r115b_magformer_r114warm_fulltarget200_oracle_2000/checkpoint_iter_0000299.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r115b_magformer_r114warm_fulltarget200_oracle_2000/checkpoint_iter_0000399.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r115b_magformer_r114warm_fulltarget200_oracle_2000/checkpoint_iter_0000499.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r115b_magformer_r114warm_fulltarget200_oracle_2000/checkpoint_iter_0000599.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r115b_magformer_r114warm_fulltarget200_oracle_2000/checkpoint_iter_0000699.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r115b_magformer_r114warm_fulltarget200_oracle_2000/checkpoint_iter_0000799.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r115b_magformer_r114warm_fulltarget200_oracle_2000/checkpoint_iter_0000899.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r115b_magformer_r114warm_fulltarget200_oracle_2000/checkpoint_iter_0001099.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r115b_magformer_r114warm_fulltarget200_oracle_2000/checkpoint_iter_0001199.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r115b_magformer_r114warm_fulltarget200_oracle_2000/checkpoint_iter_0001299.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r115b_magformer_r114warm_fulltarget200_oracle_2000/checkpoint_iter_0001399.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r115b_magformer_r114warm_fulltarget200_oracle_2000/checkpoint_iter_0001499.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r115b_magformer_r114warm_fulltarget200_oracle_2000/checkpoint_iter_0001599.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r115b_magformer_r114warm_fulltarget200_oracle_2000/checkpoint_iter_0001699.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r115b_magformer_r114warm_fulltarget200_oracle_2000/checkpoint_iter_0001799.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r115b_magformer_r114warm_fulltarget200_oracle_2000/checkpoint_iter_0001899.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r115b_magformer_r114warm_fulltarget200_oracle_2000/checkpoint_iter_0001999.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r132_official_m2f_32k_cache_continue_g67/model_0001999.pth` | 503.9 MiB | unreferenced checkpoint |
| `output/baseline/r132_official_m2f_32k_cache_continue_g67/model_0002499.pth` | 503.9 MiB | unreferenced checkpoint |
| `output/baseline/r132_official_m2f_32k_cache_continue_g67/model_0002999.pth` | 503.9 MiB | unreferenced checkpoint |
| `output/baseline/r134_official_m2f_32k_cache_continue_3000_5000_g67/model_0003499.pth` | 503.9 MiB | unreferenced checkpoint |
| `output/baseline/r134_official_m2f_32k_cache_continue_3000_5000_g67/model_0003999.pth` | 503.9 MiB | unreferenced checkpoint |
| `output/baseline/r134_official_m2f_32k_cache_continue_3000_5000_g67/model_0004999.pth` | 503.9 MiB | unreferenced checkpoint |
| `output/baseline/r135_official_m2f_r134iter4500_target150_smoke_g67/model_0000199.pth` | 503.9 MiB | unreferenced checkpoint |
| `output/baseline/r136_official_m2f_r134iter4500_target150_continue_200_1000_g67/model_0000399.pth` | 503.9 MiB | unreferenced checkpoint |
| `output/baseline/r136_official_m2f_r134iter4500_target150_continue_200_1000_g67/model_0000599.pth` | 503.9 MiB | unreferenced checkpoint |
| `output/baseline/r136_official_m2f_r134iter4500_target150_continue_200_1000_g67/model_0000799.pth` | 503.9 MiB | unreferenced checkpoint |
| `output/baseline/r136_official_m2f_r134iter4500_target150_continue_200_1000_g67/model_0000999.pth` | 503.9 MiB | unreferenced checkpoint |
| `output/baseline/r137_official_m2f_r136iter1000_target150_continue_1000_2000_g67/model_0001199.pth` | 503.9 MiB | unreferenced checkpoint |
| `output/baseline/r137_official_m2f_r136iter1000_target150_continue_1000_2000_g67/model_0001399.pth` | 503.9 MiB | unreferenced checkpoint |
| `output/baseline/r137_official_m2f_r136iter1000_target150_continue_1000_2000_g67/model_0001599.pth` | 503.9 MiB | unreferenced checkpoint |
| `output/baseline/r137_official_m2f_r136iter1000_target150_continue_1000_2000_g67/model_0001799.pth` | 503.9 MiB | unreferenced checkpoint |
| `output/baseline/r137_official_m2f_r136iter1000_target150_continue_1000_2000_g67/model_0001999.pth` | 503.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000019.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000039.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000059.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000079.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000099.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000119.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000139.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000159.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000179.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000199.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000219.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000239.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000259.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000279.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000299.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000319.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000339.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000359.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000379.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000399.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000419.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000439.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000459.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000479.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000499.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000519.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000539.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000559.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000579.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000599.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000619.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000639.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000659.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000679.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000699.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000719.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000739.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000759.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000779.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000799.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000819.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000839.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000859.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000879.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000899.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000919.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000939.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000959.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000979.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0000999.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001019.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001039.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001059.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001079.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001099.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001119.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001139.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001159.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001179.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001199.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001219.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001239.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001259.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001279.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001299.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001319.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001339.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001359.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001379.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001399.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001419.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001439.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001459.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001479.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001499.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001519.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001539.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001559.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001579.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001599.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001619.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001639.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001659.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001679.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001699.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001719.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001739.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001759.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001779.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001799.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001819.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001839.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001859.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001879.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001899.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001919.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001939.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001959.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001979.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0001999.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_smoke_20260519/checkpoint_iter_0000019.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_smoke_20260519/checkpoint_iter_0000020.pth` | 554.9 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000019.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000039.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000059.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000079.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000099.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000119.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000139.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000159.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000179.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000199.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000219.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000239.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000259.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000279.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000299.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000319.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000339.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000359.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000379.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000399.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000419.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000439.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000459.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000479.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000499.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000519.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000539.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000559.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000579.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000599.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000619.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000639.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000659.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000679.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000699.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000719.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000739.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000759.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000779.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000799.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000819.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000839.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000859.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000879.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000899.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000919.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000939.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000959.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000979.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0000999.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001019.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001039.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001059.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001079.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001099.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001119.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001139.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001159.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001179.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001199.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001219.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001239.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001259.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001279.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001299.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001319.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001339.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001359.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001379.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001399.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001419.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001439.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001459.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001479.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001499.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001519.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001539.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001559.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001579.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001599.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001619.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001639.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001659.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001679.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001699.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001719.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001739.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001759.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001779.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001799.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001819.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001839.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001859.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001879.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001899.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001919.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001939.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001959.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001979.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0001999.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/baseline/r92_official_mask2former_1k1024_smoke_mask2former_env/model_0000009.pth` | 504.5 MiB | unreferenced checkpoint |
| `output/baseline/r94_official_m2f_target_labeled25_smoke/model_0000199.pth` | 504.5 MiB | unreferenced checkpoint |
| `output/baseline/r95_official_m2f_source_1k1024_short/model_0000499.pth` | 503.9 MiB | unreferenced checkpoint |
| `output/baseline/r96_official_m2f_32k_cache_smoke/model_0000009.pth` | 503.9 MiB | unreferenced checkpoint |
| `output/baseline/r97_official_m2f_32k_cache_short/model_0000499.pth` | 503.9 MiB | unreferenced checkpoint |
| `output/baseline/r97_official_m2f_32k_cache_short/model_0000999.pth` | 503.9 MiB | unreferenced checkpoint |
| `output/baseline/r97_official_m2f_32k_cache_short/model_0001499.pth` | 503.9 MiB | unreferenced checkpoint |
| `output/baseline/r99_magformer_r98ckpt_target_labeled25_1024/checkpoint_iter_0000500.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/diagnostics/debug_target_labeled_clean_overfit_1024_teacher8499_20260515/checkpoint_iter_0001999.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/diagnostics/debug_target_labeled_clean_overfit_1024_teacher8499_20260515/checkpoint_iter_0002000.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/diagnostics/debug_target_labeled_overfit_1024_teacher8499_20260515/checkpoint_iter_0000499.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/diagnostics/debug_target_labeled_overfit_1024_teacher8499_20260515/checkpoint_iter_0000999.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/diagnostics/r86_32k512_source_smoke_clean_20260517/checkpoint_iter_0000500.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/diagnostics/r86_32k512_source_smoke_clean_20260517/checkpoint_iter_0000501.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/diagnostics/r88_32k1024_cache_source_smoke_clean_20260517/checkpoint_iter_0000501.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008999.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/experiments/vc_suda_partial_label_loss_smoke_20260515/checkpoint_iter_0000003.pth` | 383.2 MiB | unreferenced checkpoint |
| `output/experiments/vc_suda_partial_label_loss_smoke_rerun_20260515/checkpoint_iter_0000003.pth` | 383.2 MiB | unreferenced checkpoint |
| `output/experiments/vc_suda_stage_a_512_g4_7_20260513_1935/checkpoint_iter_0000315.pth` | 912.9 MiB | unreferenced checkpoint |
| `output/experiments/vc_suda_stage_a_512_g4_7_resume_earlystopfix_20260513_2006/checkpoint_iter_0002448.pth` | 912.9 MiB | unreferenced checkpoint |
| `output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0009000.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/experiments/vc_suda_stage_b_1024_teacher8499_g4_7_retry_20260513_2207/checkpoint_iter_0000999.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/experiments/vc_suda_stage_b_1024_teacher8499_g4_7_retry_20260513_2207/checkpoint_iter_0001499.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/experiments/vc_suda_stage_c_r3_a10_1024_teacher8499_smoke_20260514/checkpoint_iter_0000003.pth` | 383.2 MiB | unreferenced checkpoint |
| `output/experiments/vc_suda_stage_c_r3_b10_u001_smoke_20260514_212733/checkpoint_iter_0000003.pth` | 383.2 MiB | unreferenced checkpoint |
| `output/experiments/vc_suda_stage_c_r4_partial_a10_smoke_20260515/checkpoint_iter_0000003.pth` | 383.2 MiB | unreferenced checkpoint |
| `output/experiments/vc_suda_stage_c_r5_tl05_a10_smoke_20260515/checkpoint_iter_0000003.pth` | 383.2 MiB | unreferenced checkpoint |
| `output/experiments/vc_suda_tl_weight_smoke_20260515/checkpoint_iter_0000003.pth` | 191.7 MiB | unreferenced checkpoint |
| `output/upper_bound/target160_supervised_1024_teacher8499/checkpoint_iter_0002000.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/vc_suda/r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075/checkpoint_iter_0000001.pth` | 191.7 MiB | unreferenced checkpoint |
| `output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300/checkpoint_iter_0000199.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300/checkpoint_iter_0000299.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300/checkpoint_iter_0000300.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/vc_suda/r83_offline_smoke_20260517_141142/checkpoint_iter_0000501.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_a_r26_32k_512_smoke/checkpoint_iter_0000002.pth` | 304.5 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r10_r8b_ckpt999_no_depth_noise_continue_1024_teacher8499/checkpoint_iter_0000249.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r11_r8b_ckpt999_mask_loss75_continue_1024_teacher8499/checkpoint_iter_0000249.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r11_r8b_ckpt999_mask_loss75_continue_1024_teacher8499/checkpoint_iter_0000499.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0001000.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r20_exterior_ring_r12_ckpt499_1024_teacher8499/checkpoint_iter_0000249.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r20_exterior_ring_r12_ckpt499_1024_teacher8499/checkpoint_iter_0000500.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r27_32k512_smoke_teacher8499/checkpoint_iter_0000001.pth` | 383.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r28_32k512_short_teacher8499/checkpoint_iter_0000249.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r28_32k512_short_teacher8499/checkpoint_iter_0000250.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r31_target_labeled_plus25_1024_smoke1_cuda_msda/checkpoint_iter_0000001.pth` | 383.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r31_target_labeled_plus25_1024_teacher8499/checkpoint_iter_0000250.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r32_plus25_unsup0_1024_teacher8499/checkpoint_iter_0000249.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r32_plus25_unsup0_1024_teacher8499/checkpoint_iter_0000250.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r34_plus25_true_resume_1024_teacher8499/checkpoint_iter_0000749.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r34_plus25_true_resume_1024_teacher8499/checkpoint_iter_0000750.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r35_original_split_true_resume_1024_teacher8499/checkpoint_iter_0000749.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r37_balanced_plus25_true_resume_1024_teacher8499/checkpoint_iter_0000749.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r37_balanced_plus25_true_resume_1024_teacher8499/checkpoint_iter_0000750.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r3_a10_resume2000_1024_teacher8499/checkpoint_iter_0002000.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r3_b10_u001_1024_teacher8499/checkpoint_iter_0000999.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r3_c05_1024_teacher8499/checkpoint_iter_0000999.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r42_point65536_original_split_true_resume_1024_teacher8499/checkpoint_iter_0000749.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r42_point65536_original_split_true_resume_1024_teacher8499_smoke/checkpoint_iter_0000500.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r4_partial_a10_1024_teacher8499/checkpoint_iter_0000999.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r6_a10_nocontrast_1024_teacher8499/checkpoint_iter_0000499.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r74_balanced_ce_minfg005_1024/checkpoint_iter_0000749.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r76_balanced_ce_minfg003_1024/checkpoint_iter_0000749.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r77_minfg005_target_sampling_1024/checkpoint_iter_0000749.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r7_a10_lsj10_1024_teacher8499/checkpoint_iter_0002000.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r80_unsup_schedule_1024/checkpoint_iter_0000749.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r84_offline_tta_bank_w012_1024/checkpoint_iter_0000749.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r86_32k512_source_1024/checkpoint_iter_0000749.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r88_32k1024_cache_source_1024/checkpoint_iter_0000749.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/stage_c_r8b_lsj10_low_lr_continue_1024_teacher8499/checkpoint_iter_0001000.pth` | 747.2 MiB | unreferenced checkpoint |
| `output/vc_suda/vc_suda_stage_b_r59_teacher8499_retention_min_1024/checkpoint_iter_0000249.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/vc_suda/vc_suda_stage_b_r62_original_source_tl05_1024/checkpoint_iter_0000249.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/vc_suda/vc_suda_stage_b_r67_multisource_freeze_backbones_1024/checkpoint_iter_0000499.pth` | 343.9 MiB | unreferenced checkpoint |
| `output/vc_suda/vc_suda_stage_b_r67_multisource_freeze_backbones_1024/checkpoint_iter_0000500.pth` | 343.9 MiB | unreferenced checkpoint |
| `output/vc_suda/vc_suda_stage_b_r69_multisource_l2sp_1024/checkpoint_iter_0000499.pth` | 555.7 MiB | unreferenced checkpoint |
| `output/vc_suda/vc_suda_stage_b_r69_multisource_l2sp_1024/checkpoint_iter_0000500.pth` | 555.7 MiB | unreferenced checkpoint |

## Kept Checkpoints

| Path | Size | Reason |
| --- | ---: | --- |
| `output/baseline/r100_magformer_r98ckpt_target_labeled25_cleanfit_1024/checkpoint_iter_0000500.pth` | 555.7 MiB | referenced by configs/baseline_supervised_r103_magformer_r100_resume_target_labeled25_1000.yaml, configs/baseline_supervised_r104_magformer_r100_resume_target_labeled50_balanced_1000.yaml, configs/baseline_supervised_r105_magformer_fulltarget200_oracle_1000.yaml, configs/baseline_supervised_r106_magformer_fulltarget200_oracle_warmstart_1000.yaml, configs/baseline_supervised_r107_magformer_fulltarget200_oracle_randompoints_1000.yaml, configs/baseline_supervised_r108_magformer_fulltarget200_oracle_dice75_1000.yaml, configs/baseline_supervised_r109_magformer_fulltarget200_oracle_dice100_1000.yaml, configs/baseline_supervised_r110_magformer_fulltarget200_oracle_dice125_1000.yaml, configs/baseline_supervised_r111_magformer_r109recipe_target_labeled50_balanced_1000.yaml, configs/baseline_supervised_r112_magformer_r111recipe_target_labeled100_balanced_1000.yaml, docs/results/baseline_r103_magformer_cleanfit_resume_20260518.md, docs/results/baseline_r104_magformer_target_labeled50_balanced_20260518.md, docs/results/baseline_r105_magformer_fulltarget200_oracle_20260518.md, docs/results/baseline_r106_magformer_fulltarget200_oracle_warmstart_20260518.md, docs/results/baseline_r107_magformer_fulltarget200_oracle_randompoints_20260518.md, docs/results/baseline_r108_magformer_fulltarget200_oracle_dice75_20260518.md, docs/results/baseline_r111_magformer_r109recipe_target_labeled50_balanced_20260518.md, docs/results/baseline_r112_magformer_r111recipe_target_labeled100_balanced_20260518.md |
| `output/baseline/r101_magformer_cleanfit_r99solver_1024/checkpoint_iter_0000500.pth` | 555.7 MiB | referenced by docs/results/baseline_r101_magformer_cleanfit_r99solver_20260518.md |
| `output/baseline/r102_magformer_augnoise_strongsolver_1024/checkpoint_iter_0000500.pth` | 555.7 MiB | referenced by docs/results/baseline_r102_magformer_augnoise_strongsolver_20260518.md |
| `output/baseline/r103_magformer_r100_resume_target_labeled25_1000/checkpoint_iter_0001000.pth` | 555.7 MiB | referenced by docs/results/baseline_r103_magformer_cleanfit_resume_20260518.md |
| `output/baseline/r104_magformer_r100_resume_target_labeled50_balanced_1000/checkpoint_iter_0001000.pth` | 555.7 MiB | referenced by docs/results/baseline_r104_magformer_target_labeled50_balanced_20260518.md |
| `output/baseline/r111_magformer_r109recipe_target_labeled50_balanced_1000/checkpoint_iter_0000499.pth` | 555.7 MiB | final/gate:r111-r112-iters0499-0799-1000 |
| `output/baseline/r111_magformer_r109recipe_target_labeled50_balanced_1000/checkpoint_iter_0000799.pth` | 555.7 MiB | final/gate:r111-r112-iters0499-0799-1000 |
| `output/baseline/r111_magformer_r109recipe_target_labeled50_balanced_1000/checkpoint_iter_0001000.pth` | 555.7 MiB | referenced by docs/results/baseline_r111_magformer_r109recipe_target_labeled50_balanced_20260518.md; final/gate:r111-r112-iters0499-0799-1000 |
| `output/baseline/r112_magformer_r111recipe_target_labeled100_balanced_1000/checkpoint_iter_0000499.pth` | 555.7 MiB | final/gate:r111-r112-iters0499-0799-1000 |
| `output/baseline/r112_magformer_r111recipe_target_labeled100_balanced_1000/checkpoint_iter_0000799.pth` | 555.7 MiB | final/gate:r111-r112-iters0499-0799-1000 |
| `output/baseline/r112_magformer_r111recipe_target_labeled100_balanced_1000/checkpoint_iter_0001000.pth` | 555.7 MiB | referenced by configs/baseline_supervised_r113_magformer_r112_target100_resume2000.yaml, docs/results/baseline_r113_magformer_r112_target100_resume2000_20260518.md; final/gate:r111-r112-iters0499-0799-1000 |
| `output/baseline/r113_magformer_r112_target100_resume2000/checkpoint_iter_0001499.pth` | 555.7 MiB | final/gate:r113-iters1499-2000 |
| `output/baseline/r113_magformer_r112_target100_resume2000/checkpoint_iter_0002000.pth` | 555.7 MiB | referenced by configs/baseline_supervised_r114_magformer_r113warm_target150_balanced_2000.yaml, docs/results/baseline_r114_magformer_r113warm_target150_balanced_20260518.md; final/gate:r113-iters1499-2000 |
| `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0000999.pth` | 555.7 MiB | final/gate:r114-iters0999-1499-2000 |
| `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0001499.pth` | 555.7 MiB | final/gate:r114-iters0999-1499-2000 |
| `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0002000.pth` | 555.7 MiB | referenced by configs/baseline_supervised_r115_magformer_r114warm_fulltarget200_oracle_2000.yaml, configs/baseline_vc_suda_r118_magformer_r114warm_pseudo300.yaml, configs/baseline_vc_suda_r121_depth_boundary_w001_smoke.yaml, configs/baseline_vc_suda_r121_depth_boundary_w001_smoke_b1_iter1_loss_only.yaml, configs/baseline_vc_suda_r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075.yaml, configs/baseline_vc_suda_r122_depth_boundary_w001_pseudo300.yaml, docs/results/baseline_r114_magformer_r113warm_target150_balanced_20260518.md, docs/results/baseline_r115_magformer_r114warm_fulltarget200_oracle_20260518.md, docs/results/vc_suda_r118_stage_c_smallstep_20260518.md; final/gate:r114-iters0999-1499-2000 |
| `output/baseline/r115b_magformer_r114warm_fulltarget200_oracle_2000/checkpoint_iter_0000999.pth` | 555.7 MiB | final/gate:r115b-iters0999-2000 |
| `output/baseline/r115b_magformer_r114warm_fulltarget200_oracle_2000/checkpoint_iter_0002000.pth` | 555.7 MiB | referenced by docs/results/baseline_r115_magformer_r114warm_fulltarget200_oracle_20260518.md; final/gate:r115b-iters0999-2000 |
| `output/baseline/r134_official_m2f_32k_cache_continue_3000_5000_g67/model_0002999.pth` | 503.9 MiB | referenced by docs/results/baseline_r134_official_mask2former_32k_continue_3000_5000_g67_20260519.md |
| `output/baseline/r134_official_m2f_32k_cache_continue_3000_5000_g67/model_0004499.pth` | 503.9 MiB | referenced by docs/results/baseline_r135_official_mask2former_r134iter4500_target150_smoke_20260519.md, docs/results/vc_suda_baseline_gap_protocol_20260519.md |
| `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0002000.pth` | 554.9 MiB | referenced by docs/results/magformer_r139_rgb_only_target150_20260519.md; final/gate:r139-r141-iter2000 |
| `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0002000.pth` | 555.7 MiB | referenced by docs/results/magformer_r141_rgbd_r98warm_target150_20260519.md; final/gate:r139-r141-iter2000 |
| `output/baseline/r90_supervised_32k1024_cache_teacher8499_smoke/checkpoint_iter_0000150.pth` | 555.7 MiB | referenced by docs/results/baseline_r90_supervised_32k1024_smoke_20260517.md |
| `output/baseline/r91_supervised_32k1024_cache_teacher8499_lr1e6_smoke/checkpoint_iter_0000150.pth` | 555.7 MiB | referenced by docs/results/baseline_r91_supervised_32k1024_lr1e6_smoke_20260517.md |
| `output/baseline/r98_magformer_32k_source_short/checkpoint_iter_0000499.pth` | 555.7 MiB | referenced by configs/baseline_supervised_r100_magformer_r98ckpt_target_labeled25_cleanfit_1024.yaml, configs/baseline_supervised_r101_magformer_cleanfit_r99solver_1024.yaml, configs/baseline_supervised_r102_magformer_augnoise_strongsolver_1024.yaml, configs/baseline_supervised_r139_magformer_rgb_only_r98warm_target150.yaml, configs/baseline_supervised_r139_magformer_rgb_only_r98warm_target150_smoke.yaml, configs/baseline_supervised_r141_magformer_rgbd_r98warm_target150.yaml, configs/baseline_supervised_r99_magformer_r98ckpt_target_labeled25_1024.yaml, docs/results/baseline_r100_magformer_target_labeled25_cleanfit_20260518.md, docs/results/baseline_r102_magformer_augnoise_strongsolver_20260518.md, docs/results/baseline_r98_magformer_32k_source_short_20260518.md, docs/results/baseline_r99_magformer_r98_target_labeled25_20260518.md, docs/results/magformer_r138_r139_full32k_ablation_plan_20260519.md, docs/results/magformer_r139_rgb_only_target150_20260519.md, docs/results/magformer_r141_rgbd_r98warm_target150_20260519.md; warm-start:r12-r98-iter0499 |
| `output/baseline/r99_magformer_r98ckpt_target_labeled25_1024/checkpoint_iter_0000499.pth` | 555.7 MiB | referenced by docs/results/baseline_r99_magformer_r98_target_labeled25_20260518.md |
| `output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth` | 747.2 MiB | referenced by configs/baseline_supervised_r90_32k1024_cache_teacher8499_smoke.yaml, configs/baseline_supervised_r91_32k1024_cache_teacher8499_lr1e6_smoke.yaml, configs/baseline_supervised_r98_magformer_32k1024_cache_teacher8499_short.yaml, configs/debug_target_labeled_clean_overfit_1024_teacher8499.yaml, configs/debug_target_labeled_overfit_1024_teacher8499.yaml, configs/vc_suda_stage_b_1024_teacher8499.yaml, configs/vc_suda_stage_b_r59_teacher8499_retention_min_1024.yaml, configs/vc_suda_stage_b_r62_original_source_tl05_1024.yaml, configs/vc_suda_stage_b_r65_multisource_retention_1024.yaml, configs/vc_suda_stage_b_r67_multisource_freeze_backbones_1024.yaml, configs/vc_suda_stage_b_r69_multisource_l2sp_1024.yaml, docs/results/baseline_r90_supervised_32k1024_smoke_20260517.md, docs/results/baseline_r91_supervised_32k1024_lr1e6_smoke_20260517.md, docs/results/baseline_r98_magformer_32k_source_short_20260518.md, docs/results/vc_suda_r44_teacher_first50_protocol_fix_20260516.md, docs/results/vc_suda_r54_eval_protocol_checker_20260516.md, docs/results/vc_suda_r59_teacher8499_retention_min_20260516.md, docs/results/vc_suda_r62_original_source_tl05_20260516.md, docs/results/vc_suda_r65_multisource_retention_20260516.md, docs/results/vc_suda_r67_multisource_freeze_backbones_20260517.md, docs/results/vc_suda_r69_multisource_l2sp_20260517.md, docs/results/vc_suda_stage_b_launch_20260514.md, docs/results/vc_suda_stage_b_retry_launch_20260513_2207.md, docs/results/vc_suda_stage_b_training_launch_20260513_2131.md, docs/results/vc_suda_teacher_baseline_20260514.md, docs/results/vc_suda_teacher_eval_and_data_sanity_20260515.md, docs/results/vc_suda_teacher_pseudoreal_anomaly_20260515.md, tools/__pycache__/check_eval_protocol.cpython-311.pyc, tools/check_eval_protocol.py |
| `output/experiments/vc_suda_stage_a_512_g4_7_20260513_1935/checkpoint_iter_0000394.pth` | 912.9 MiB | referenced by docs/results/vc_suda_resume_training_launch_20260513_2006.md, docs/results/vc_suda_training_launch_20260513_1935.md |
| `output/experiments/vc_suda_stage_a_512_g4_7_resume_earlystopfix_20260513_2006/checkpoint_iter_0002527.pth` | 912.9 MiB | referenced by configs/vc_suda_stage_a_r26_32k_512_smoke.yaml |
| `output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth` | 555.7 MiB | referenced by configs/upper_bound_target160_supervised_1024_teacher8499.yaml, configs/vc_suda_stage_c_1024_teacher8499.yaml, configs/vc_suda_stage_c_r2_1024_teacher8499.yaml, configs/vc_suda_stage_c_r3_a10_1024_teacher8499.yaml, configs/vc_suda_stage_c_r3_b10_u001_1024_teacher8499.yaml, configs/vc_suda_stage_c_r3_c05_1024_teacher8499.yaml, configs/vc_suda_stage_c_r4_partial_a10_1024_teacher8499.yaml, configs/vc_suda_stage_c_r5_tl05_a10_1024_teacher8499.yaml, configs/vc_suda_stage_c_r6_a10_nocontrast_1024_teacher8499.yaml, configs/vc_suda_stage_c_r7_a10_lsj10_1024_teacher8499.yaml, docs/results/target_hidden_gt_upper_bound_plan_20260515.md, docs/results/vc_suda_stage_b_launch_20260514.md, docs/results/vc_suda_stage_b_post_eval_20260514.md, docs/results/vc_suda_stage_b_post_eval_20260514_metrics.json, docs/results/vc_suda_stage_b_to_c_handoff_20260514.md, docs/results/vc_suda_stage_c_launch_20260514.md, docs/results/vc_suda_stage_c_preflight_20260514.md, docs/results/vc_suda_stage_c_r2_plan_20260514.md, docs/results/vc_suda_stage_c_r2_status_20260514.md, docs/results/vc_suda_stage_c_r3_plan_20260514.md, docs/results/vc_suda_stage_c_r4_plan_20260515.md, docs/results/vc_suda_stage_c_r5_plan_20260515.md, docs/results/vc_suda_stage_c_r6_nocontrast_plan_20260515.md, docs/results/vc_suda_stage_c_r7_lsj10_plan_20260515.md, docs/results/vc_suda_stage_c_smoke_20260514.md, docs/results/vc_suda_teacher_pseudoreal_anomaly_20260515.md |
| `output/experiments/vc_suda_stage_c_1024_teacher8499_20260514_0734/checkpoint_iter_0008999.pth` | 747.2 MiB | referenced by docs/results/vc_suda_stage_c_final_eval_20260514_metrics.json |
| `output/experiments/vc_suda_stage_c_1024_teacher8499_20260514_0734/checkpoint_iter_0009000.pth` | 747.2 MiB | referenced by docs/results/vc_suda_stage_c_final_eval_20260514_metrics.json |
| `output/experiments/vc_suda_stage_c_1024_teacher8499_aw_smoke_20260514/checkpoint_iter_0000003.pth` | 383.2 MiB | referenced by docs/results/vc_suda_stage_c_smoke_20260514.md |
| `output/upper_bound/target160_supervised_1024_teacher8499/checkpoint_iter_0001999.pth` | 555.7 MiB | referenced by docs/results/target_hidden_gt_upper_bound_plan_20260515.md |
| `output/vc_suda/r118_magformer_r114warm_pseudo300/checkpoint_iter_0000099.pth` | 555.7 MiB | referenced by docs/results/vc_suda_r118_stage_c_smallstep_20260518.md |
| `output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300/checkpoint_iter_0000099.pth` | 555.7 MiB | referenced by docs/results/vc_suda_r124_depth_boundary_resume_runbook_20260518.md, docs/results/vc_suda_r128_gated_r122_evaluator_20260518.md |
| `output/vc_suda/stage_c_r11_r8b_ckpt999_mask_loss75_continue_1024_teacher8499/checkpoint_iter_0000749.pth` | 747.2 MiB | referenced by docs/results/vc_suda_r39_checkpoint_inventory_eval_20260516.md |
| `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000249.pth` | 747.2 MiB | referenced by docs/results/vc_suda_r58_r8b_r12_early_source_sanity_20260516.md, docs/results/vc_suda_stage_c_r12_32k_source_plan_20260515.md |
| `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth` | 747.2 MiB | referenced by configs/vc_suda_stage_c_r142_32254_train25654_source_target150_fixed.yaml, configs/vc_suda_stage_c_r18_pseudo_unmatched_neg_r12_ckpt499_1024_teacher8499.yaml, configs/vc_suda_stage_c_r20_exterior_ring_r12_ckpt499_1024_teacher8499.yaml, configs/vc_suda_stage_c_r31_target_labeled_plus25_1024_smoke1_cuda_msda.yaml, configs/vc_suda_stage_c_r31_target_labeled_plus25_1024_teacher8499.yaml, configs/vc_suda_stage_c_r32_plus25_unsup0_1024_teacher8499.yaml, configs/vc_suda_stage_c_r34_plus25_true_resume_1024_teacher8499.yaml, configs/vc_suda_stage_c_r35_original_split_true_resume_1024_teacher8499.yaml, configs/vc_suda_stage_c_r37_balanced_plus25_true_resume_1024_teacher8499.yaml, configs/vc_suda_stage_c_r42_point65536_original_split_true_resume_1024_teacher8499.yaml, configs/vc_suda_stage_c_r42_point65536_original_split_true_resume_1024_teacher8499_smoke.yaml, configs/vc_suda_stage_c_r46_pseudoreal_source_true_resume_1024_teacher8499.yaml, configs/vc_suda_stage_c_r50_scale024_source_true_resume_1024_teacher8499.yaml, configs/vc_suda_stage_c_r51_scale024_source1k_true_resume_1024_teacher8499.yaml, configs/vc_suda_stage_c_r52_pseudoreal_source_target_sampling_true_resume_1024_teacher8499.yaml, configs/vc_suda_stage_c_r71_r46_balanced_ce_off_1024.yaml, configs/vc_suda_stage_c_r73_balanced_ce_off_target_sampling_1024.yaml, configs/vc_suda_stage_c_r74_balanced_ce_minfg005_1024.yaml, configs/vc_suda_stage_c_r76_balanced_ce_minfg003_1024.yaml, configs/vc_suda_stage_c_r77_minfg005_target_sampling_1024.yaml, configs/vc_suda_stage_c_r80_unsup_schedule_1024.yaml, configs/vc_suda_stage_c_r83_offline_tta_bank_1024.yaml, configs/vc_suda_stage_c_r84_offline_tta_bank_w012_1024.yaml, configs/vc_suda_stage_c_r86_32k512_source_1024.yaml, configs/vc_suda_stage_c_r88_32k1024_cache_source_1024.yaml, docs/results/vc_suda_r33_r12_ckpt499_sanity_20260516.md, docs/results/vc_suda_r41_eval1536_teacher_r12_20260516.md, docs/results/vc_suda_r42_point65536_original_split_plan_20260516.md, docs/results/vc_suda_r46_pseudoreal_source_true_resume_20260516.md, docs/results/vc_suda_r50_scale024_source_true_resume_20260516.md, docs/results/vc_suda_r51_scale024_source1k_true_resume_20260516.md, docs/results/vc_suda_r52_target_unlabeled_sampling_gate_20260516.md, docs/results/vc_suda_r56_source_sanity_timeline_20260516.md, docs/results/vc_suda_r71_balanced_ce_off_20260517.md, docs/results/vc_suda_r74_balanced_ce_minfg005_20260517.md, docs/results/vc_suda_r76_balanced_ce_minfg003_20260517.md, docs/results/vc_suda_stage_c_r12_32k_source_plan_20260515.md, docs/results/vc_suda_stage_c_r13_dense_geometry_audit_20260515.md, docs/results/vc_suda_stage_c_r14_inference_instrumentation_20260515.md, docs/results/vc_suda_stage_c_r15_topk200_maxdets200_20260515.md, docs/results/vc_suda_stage_c_r17_signal_diagnostics_20260515.md, docs/results/vc_suda_stage_c_r18_pseudo_unmatched_negative_plan_20260515.md, docs/results/vc_suda_stage_c_r19_exterior_ring_dryrun_20260515.md, docs/results/vc_suda_stage_c_r20_exterior_ring_plan_20260515.md, docs/results/vc_suda_stage_c_r31_target_labeled_plus25_plan_20260516.md, docs/results/vc_suda_stage_c_r32_plus25_unsup0_plan_20260516.md, docs/results/vc_suda_stage_c_r34_plus25_true_resume_plan_20260516.md, docs/results/vc_suda_stage_c_r35_original_split_true_resume_plan_20260516.md, docs/results/vc_suda_stage_c_r37_balanced_plus25_plan_20260516.md, tools/__pycache__/plan_checkpoint_cleanup.cpython-312.pyc, tools/plan_checkpoint_cleanup.py; explicit keep path; warm-start:r12-r98-iter0499 |
| `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000749.pth` | 747.2 MiB | referenced by docs/results/vc_suda_r39_checkpoint_inventory_eval_20260516.md |
| `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000999.pth` | 747.2 MiB | referenced by docs/results/vc_suda_stage_c_r12_32k_source_plan_20260515.md |
| `output/vc_suda/stage_c_r18_pseudo_unmatched_neg_r12_ckpt499_1024_teacher8499/checkpoint_iter_0000249.pth` | 747.2 MiB | referenced by docs/results/vc_suda_stage_c_r18_pseudo_unmatched_negative_plan_20260515.md |
| `output/vc_suda/stage_c_r20_exterior_ring_r12_ckpt499_1024_teacher8499/checkpoint_iter_0000499.pth` | 747.2 MiB | warm-start:r12-r98-iter0499 |
| `output/vc_suda/stage_c_r2_1024_teacher8499/checkpoint_iter_0000999.pth` | 747.2 MiB | referenced by docs/results/vc_suda_stage_c_r2_status_20260514.md |
| `output/vc_suda/stage_c_r2_1024_teacher8499/checkpoint_iter_0001999.pth` | 747.2 MiB | referenced by docs/results/vc_suda_stage_c_r2_status_20260514.md |
| `output/vc_suda/stage_c_r31_target_labeled_plus25_1024_teacher8499/checkpoint_iter_0000249.pth` | 747.2 MiB | referenced by docs/results/vc_suda_stage_c_r31_target_labeled_plus25_plan_20260516.md |
| `output/vc_suda/stage_c_r35_original_split_true_resume_1024_teacher8499/checkpoint_iter_0000750.pth` | 747.2 MiB | referenced by docs/results/vc_suda_r39_checkpoint_inventory_eval_20260516.md, docs/results/vc_suda_r56_source_sanity_timeline_20260516.md |
| `output/vc_suda/stage_c_r3_a10_1024_teacher8499/checkpoint_iter_0000999.pth` | 747.2 MiB | referenced by docs/results/vc_suda_stage_c_r3_plan_20260514.md, docs/results/vc_suda_stage_c_r4_plan_20260515.md, docs/results/vc_suda_stage_c_r5_plan_20260515.md, docs/results/vc_suda_teacher_pseudoreal_anomaly_20260515.md |
| `output/vc_suda/stage_c_r3_a10_resume2000_1024_teacher8499/checkpoint_iter_0001999.pth` | 747.2 MiB | referenced by docs/results/vc_suda_stage_c_r3_plan_20260514.md |
| `output/vc_suda/stage_c_r42_point65536_original_split_true_resume_1024_teacher8499/checkpoint_iter_0000750.pth` | 747.2 MiB | referenced by docs/results/vc_suda_r42_point65536_original_split_plan_20260516.md |
| `output/vc_suda/stage_c_r46_pseudoreal_source_true_resume_1024_teacher8499/checkpoint_iter_0000749.pth` | 747.2 MiB | referenced by docs/results/vc_suda_r46_pseudoreal_source_true_resume_20260516.md |
| `output/vc_suda/stage_c_r46_pseudoreal_source_true_resume_1024_teacher8499/checkpoint_iter_0000750.pth` | 747.2 MiB | referenced by docs/results/vc_suda_r46_pseudoreal_source_true_resume_20260516.md, docs/results/vc_suda_r54_eval_protocol_checker_20260516.md, docs/results/vc_suda_r55_r46_original_first50_eval_20260516.md, docs/results/vc_suda_r56_source_sanity_timeline_20260516.md |
| `output/vc_suda/stage_c_r50_scale024_source_true_resume_1024_teacher8499/checkpoint_iter_0000749.pth` | 747.2 MiB | referenced by docs/results/vc_suda_r50_scale024_source_true_resume_20260516.md |
| `output/vc_suda/stage_c_r50_scale024_source_true_resume_1024_teacher8499/checkpoint_iter_0000750.pth` | 747.2 MiB | referenced by docs/results/vc_suda_r50_scale024_source_true_resume_20260516.md |
| `output/vc_suda/stage_c_r51_scale024_source1k_true_resume_1024_teacher8499/checkpoint_iter_0000749.pth` | 747.2 MiB | referenced by docs/results/vc_suda_r51_scale024_source1k_true_resume_20260516.md |
| `output/vc_suda/stage_c_r51_scale024_source1k_true_resume_1024_teacher8499/checkpoint_iter_0000750.pth` | 747.2 MiB | referenced by docs/results/vc_suda_r51_scale024_source1k_true_resume_20260516.md |
| `output/vc_suda/stage_c_r52_pseudoreal_source_target_sampling_true_resume_1024_teacher8499/checkpoint_iter_0000749.pth` | 747.2 MiB | referenced by docs/results/vc_suda_r52_target_unlabeled_sampling_gate_20260516.md |
| `output/vc_suda/stage_c_r52_pseudoreal_source_target_sampling_true_resume_1024_teacher8499/checkpoint_iter_0000750.pth` | 747.2 MiB | referenced by docs/results/vc_suda_r52_target_unlabeled_sampling_gate_20260516.md, docs/results/vc_suda_r56_source_sanity_timeline_20260516.md |
| `output/vc_suda/stage_c_r5_tl05_a10_1024_teacher8499/checkpoint_iter_0000999.pth` | 747.2 MiB | referenced by docs/results/vc_suda_stage_c_r5_plan_20260515.md |
| `output/vc_suda/stage_c_r71_r46_balanced_ce_off_1024/checkpoint_iter_0000749.pth` | 747.2 MiB | referenced by docs/results/vc_suda_r71_balanced_ce_off_20260517.md |
| `output/vc_suda/stage_c_r71_r46_balanced_ce_off_1024/checkpoint_iter_0000750.pth` | 747.2 MiB | referenced by docs/results/vc_suda_r71_balanced_ce_off_20260517.md |
| `output/vc_suda/stage_c_r73_balanced_ce_off_target_sampling_1024/checkpoint_iter_0000749.pth` | 747.2 MiB | referenced by docs/results/vc_suda_r73_balanced_ce_off_target_sampling_20260517.md |
| `output/vc_suda/stage_c_r73_balanced_ce_off_target_sampling_1024/checkpoint_iter_0000750.pth` | 747.2 MiB | referenced by docs/results/vc_suda_r73_balanced_ce_off_target_sampling_20260517.md |
| `output/vc_suda/stage_c_r74_balanced_ce_minfg005_1024/checkpoint_iter_0000750.pth` | 747.2 MiB | referenced by docs/results/vc_suda_r74_balanced_ce_minfg005_20260517.md, docs/results/vc_suda_r79_pseudo_signal_diagnosis_20260517.md, docs/results/vc_suda_r81_pseudo_candidate_objective_20260517.md |
| `output/vc_suda/stage_c_r76_balanced_ce_minfg003_1024/checkpoint_iter_0000750.pth` | 747.2 MiB | referenced by docs/results/vc_suda_r76_balanced_ce_minfg003_20260517.md |
| `output/vc_suda/stage_c_r77_minfg005_target_sampling_1024/checkpoint_iter_0000750.pth` | 747.2 MiB | referenced by docs/results/vc_suda_r77_minfg005_target_sampling_20260517.md |
| `output/vc_suda/stage_c_r7_a10_lsj10_1024_teacher8499/checkpoint_iter_0001999.pth` | 747.2 MiB | referenced by configs/vc_suda_stage_c_r8b_lsj10_low_lr_continue_1024_teacher8499.yaml, docs/results/vc_suda_stage_c_r8b_low_lr_continue_plan_20260515.md |
| `output/vc_suda/stage_c_r80_unsup_schedule_1024/checkpoint_iter_0000750.pth` | 747.2 MiB | referenced by docs/results/vc_suda_r80_unsup_schedule_20260517.md, docs/results/vc_suda_r81_pseudo_candidate_objective_20260517.md, docs/results/vc_suda_r82_tta_candidate_bank_20260517.md, docs/results/vc_suda_r83_tta_coco_bank_20260517.md, tools/__pycache__/build_r83_tta_coco_bank.cpython-311.pyc, tools/__pycache__/diagnose_r82_tta_candidate_bank.cpython-311.pyc, tools/build_r83_tta_coco_bank.py, tools/diagnose_r82_tta_candidate_bank.py |
| `output/vc_suda/stage_c_r84_offline_tta_bank_w012_1024/checkpoint_iter_0000750.pth` | 555.7 MiB | referenced by docs/results/vc_suda_r84_offline_tta_bank_w012_20260517.md |
| `output/vc_suda/stage_c_r86_32k512_source_1024/checkpoint_iter_0000750.pth` | 747.2 MiB | referenced by docs/results/vc_suda_r86_32k512_source_20260517.md |
| `output/vc_suda/stage_c_r88_32k1024_cache_source_1024/checkpoint_iter_0000750.pth` | 747.2 MiB | referenced by docs/results/vc_suda_r88_32k1024_cache_source_20260517.md |
| `output/vc_suda/stage_c_r8b_lsj10_low_lr_continue_1024_teacher8499/checkpoint_iter_0000999.pth` | 747.2 MiB | referenced by configs/vc_suda_stage_c_r10_r8b_ckpt999_no_depth_noise_continue_1024_teacher8499.yaml, configs/vc_suda_stage_c_r11_r8b_ckpt999_mask_loss75_continue_1024_teacher8499.yaml, configs/vc_suda_stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499.yaml, configs/vc_suda_stage_c_r12b_32k_source_r8b_ckpt999_continue_1024_teacher8499.yaml, configs/vc_suda_stage_c_r27_32k512_smoke_teacher8499.yaml, configs/vc_suda_stage_c_r28_32k512_short_teacher8499.yaml, docs/results/vc_suda_r57_r12_source_sanity_collapse_audit_20260516.md, docs/results/vc_suda_r58_r8b_r12_early_source_sanity_20260516.md, docs/results/vc_suda_stage_c_r10_no_depth_noise_plan_20260515.md, docs/results/vc_suda_stage_c_r11_mask_loss_plan_20260515.md, docs/results/vc_suda_stage_c_r12_32k_source_plan_20260515.md, docs/results/vc_suda_stage_c_r17_signal_diagnostics_20260515.md, docs/results/vc_suda_stage_c_r28_32k512_short_plan_20260516.md |
| `output/vc_suda/vc_suda_stage_b_r59_teacher8499_retention_min_1024/checkpoint_iter_0000250.pth` | 555.7 MiB | referenced by docs/results/vc_suda_r59_teacher8499_retention_min_20260516.md |
| `output/vc_suda/vc_suda_stage_b_r62_original_source_tl05_1024/checkpoint_iter_0000250.pth` | 555.7 MiB | referenced by docs/results/vc_suda_r62_original_source_tl05_20260516.md |
| `output/vc_suda/vc_suda_stage_b_r65_multisource_retention_1024/checkpoint_iter_0000499.pth` | 555.7 MiB | referenced by docs/results/vc_suda_r65_multisource_retention_20260516.md |
| `output/vc_suda/vc_suda_stage_b_r65_multisource_retention_1024/checkpoint_iter_0000999.pth` | 555.7 MiB | referenced by docs/results/vc_suda_r65_multisource_retention_20260516.md |
| `output/vc_suda/vc_suda_stage_b_r65_multisource_retention_1024/checkpoint_iter_0001000.pth` | 555.7 MiB | referenced by docs/results/vc_suda_r65_multisource_retention_20260516.md |
