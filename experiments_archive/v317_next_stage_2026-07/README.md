v317 / next_stage generation (2026-07, 4028). This is the code lineage that
trained c0/v317_merge (fullval 0.8603). Its 66-commit git history is preserved
at archive_20260906/backup/magformer_4028_all_refs_20260906.bundle. Run configs
in source/configs/next_stage/ (c0/d0/p1/p2/dpe_off + rdi 300K family). Key
finding of the generation: DPE (depth positional encoding) is worth +0.9pt;
modality_fusion (dccg) gating changes forward behavior without changing the
parameter set - the 2026-09-06 caliber incident (0.7565 vs 0.8603) was exactly
this flag bypassing fusion in an eval config.
