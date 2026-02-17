# Comparison Report Update (M1)

## What was corrected

- Fixed parameter grouping logic in `scripts/compare_model_params.py`.
- Previously, many MagFormer `decoder.*` parameters were incorrectly grouped as `other`.
- Grouping now explicitly maps:
  - `decoder.transformer.encoder.*` -> `pixel_decoder`
  - `sem_seg_head.predictor.*` -> `transformer_decoder`

## Provenance clarification

- The previously quoted Mask2Former high AP (`~85.64`) was from:
  `mask2former/MGM_Mask2Former/output/0909_512_0.12K/20260214_train2000/`
- That run used warm-start (`MODEL.FINETUNE_WEIGHTS`) and recipe overrides,
  not a strict one-to-one execution of `configs/mgm_aligned_comparison.yaml`.

## Impact

- The `pixel decoder -57%` claim from old grouped output is no longer treated as a
  reliable architecture conclusion by itself.
- Architecture/runtime probes remain the source of truth for concrete mismatches
  (e.g. DPE wiring and pixel-decoder FFN dimension).
