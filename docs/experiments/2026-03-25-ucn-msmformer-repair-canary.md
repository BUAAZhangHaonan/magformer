# UCN And MSMFormer Repair Canary

Date: 2026-03-25

## Scope

This note records the first post-repair canaries after fixing two wrapper-level bugs:

- MSMFormer was accidentally training with Detectron2 default `SGD` instead of the vendor-style `AdamW` optimizer.
- Both UCN and MSMFormer were feeding the RGBD backbone the wrong depth representation. The wrappers were converting scalar depth into repeated channels and global min-max features instead of the XYZ-style geometry expected by the UCN family.

Additional repair landed during this round:

- MSMFormer RGBD mapper now emits an instance-id `label` map so the vendor embedding-loss path can be exercised.
- UCN and MSMFormer tests were expanded to pin the corrected RGB normalization, depth geometry, optimizer behavior, and pretrained-checkpoint selection.
- Official UCN and MSMFormer checkpoints were downloaded from the upstream Box shares and cached under `output/pretrained/`.

## Commands

### UCN canary with official checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n magformer python baselines/run_ucn_ecc.py \
  --register 0831 \
  --dataset-root /home/team/zhanghaonan/magformer/magformer_datasets/0831_1K \
  --output-dir /home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/output/debug/ucn_probe_pretrained \
  --epochs 1 \
  --batch 4 \
  --img-size 256
```

### MSMFormer eval-only with official RGBD checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n magformer python baselines/run_msmformer_ecc.py \
  --register 0831 \
  --dataset-root /home/team/zhanghaonan/magformer/magformer_datasets/0831_1K \
  --msmformer-root /home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/baselines/msmformer/MSMFormer \
  -- \
  --num-gpus 1 \
  --eval-only \
  --config-file /home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/configs/baselines/msmformer_0831_1k_tracks.yaml \
  MODEL.WEIGHTS /home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/output/pretrained/norm_RGBD_pretrained.pth \
  OUTPUT_DIR /home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/output/debug/msmformer_pretrained_eval
```

### MSMFormer 20-iter finetune probe with official RGBD checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n magformer python baselines/run_msmformer_ecc.py \
  --register 0831 \
  --dataset-root /home/team/zhanghaonan/magformer/magformer_datasets/0831_1K \
  --msmformer-root /home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/baselines/msmformer/MSMFormer \
  -- \
  --num-gpus 1 \
  --config-file /home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/configs/baselines/msmformer_0831_1k_tracks.yaml \
  MODEL.WEIGHTS /home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/output/pretrained/norm_RGBD_pretrained.pth \
  SOLVER.MAX_ITER 20 \
  SOLVER.STEPS '()' \
  SOLVER.WARMUP_ITERS 0 \
  TEST.EVAL_PERIOD 20 \
  SOLVER.CHECKPOINT_PERIOD 1000 \
  DATALOADER.NUM_WORKERS 0 \
  OUTPUT_DIR /home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/output/debug/msmformer_pretrained_train20_nw0
```

### MSMFormer scratch overfit probes

Single-batch 5-step overfit probe on GPU 0 with:

- corrected RGB normalization
- corrected XYZ depth geometry
- vendor-style `AdamW`

Observed scratch-only collapse:

- UCN-backbone route:
  `step 0 score_max=0.3387 mask_pos_ratio=0.6826`
  `step 1 score_max=0.4839 mask_pos_ratio=0.6665`
  `step 2 score_max=0.4894 mask_pos_ratio=0.0052`
  `step 3+ score_max=0.0 mask_pos_ratio=0.0`
- other-backbone route:
  `step 0 score_max=0.5956 mask_pos_ratio=0.6683`
  `step 1 score_max=0.5219 mask_pos_ratio=0.0451`
  `step 2 score_max=0.4806 mask_pos_ratio=0.000229`
  `step 3+ score_max=0.0 mask_pos_ratio=0.0`
- embedding-loss route:
  `step 0 score_max=0.3387 mask_pos_ratio=0.6826`
  `step 1 score_max=0.4862 mask_pos_ratio=0.7155`
  `step 2 score_max=0.4926 mask_pos_ratio=0.0469`
  `step 3+ score_max=0.0 mask_pos_ratio=0.0`

## Outputs

- UCN output dir:
  `/home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/output/debug/ucn_probe_pretrained`
- MSMFormer official-pretrain eval dir:
  `/home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/output/debug/msmformer_pretrained_eval`
- MSMFormer official-pretrain finetune dir:
  `/home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/output/debug/msmformer_pretrained_train20_nw0`
- MSMFormer scratch probe dirs:
  `/home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/output/debug/msmformer_probe`
  `/home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/output/debug/msmformer_probe_other`
  `/home/team/zhanghaonan/magformer/.worktrees/ucn-msmformer-repair/output/debug/msmformer_probe_embed`

## Results

### UCN

UCN is no longer in the completely dead regime.

Metric caveat:

- the older repaired-scratch `metrics.json` stores raw AP units
- the newer pretrained `metrics.json` stores COCO percentage points
- the comparisons below normalize both runs back to raw AP units

- `coco_instances_results.json` exists and is non-empty.
- repaired scratch canary: `segm/AP = 0.0086`
- official-pretrained canary: `segm/AP = 0.0105`
- repaired scratch canary: `bbox/AP = 0.1155`
- official-pretrained canary: `segm/AP50 = 0.0531`
- official-pretrained canary: `bbox/AP = 0.0290`

So the pretrained 1-epoch probe slightly improves `segm/AP`, but it does not produce a clean across-the-board improvement yet. This is still far below a paper-ready baseline, but it is materially different from the previous all-zero failure mode.

### MSMFormer

MSMFormer is no longer stuck at `0.000` once the official pretrained checkpoint is used.

What is now verified:

- the wrapper-level optimizer bug is fixed
- the RGB normalization bug is fixed
- the depth geometry bug is fixed
- the embedding-loss path can now receive label maps
- the official RGBD full-model checkpoint loads cleanly except for the expected single-class head mismatch
- eval-only on ECC is non-zero with the official checkpoint
- short finetuning on ECC remains non-zero and does not immediately collapse to empty masks

What is still happening under scratch-only probes:

- both the UCN-backbone route and the other-backbone route still collapse to empty masks after a few gradient steps
- adding the embedding loss does not remove the collapse

Official-pretrain results observed so far:

- eval-only: `segm/AP = 0.0114`, `bbox/AP = 0.0133`
- 20-iter finetune: `segm/AP = 0.0846`, `bbox/AP = 0.0844`

## Interpretation

The current evidence supports this explanation:

1. The old `0.000` result was partly caused by genuine wrapper bugs.
2. Those bugs were real and are now repaired.
3. After those repairs, UCN can train/evaluate into a non-zero regime. Official UCN pretraining keeps it in that non-zero regime, with slightly better `segm/AP` on this 1-epoch probe but not a clean overall win yet.
4. MSMFormer scratch training still collapses, which points to a deeper training-dynamics or architecture mismatch rather than just a bad wrapper default.
5. Official MSMFormer pretraining is sufficient to break out of the all-zero regime on ECC, both for eval-only transfer and short finetuning.

In other words, “MSMFormer is not an instance-segmentation model” is not supported by the new evidence. The more defensible explanation is that scratch training was unstable on ECC, while the official pretrained initialization restores viable instance predictions.

## Residual Risk

- UCN is still only validated with a short 1-epoch 256px canary, not a full canonical run.
- MSMFormer is only validated with eval-only transfer plus a 20-iter finetune probe, not yet a full paper-budget run.
- The ECC dataset currently has no explicit camera metadata bundled with each sample, so the XYZ conversion uses a centered-pinhole fallback geometry rather than measured intrinsics.
