#!/bin/bash
# Front A round-2: D2 concat in-house positive control @5K, subset-1000 protocol
set -u
GPU=$1
cd $HOME/magformer
OUT=$HOME/magformer/output/experiments/audit-control/concat5k_control
mkdir -p $OUT
CUDA_VISIBLE_DEVICES=$GPU nohup $HOME/miniconda3/envs/magformer/bin/python baselines/run_concat5k_control.py \
  -- \
  --config-file configs/coco/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml \
  MODEL.SEM_SEG_HEAD.NUM_CLASSES 1 \
  MODEL.BACKBONE.NAME D2SwinTransformer4CH \
  MODEL.PIXEL_MEAN "[123.675,116.280,103.530,125.628]" \
  MODEL.PIXEL_STD "[58.395,57.120,57.375,15.157]" \
  MODEL.WEIGHTS $HOME/magformer/output/pretrained/swin_t_4ch_init_for_rgbd_concat.pth \
  INPUT.FORMAT RGB \
  INPUT.DATASET_MAPPER_NAME coco_instance_lsj_rgbd \
  INPUT.IMAGE_SIZE 1024 \
  INPUT.MIN_SIZE_TRAIN "(640, 672, 704, 736, 768, 800)" \
  INPUT.MIN_SIZE_TEST 800 \
  DATASETS.TRAIN '("ecc20260318_1k_rgbd_train",)' \
  DATASETS.TEST '("ecc20260318_1k_rgbd_val_sub1000",)' \
  DATALOADER.NUM_WORKERS 2 \
  SOLVER.IMS_PER_BATCH 2 \
  SOLVER.BASE_LR 0.00005 \
  SOLVER.MAX_ITER 5000 \
  SOLVER.STEPS "(999999,)" \
  SOLVER.CHECKPOINT_PERIOD 2500 \
  TEST.EVAL_PERIOD 2500 \
  OUTPUT_DIR $OUT \
  > $OUT/console.log 2>&1 &
echo "pid=$! gpu=$GPU out=$OUT"
