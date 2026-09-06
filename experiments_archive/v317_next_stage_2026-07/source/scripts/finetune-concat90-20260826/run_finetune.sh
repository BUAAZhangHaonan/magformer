#!/bin/bash
# Front B: concat90 (model_0264999, full-val AP 90.6310@264979) D2-native finetune
# usage: run_finetune.sh <gpu> <lr> <tag>
set -u
GPU=$1; LR=$2; TAG=$3
cd $HOME/magformer
OUT=$HOME/magformer/output/experiments/finetune-concat90-20260826/ft_${TAG}
mkdir -p $OUT
CUDA_VISIBLE_DEVICES=$GPU nohup $HOME/miniconda3/envs/magformer/bin/python baselines/run_official_mask2former_rgbd_concat_ecc.py \
  -- \
  --config-file configs/coco/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml \
  MODEL.SEM_SEG_HEAD.NUM_CLASSES 1 \
  MODEL.BACKBONE.NAME D2SwinTransformer4CH \
  MODEL.PIXEL_MEAN "[123.675,116.280,103.530,125.628]" \
  MODEL.PIXEL_STD "[58.395,57.120,57.375,15.157]" \
  MODEL.WEIGHTS $HOME/magformer/output/experiments/baselines_v2/m2f_swin_t_rgbd_concat/model_0264999.pth \
  INPUT.FORMAT RGB \
  INPUT.DATASET_MAPPER_NAME coco_instance_lsj_rgbd \
  INPUT.IMAGE_SIZE 1024 \
  INPUT.MIN_SIZE_TRAIN "(640, 672, 704, 736, 768, 800)" \
  INPUT.MIN_SIZE_TEST 800 \
  DATASETS.TRAIN '("ecc20260318_1k_rgbd_train",)' \
  DATASETS.TEST '("ecc20260318_1k_rgbd_val",)' \
  DATALOADER.NUM_WORKERS 2 \
  SOLVER.IMS_PER_BATCH 2 \
  SOLVER.BASE_LR $LR \
  SOLVER.MAX_ITER 20000 \
  SOLVER.STEPS "(999999,)" \
  SOLVER.CHECKPOINT_PERIOD 5000 \
  TEST.EVAL_PERIOD 5000 \
  OUTPUT_DIR $OUT \
  > $OUT/console.log 2>&1 &
echo "pid=$! gpu=$GPU lr=$LR out=$OUT"
