#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"
source "${SCRIPT_DIR}/ecc_common.sh"
HF_ENV_PREFIX="$(runner_hf_env_prefix)"

MODE="run"
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/20260406_smoke_matrix"
GPU="0"
DDP_GPUS="0,1"
REGISTER="20260321_ddp_smoke"
FAKE_DATASET_ROOT="${OUTPUT_ROOT}/_shared/fake_dataset"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      OUTPUT_ROOT="$2"
      FAKE_DATASET_ROOT="${OUTPUT_ROOT}/_shared/fake_dataset"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --ddp-gpus)
      DDP_GPUS="$2"
      shift 2
      ;;
    --run)
      MODE="run"
      shift
      ;;
    --dry-run)
      MODE="dry-run"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${OUTPUT_ROOT}"
RUN_LOG="$(runner_setup_log "${OUTPUT_ROOT}" "${MODE}")"

runner_log "${MODE}" "${RUN_LOG}" "[smoke-matrix] output_root=${OUTPUT_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[smoke-matrix] gpu=${GPU}"
runner_log "${MODE}" "${RUN_LOG}" "[smoke-matrix] ddp_gpus=${DDP_GPUS}"
runner_log "${MODE}" "${RUN_LOG}" "[smoke-matrix] fake_dataset_root=${FAKE_DATASET_ROOT}"

runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/create_fake_dataset.py --output '${FAKE_DATASET_ROOT}' --num-images 8 --img-size 512"

read -r DATASET_NAME_TRAIN DATASET_NAME_VAL < <(ecc_dataset_names_coco "${REGISTER}" "${FAKE_DATASET_ROOT}")
read -r RGBD_DATASET_NAME_TRAIN RGBD_DATASET_NAME_VAL < <(ecc_dataset_names_coco_rgbd "${REGISTER}" "${FAKE_DATASET_ROOT}")
if [[ "${MODE}" == "dry-run" ]]; then
  PIXEL_MEAN_RGB="[0.0,0.0,0.0]"
  PIXEL_STD_RGB="[1.0,1.0,1.0]"
  PIXEL_MEAN_BGR="[0.0,0.0,0.0]"
  PIXEL_STD_BGR="[1.0,1.0,1.0]"
  DEPTH_CLIP_MIN="0.0"
  DEPTH_CLIP_MAX="1.0"
else
  read -r PIXEL_MEAN_RGB PIXEL_STD_RGB < <(ecc_read_rgb_stats_rgb "${REGISTER}" "${FAKE_DATASET_ROOT}")
  read -r PIXEL_MEAN_BGR PIXEL_STD_BGR < <(ecc_read_rgb_stats_bgr "${REGISTER}" "${FAKE_DATASET_ROOT}")
  read -r DEPTH_CLIP_MIN DEPTH_CLIP_MAX < <(ecc_read_depth_clip "${REGISTER}" "${FAKE_DATASET_ROOT}")
fi

run_magformer_smoke() {
  local model_id="$1"
  local base_cfg="$2"
  shift 2
  local out_dir="${OUTPUT_ROOT}/group_a/${model_id}"
  local runtime_cfg="${out_dir}/runtime.yaml"
  local override_args="--override data.image_size=512"
  local ov=""
  for ov in "$@"; do
    override_args="${override_args} --override $(printf '%q' "${ov}")"
  done
  runner_exec "${MODE}" "${RUN_LOG}" "mkdir -p '${out_dir}'"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/analysis/render_magformer_runtime_config.py \
    --base-config '${base_cfg}' \
    --out-config '${runtime_cfg}' \
    --output-dir '${out_dir}' \
    --run-name 'smoke_${model_id}' \
    --base-lr 0.00005 \
    --max-iter 2 \
    --steps '1,1' \
    --warmup-iters 0 \
    --ims-per-batch 2 \
    --eval-period 2 \
    --checkpoint-period 2 \
    --num-workers 0 \
    --dataset-root '${FAKE_DATASET_ROOT}' \
    --override runtime.ddp_enabled=false \
    --override runtime.gpus=[0] \
    --override runtime.skip_depth_sanity=true \
    --override runtime.logger.type='tensorboard' \
    ${override_args}"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} ${HF_ENV_PREFIX}conda run -n magformer python tools/train.py --config '${runtime_cfg}' --dataset-root '${FAKE_DATASET_ROOT}' --output-dir '${out_dir}' --num-workers 0"
}

run_magformer_smoke \
  "magformer_depthnorm_on" \
  "${REPO_ROOT}/configs/magformer_0831_1k_20ep_1024_depthnorm_on.yaml"
run_magformer_smoke \
  "magformer_nodpth_ref_fair" \
  "${REPO_ROOT}/configs/magformer_0831_1k_20ep_1024_nodpth_ref_fair.yaml"
run_magformer_smoke \
  "magformer_lightdepth_convnextlite_spatialgate_edge_validhole" \
  "${REPO_ROOT}/configs/magformer_0831_1k_20ep_1024_lightdepth_convnextlite.yaml"
run_magformer_smoke \
  "magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole" \
  "${REPO_ROOT}/configs/magformer_0831_1k_20ep_1024_lightdepth_mobilenetv3.yaml" \
  "model.magformer.modality_fusion.mode=spatial_gate" \
  "model.magformer.modality_fusion.priors=[edge,valid-hole]" \
  "model.magformer.modality_fusion.prior.use_gradient=true" \
  "model.magformer.modality_fusion.prior.use_variance=false" \
  "model.magformer.modality_fusion.prior.use_valid_hole=true" \
  "model.magformer.modality_fusion.prior.use_rgb_edge=false"
run_magformer_smoke \
  "magformer_lightdepth_mobilenetv3_sagate_edge_validhole" \
  "${REPO_ROOT}/configs/magformer_0831_1k_20ep_1024_lightdepth_mobilenetv3.yaml" \
  "model.magformer.modality_fusion.mode=sa_gate" \
  "model.magformer.modality_fusion.priors=[edge,valid-hole]" \
  "model.magformer.modality_fusion.prior.use_gradient=true" \
  "model.magformer.modality_fusion.prior.use_variance=false" \
  "model.magformer.modality_fusion.prior.use_valid_hole=true" \
  "model.magformer.modality_fusion.prior.use_rgb_edge=false"

OFFICIAL_MASK2FORMER_OUT="${OUTPUT_ROOT}/group_b/official_mask2former_pretrained"
runner_exec "${MODE}" "${RUN_LOG}" "mkdir -p '${OFFICIAL_MASK2FORMER_OUT}'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} ${HF_ENV_PREFIX}conda run -n magformer python baselines/run_official_mask2former_ecc.py \
  --register '${REGISTER}' \
  --dataset-root '${FAKE_DATASET_ROOT}' \
  --mask2former-root '${REPO_ROOT}/baselines/Mask2Former' \
  -- \
  --num-gpus 1 \
  --config-file 'configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml' \
  OUTPUT_DIR '${OFFICIAL_MASK2FORMER_OUT}' \
  DATASETS.TRAIN \"('${DATASET_NAME_TRAIN}',)\" \
  DATASETS.TEST \"('${DATASET_NAME_VAL}',)\" \
  SOLVER.IMS_PER_BATCH 4 \
  SOLVER.BASE_LR 0.0001 \
  SOLVER.WARMUP_ITERS 0 \
  SOLVER.MAX_ITER 2 \
  SOLVER.STEPS '(100,200)' \
  TEST.EVAL_PERIOD 2 \
  SOLVER.CHECKPOINT_PERIOD 2 \
  INPUT.MIN_SIZE_TRAIN '(512,)' \
  INPUT.MAX_SIZE_TRAIN 512 \
  INPUT.MIN_SIZE_TEST 512 \
  INPUT.MAX_SIZE_TEST 512 \
  INPUT.MASK_FORMAT bitmask \
  MODEL.WEIGHTS '' \
  MODEL.PIXEL_MEAN '${PIXEL_MEAN_BGR}' \
  MODEL.PIXEL_STD '${PIXEL_STD_BGR}' \
  MODEL.SEM_SEG_HEAD.NUM_CLASSES 1 \
  MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100"

MASKRCNN_OUT="${OUTPUT_ROOT}/group_b/maskrcnn_pretrained"
runner_exec "${MODE}" "${RUN_LOG}" "mkdir -p '${MASKRCNN_OUT}'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} ${HF_ENV_PREFIX}conda run -n magformer python baselines/run_detectron2_ecc.py \
  --register '${REGISTER}' \
  --dataset-root '${FAKE_DATASET_ROOT}' \
  --detectron2-root '${REPO_ROOT}/baselines/detectron2' \
  -- \
  --num-gpus 1 \
  --config-file 'configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml' \
  OUTPUT_DIR '${MASKRCNN_OUT}' \
  DATASETS.TRAIN \"('${DATASET_NAME_TRAIN}',)\" \
  DATASETS.TEST \"('${DATASET_NAME_VAL}',)\" \
  SOLVER.IMS_PER_BATCH 4 \
  SOLVER.BASE_LR 0.0001 \
  SOLVER.WARMUP_ITERS 0 \
  SOLVER.MAX_ITER 2 \
  SOLVER.STEPS '(100,200)' \
  TEST.EVAL_PERIOD 2 \
  SOLVER.CHECKPOINT_PERIOD 2 \
  INPUT.MIN_SIZE_TRAIN '(512,)' \
  INPUT.MAX_SIZE_TRAIN 512 \
  INPUT.MIN_SIZE_TEST 512 \
  INPUT.MAX_SIZE_TEST 512 \
  INPUT.MASK_FORMAT bitmask \
  MODEL.WEIGHTS '' \
  MODEL.PIXEL_MEAN '${PIXEL_MEAN_BGR}' \
  MODEL.PIXEL_STD '${PIXEL_STD_BGR}' \
  MODEL.ROI_HEADS.NUM_CLASSES 1"

YOLO_DATA_DIR="${OUTPUT_ROOT}/_shared/yolo_fake"
YOLO_MODEL="${REPO_ROOT}/output/pretrained/yolov8n-seg.pt"
if [[ ! -f "${YOLO_MODEL}" ]]; then
  YOLO_MODEL="yolov8n-seg.pt"
fi
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python baselines/ultralytics_tools/convert_coco_to_yolo_seg.py --dataset-root '${FAKE_DATASET_ROOT}' --output-root '${YOLO_DATA_DIR}'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} ${HF_ENV_PREFIX}conda run -n magformer python baselines/run_yolo_seg_ecc.py \
  --model '${YOLO_MODEL}' \
  --data '${YOLO_DATA_DIR}/dataset.yaml' \
  --imgsz 512 \
  --batch 4 \
  --epochs 1 \
  --device '0' \
  --workers 0 \
  --pretrained 'True' \
  --lr0 0.01 \
  --warmup-epochs 0 \
  --cos-lr 'False' \
  --plots 'False' \
  --project '${OUTPUT_ROOT}/group_b/yolov8_seg_n_pretrained' \
  --name 'train' \
  --rgb-mean '${PIXEL_MEAN_RGB}' \
  --rgb-std '${PIXEL_STD_RGB}'"

run_mgm_smoke() {
  local model_id="$1"
  local depth_enabled="$2"
  local mgm_enabled="$3"
  local dpe_enabled="$4"
  local depth_per_sample_norm="$5"
  local out_dir="${OUTPUT_ROOT}/group_b/${model_id}"
  runner_exec "${MODE}" "${RUN_LOG}" "mkdir -p '${out_dir}'"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}/baselines/MGM_Mask2Former' && CUDA_VISIBLE_DEVICES=${GPU} ${HF_ENV_PREFIX}conda run -n magformer python train_net_mgm_0831.py --num-gpus 1 --config-file '${REPO_ROOT}/baselines/MGM_Mask2Former/configs/mgm_swin_convnext_tiny.yaml' \
    INPUT.DATASET_ROOT '${FAKE_DATASET_ROOT}' \
    OUTPUT_DIR '${out_dir}' \
    DDP.FIND_UNUSED_PARAMETERS False \
    MODEL.FINETUNE_WEIGHTS '' \
    MODEL.WEIGHTS '' \
    MODEL.DEPTH_BACKBONE.WEIGHTS '' \
    MODEL.PIXEL_MEAN '${PIXEL_MEAN_RGB}' \
    MODEL.PIXEL_STD '${PIXEL_STD_RGB}' \
    MODEL.DEPTH_BACKBONE.ENABLED ${depth_enabled} \
    MODEL.MGM.ENABLED ${mgm_enabled} \
    MODEL.DPE.ENABLED ${dpe_enabled} \
    INPUT.IMAGE_SIZE 512 \
    INPUT.MIN_SCALE 0.1 \
    INPUT.MAX_SCALE 2.0 \
    INPUT.RANDOM_FLIP 'horizontal' \
    INPUT.DEPTH_SCALE 1.0 \
    INPUT.DEPTH_SHIFT 0.0 \
    INPUT.DEPTH_CLIP_MIN ${DEPTH_CLIP_MIN} \
    INPUT.DEPTH_CLIP_MAX ${DEPTH_CLIP_MAX} \
    INPUT.DEPTH_NORM 'minmax' \
    INPUT.DEPTH_PER_SAMPLE_NORM ${depth_per_sample_norm} \
    INPUT.DEPTH_NOISE.ENABLED False \
    SOLVER.MAX_ITER 2 \
    SOLVER.STEPS '(100,200)' \
    SOLVER.BASE_LR 0.00005 \
    SOLVER.WARMUP_ITERS 0 \
    SOLVER.IMS_PER_BATCH 4 \
    SOLVER.CHECKPOINT_PERIOD 2 \
    TEST.EVAL_PERIOD 2 \
    DATALOADER.NUM_WORKERS 0 \
    MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100 \
    MODEL.MGM.PRIOR.COMPUTE_ON 'full'"
}

run_mgm_smoke "mgm_mask2former_depthnorm_on" "True" "True" "True" "True"
run_mgm_smoke "mgm_mask2former_nodpth_ref" "False" "False" "False" "False"

MSMFORMER_OUT="${OUTPUT_ROOT}/group_b/msmformer"
runner_exec "${MODE}" "${RUN_LOG}" "mkdir -p '${MSMFORMER_OUT}'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} conda run -n magformer python baselines/run_msmformer_ecc.py \
  --register '${REGISTER}' \
  --dataset-root '${FAKE_DATASET_ROOT}' \
  --msmformer-root '${REPO_ROOT}/baselines/msmformer/MSMFormer' \
  --pretrained none \
  -- \
  --num-gpus 1 \
  --config-file '${REPO_ROOT}/configs/baselines/msmformer_0831_1k_tracks.yaml' \
  DATASETS.TRAIN \"('${RGBD_DATASET_NAME_TRAIN}',)\" \
  DATASETS.TEST \"('${RGBD_DATASET_NAME_VAL}',)\" \
  MODEL.PIXEL_MEAN '${PIXEL_MEAN_BGR}' \
  MODEL.PIXEL_STD '${PIXEL_STD_BGR}' \
  MODEL.BACKBONE.FREEZE_AT 0 \
  SOLVER.MAX_ITER 2 \
  SOLVER.STEPS '(100,200)' \
  SOLVER.BASE_LR 0.0001 \
  SOLVER.WARMUP_ITERS 0 \
  SOLVER.IMS_PER_BATCH 4 \
  SOLVER.CHECKPOINT_PERIOD 2 \
  TEST.EVAL_PERIOD 2 \
  INPUT.MIN_SIZE_TRAIN '(512,)' \
  INPUT.MAX_SIZE_TRAIN 512 \
  INPUT.MIN_SIZE_TEST 512 \
  INPUT.MAX_SIZE_TEST 512 \
  OUTPUT_DIR '${MSMFORMER_OUT}'"

UCN_OUT="${OUTPUT_ROOT}/group_b/ucn"
runner_exec "${MODE}" "${RUN_LOG}" "mkdir -p '${UCN_OUT}'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} conda run -n magformer python baselines/run_ucn_ecc.py \
  --register '${REGISTER}' \
  --dataset-root '${FAKE_DATASET_ROOT}' \
  --output-dir '${UCN_OUT}' \
  --pretrained none \
  --epochs 1 \
  --batch 4 \
  --img-size 512 \
  --lr 0.00001 \
  --kappa 20 \
  --num-seeds 100"

UOAIS_OUT="${OUTPUT_ROOT}/group_b/uoais"
runner_exec "${MODE}" "${RUN_LOG}" "mkdir -p '${UOAIS_OUT}'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} ${HF_ENV_PREFIX}conda run -n magformer python baselines/run_uoais_ecc.py \
  --register '${REGISTER}' \
  --dataset-root '${FAKE_DATASET_ROOT}' \
  --uoais-root '${REPO_ROOT}/baselines/uoais' \
  -- \
  --num-gpus 1 \
  --config-file '${REPO_ROOT}/configs/baselines/uoais_0831_1k_tracks.yaml' \
  DATASETS.TRAIN \"('${RGBD_DATASET_NAME_TRAIN}',)\" \
  DATASETS.TEST \"('${RGBD_DATASET_NAME_VAL}',)\" \
  MODEL.PIXEL_MEAN '${PIXEL_MEAN_RGB}' \
  MODEL.PIXEL_STD '${PIXEL_STD_RGB}' \
  INPUT.DEPTH_RANGE '[${DEPTH_CLIP_MIN},${DEPTH_CLIP_MAX}]' \
  SOLVER.MAX_ITER 2 \
  SOLVER.STEPS '(100,200)' \
  SOLVER.BASE_LR 0.0001 \
  SOLVER.WARMUP_ITERS 0 \
  SOLVER.IMS_PER_BATCH 4 \
  SOLVER.CHECKPOINT_PERIOD 2 \
  TEST.EVAL_PERIOD 2 \
  INPUT.IMG_SIZE '(512,512)' \
  OUTPUT_DIR '${UOAIS_OUT}'"

run_unet_smoke() {
  local variant="$1"
  local out_dir="${OUTPUT_ROOT}/group_b/${variant}"
  runner_exec "${MODE}" "${RUN_LOG}" "mkdir -p '${out_dir}'"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} conda run -n magformer python baselines/run_unet_instance_ecc.py \
    --dataset-root '${FAKE_DATASET_ROOT}' \
    --output-dir '${out_dir}' \
    --variant '${variant}' \
    --image-size 512 \
    --epochs 1 \
    --batch 4 \
    --num-workers 0 \
    --max-train-steps 2 \
    --max-val-images 2 \
    --rgb-mean '${PIXEL_MEAN_RGB}' \
    --rgb-std '${PIXEL_STD_RGB}' \
    --depth-clip-min ${DEPTH_CLIP_MIN} \
    --depth-clip-max ${DEPTH_CLIP_MAX}"
}

run_unet_smoke "unet_boundary_inst"
run_unet_smoke "unet_semantic_inst"
run_unet_smoke "unetpp_boundary_inst"

runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${DDP_GPUS} bash '${REPO_ROOT}/scripts/experiments/run_20260321_ddp_smoke_canary.sh' --output-root '${OUTPUT_ROOT}/group_c/ddp_canary' --magformer-only --${MODE}"

runner_log "${MODE}" "${RUN_LOG}" "[smoke-matrix] done"
