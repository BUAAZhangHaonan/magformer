#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"
source "${SCRIPT_DIR}/ecc_common.sh"
HF_ENV_PREFIX="$(runner_hf_env_prefix)"

MODE="run"
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/20260406_baseline_repair_smokes"
REGISTER="20260321_ddp_smoke"
FAKE_DATASET_ROOT="${OUTPUT_ROOT}/_shared/fake_dataset"
SMOKE_GPU="${SMOKE_GPU:-0}"
DDP_GPUS="${DDP_GPUS:-0,1}"
MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 20000))}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      OUTPUT_ROOT="$2"
      FAKE_DATASET_ROOT="${OUTPUT_ROOT}/_shared/fake_dataset"
      shift 2
      ;;
    --smoke-gpu)
      SMOKE_GPU="$2"
      shift 2
      ;;
    --ddp-gpus)
      DDP_GPUS="$2"
      shift 2
      ;;
    --master-port)
      MASTER_PORT="$2"
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

runner_log "${MODE}" "${RUN_LOG}" "[baseline-repair-smokes] output_root=${OUTPUT_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[baseline-repair-smokes] fake_dataset_root=${FAKE_DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[baseline-repair-smokes] smoke_gpu=${SMOKE_GPU}"
runner_log "${MODE}" "${RUN_LOG}" "[baseline-repair-smokes] ddp_gpus=${DDP_GPUS}"
runner_log "${MODE}" "${RUN_LOG}" "[baseline-repair-smokes] master_port=${MASTER_PORT}"

runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/create_fake_dataset.py --output '${FAKE_DATASET_ROOT}' --num-images 8 --img-size 512"

read -r DATASET_NAME_TRAIN DATASET_NAME_VAL < <(ecc_dataset_names_coco "${REGISTER}" "${FAKE_DATASET_ROOT}")
read -r DATASET_NAME_TRAIN_RGBD DATASET_NAME_VAL_RGBD < <(ecc_dataset_names_coco_rgbd "${REGISTER}" "${FAKE_DATASET_ROOT}")
if [[ "${MODE}" == "run" ]]; then
  read -r PIXEL_MEAN_RGB PIXEL_STD_RGB < <(ecc_read_rgb_stats_rgb_for_dataset_root "${FAKE_DATASET_ROOT}")
  read -r PIXEL_MEAN_BGR PIXEL_STD_BGR < <(ecc_read_rgb_stats_bgr_for_dataset_root "${FAKE_DATASET_ROOT}")
  read -r PIXEL_MEAN_BGR6 PIXEL_STD_BGR6 < <(ecc_read_rgb_stats_bgr6_depth1275_for_dataset_root "${FAKE_DATASET_ROOT}")
  read -r DEPTH_CLIP_MIN DEPTH_CLIP_MAX < <(ecc_read_depth_clip_for_dataset_root "${FAKE_DATASET_ROOT}")
else
  PIXEL_MEAN_RGB='[123.6750,116.2800,103.5300]'
  PIXEL_STD_RGB='[58.3950,57.1200,57.3750]'
  PIXEL_MEAN_BGR='[103.5300,116.2800,123.6750]'
  PIXEL_STD_BGR='[57.3750,57.1200,58.3950]'
  PIXEL_MEAN_BGR6='[103.5300,116.2800,123.6750,127.5000,127.5000,127.5000]'
  PIXEL_STD_BGR6='[57.3750,57.1200,58.3950,127.5000,127.5000,127.5000]'
  DEPTH_CLIP_MIN='0.1'
  DEPTH_CLIP_MAX='1.0'
fi
DEPTH_RANGE="[${DEPTH_CLIP_MIN}, ${DEPTH_CLIP_MAX}]"

MASK2FORMER_ROOT="${REPO_ROOT}/baselines/Mask2Former"
DETECTRON2_ROOT="${REPO_ROOT}/baselines/detectron2"
MGM_ROOT="${REPO_ROOT}/baselines/MGM_Mask2Former"
MSMFORMER_ROOT="${REPO_ROOT}/baselines/msmformer/MSMFormer"
UOAIS_ROOT="${REPO_ROOT}/baselines/uoais"

MASK2FORMER_CFG_REL="configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
MASKRCNN_CFG_REL="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
MSMFORMER_CFG="${REPO_ROOT}/configs/baselines/msmformer_0831_1k_tracks.yaml"
UOAIS_CFG="${REPO_ROOT}/configs/baselines/uoais_0831_1k_tracks.yaml"
MGM_CFG="${MGM_ROOT}/configs/mgm_swin_convnext_tiny.yaml"

MASK2FORMER_WEIGHTS="${REPO_ROOT}/output/pretrained/model_final_3c8ec9_1class_v2.pth"
MASKRCNN_WEIGHTS="${REPO_ROOT}/output/pretrained/model_final_f10217.pkl"
MGM_RGB_WEIGHTS="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_tiny_bs16_50ep/model_final_86143f.pkl"
MGM_DEPTH_WEIGHTS="${REPO_ROOT}/output/pretrained/convnext_tiny_imagenet_in1_mgm_depth_backbone.pth"
YOLO_MODEL="${REPO_ROOT}/output/pretrained/yolov8n-seg.pt"
YOLO_DATA_DIR="${OUTPUT_ROOT}/_shared/yolo_fake"
YOLO_DATA_YAML="${YOLO_DATA_DIR}/dataset.yaml"
ANN_VAL="${FAKE_DATASET_ROOT}/annotations/instances_val.json"

verify_no_nan() {
  local out_dir="$1"
  runner_exec "${MODE}" "${RUN_LOG}" "python3 - <<'PY'
from pathlib import Path
import re
out_dir = Path(r'''${out_dir}''')
for name in ('run.log', 'log.txt'):
    path = out_dir / name
    if not path.exists():
        continue
    text = path.read_text(encoding='utf-8', errors='ignore')
    bad_patterns = [
        r'(?i)\\bloss\\s*[=:]\\s*nan\\b',
        r'(?i)contain inf/nan',
        r'(?i)training has diverged',
        r'(?i)floatingpointerror',
        r'(?i)\\bgrad(?:ient)?\\s*[=:]\\s*nan\\b',
    ]
    if any(re.search(pattern, text) for pattern in bad_patterns):
        raise SystemExit(f'divergence signature found in {path}')
PY"
}

verify_metrics_file() {
  local out_dir="$1"
  runner_exec "${MODE}" "${RUN_LOG}" "python3 - <<'PY'
from pathlib import Path
import json
out_dir = Path(r'''${out_dir}''')
metrics = out_dir / 'metrics.cocoeval.json'
if not metrics.exists():
    raise SystemExit(f'missing {metrics}')
payload = json.loads(metrics.read_text(encoding='utf-8'))
if not isinstance(payload, dict):
    raise SystemExit(f'invalid metrics payload: {metrics}')
PY"
}

prepare_out_dir() {
  local out_dir="$1"
  runner_exec "${MODE}" "${RUN_LOG}" "rm -rf '${out_dir}' && mkdir -p '${out_dir}'"
}

run_magformer_smoke() {
  local model_id="$1"
  local base_cfg="$2"
  shift 2
  local out="${OUTPUT_ROOT}/${model_id}"
  local runtime_cfg="${out}/runtime.yaml"
  prepare_out_dir "${out}"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/analysis/render_magformer_runtime_config.py \
    --base-config '${base_cfg}' \
    --out-config '${runtime_cfg}' \
    --output-dir '${out}' \
    --run-name 'smoke_${model_id}' \
    --base-lr 0.00005 \
    --max-iter 2 \
    --steps '1,1' \
    --warmup-iters 0 \
    --ims-per-batch 4 \
    --eval-period 2 \
    --checkpoint-period 2 \
    --num-workers 0 \
    --dataset-root '${FAKE_DATASET_ROOT}' \
    --override runtime.gpus=[0] \
    --override runtime.ddp_enabled=false \
    --override runtime.find_unused_parameters=false \
    --override runtime.skip_depth_sanity=true \
    --override runtime.log_period=1 \
    --override runtime.logger.type='tensorboard' \
    --override data.image_size=512 \
    $*"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES='${SMOKE_GPU}' ${HF_ENV_PREFIX}conda run -n magformer python tools/train.py \
    --config '${runtime_cfg}' \
    --dataset-root '${FAKE_DATASET_ROOT}' \
    --output-dir '${out}' \
    --num-workers 0"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/experiments/postprocess_cocoeval.py \
    --dataset-root '${FAKE_DATASET_ROOT}' \
    --out-dir '${out}' \
    --metrics-out 'metrics.cocoeval.json'"
  verify_no_nan "${out}"
  verify_metrics_file "${out}"
}

run_magformer_smoke "magformer_depthnorm_on" \
  "${REPO_ROOT}/configs/magformer_0831_1k_20ep_1024_depthnorm_on.yaml"
run_magformer_smoke "magformer_nodpth_ref_fair" \
  "${REPO_ROOT}/configs/magformer_0831_1k_20ep_1024_nodpth_ref_fair.yaml"
run_magformer_smoke "magformer_lightdepth_convnextlite_spatialgate_edge_validhole" \
  "${REPO_ROOT}/configs/magformer_0831_1k_20ep_1024_lightdepth_convnextlite.yaml"
run_magformer_smoke "magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole" \
  "${REPO_ROOT}/configs/magformer_0831_1k_20ep_1024_lightdepth_mobilenetv3.yaml" \
  "--override model.magformer.modality_fusion.mode='spatial_gate'"
run_magformer_smoke "magformer_lightdepth_mobilenetv3_sagate_edge_validhole" \
  "${REPO_ROOT}/configs/magformer_0831_1k_20ep_1024_lightdepth_mobilenetv3.yaml" \
  "--override model.magformer.modality_fusion.mode='sa_gate'"

OFFICIAL_OUT="${OUTPUT_ROOT}/official_mask2former_pretrained"
prepare_out_dir "${OFFICIAL_OUT}"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES='${SMOKE_GPU}' ${HF_ENV_PREFIX}conda run -n magformer python baselines/run_official_mask2former_ecc.py \
  --register '${REGISTER}' \
  --dataset-root '${FAKE_DATASET_ROOT}' \
  --mask2former-root '${MASK2FORMER_ROOT}' \
  -- \
  --num-gpus 1 \
  --config-file '${MASK2FORMER_CFG_REL}' \
  OUTPUT_DIR '${OFFICIAL_OUT}' \
  DATASETS.TRAIN \"('${DATASET_NAME_TRAIN}',)\" \
  DATASETS.TEST \"('${DATASET_NAME_VAL}',)\" \
  SOLVER.IMS_PER_BATCH 4 \
  SOLVER.BASE_LR 0.0001 \
  SOLVER.WARMUP_ITERS 0 \
  SOLVER.MAX_ITER 2 \
  SOLVER.STEPS '(1,)' \
  SOLVER.CHECKPOINT_PERIOD 2 \
  TEST.EVAL_PERIOD 2 \
  DATALOADER.NUM_WORKERS 0 \
  INPUT.MIN_SIZE_TRAIN '(512,)' \
  INPUT.MAX_SIZE_TRAIN 512 \
  INPUT.MIN_SIZE_TEST 512 \
  INPUT.MAX_SIZE_TEST 512 \
  INPUT.MASK_FORMAT bitmask \
  MODEL.WEIGHTS '${MASK2FORMER_WEIGHTS}' \
  MODEL.PIXEL_MEAN '${PIXEL_MEAN_BGR}' \
  MODEL.PIXEL_STD '${PIXEL_STD_BGR}' \
  MODEL.SEM_SEG_HEAD.NUM_CLASSES 1 \
  MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${FAKE_DATASET_ROOT}' --out-dir '${OFFICIAL_OUT}' --metrics-out 'metrics.cocoeval.json'"
verify_no_nan "${OFFICIAL_OUT}"
verify_metrics_file "${OFFICIAL_OUT}"

MASKRCNN_OUT="${OUTPUT_ROOT}/maskrcnn_pretrained"
prepare_out_dir "${MASKRCNN_OUT}"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES='${SMOKE_GPU}' ${HF_ENV_PREFIX}conda run -n magformer python baselines/run_detectron2_ecc.py \
  --register '${REGISTER}' \
  --dataset-root '${FAKE_DATASET_ROOT}' \
  --detectron2-root '${DETECTRON2_ROOT}' \
  -- \
  --num-gpus 1 \
  --config-file '${MASKRCNN_CFG_REL}' \
  OUTPUT_DIR '${MASKRCNN_OUT}' \
  DATASETS.TRAIN \"('${DATASET_NAME_TRAIN}',)\" \
  DATASETS.TEST \"('${DATASET_NAME_VAL}',)\" \
  SOLVER.IMS_PER_BATCH 4 \
  SOLVER.BASE_LR 0.01 \
  SOLVER.WARMUP_ITERS 0 \
  SOLVER.MAX_ITER 2 \
  SOLVER.STEPS '(1,)' \
  SOLVER.CHECKPOINT_PERIOD 2 \
  TEST.EVAL_PERIOD 2 \
  DATALOADER.NUM_WORKERS 0 \
  INPUT.MIN_SIZE_TRAIN '(512,)' \
  INPUT.MAX_SIZE_TRAIN 512 \
  INPUT.MIN_SIZE_TEST 512 \
  INPUT.MAX_SIZE_TEST 512 \
  INPUT.MASK_FORMAT bitmask \
  MODEL.WEIGHTS '${MASKRCNN_WEIGHTS}' \
  MODEL.PIXEL_MEAN '${PIXEL_MEAN_BGR}' \
  MODEL.PIXEL_STD '${PIXEL_STD_BGR}' \
  MODEL.ROI_HEADS.NUM_CLASSES 1"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${FAKE_DATASET_ROOT}' --out-dir '${MASKRCNN_OUT}' --metrics-out 'metrics.cocoeval.json'"
verify_no_nan "${MASKRCNN_OUT}"
verify_metrics_file "${MASKRCNN_OUT}"

if [[ ! -f "${YOLO_DATA_YAML}" ]]; then
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python baselines/ultralytics_tools/convert_coco_to_yolo_seg.py --dataset-root '${FAKE_DATASET_ROOT}' --output-root '${YOLO_DATA_DIR}'"
fi
YOLO_OUT="${OUTPUT_ROOT}/yolov8_seg_n_pretrained"
prepare_out_dir "${YOLO_OUT}"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES='${SMOKE_GPU}' ${HF_ENV_PREFIX}conda run -n magformer python baselines/run_yolo_seg_ecc.py \
  --model '${YOLO_MODEL}' \
  --data '${YOLO_DATA_YAML}' \
  --imgsz 512 \
  --batch 4 \
  --epochs 1 \
  --device 0 \
  --workers 0 \
  --project '${YOLO_OUT}' \
  --name 'train' \
  --pretrained 'True' \
  --lr0 0.01 \
  --warmup-epochs 0 \
  --cos-lr 'False' \
  --plots 'False' \
  --rgb-mean '${PIXEL_MEAN_RGB}' \
  --rgb-std '${PIXEL_STD_RGB}'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python baselines/yolo_export_coco.py --dataset-root '${FAKE_DATASET_ROOT}' --ann-file '${ANN_VAL}' --split val --output-json '${YOLO_OUT}/coco_instances_results.json'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${FAKE_DATASET_ROOT}' --out-dir '${YOLO_OUT}' --metrics-out 'metrics.cocoeval.json'"
verify_no_nan "${YOLO_OUT}"
verify_metrics_file "${YOLO_OUT}"

MGM_DEPTH_OUT="${OUTPUT_ROOT}/mgm_mask2former_depthnorm_on"
prepare_out_dir "${MGM_DEPTH_OUT}"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${MGM_ROOT}' && CUDA_VISIBLE_DEVICES='${SMOKE_GPU}' ${HF_ENV_PREFIX}conda run -n magformer python train_net_mgm_0831.py --num-gpus 1 --config-file '${MGM_CFG}' \
  INPUT.DATASET_ROOT '${FAKE_DATASET_ROOT}' \
  OUTPUT_DIR '${MGM_DEPTH_OUT}' \
  DDP.FIND_UNUSED_PARAMETERS False \
  MODEL.FINETUNE_WEIGHTS '' \
  MODEL.WEIGHTS '${MGM_RGB_WEIGHTS}' \
  MODEL.DEPTH_BACKBONE.WEIGHTS '${MGM_DEPTH_WEIGHTS}' \
  MODEL.PIXEL_MEAN '${PIXEL_MEAN_RGB}' \
  MODEL.PIXEL_STD '${PIXEL_STD_RGB}' \
  MODEL.DEPTH_BACKBONE.ENABLED True \
  MODEL.MGM.ENABLED True \
  MODEL.DPE.ENABLED True \
  INPUT.IMAGE_SIZE 512 \
  INPUT.MIN_SCALE 0.1 \
  INPUT.MAX_SCALE 2.0 \
  INPUT.RANDOM_FLIP 'horizontal' \
  INPUT.DEPTH_SCALE 1.0 \
  INPUT.DEPTH_SHIFT 0.0 \
  INPUT.DEPTH_CLIP_MIN ${DEPTH_CLIP_MIN} \
  INPUT.DEPTH_CLIP_MAX ${DEPTH_CLIP_MAX} \
  INPUT.DEPTH_NORM 'minmax' \
  INPUT.DEPTH_PER_SAMPLE_NORM True \
  INPUT.DEPTH_NOISE.ENABLED False \
  SOLVER.MAX_ITER 2 \
  SOLVER.STEPS '(1,)' \
  SOLVER.BASE_LR 0.00005 \
  SOLVER.WARMUP_ITERS 0 \
  SOLVER.IMS_PER_BATCH 4 \
  SOLVER.CHECKPOINT_PERIOD 2 \
  TEST.EVAL_PERIOD 2 \
  DATALOADER.NUM_WORKERS 0 \
  MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100 \
  MODEL.MGM.PRIOR.COMPUTE_ON 'full'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${FAKE_DATASET_ROOT}' --out-dir '${MGM_DEPTH_OUT}' --metrics-out 'metrics.cocoeval.json'"
verify_no_nan "${MGM_DEPTH_OUT}"
verify_metrics_file "${MGM_DEPTH_OUT}"

MGM_RGB_OUT="${OUTPUT_ROOT}/mgm_mask2former_nodpth_ref"
prepare_out_dir "${MGM_RGB_OUT}"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${MGM_ROOT}' && CUDA_VISIBLE_DEVICES='${SMOKE_GPU}' ${HF_ENV_PREFIX}conda run -n magformer python train_net_mgm_0831.py --num-gpus 1 --config-file '${MGM_CFG}' \
  INPUT.DATASET_ROOT '${FAKE_DATASET_ROOT}' \
  OUTPUT_DIR '${MGM_RGB_OUT}' \
  DDP.FIND_UNUSED_PARAMETERS False \
  MODEL.FINETUNE_WEIGHTS '' \
  MODEL.WEIGHTS '${MGM_RGB_WEIGHTS}' \
  MODEL.DEPTH_BACKBONE.WEIGHTS '${MGM_DEPTH_WEIGHTS}' \
  MODEL.PIXEL_MEAN '${PIXEL_MEAN_RGB}' \
  MODEL.PIXEL_STD '${PIXEL_STD_RGB}' \
  MODEL.DEPTH_BACKBONE.ENABLED False \
  MODEL.MGM.ENABLED False \
  MODEL.DPE.ENABLED False \
  INPUT.IMAGE_SIZE 512 \
  INPUT.MIN_SCALE 0.1 \
  INPUT.MAX_SCALE 2.0 \
  INPUT.RANDOM_FLIP 'horizontal' \
  INPUT.DEPTH_SCALE 1.0 \
  INPUT.DEPTH_SHIFT 0.0 \
  INPUT.DEPTH_CLIP_MIN ${DEPTH_CLIP_MIN} \
  INPUT.DEPTH_CLIP_MAX ${DEPTH_CLIP_MAX} \
  INPUT.DEPTH_NORM 'minmax' \
  INPUT.DEPTH_PER_SAMPLE_NORM False \
  INPUT.DEPTH_NOISE.ENABLED False \
  SOLVER.MAX_ITER 2 \
  SOLVER.STEPS '(1,)' \
  SOLVER.BASE_LR 0.00005 \
  SOLVER.WARMUP_ITERS 0 \
  SOLVER.IMS_PER_BATCH 4 \
  SOLVER.CHECKPOINT_PERIOD 2 \
  TEST.EVAL_PERIOD 2 \
  DATALOADER.NUM_WORKERS 0 \
  MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100 \
  MODEL.MGM.PRIOR.COMPUTE_ON 'full'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${FAKE_DATASET_ROOT}' --out-dir '${MGM_RGB_OUT}' --metrics-out 'metrics.cocoeval.json'"
verify_no_nan "${MGM_RGB_OUT}"
verify_metrics_file "${MGM_RGB_OUT}"

MSM_OUT="${OUTPUT_ROOT}/msmformer_scratch"
prepare_out_dir "${MSM_OUT}"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES='${SMOKE_GPU}' conda run -n magformer python baselines/run_msmformer_ecc.py \
  --register '${REGISTER}' \
  --dataset-root '${FAKE_DATASET_ROOT}' \
  --msmformer-root '${MSMFORMER_ROOT}' \
  -- \
  --num-gpus 1 \
  --config-file '${MSMFORMER_CFG}' \
  DATASETS.TRAIN \"('${DATASET_NAME_TRAIN_RGBD}',)\" \
  DATASETS.TEST \"('${DATASET_NAME_VAL_RGBD}',)\" \
  MODEL.PIXEL_MEAN '${PIXEL_MEAN_BGR}' \
  MODEL.PIXEL_STD '${PIXEL_STD_BGR}' \
  MODEL.BACKBONE.FREEZE_AT 0 \
  SOLVER.MAX_ITER 2 \
  SOLVER.STEPS '(1,)' \
  SOLVER.BASE_LR 0.0001 \
  SOLVER.WARMUP_ITERS 0 \
  SOLVER.IMS_PER_BATCH 4 \
  SOLVER.CHECKPOINT_PERIOD 2 \
  TEST.EVAL_PERIOD 2 \
  DATALOADER.NUM_WORKERS 0 \
  INPUT.MIN_SIZE_TRAIN '(512,)' \
  INPUT.MAX_SIZE_TRAIN 512 \
  INPUT.MIN_SIZE_TEST 512 \
  INPUT.MAX_SIZE_TEST 512 \
  OUTPUT_DIR '${MSM_OUT}'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${FAKE_DATASET_ROOT}' --out-dir '${MSM_OUT}' --metrics-out 'metrics.cocoeval.json'"
verify_no_nan "${MSM_OUT}"
verify_metrics_file "${MSM_OUT}"

UCN_OUT="${OUTPUT_ROOT}/ucn_scratch"
prepare_out_dir "${UCN_OUT}"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES='${SMOKE_GPU}' conda run -n magformer python baselines/run_ucn_ecc.py \
  --register '${REGISTER}' \
  --dataset-root '${FAKE_DATASET_ROOT}' \
  --output-dir '${UCN_OUT}' \
  --epochs 1 \
  --batch 4 \
  --img-size 512"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${FAKE_DATASET_ROOT}' --out-dir '${UCN_OUT}' --metrics-out 'metrics.cocoeval.json'"
verify_no_nan "${UCN_OUT}"
verify_metrics_file "${UCN_OUT}"

UOAIS_OUT="${OUTPUT_ROOT}/uoais_scratch"
prepare_out_dir "${UOAIS_OUT}"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES='${SMOKE_GPU}' ${HF_ENV_PREFIX}conda run -n magformer python baselines/run_uoais_ecc.py \
  --register '${REGISTER}' \
  --dataset-root '${FAKE_DATASET_ROOT}' \
  --uoais-root '${UOAIS_ROOT}' \
  -- \
  --num-gpus 1 \
  --config-file '${UOAIS_CFG}' \
  DATASETS.TRAIN \"('${DATASET_NAME_TRAIN_RGBD}',)\" \
  DATASETS.TEST \"('${DATASET_NAME_VAL_RGBD}',)\" \
  MODEL.PIXEL_MEAN '${PIXEL_MEAN_BGR6}' \
  MODEL.PIXEL_STD '${PIXEL_STD_BGR6}' \
  INPUT.DEPTH_RANGE '${DEPTH_RANGE}' \
  SOLVER.MAX_ITER 2 \
  SOLVER.STEPS '(1,)' \
  SOLVER.BASE_LR 0.0001 \
  SOLVER.WARMUP_ITERS 0 \
  SOLVER.IMS_PER_BATCH 4 \
  SOLVER.CHECKPOINT_PERIOD 2 \
  TEST.EVAL_PERIOD 2 \
  INPUT.IMG_SIZE '(512,512)' \
  OUTPUT_DIR '${UOAIS_OUT}'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${FAKE_DATASET_ROOT}' --out-dir '${UOAIS_OUT}' --metrics-out 'metrics.cocoeval.json'"
verify_no_nan "${UOAIS_OUT}"
verify_metrics_file "${UOAIS_OUT}"

for variant in unet_boundary_inst unet_semantic_inst unetpp_boundary_inst; do
  out="${OUTPUT_ROOT}/${variant}"
  prepare_out_dir "${out}"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES='${SMOKE_GPU}' conda run -n magformer python baselines/run_unet_instance_ecc.py \
    --dataset-root '${FAKE_DATASET_ROOT}' \
    --output-dir '${out}' \
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
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${FAKE_DATASET_ROOT}' --out-dir '${out}' --metrics-out 'metrics.cocoeval.json'"
  verify_no_nan "${out}"
  verify_metrics_file "${out}"
done

runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES='${DDP_GPUS}' MASTER_PORT='${MASTER_PORT}' bash scripts/experiments/run_20260321_ddp_smoke_canary.sh --output-root '${OUTPUT_ROOT}/ddp_canary' --magformer-only --run"

runner_log "${MODE}" "${RUN_LOG}" "[baseline-repair-smokes] done"
