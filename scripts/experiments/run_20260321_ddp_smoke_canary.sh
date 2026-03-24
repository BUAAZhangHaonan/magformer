#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"
source "${SCRIPT_DIR}/ecc_common.sh"

MODE="run"
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/20260321_ddp_smoke_canary"
REGISTER="20260321_ddp_smoke"
FAKE_DATASET_ROOT="${OUTPUT_ROOT}/_shared/fake_dataset"
HF_ENV_PREFIX="$(runner_hf_env_prefix)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      OUTPUT_ROOT="$2"
      FAKE_DATASET_ROOT="${OUTPUT_ROOT}/_shared/fake_dataset"
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

runner_log "${MODE}" "${RUN_LOG}" "[ddp-smoke] output_root=${OUTPUT_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[ddp-smoke] fake_dataset_root=${FAKE_DATASET_ROOT}"

runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/create_fake_dataset.py --output '${FAKE_DATASET_ROOT}' --num-images 8 --img-size 512"

MAG_OUT="${OUTPUT_ROOT}/magformer_ddp_nodpth_ref"
MAG_CFG="${MAG_OUT}/runtime.yaml"
runner_exec "${MODE}" "${RUN_LOG}" "mkdir -p '${MAG_OUT}'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/analysis/render_magformer_runtime_config.py \
  --base-config '${REPO_ROOT}/configs/magformer_0831_1k_20ep_1024_nodpth_ref.yaml' \
  --out-config '${MAG_CFG}' \
  --output-dir '${MAG_OUT}' \
  --run-name 'ddp_smoke_magformer' \
  --base-lr 0.00005 \
  --max-iter 2 \
  --steps '1,1' \
  --warmup-iters 0 \
  --ims-per-batch 2 \
  --eval-period 9999 \
  --checkpoint-period 9999 \
  --num-workers 0 \
  --dataset-root '${FAKE_DATASET_ROOT}' \
  --override runtime.ddp_enabled=true \
  --override runtime.gpus=[0,1] \
  --override runtime.find_unused_parameters=true \
  --override runtime.skip_depth_sanity=true \
  --override runtime.logger.type='tensorboard'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer torchrun --nproc_per_node=2 tools/train.py --config '${MAG_CFG}' --dataset-root '${FAKE_DATASET_ROOT}' --output-dir '${MAG_OUT}' --gpus 0,1 --num-workers 0"

MGM_OUT="${OUTPUT_ROOT}/mgm_ddp_depthnorm_on"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}/baselines/MGM_Mask2Former' && ${HF_ENV_PREFIX}conda run -n magformer python train_net_mgm_0831.py --num-gpus 2 --config-file '${REPO_ROOT}/baselines/MGM_Mask2Former/configs/mgm_swin_convnext_tiny.yaml' \
  INPUT.DATASET_ROOT '${FAKE_DATASET_ROOT}' OUTPUT_DIR '${MGM_OUT}' DDP.FIND_UNUSED_PARAMETERS True MODEL.FINETUNE_WEIGHTS '' MODEL.WEIGHTS '' \
  SOLVER.MAX_ITER 3 SOLVER.STEPS '(100,200)' SOLVER.WARMUP_ITERS 0 SOLVER.IMS_PER_BATCH 2 TEST.EVAL_PERIOD 9999 DATALOADER.NUM_WORKERS 0"

UOAIS_OUT="${OUTPUT_ROOT}/uoais_ddp_smoke"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python baselines/run_uoais_ecc.py \
  --register '${REGISTER}' --dataset-root '${FAKE_DATASET_ROOT}' --uoais-root '${REPO_ROOT}/baselines/uoais' -- \
  --num-gpus 1 --config-file '${REPO_ROOT}/configs/baselines/uoais_0831_1k_tracks.yaml' \
  SOLVER.MAX_ITER 3 SOLVER.STEPS '(100,200)' SOLVER.WARMUP_ITERS 0 SOLVER.CHECKPOINT_PERIOD 9999 TEST.EVAL_PERIOD 9999 DATALOADER.NUM_WORKERS 0 OUTPUT_DIR '${UOAIS_OUT}'"

YOLO_DATA_DIR="${OUTPUT_ROOT}/_shared/yolo_fake"
YOLO_OUT="${OUTPUT_ROOT}/yolov8n_ddp_smoke"
YOLO_MODEL="${REPO_ROOT}/output/pretrained/yolov8n-seg.pt"
if [[ ! -f "${YOLO_MODEL}" ]]; then
  YOLO_MODEL="yolov8n-seg.pt"
fi
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python baselines/ultralytics_tools/convert_coco_to_yolo_seg.py --dataset-root '${FAKE_DATASET_ROOT}' --output-root '${YOLO_DATA_DIR}'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer yolo segment train model='${YOLO_MODEL}' data='${YOLO_DATA_DIR}/dataset.yaml' imgsz=512 batch=4 epochs=1 device=0,1 workers=0 project='${YOLO_OUT}' name='train'"

UNET_OUT="${OUTPUT_ROOT}/unet_smp_smoke"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python baselines/run_unet_instance_ecc.py --dataset-root '${FAKE_DATASET_ROOT}' --output-dir '${UNET_OUT}' --variant smp_unet_mobilenetv2_boundary_inst --image-size 512 --epochs 1 --batch 2 --num-workers 0 --max-train-steps 2 --max-val-images 2"

runner_log "${MODE}" "${RUN_LOG}" "[ddp-smoke] done"
