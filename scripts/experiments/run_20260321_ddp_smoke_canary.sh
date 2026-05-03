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
MAGFORMER_ONLY=0
MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 20000))}"

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
    --magformer-only)
      MAGFORMER_ONLY=1
      shift
      ;;
    --master-port)
      MASTER_PORT="$2"
      shift 2
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
runner_log "${MODE}" "${RUN_LOG}" "[ddp-smoke] master_port=${MASTER_PORT}"

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
  --max-iter 1 \
  --steps '1,1' \
  --warmup-iters 0 \
  --ims-per-batch 2 \
  --eval-period 1 \
  --checkpoint-period 1 \
  --num-workers 0 \
  --dataset-root '${FAKE_DATASET_ROOT}' \
  --override runtime.ddp_enabled=true \
  --override runtime.gpus=[0,1] \
  --override runtime.find_unused_parameters=true \
  --override runtime.skip_depth_sanity=true \
  --override runtime.logger.type='tensorboard'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer torchrun --nproc_per_node=2 --master_port ${MASTER_PORT} tools/train.py --config '${MAG_CFG}' --dataset-root '${FAKE_DATASET_ROOT}' --output-dir '${MAG_OUT}' --gpus 0,1 --num-workers 0"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${FAKE_DATASET_ROOT}' --out-dir '${MAG_OUT}' --metrics-out 'metrics.cocoeval.json'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python -c \"from pathlib import Path; import json; out = Path(r'${MAG_OUT}'); metrics = json.loads((out / 'metrics.cocoeval.json').read_text(encoding='utf-8')); log_rows = [json.loads(line) for line in (out / 'metrics_log.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]; val_rows = [row for row in log_rows if row.get('phase') == 'val']; assert val_rows, 'missing validation rows'; last_val = val_rows[-1]; assert (out / 'checkpoint_iter_0000001.pth').exists(), 'missing checkpoint_iter_0000001.pth'; assert 'segm/AP' in metrics and metrics['segm/AP'] is not None, 'missing segm/AP'; assert 'val/diag_num_predictions' in last_val, 'missing validation diagnostics'; assert 'val/loss' not in json.dumps(last_val), 'unexpected val/loss in validation metrics'; print({'checkpoint': str(out / 'checkpoint_iter_0000001.pth'), 'diag_num_predictions': last_val['val/diag_num_predictions'], 'segm_AP': metrics['segm/AP']})\""
MAG_EVAL_CFG="${MAG_OUT}/eval_runtime.yaml"
MAG_EVAL_OUT="${OUTPUT_ROOT}/magformer_eval_from_config"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python -c \"from pathlib import Path; import yaml; cfg_path = Path(r'${MAG_CFG}'); out_cfg = Path(r'${MAG_EVAL_CFG}'); payload = yaml.safe_load(cfg_path.read_text(encoding='utf-8')); payload.setdefault('model', {})['weights'] = str(Path(r'${MAG_OUT}') / 'checkpoint_iter_0000001.pth'); payload.setdefault('runtime', {})['output_dir'] = str(Path(r'${MAG_EVAL_OUT}')); out_cfg.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding='utf-8')\""
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python tools/evaluate.py --config-file '${MAG_EVAL_CFG}' --dataset-root '${FAKE_DATASET_ROOT}' --output '${MAG_EVAL_OUT}' --batch-size 1 --num-workers 0"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${FAKE_DATASET_ROOT}' --out-dir '${MAG_EVAL_OUT}' --metrics-out 'metrics.cocoeval.json'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python -c \"from pathlib import Path; import json; out = Path(r'${MAG_EVAL_OUT}'); metrics = json.loads((out / 'metrics.cocoeval.json').read_text(encoding='utf-8')); assert (out / 'coco_instances_results.json').exists(), 'offline eval missing coco_instances_results.json'; assert 'segm/AP' in metrics and metrics['segm/AP'] is not None, 'offline eval missing segm/AP'; print({'offline_eval_dir': str(out), 'offline_segm_AP': metrics['segm/AP']})\""

if [[ "${MAGFORMER_ONLY}" == "1" ]]; then
  runner_log "${MODE}" "${RUN_LOG}" "[ddp-smoke] parity-token TEST.EVAL_PERIOD 1"
  runner_log "${MODE}" "${RUN_LOG}" "[ddp-smoke] magformer-only completed"
  exit 0
fi

MGM_OUT="${OUTPUT_ROOT}/mgm_ddp_depthnorm_on"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}/baselines/MGM_Mask2Former' && ${HF_ENV_PREFIX}conda run -n magformer python train_net_mgm_0831.py --num-gpus 2 --config-file '${REPO_ROOT}/baselines/MGM_Mask2Former/configs/mgm_swin_convnext_tiny.yaml' \
  INPUT.DATASET_ROOT '${FAKE_DATASET_ROOT}' OUTPUT_DIR '${MGM_OUT}' DDP.FIND_UNUSED_PARAMETERS True MODEL.FINETUNE_WEIGHTS '' MODEL.WEIGHTS '' \
  SOLVER.MAX_ITER 3 SOLVER.STEPS '(100,200)' SOLVER.WARMUP_ITERS 0 SOLVER.IMS_PER_BATCH 2 TEST.EVAL_PERIOD 1 DATALOADER.NUM_WORKERS 0"

UOAIS_OUT="${OUTPUT_ROOT}/uoais_ddp_smoke"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python baselines/run_uoais_ecc.py \
  --register '${REGISTER}' --dataset-root '${FAKE_DATASET_ROOT}' --uoais-root '${REPO_ROOT}/baselines/uoais' -- \
  --num-gpus 1 --config-file '${REPO_ROOT}/configs/baselines/uoais_0831_1k_tracks.yaml' \
  SOLVER.MAX_ITER 3 SOLVER.STEPS '(100,200)' SOLVER.WARMUP_ITERS 0 SOLVER.CHECKPOINT_PERIOD 9999 TEST.EVAL_PERIOD 1 DATALOADER.NUM_WORKERS 0 OUTPUT_DIR '${UOAIS_OUT}'"

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
