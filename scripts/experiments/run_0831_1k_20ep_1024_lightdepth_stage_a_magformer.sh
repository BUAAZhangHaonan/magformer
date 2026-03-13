#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"
source "${SCRIPT_DIR}/ecc_common.sh"

DATASET_ROOT="${PROJECT_ROOT}/magformer_datasets/0831_1K"
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/0831_1k_20ep_1024_lightdepth_stage_a"
MODE="run"
SMOKE=0
VARIANT="mobilenetv3_gatedadd_edge"
NUM_WORKERS=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    --smoke)
      SMOKE=1
      shift
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

CFG_BASE=""
MODEL_ID=""
declare -a EXTRA_OVERRIDES=()

case "${VARIANT}" in
  mobilenetv3_directadd_edge)
    MODEL_ID="magformer_lightdepth_mobilenetv3_directadd_edge"
    CFG_BASE="${REPO_ROOT}/configs/magformer_0831_1k_20ep_1024_lightdepth_mobilenetv3.yaml"
    EXTRA_OVERRIDES+=(
      "model.magformer.modality_fusion.mode=direct_add"
      "model.magformer.modality_fusion.priors=[edge]"
      "model.magformer.modality_fusion.prior.use_gradient=true"
      "model.magformer.modality_fusion.prior.use_variance=false"
      "model.magformer.modality_fusion.prior.use_valid_hole=false"
      "model.magformer.modality_fusion.prior.use_rgb_edge=false"
    )
    ;;
  mobilenetv3_gatedadd_edge)
    MODEL_ID="magformer_lightdepth_mobilenetv3_gatedadd_edge"
    CFG_BASE="${REPO_ROOT}/configs/magformer_0831_1k_20ep_1024_lightdepth_mobilenetv3.yaml"
    EXTRA_OVERRIDES+=(
      "model.magformer.modality_fusion.mode=gated_add"
      "model.magformer.modality_fusion.priors=[edge]"
      "model.magformer.modality_fusion.prior.use_gradient=true"
      "model.magformer.modality_fusion.prior.use_variance=false"
      "model.magformer.modality_fusion.prior.use_valid_hole=false"
      "model.magformer.modality_fusion.prior.use_rgb_edge=false"
    )
    ;;
  resnet18_gatedadd_edge)
    MODEL_ID="magformer_lightdepth_resnet18_gatedadd_edge"
    CFG_BASE="${REPO_ROOT}/configs/magformer_0831_1k_20ep_1024_lightdepth_resnet18.yaml"
    EXTRA_OVERRIDES+=(
      "model.magformer.modality_fusion.mode=gated_add"
      "model.magformer.modality_fusion.priors=[edge]"
      "model.magformer.modality_fusion.prior.use_gradient=true"
      "model.magformer.modality_fusion.prior.use_variance=false"
      "model.magformer.modality_fusion.prior.use_valid_hole=false"
      "model.magformer.modality_fusion.prior.use_rgb_edge=false"
    )
    ;;
  mobilenetv3_film_edge_validhole)
    MODEL_ID="magformer_lightdepth_mobilenetv3_film_edge_validhole"
    CFG_BASE="${REPO_ROOT}/configs/magformer_0831_1k_20ep_1024_lightdepth_film.yaml"
    EXTRA_OVERRIDES+=(
      "model.magformer.modality_fusion.mode=film"
      "model.magformer.modality_fusion.priors=[edge,valid-hole]"
      "model.magformer.modality_fusion.prior.use_gradient=true"
      "model.magformer.modality_fusion.prior.use_variance=false"
      "model.magformer.modality_fusion.prior.use_valid_hole=true"
      "model.magformer.modality_fusion.prior.use_rgb_edge=false"
    )
    ;;
  mobilenetv3_channelattn_edge)
    MODEL_ID="magformer_lightdepth_mobilenetv3_channelattn_edge"
    CFG_BASE="${REPO_ROOT}/configs/magformer_0831_1k_20ep_1024_lightdepth_mobilenetv3.yaml"
    EXTRA_OVERRIDES+=(
      "model.magformer.modality_fusion.mode=channel_attn"
      "model.magformer.modality_fusion.priors=[edge]"
      "model.magformer.modality_fusion.prior.use_gradient=true"
      "model.magformer.modality_fusion.prior.use_variance=false"
      "model.magformer.modality_fusion.prior.use_valid_hole=false"
      "model.magformer.modality_fusion.prior.use_rgb_edge=false"
    )
    ;;
  mobilenetv3_spatialgate_edge_validhole)
    MODEL_ID="magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole"
    CFG_BASE="${REPO_ROOT}/configs/magformer_0831_1k_20ep_1024_lightdepth_mobilenetv3.yaml"
    EXTRA_OVERRIDES+=(
      "model.magformer.modality_fusion.mode=spatial_gate"
      "model.magformer.modality_fusion.priors=[edge,valid-hole]"
      "model.magformer.modality_fusion.prior.use_gradient=true"
      "model.magformer.modality_fusion.prior.use_variance=false"
      "model.magformer.modality_fusion.prior.use_valid_hole=true"
      "model.magformer.modality_fusion.prior.use_rgb_edge=false"
    )
    ;;
  mobilenetv3_sagate_edge_validhole)
    MODEL_ID="magformer_lightdepth_mobilenetv3_sagate_edge_validhole"
    CFG_BASE="${REPO_ROOT}/configs/magformer_0831_1k_20ep_1024_lightdepth_mobilenetv3.yaml"
    EXTRA_OVERRIDES+=(
      "model.magformer.modality_fusion.mode=sa_gate"
      "model.magformer.modality_fusion.priors=[edge,valid-hole]"
      "model.magformer.modality_fusion.prior.use_gradient=true"
      "model.magformer.modality_fusion.prior.use_variance=false"
      "model.magformer.modality_fusion.prior.use_valid_hole=true"
      "model.magformer.modality_fusion.prior.use_rgb_edge=false"
    )
    ;;
  mobilenetv3_esanetctx_edge_validhole)
    MODEL_ID="magformer_lightdepth_mobilenetv3_esanetctx_edge_validhole"
    CFG_BASE="${REPO_ROOT}/configs/magformer_0831_1k_20ep_1024_lightdepth_mobilenetv3.yaml"
    EXTRA_OVERRIDES+=(
      "model.magformer.modality_fusion.mode=esanet_ctx"
      "model.magformer.modality_fusion.priors=[edge,valid-hole]"
      "model.magformer.modality_fusion.prior.use_gradient=true"
      "model.magformer.modality_fusion.prior.use_variance=false"
      "model.magformer.modality_fusion.prior.use_valid_hole=true"
      "model.magformer.modality_fusion.prior.use_rgb_edge=false"
    )
    ;;
  *)
    echo "Unsupported --variant: ${VARIANT}" >&2
    exit 1
    ;;
esac

OUT="${OUTPUT_ROOT}/${MODEL_ID}"
mkdir -p "${OUT}/visualizations"
OUT="$(cd "${OUT}" && pwd)"
DATASET_ROOT="$(cd "${DATASET_ROOT}" && pwd)"
RUN_LOG="$(runner_setup_log "${OUT}" "${MODE}")"

runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-a-magformer] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-a-magformer] smoke=${SMOKE}"
runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-a-magformer] variant=${VARIANT}"
runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-a-magformer] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-a-magformer] output_dir=${OUT}"

IMS_PER_BATCH=4
EPOCHS=20
BASE_LR="0.00005"
if [[ "${SMOKE}" == "1" ]]; then
  IMS_PER_BATCH=1
  EPOCHS=1
  NUM_WORKERS=2
fi

compute_budget() {
  local ims_per_batch="$1"
  local epochs="$2"
  local num_images
  num_images="$(ecc_num_train_images "${DATASET_ROOT}")"
  local iters_per_epoch
  iters_per_epoch="$(ecc_iters_per_epoch "${num_images}" "${ims_per_batch}")"
  local max_iter=$(( iters_per_epoch * epochs ))
  local step1=$(( max_iter * 8 / 10 ))
  local step2=$(( max_iter * 9 / 10 ))
  local warmup="${iters_per_epoch}"
  echo "${iters_per_epoch} ${max_iter} ${step1},${step2} ${warmup} ${iters_per_epoch} ${iters_per_epoch}"
}

read -r ITERS_PER_EPOCH MAX_ITER STEPS WARMUP_ITERS EVAL_PERIOD CHECKPOINT_PERIOD < <(compute_budget "${IMS_PER_BATCH}" "${EPOCHS}")
if [[ "${SMOKE}" == "1" ]]; then
  MAX_ITER=20
  STEPS="15,18"
  WARMUP_ITERS=10
  EVAL_PERIOD=10
  CHECKPOINT_PERIOD=10
fi

METADATA_ARGS=(
  bash
  "$(basename "${BASH_SOURCE[0]}")"
  --dataset-root
  "${DATASET_ROOT}"
  --output-root
  "${OUTPUT_ROOT}"
  --variant
  "${VARIANT}"
)
if [[ "${MODE}" == "run" ]]; then
  METADATA_ARGS+=(--run)
else
  METADATA_ARGS+=(--dry-run)
fi
if [[ "${SMOKE}" == "1" ]]; then
  METADATA_ARGS+=(--smoke)
fi
for ov in "${EXTRA_OVERRIDES[@]}"; do
  METADATA_ARGS+=(--override "${ov}")
done
METADATA_CMD="$(printf "%q " "${METADATA_ARGS[@]}")"
METADATA_CMD="${METADATA_CMD% }"

runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/write_run_metadata.py \
  --phase start \
  --out-dir '${OUT}' \
  --track 0831_1k_20ep_1024_lightdepth_stage_a \
  --register '0831' \
  --dataset-root '${DATASET_ROOT}' \
  --model-id '${MODEL_ID}' \
  --candidate-id '${VARIANT}' \
  --run-tag 'final' \
  --command \"${METADATA_CMD}\" \
  --iters-per-epoch ${ITERS_PER_EPOCH} \
  --max-iter ${MAX_ITER} \
  --epochs ${EPOCHS} \
  --ims-per-batch ${IMS_PER_BATCH}"

RUNTIME_CFG="${OUT}/magformer_runtime_config.yaml"
RUN_NAME="0831_1k_20ep_1024_${VARIANT}"

render_cfg() {
  local ims_per_batch="$1"
  local max_iter="$2"
  local steps="$3"
  local warmup_iters="$4"
  local eval_period="$5"
  local checkpoint_period="$6"
  local override_args=""
  for ov in "${EXTRA_OVERRIDES[@]}"; do
    override_args="${override_args} --override '${ov}'"
  done
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/render_magformer_runtime_config.py \
    --base-config '${CFG_BASE}' \
    --out-config '${RUNTIME_CFG}' \
    --output-dir '${OUT}' \
    --run-name '${RUN_NAME}' \
    --base-lr ${BASE_LR} \
    --max-iter ${max_iter} \
    --steps '${steps}' \
    --warmup-iters ${warmup_iters} \
    --ims-per-batch ${ims_per_batch} \
    --eval-period ${eval_period} \
    --checkpoint-period ${checkpoint_period} \
    --num-workers ${NUM_WORKERS} \
    ${override_args}"
}

run_train_once() {
  local cmd="cd '${REPO_ROOT}' && conda run -n magformer python tools/train.py --config '${RUNTIME_CFG}' --dataset-root '${DATASET_ROOT}' --output-dir '${OUT}' --num-workers ${NUM_WORKERS}"
  runner_log "${MODE}" "${RUN_LOG}" "+ ${cmd}"
  if [[ "${MODE}" != "run" ]]; then
    return 0
  fi
  set +e
  eval "${cmd}" 2>&1 | tee -a "${RUN_LOG}"
  local rc=${PIPESTATUS[0]}
  set -e
  return "${rc}"
}

SECONDS=0
render_cfg "${IMS_PER_BATCH}" "${MAX_ITER}" "${STEPS}" "${WARMUP_ITERS}" "${EVAL_PERIOD}" "${CHECKPOINT_PERIOD}"
if run_train_once; then
  :
else
  if [[ "${MODE}" != "run" || "${SMOKE}" == "1" ]]; then
    runner_log "${MODE}" "${RUN_LOG}" "FAILED rc=1"
    exit 1
  fi

  if rg -qi "outofmemoryerror|cuda out of memory" "${RUN_LOG}"; then
    runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-a-magformer] OOM detected, retry with batch=2"
    read -r _ipe fallback_iter fallback_steps fallback_warmup fallback_eval fallback_ckpt < <(compute_budget "2" "${EPOCHS}")
    : "${fallback_iter:?missing fallback_iter}"
    : "${fallback_steps:?missing fallback_steps}"
    : "${fallback_warmup:?missing fallback_warmup}"
    : "${fallback_eval:?missing fallback_eval}"
    : "${fallback_ckpt:?missing fallback_ckpt}"
    cat > "${OUT}/notes_oom.txt" <<EON
OOM fallback activated for ${MODEL_ID}.
Original: batch=${IMS_PER_BATCH} max_iter=${MAX_ITER} steps=${STEPS} warmup_iters=${WARMUP_ITERS}
Fallback: batch=2 max_iter=${fallback_iter} steps=${fallback_steps} warmup_iters=${fallback_warmup}
EON
    rm -f "${OUT}"/checkpoint_iter_*.pth "${OUT}/model_best.pth" "${OUT}/coco_instances_results.json" "${OUT}/metrics.cocoeval.json" "${OUT}/metrics_std.jsonl" "${OUT}/metrics_std.csv" || true
    render_cfg "2" "${fallback_iter}" "${fallback_steps}" "${fallback_warmup}" "${fallback_eval}" "${fallback_ckpt}"
    if ! run_train_once; then
      runner_log "${MODE}" "${RUN_LOG}" "FAILED rc=1 (fallback also failed)"
      exit 1
    fi
  else
    runner_log "${MODE}" "${RUN_LOG}" "FAILED rc=1"
    exit 1
  fi
fi

echo "${SECONDS}" > "${OUT}/wall_time_sec.txt"

if [[ "${MODE}" == "run" ]]; then
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/write_params_from_magformer_ckpt.py --out-dir '${OUT}'"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${DATASET_ROOT}' --out-dir '${OUT}' --metrics-out 'metrics.cocoeval.json'"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/write_metrics_std.py --out-dir '${OUT}' --iters-per-epoch ${ITERS_PER_EPOCH}"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/prune_checkpoints.py --out-dir '${OUT}' --framework magformer"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/write_run_metadata.py --phase end --out-dir '${OUT}'"
fi

runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-a-magformer] done"
