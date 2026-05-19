#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="/home/hdd3/zhanghaonan/magformer"
PYTHON="/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python"
CURRENT_STAGE_C_CONFIG="configs/vc_suda_stage_c_r142_32k_source_target150_fixed.yaml"
CURRENT_STAGE_C_RUN="vc_suda_stage_c_r142_32254_train25654_source_target150_fixed"
CURRENT_SOURCE_TRAIN_CACHE="magformer_datasets/20260318_1K_32254/cache/coco_loader/instances_train.sqlite"
START_LOG="${REPO_ROOT}/output/diagnostics/start_vc_suda_watchers_20260519.log"

cd "${REPO_ROOT}"

main_log() {
  mkdir -p "$(dirname "${START_LOG}")"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*" | tee -a "${START_LOG}"
}

report_disabled_historical_watchers() {
  main_log "VC-SUDA historical invalid watcher recovery is disabled."
  main_log "R118/R121/R122 historical invalid configs are fail-fast isolated: source/target collapse and target_labeled_weight=0."
  main_log "Current Stage C positive config: ${CURRENT_STAGE_C_CONFIG}"
  main_log "Current Stage C run name: ${CURRENT_STAGE_C_RUN}"
  main_log "R142 source cache: ${CURRENT_SOURCE_TRAIN_CACHE}"
  main_log "This script does not launch training or evaluation jobs."
  return 2
}

case "${1:-}" in
  ""|--r126-watcher|--r127-watcher)
    report_disabled_historical_watchers
    ;;
  *)
    printf 'usage: %s [--r126-watcher|--r127-watcher]\n' "$0" >&2
    exit 2
    ;;
esac
