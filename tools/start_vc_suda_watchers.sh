#!/usr/bin/env bash
set -euo pipefail

CURRENT_STAGE_C_CONFIG="configs/vc_suda_stage_c_r142_32254_train25654_source_target150_fixed.yaml"
CURRENT_STAGE_C_RUN="vc_suda_stage_c_r142_32254_train25654_source_target150_fixed"
CURRENT_SOURCE_TRAIN_CACHE="magformer_datasets/20260318_1K_32254/cache/coco_loader/instances_train.sqlite"
CURRENT_MANUAL_LAUNCH_DOC="docs/results/vc_suda_stage_c_repair_notes_20260519.md"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*"
}

log "VC-SUDA watcher entrypoint is retired and is now a no-op."
log "Historical invalid R118/R121/R122 watcher recovery is disabled."
log "Those historical configs are fail-fast isolated because of source/target collapse and target_labeled_weight=0."
log "Current Stage C positive config: ${CURRENT_STAGE_C_CONFIG}"
log "Current Stage C run name: ${CURRENT_STAGE_C_RUN}"
log "R142 source cache: ${CURRENT_SOURCE_TRAIN_CACHE}"
log "Use R142/current manual launch docs instead: ${CURRENT_MANUAL_LAUNCH_DOC}"
log "No training or evaluation job was started."

exit 0
