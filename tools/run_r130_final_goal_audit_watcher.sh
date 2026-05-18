#!/usr/bin/env bash
set -u

REPO_ROOT="/home/hdd3/zhanghaonan/magformer"
PYTHON="/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python"
SLEEP_SECONDS=300
LOG_PATH="output/diagnostics/r130_final_goal_audit_watcher_20260518.log"

GO_NO_GO_JSON="output/diagnostics/r122_depth_boundary_w001_go_no_go_20260518/go_no_go.json"
AUDIT_DIR="output/diagnostics/r130_final_goal_audit_20260518"
AUDIT_JSON="${AUDIT_DIR}/final_goal_audit.json"
AUDIT_MD="${AUDIT_DIR}/final_goal_audit.md"

cd "${REPO_ROOT}" || exit 2
mkdir -p "$(dirname "${LOG_PATH}")"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*" | tee -a "${LOG_PATH}"
}

log "R130 final goal audit watcher started; final audit only; no training/eval/go-no-go/update_goal will be launched"

while true; do
  if [[ -e "${AUDIT_JSON}" || -e "${AUDIT_MD}" ]]; then
    log "audit output already exists; not overwriting: ${AUDIT_JSON} ${AUDIT_MD}; exiting"
    exit 0
  fi

  if [[ ! -s "${GO_NO_GO_JSON}" ]]; then
    log "missing go_no_go or empty: ${GO_NO_GO_JSON}; sleeping ${SLEEP_SECONDS}s"
    sleep "${SLEEP_SECONDS}"
    continue
  fi

  mkdir -p "${AUDIT_DIR}"
  log "go_no_go present; running final audit: ${AUDIT_JSON}"
  CUDA_VISIBLE_DEVICES="" "${PYTHON}" tools/audit_vc_suda_goal.py \
    --repo-root "${REPO_ROOT}" \
    --output-json "${AUDIT_JSON}" \
    --output-md "${AUDIT_MD}" >>"${LOG_PATH}" 2>&1
  audit_status=$?

  if [[ ${audit_status} -ne 0 ]]; then
    log "final audit failed with exit ${audit_status}; exiting"
    exit 2
  fi

  status="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", "UNKNOWN"))' "${AUDIT_JSON}")"
  log "final audit completed with status ${status}; update_goal not called; exiting"
  exit 0
done
