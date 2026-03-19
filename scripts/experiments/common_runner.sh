#!/usr/bin/env bash
set -euo pipefail

runner_now_ts() {
  date '+%Y-%m-%d %H:%M:%S'
}

runner_setup_log() {
  local out_dir="$1"
  local mode="$2"
  mkdir -p "${out_dir}"
  local run_log="${out_dir}/run.log"
  if [[ "${mode}" == "run" ]]; then
    : > "${run_log}"
  fi
  printf '%s\n' "${run_log}"
}

runner_log() {
  local mode="$1"
  local run_log="$2"
  shift 2
  local msg="$*"
  local line="[$(runner_now_ts)] ${msg}"
  if [[ "${mode}" == "run" ]]; then
    echo "${line}" | tee -a "${run_log}"
  else
    echo "${line}"
  fi
}

runner_exec() {
  local mode="$1"
  local run_log="$2"
  shift 2
  local cmd="$*"
  runner_log "${mode}" "${run_log}" "+ ${cmd}"
  if [[ "${mode}" != "run" ]]; then
    return 0
  fi

  set +e
  eval "${cmd}" 2>&1 | tee -a "${run_log}"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ ${rc} -ne 0 ]]; then
    runner_log "${mode}" "${run_log}" "FAILED rc=${rc}"
    exit "${rc}"
  fi
}

runner_gpu_free_mb() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "-1"
    return 0
  fi
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' '
}

runner_wait_for_free_gpu_mb() {
  local mode="$1"
  local run_log="$2"
  local threshold_mb="$3"
  local sleep_sec="$4"
  local label="${5:-job}"

  if [[ "${mode}" != "run" ]]; then
    return 0
  fi
  if [[ -z "${threshold_mb}" || "${threshold_mb}" -le 0 ]]; then
    return 0
  fi

  while true; do
    local free_mb
    free_mb="$(runner_gpu_free_mb)"
    if [[ -z "${free_mb}" || "${free_mb}" == "-1" ]]; then
      runner_log "${mode}" "${run_log}" "[gpu-wait] nvidia-smi unavailable, skip wait for ${label}"
      return 0
    fi
    if [[ "${free_mb}" -ge "${threshold_mb}" ]]; then
      runner_log "${mode}" "${run_log}" "[gpu-wait] ready for ${label}: free_mb=${free_mb} threshold_mb=${threshold_mb}"
      return 0
    fi
    runner_log "${mode}" "${run_log}" "[gpu-wait] waiting for ${label}: free_mb=${free_mb} threshold_mb=${threshold_mb} sleep_sec=${sleep_sec}"
    sleep "${sleep_sec}"
  done
}
