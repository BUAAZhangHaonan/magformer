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

runner_resolve_hf_endpoint() {
  local endpoint="${HF_ENDPOINT:-${HF_MIRROR:-${hf_mirror:-}}}"
  if [[ -z "${endpoint}" ]]; then
    return 0
  fi
  if [[ "${endpoint}" != http://* && "${endpoint}" != https://* ]]; then
    endpoint="https://${endpoint}"
  fi
  printf '%s\n' "${endpoint}"
}

runner_hf_env_prefix() {
  local endpoint
  endpoint="$(runner_resolve_hf_endpoint)"
  if [[ -n "${endpoint}" ]]; then
    printf "HF_ENDPOINT=%q " "${endpoint}"
  fi
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

runner_gpu_query_id() {
  local visible_devices="${CUDA_VISIBLE_DEVICES:-}"
  local first_visible=""
  if [[ -z "${visible_devices}" ]]; then
    return 0
  fi

  IFS=',' read -r first_visible _ <<< "${visible_devices}"
  first_visible="${first_visible//[[:space:]]/}"
  if [[ -z "${first_visible}" || "${first_visible}" == "-1" ]]; then
    return 0
  fi
  printf '%s\n' "${first_visible}"
}

runner_gpu_free_mb() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "-1"
    return 0
  fi
  local gpu_id=""
  local free_mb=""
  gpu_id="$(runner_gpu_query_id)"
  if [[ -n "${gpu_id}" ]]; then
    free_mb="$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')"
    if [[ -n "${free_mb}" ]]; then
      printf '%s\n' "${free_mb}"
      return 0
    fi
  fi
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' '
}

runner_meminfo_path() {
  printf '%s\n' "${RUNNER_MEMINFO_PATH:-/proc/meminfo}"
}

runner_mem_available_mb() {
  local meminfo
  meminfo="$(runner_meminfo_path)"
  awk '/^MemAvailable:/ { printf "%d\n", int($2 / 1024); found=1 } END { if (!found) print "" }' "${meminfo}" 2>/dev/null
}

runner_mem_total_mb() {
  local meminfo
  meminfo="$(runner_meminfo_path)"
  awk '/^MemTotal:/ { printf "%d\n", int($2 / 1024); found=1 } END { if (!found) print "" }' "${meminfo}" 2>/dev/null
}

runner_ram_used_pct() {
  local total available
  total="$(runner_mem_total_mb)"
  available="$(runner_mem_available_mb)"
  if [[ -z "${total}" || -z "${available}" || "${total}" -le 0 ]]; then
    echo ""
    return 0
  fi
  awk -v total="${total}" -v available="${available}" 'BEGIN { printf "%d\n", int(((total - available) * 100) / total) }'
}

runner_swap_used_mb() {
  local meminfo
  meminfo="$(runner_meminfo_path)"
  awk '
    /^SwapTotal:/ { total=$2 }
    /^SwapFree:/ { free=$2 }
    END {
      if (total == "") {
        print ""
      } else {
        used = total - free
        if (used < 0) used = 0
        printf "%d\n", int(used / 1024)
      }
    }
  ' "${meminfo}" 2>/dev/null
}

runner_log_launch_guard_snapshot() {
  local mode="$1"
  local run_log="$2"
  local label="${3:-job}"
  local gpu_free_mb mem_available_mb mem_total_mb ram_used_pct swap_used_mb sessions
  gpu_free_mb="$(runner_gpu_free_mb)"
  mem_available_mb="$(runner_mem_available_mb)"
  mem_total_mb="$(runner_mem_total_mb)"
  ram_used_pct="$(runner_ram_used_pct)"
  swap_used_mb="$(runner_swap_used_mb)"
  sessions="$(tmux list-sessions -F '#S' 2>/dev/null | paste -sd ',' - || true)"
  runner_log "${mode}" "${run_log}" "[launch-guard] ${label}: gpu_free_mb=${gpu_free_mb:-unknown} mem_available_mb=${mem_available_mb:-unknown} mem_total_mb=${mem_total_mb:-unknown} ram_used_pct=${ram_used_pct:-unknown} swap_used_mb=${swap_used_mb:-unknown} tmux_sessions=${sessions:-none}"
}

runner_wait_for_system_resources() {
  local mode="$1"
  local run_log="$2"
  local min_ram_mb="$3"
  local max_swap_used_mb="$4"
  local sleep_sec="$5"
  local label="${6:-job}"

  if [[ "${mode}" != "run" ]]; then
    return 0
  fi

  while true; do
    local available_mb swap_used_mb ram_ok swap_ok
    available_mb="$(runner_mem_available_mb)"
    swap_used_mb="$(runner_swap_used_mb)"
    ram_ok=1
    swap_ok=1
    if [[ -n "${min_ram_mb}" && "${min_ram_mb}" -gt 0 && -n "${available_mb}" && "${available_mb}" -lt "${min_ram_mb}" ]]; then
      ram_ok=0
    fi
    if [[ -n "${max_swap_used_mb}" && "${max_swap_used_mb}" -ge 0 && -n "${swap_used_mb}" && "${swap_used_mb}" -gt "${max_swap_used_mb}" ]]; then
      swap_ok=0
    fi
    if [[ "${ram_ok}" -eq 1 && "${swap_ok}" -eq 1 ]]; then
      runner_log "${mode}" "${run_log}" "[resource-wait] ready for ${label}: mem_available_mb=${available_mb:-unknown} min_ram_mb=${min_ram_mb} swap_used_mb=${swap_used_mb:-unknown} max_swap_used_mb=${max_swap_used_mb}"
      return 0
    fi
    runner_log "${mode}" "${run_log}" "[resource-wait] waiting for ${label}: mem_available_mb=${available_mb:-unknown} min_ram_mb=${min_ram_mb} swap_used_mb=${swap_used_mb:-unknown} max_swap_used_mb=${max_swap_used_mb} sleep_sec=${sleep_sec}"
    sleep "${sleep_sec}"
  done
}

runner_acquire_output_lock() {
  local mode="$1"
  local run_log="$2"
  local lock_dir="$3"
  local sleep_sec="$4"
  local label="${5:-job}"

  if [[ "${mode}" != "run" ]]; then
    runner_log "${mode}" "${run_log}" "[output-lock] dry-run ${label}: ${lock_dir}"
    return 0
  fi

  mkdir -p "$(dirname "${lock_dir}")"
  while true; do
    if mkdir "${lock_dir}" 2>/dev/null; then
      printf '%s\n' "$$" > "${lock_dir}/pid"
      runner_log "${mode}" "${run_log}" "[output-lock] acquired ${label}: ${lock_dir}"
      return 0
    fi
    local lock_pid
    lock_pid="$(cat "${lock_dir}/pid" 2>/dev/null || true)"
    if [[ -n "${lock_pid}" ]] && kill -0 "${lock_pid}" 2>/dev/null; then
      runner_log "${mode}" "${run_log}" "[output-lock] waiting for ${label}: lock_dir=${lock_dir} pid=${lock_pid} sleep_sec=${sleep_sec}"
      sleep "${sleep_sec}"
    else
      runner_log "${mode}" "${run_log}" "[output-lock] removing stale lock for ${label}: ${lock_dir}"
      rm -rf "${lock_dir}"
    fi
  done
}

runner_release_output_lock() {
  local mode="$1"
  local run_log="$2"
  local lock_dir="$3"
  local label="${4:-job}"
  if [[ "${mode}" != "run" ]]; then
    return 0
  fi
  rm -rf "${lock_dir}"
  runner_log "${mode}" "${run_log}" "[output-lock] released ${label}: ${lock_dir}"
}

runner_exec_locked() {
  local mode="$1"
  local run_log="$2"
  local lock_dir="$3"
  local label="$4"
  shift 4
  local cmd="$*"
  if [[ "${mode}" != "run" ]]; then
    runner_acquire_output_lock "${mode}" "${run_log}" "${lock_dir}" 0 "${label}"
    runner_exec "${mode}" "${run_log}" "${cmd}"
    return 0
  fi

  runner_acquire_output_lock "${mode}" "${run_log}" "${lock_dir}" 30 "${label}"
  runner_log "${mode}" "${run_log}" "+ ${cmd}"
  set +e
  eval "${cmd}" 2>&1 | tee -a "${run_log}"
  local rc=${PIPESTATUS[0]}
  set -e
  runner_release_output_lock "${mode}" "${run_log}" "${lock_dir}" "${label}"
  if [[ ${rc} -ne 0 ]]; then
    runner_log "${mode}" "${run_log}" "FAILED rc=${rc}"
    exit "${rc}"
  fi
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
