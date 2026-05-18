#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="/home/hdd3/zhanghaonan/magformer"
PYTHON="/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python"
SCRIPT_PATH="${REPO_ROOT}/tools/start_vc_suda_watchers.sh"
START_LOG="${REPO_ROOT}/output/diagnostics/start_vc_suda_watchers_20260518.log"
SLEEP_SECONDS=300

cd "${REPO_ROOT}"

main_log() {
  mkdir -p "$(dirname "${START_LOG}")"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*" | tee -a "${START_LOG}"
}

watcher_log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*"
}

training_processes() {
  ps -u "$(id -u)" -o pid=,args= | "${PYTHON}" -c '
import re
import sys

job_pattern = re.compile(r"(^|[\s/])(train\.py|evaluate[^/\s]*\.py|eval[^/\s]*\.py|torchrun|torch\.distributed\.run)(?=\s|$)", re.IGNORECASE)
python_c_pattern = re.compile(r"(^|\s)\S*python\S*\s+-c(?=\s|$)", re.IGNORECASE)

for line in sys.stdin:
    command = re.sub(r"^\s*\d+\s+", "", line.rstrip())
    if "start_vc_suda_watchers.sh" in command:
        continue
    if python_c_pattern.search(command):
        continue
    if job_pattern.search(command):
        print(line.rstrip())
'
}

state_from_json() {
  local state_json="$1"
  "${PYTHON}" - "${state_json}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
print(payload.get("state", ""))
PY
}

log_state_summary() {
  local state_json="$1"
  "${PYTHON}" - "${state_json}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
print(f"state={payload.get('state', '')}")
print(f"next_action={payload.get('next_action', '')}")
for reason in payload.get("reasons", []):
    print(f"reason={reason}")
PY
}

probe_gpu() {
  local gpu_id="$1"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON}" <<'PY'
import torch

print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
x = torch.ones((2, 2), device="cuda")
print("tensor_sum", float(x.sum().item()))
print("device_name", torch.cuda.get_device_name(0))
PY
}

r126_watcher() {
  local log_path="output/diagnostics/r126_cuda_resume_r121_watcher_g4567_20260518.log"
  local output_dir="output/vc_suda/r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075"
  local retry_glob="output/vc_suda/r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075.retry_gpu*.tmux.log"
  local lock_dir="output/diagnostics/r126_cuda_resume_r121_watcher_g4567.lock"

  mkdir -p "$(dirname "${log_path}")" output/vc_suda
  exec >>"${log_path}" 2>&1

  watcher_log "R126 CUDA resume watcher started; probes GPU 4,5,6,7; launches only gated R121 smoke"
  watcher_log "repo=${REPO_ROOT}"

  while true; do
    if [[ -d "${output_dir}" ]]; then
      watcher_log "R121 smoke output exists; no launch needed: ${output_dir}; exiting"
      exit 0
    fi

    if compgen -G "${retry_glob}" >/dev/null; then
      watcher_log "R121 retry log exists; refusing duplicate smoke launch"
      compgen -G "${retry_glob}" | sort
      exit 0
    fi

    local processes
    processes="$(training_processes)"
    if [[ -n "${processes}" ]]; then
      watcher_log "training process gate failed; no R121 smoke will start"
      printf '%s\n' "${processes}"
      watcher_log "sleeping ${SLEEP_SECONDS}s"
      sleep "${SLEEP_SECONDS}"
      continue
    fi

    local available_gpu=""
    local gpu_id
    for gpu_id in 4 5 6 7; do
      watcher_log "probing GPU ${gpu_id}"
      if probe_gpu "${gpu_id}"; then
        available_gpu="${gpu_id}"
        watcher_log "GPU ${gpu_id} probe passed"
        break
      fi
      watcher_log "GPU ${gpu_id} probe failed"
    done

    if [[ -z "${available_gpu}" ]]; then
      watcher_log "all GPU probes failed; sleeping ${SLEEP_SECONDS}s"
      sleep "${SLEEP_SECONDS}"
      continue
    fi

    if ! mkdir "${lock_dir}" 2>/dev/null; then
      watcher_log "lock exists; another R126 watcher may be launching; sleeping ${SLEEP_SECONDS}s"
      sleep "${SLEEP_SECONDS}"
      continue
    fi
    trap 'rmdir "${lock_dir}" 2>/dev/null || true' EXIT

    if [[ -d "${output_dir}" ]] || compgen -G "${retry_glob}" >/dev/null; then
      watcher_log "R121 output or retry log appeared after lock; exiting without duplicate launch"
      exit 0
    fi

    local launch_log
    launch_log="output/vc_suda/r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075.retry_gpu${available_gpu}_$(date +%Y%m%d_%H%M%S).tmux.log"
    watcher_log "launching gated R121 smoke on GPU ${available_gpu}; log=${launch_log}"
    CUDA_VISIBLE_DEVICES="${available_gpu}" MAGFORMER_MS_DEFORM_ATTN_BACKEND=cuda \
      "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=1 tools/train.py \
      --config configs/baseline_vc_suda_r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075.yaml \
      --gpus 0 --num-workers 0 \
      2>&1 | tee "${launch_log}"
    local rc=${PIPESTATUS[0]}
    watcher_log "R121 smoke exit_code=${rc}; exiting"
    exit "${rc}"
  done
}

r127_probe_all_gpus() {
  local gpu_id
  for gpu_id in 4 5 6 7; do
    watcher_log "probing GPU ${gpu_id}"
    if ! probe_gpu "${gpu_id}"; then
      watcher_log "GPU ${gpu_id} probe failed"
      return 1
    fi
    watcher_log "GPU ${gpu_id} probe passed"
  done
  return 0
}

r127_launch_r122() {
  local r122_log="output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300.retry_$(date +%Y%m%d_%H%M%S).tmux.log"
  mkdir -p "$(dirname "${r122_log}")"
  if [[ -e "${r122_log}" ]]; then
    watcher_log "refusing to overwrite existing R122 log: ${r122_log}"
    exit 1
  fi
  watcher_log "launching gated R122 300iter run on GPU 4,5,6,7; log=${r122_log}"
  CUDA_VISIBLE_DEVICES=4,5,6,7 MAGFORMER_MS_DEFORM_ATTN_BACKEND=cuda \
    "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=4 tools/train.py \
    --config configs/baseline_vc_suda_r122_depth_boundary_w001_pseudo300.yaml \
    --gpus 0,1,2,3 --num-workers 2 \
    2>&1 | tee "${r122_log}"
  local rc=${PIPESTATUS[0]}
  watcher_log "R122 train exit_code=${rc}; exiting"
  exit "${rc}"
}

r127_safety_gates_and_launch() {
  local output_dir="output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300"
  local processes

  processes="$(training_processes)"
  if [[ -n "${processes}" ]]; then
    watcher_log "training process gate failed; no R122 training will start"
    printf '%s\n' "${processes}"
    sleep "${SLEEP_SECONDS}"
    return 0
  fi

  if [[ -e "${output_dir}" ]]; then
    watcher_log "R122 output exists; refusing duplicate training launch: ${output_dir}; exiting"
    exit 0
  fi

  if ! r127_probe_all_gpus; then
    watcher_log "GPU probe gate failed; no R122 training will start; sleeping ${SLEEP_SECONDS}s"
    sleep "${SLEEP_SECONDS}"
    return 0
  fi

  r127_launch_r122
}

r127_watcher() {
  local log_path="output/diagnostics/r127_gated_r122_launcher_20260518.log"
  local state_json="/tmp/r127_state.json"
  local state_md="/tmp/r127_state.md"

  mkdir -p "$(dirname "${log_path}")"
  exec >>"${log_path}" 2>&1

  watcher_log "R127 gated R122 launcher started; no R121 launch; R122 launch only on NEED_R122_TRAIN"
  watcher_log "repo=${REPO_ROOT}"
  watcher_log "state_json=${state_json} state_md=${state_md}"

  while true; do
    watcher_log "running R121/R122 resume state checker"
    "${PYTHON}" tools/check_r121_r122_resume_state.py --output-json "${state_json}" --output-md "${state_md}"
    local rc=$?
    if [[ "${rc}" -ne 0 ]]; then
      watcher_log "state checker failed exit_code=${rc}; exiting"
      exit "${rc}"
    fi

    log_state_summary "${state_json}"
    local state
    state="$(state_from_json "${state_json}")"

    case "${state}" in
      BLOCKED_CUDA|NEED_R121_SMOKE|NEED_R121_TRAIN|R122_TRAINING)
        watcher_log "state ${state}; sleeping ${SLEEP_SECONDS}s"
        sleep "${SLEEP_SECONDS}"
        ;;
      R121_FAILED|NEED_R122_EVAL|NEED_GO_NO_GO|READY_TO_DECIDE)
        watcher_log "terminal non-launch state ${state}; exiting without R122 training"
        exit 0
        ;;
      NEED_R122_TRAIN)
        watcher_log "state NEED_R122_TRAIN; evaluating R122 safety gates"
        r127_safety_gates_and_launch
        ;;
      *)
        watcher_log "unknown state ${state}; exiting"
        exit 2
        ;;
    esac
  done
}

session_exists() {
  local session_name="$1"
  tmux has-session -t "${session_name}" >/dev/null 2>&1
}

start_tmux_session() {
  local session_name="$1"
  local command="$2"

  if session_exists "${session_name}"; then
    main_log "skip existing tmux session: ${session_name}"
    return 0
  fi

  main_log "starting tmux session: ${session_name}"
  tmux new-session -d -s "${session_name}" "${command}"
  main_log "started tmux session: ${session_name}"
}

start_all_watchers() {
  mkdir -p "$(dirname "${START_LOG}")"
  main_log "VC-SUDA reboot watcher recovery start"
  main_log "repo=${REPO_ROOT}"
  main_log "log=${START_LOG}"

  start_tmux_session \
    "r126_cuda_resume_r121_watcher_g4567" \
    "cd '${REPO_ROOT}' && exec bash '${SCRIPT_PATH}' --r126-watcher"
  start_tmux_session \
    "r127_gated_r122_launcher" \
    "cd '${REPO_ROOT}' && exec bash '${SCRIPT_PATH}' --r127-watcher"
  start_tmux_session \
    "r128_gated_r122_evaluator" \
    "cd '${REPO_ROOT}' && exec bash tools/run_r128_gated_r122_evaluator.sh"
  start_tmux_session \
    "r129_gated_go_no_go" \
    "cd '${REPO_ROOT}' && exec bash tools/run_r129_gated_go_no_go.sh"
  start_tmux_session \
    "r130_final_goal_audit_watcher" \
    "cd '${REPO_ROOT}' && exec bash tools/run_r130_final_goal_audit_watcher.sh"

  main_log "VC-SUDA reboot watcher recovery complete"
}

case "${1:-}" in
  "")
    start_all_watchers
    ;;
  --r126-watcher)
    r126_watcher
    ;;
  --r127-watcher)
    r127_watcher
    ;;
  *)
    printf 'usage: %s [--r126-watcher|--r127-watcher]\n' "$0" >&2
    exit 2
    ;;
esac
