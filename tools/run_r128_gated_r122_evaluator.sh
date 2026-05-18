#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="/home/hdd3/zhanghaonan/magformer"
PYTHON="/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python"
CONDA_NVJITLINK_LIB="/home/hdd3/zhanghaonan/anaconda3/envs/magformer/lib/python3.11/site-packages/nvidia/nvjitlink/lib"
LOG_PATH="output/diagnostics/r128_gated_r122_evaluator_20260518.log"
STATE_JSON="/tmp/r128_state.json"
STATE_MD="/tmp/r128_state.md"
SLEEP_SECONDS=300

R122_CHECKPOINT="output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300/checkpoint_iter_0000099.pth"
REMAINING75_DIR="output/diagnostics/r122_depth_boundary_w001_iter0099_remaining75_1024_backmap_topk200_20260518"
VAL28_DIR="output/diagnostics/r122_depth_boundary_w001_iter0099_val28_1024_backmap_topk200_20260518"

cd "${REPO_ROOT}" || exit 2
export LD_LIBRARY_PATH="${CONDA_NVJITLINK_LIB}:${LD_LIBRARY_PATH:-}"
mkdir -p output/diagnostics
exec >>"${LOG_PATH}" 2>&1

log() {
  printf '[%(%Y-%m-%d %H:%M:%S %z)T] %s\n' -1 "$*"
}

state_from_json() {
  "${PYTHON}" - "$STATE_JSON" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
print(payload.get("state", ""))
PY
}

log_state_summary() {
  "${PYTHON}" - "$STATE_JSON" <<'PY'
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

training_processes() {
  ps -u "$(id -u)" -o pid=,args= | "${PYTHON}" -c '
import re
import sys

job_pattern = re.compile(r"(^|[\s/])(train\.py|evaluate[^/\s]*\.py|eval[^/\s]*\.py|torchrun|torch\.distributed\.run)(?=\s|$)", re.IGNORECASE)
python_c_pattern = re.compile(r"(^|\s)\S*python\S*\s+-c(?=\s|$)", re.IGNORECASE)

for line in sys.stdin:
    command = re.sub(r"^\s*\d+\s+", "", line.rstrip())
    if "run_r128_gated_r122_evaluator.sh" in command:
        continue
    if python_c_pattern.search(command):
        continue
    if job_pattern.search(command):
        print(line.rstrip())
'
}


r122_eval_blocking_processes() {
  training_processes | REPO_ROOT_FOR_GATE="${REPO_ROOT}" \
    R122_CONFIG_FOR_GATE="configs/baseline_vc_suda_r122_depth_boundary_w001_pseudo300.yaml" \
    R122_OUTPUT_FOR_GATE="output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300" \
    "${PYTHON}" -c '
import os
import re
import shlex
import sys

repo_root = os.environ["REPO_ROOT_FOR_GATE"]
r122_config = os.environ["R122_CONFIG_FOR_GATE"]
r122_output = os.environ["R122_OUTPUT_FOR_GATE"]
physical_guard_gpus = {"6", "7"}
visible_devices_pattern = re.compile(r"(?:^|\s)(?:CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES)=([^\s]+)")


def command_from_ps_line(line: str) -> str:
    return re.sub(r"^\s*\d+\s+", "", line.rstrip())


def pid_from_ps_line(line: str) -> str:
    match = re.match(r"^\s*(\d+)\s+", line)
    return match.group(1) if match else ""


def split_gpu_ids(value: str) -> set[str]:
    value = value.strip(chr(34) + chr(39) + " ")
    if value.lower() == "all":
        return set(physical_guard_gpus)
    return {part.strip() for part in re.split(r"[,;]", value) if part.strip()}


def env_targets_guard_gpu(command: str) -> bool:
    for match in visible_devices_pattern.finditer(command):
        if split_gpu_ids(match.group(1)) & physical_guard_gpus:
            return True
    return False


def option_targets_guard_gpu(command: str) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    gpu_options = {"--gpu", "--gpus", "--device", "--devices"}
    for index, token in enumerate(tokens):
        option = token
        value = ""
        if "=" in token:
            option, value = token.split("=", 1)
        elif token in gpu_options and index + 1 < len(tokens):
            value = tokens[index + 1]
        if option in gpu_options and split_gpu_ids(value) & physical_guard_gpus:
            return True
    return False


def process_cwd(pid: str) -> str:
    if not pid:
        return ""
    try:
        return os.path.realpath(os.readlink(f"/proc/{pid}/cwd"))
    except OSError:
        return ""


def is_repo_job(command: str, pid: str) -> bool:
    if repo_root in command:
        return True
    cwd = process_cwd(pid)
    return cwd == repo_root or cwd.startswith(repo_root + os.sep)


def is_r122_job(command: str) -> bool:
    return r122_config in command or r122_output in command


for line in sys.stdin:
    command = command_from_ps_line(line)
    pid = pid_from_ps_line(line)
    if is_r122_job(command):
        print(line.rstrip())
        continue
    if env_targets_guard_gpu(command) or option_targets_guard_gpu(command):
        print(line.rstrip())
        continue
    if is_repo_job(command, pid) and "CUDA_VISIBLE_DEVICES=" not in command and "NVIDIA_VISIBLE_DEVICES=" not in command:
        print(line.rstrip())
'
}

checkpoint_size() {
  stat -c '%s' "$R122_CHECKPOINT"
}

probe_eval_gpus() {
  CUDA_VISIBLE_DEVICES=6,7 "${PYTHON}" <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
x = torch.ones((2, 2), device="cuda")
print("tensor_sum", float(x.sum().item()))
print("device_name", torch.cuda.get_device_name(0))
PY
}

output_complete() {
  local output_dir="$1"
  [[ -f "${output_dir}/coco_instances_results.json" && -f "${output_dir}/metrics.cocoeval.json" ]]
}

assert_output_safe() {
  local label="$1"
  local output_dir="$2"
  if [[ -d "$output_dir" ]]; then
    if output_complete "$output_dir"; then
      log "${label} output already complete; not rerunning: ${output_dir}"
      return 1
    fi
    log "${label} output dir exists but required files are missing; refusing to overwrite: ${output_dir}"
    exit 4
  fi
  return 0
}

verify_eval_output() {
  local label="$1"
  local output_dir="$2"
  if output_complete "$output_dir"; then
    log "${label} generated required eval files: ${output_dir}"
    return 0
  fi
  log "${label} finished without required eval files: ${output_dir}"
  exit 5
}

run_remaining75_eval() {
  log "starting remaining75 eval on GPU6,7"
  CUDA_VISIBLE_DEVICES=6,7 MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch \
    "${PYTHON}" tools/evaluate_1024_backmap.py \
    --base-config configs/baseline_vc_suda_r122_depth_boundary_w001_pseudo300.yaml \
    --dataset-root magformer_datasets/pseudo_real_512 \
    --ann annotations/instances_target_unlabeled_r114_balanced_minus125.json \
    --split train \
    --weights "$R122_CHECKPOINT" \
    --output-dir "$REMAINING75_DIR" \
    --image-size 1024 \
    --batch-size 4 \
    --num-workers 0 \
    --score-threshold 0.05 \
    --mask-threshold 0.5 \
    --iou-types bbox,segm \
    --inference-topk 200 \
    --max-dets 200 \
    --force-pytorch-msda
}

run_val28_eval() {
  log "starting val28 eval on GPU6,7"
  CUDA_VISIBLE_DEVICES=6,7 MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch \
    "${PYTHON}" tools/evaluate_1024_backmap.py \
    --base-config configs/baseline_vc_suda_r122_depth_boundary_w001_pseudo300.yaml \
    --dataset-root magformer_datasets/pseudo_real_512 \
    --ann annotations/instances_val.json \
    --split val \
    --weights "$R122_CHECKPOINT" \
    --output-dir "$VAL28_DIR" \
    --image-size 1024 \
    --batch-size 4 \
    --num-workers 0 \
    --score-threshold 0.05 \
    --mask-threshold 0.5 \
    --iou-types bbox,segm \
    --inference-topk 200 \
    --max-dets 200 \
    --force-pytorch-msda
}

run_safety_gates_and_eval() {
  local processes
  processes="$(r122_eval_blocking_processes)"
  if [[ -n "$processes" ]]; then
    log "R122 eval process gate failed; no eval will start"
    printf '%s\n' "$processes"
    sleep "$SLEEP_SECONDS"
    return 0
  fi
  log "training process gate passed"

  if [[ ! -f "$R122_CHECKPOINT" ]]; then
    log "checkpoint gate failed; missing ${R122_CHECKPOINT}"
    sleep "$SLEEP_SECONDS"
    return 0
  fi

  local size_one size_two
  size_one="$(checkpoint_size)" || {
    log "checkpoint gate failed; cannot stat first size"
    sleep "$SLEEP_SECONDS"
    return 0
  }
  log "checkpoint size check 1: ${size_one} bytes"
  sleep 30
  size_two="$(checkpoint_size)" || {
    log "checkpoint gate failed; cannot stat second size"
    sleep "$SLEEP_SECONDS"
    return 0
  }
  log "checkpoint size check 2: ${size_two} bytes"
  if [[ "$size_one" != "$size_two" ]]; then
    log "checkpoint gate failed; size changed across 30 seconds"
    sleep "$SLEEP_SECONDS"
    return 0
  fi
  log "checkpoint stability gate passed"

  if ! probe_eval_gpus; then
    log "CUDA probe gate failed for GPU6,7; no eval will start"
    sleep "$SLEEP_SECONDS"
    return 0
  fi
  log "CUDA probe gate passed for GPU6,7"

  local run_remaining75=1
  local run_val28=1
  if assert_output_safe "remaining75" "$REMAINING75_DIR"; then
    run_remaining75=0
  fi
  if assert_output_safe "val28" "$VAL28_DIR"; then
    run_val28=0
  fi

  if [[ "$run_remaining75" -eq 0 ]]; then
    run_remaining75_eval
    local rc=$?
    log "remaining75 eval exit_code=${rc}"
    if [[ "$rc" -ne 0 ]]; then
      exit "$rc"
    fi
    verify_eval_output "remaining75" "$REMAINING75_DIR"
  fi

  if [[ "$run_val28" -eq 0 ]]; then
    run_val28_eval
    local rc=$?
    log "val28 eval exit_code=${rc}"
    if [[ "$rc" -ne 0 ]]; then
      exit "$rc"
    fi
    verify_eval_output "val28" "$VAL28_DIR"
  fi

  if output_complete "$REMAINING75_DIR" && output_complete "$VAL28_DIR"; then
    log "R128 eval outputs are complete; exiting without go/no-go comparator"
    exit 0
  fi

  log "R128 eval outputs are not complete; exiting"
  exit 5
}

log "R128 gated R122 evaluator watcher started"
log "repo=${REPO_ROOT}"
log "state_json=${STATE_JSON} state_md=${STATE_MD}"

while true; do
  log "running R121/R122 resume state checker"
  "${PYTHON}" tools/check_r121_r122_resume_state.py --output-json "$STATE_JSON" --output-md "$STATE_MD"
  rc=$?
  if [[ "$rc" -ne 0 ]]; then
    log "state checker failed exit_code=${rc}; exiting"
    exit "$rc"
  fi

  log_state_summary
  state="$(state_from_json)"

  case "$state" in
    BLOCKED_CUDA|NEED_R121_SMOKE|NEED_R121_TRAIN|R122_TRAINING|NEED_R122_TRAIN)
      log "state ${state}; sleeping ${SLEEP_SECONDS}s"
      sleep "$SLEEP_SECONDS"
      ;;
    R121_FAILED|NEED_GO_NO_GO|READY_TO_DECIDE)
      log "terminal non-eval state ${state}; exiting without eval"
      exit 0
      ;;
    NEED_R122_EVAL)
      log "state NEED_R122_EVAL; evaluating safety gates"
      run_safety_gates_and_eval
      ;;
    *)
      log "unknown state ${state}; exiting"
      exit 2
      ;;
  esac
done
