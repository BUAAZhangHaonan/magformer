#!/usr/bin/env bash
set -euo pipefail

ecc_normalize_register() {
  local r="${1:-}"
  python3 - <<PY
import sys
from pathlib import Path
repo_root = Path("${REPO_ROOT}").resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from baselines.ecc_datasets import normalize_register
print(normalize_register("${r}"))
PY
}

ecc_default_dataset_root() {
  local raw="${1}"
  local reg
  reg="$(ecc_normalize_register "${raw}")"
  if [[ "${reg}" == "0831" ]]; then
    echo "${PROJECT_ROOT}/magformer_datasets/0831_1K"
  elif [[ "${reg}" == "0909" ]]; then
    echo "${PROJECT_ROOT}/magformer_datasets/0909_512_0.12K"
  else
    echo "${PROJECT_ROOT}/magformer_datasets/${raw}"
  fi
}

ecc_dataset_prefix() {
  local register="${1}"
  local dataset_root="${2:-}"
  python3 - <<PY
import sys
from pathlib import Path
repo_root = Path("${REPO_ROOT}").resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from baselines.ecc_datasets import dataset_prefix
print(dataset_prefix("${register}", "${dataset_root}" or None))
PY
}

ecc_dataset_names_coco() {
  local register="${1}"
  local dataset_root="${2:-}"
  python3 - <<PY
import sys
from pathlib import Path
repo_root = Path("${REPO_ROOT}").resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from baselines.ecc_datasets import dataset_name_pair_coco
train_name, val_name = dataset_name_pair_coco("${register}", "${dataset_root}" or None)
print(train_name, val_name)
PY
}

ecc_dataset_names_coco_rgbd() {
  local register="${1}"
  local dataset_root="${2:-}"
  python3 - <<PY
import sys
from pathlib import Path
repo_root = Path("${REPO_ROOT}").resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from baselines.ecc_datasets import dataset_name_pair_coco_rgbd
train_name, val_name = dataset_name_pair_coco_rgbd("${register}", "${dataset_root}" or None)
print(train_name, val_name)
PY
}

ecc_rgb_stats_json() {
  local reg
  reg="$(ecc_normalize_register "${1}")"
  if [[ "${reg}" == "0831" ]]; then
    echo "${REPO_ROOT}/configs/stats/0831_1k_rgb_stats.json"
  else
    echo "${REPO_ROOT}/configs/stats/0909_512_rgb_stats.json"
  fi
}

ecc_depth_stats_json() {
  local reg
  reg="$(ecc_normalize_register "${1}")"
  if [[ "${reg}" == "0831" ]]; then
    echo "${REPO_ROOT}/configs/stats/0831_1k_depth_stats.json"
  else
    echo "${REPO_ROOT}/configs/stats/0909_512_depth_stats.json"
  fi
}

ecc_read_rgb_stats_bgr() {
  local register="${1}"
  local dataset_root="${2:-}"
  local reg
  local p
  reg="$(ecc_normalize_register "${register}")"
  if [[ "${reg}" == "0831" || "${reg}" == "0909" ]]; then
    p="$(ecc_rgb_stats_json "${register}")"
  else
    if [[ -z "${dataset_root}" ]]; then
      echo "Custom register requires explicit dataset_root for RGB stats: ${register}" >&2
      return 1
    fi
    p="$(ecc_rgb_stats_json_for_dataset_root "${dataset_root}")"
  fi
  python3 - <<PY
import json
d=json.load(open("${p}","r",encoding="utf-8"))
mean=d["mean_bgr"]
std=d["std_bgr"]
fmt=lambda xs: "[" + ",".join(f"{float(x):.4f}" for x in xs) + "]"
print(fmt(mean), fmt(std))
PY
}

ecc_read_rgb_stats_bgr6_depth1275() {
  local register="${1}"
  local dataset_root="${2:-}"
  local reg
  local p
  reg="$(ecc_normalize_register "${register}")"
  if [[ "${reg}" == "0831" || "${reg}" == "0909" ]]; then
    p="$(ecc_rgb_stats_json "${register}")"
  else
    if [[ -z "${dataset_root}" ]]; then
      echo "Custom register requires explicit dataset_root for RGB stats: ${register}" >&2
      return 1
    fi
    p="$(ecc_rgb_stats_json_for_dataset_root "${dataset_root}")"
  fi
  python3 - <<PY
import json
d=json.load(open("${p}","r",encoding="utf-8"))
mean=list(d["mean_bgr"]) + [127.5,127.5,127.5]
std=list(d["std_bgr"]) + [127.5,127.5,127.5]
fmt=lambda xs: "[" + ",".join(f"{float(x):.4f}" for x in xs) + "]"
print(fmt(mean), fmt(std))
PY
}

ecc_read_rgb_stats_rgb() {
  local register="${1}"
  local dataset_root="${2:-}"
  local reg
  local p
  reg="$(ecc_normalize_register "${register}")"
  if [[ "${reg}" == "0831" || "${reg}" == "0909" ]]; then
    p="$(ecc_rgb_stats_json "${register}")"
  else
    if [[ -z "${dataset_root}" ]]; then
      echo "Custom register requires explicit dataset_root for RGB stats: ${register}" >&2
      return 1
    fi
    p="$(ecc_rgb_stats_json_for_dataset_root "${dataset_root}")"
  fi
  python3 - <<PY
import json
d=json.load(open("${p}","r",encoding="utf-8"))
mean=d["mean_rgb"]
std=d["std_rgb"]
fmt=lambda xs: "[" + ",".join(f"{float(x):.4f}" for x in xs) + "]"
print(fmt(mean), fmt(std))
PY
}

ecc_read_depth_clip() {
  local register="${1}"
  local dataset_root="${2:-}"
  local reg
  local p
  reg="$(ecc_normalize_register "${register}")"
  if [[ "${reg}" == "0831" || "${reg}" == "0909" ]]; then
    p="$(ecc_depth_stats_json "${register}")"
  else
    if [[ -z "${dataset_root}" ]]; then
      echo "Custom register requires explicit dataset_root for depth stats: ${register}" >&2
      return 1
    fi
    p="$(ecc_depth_stats_json_for_dataset_root "${dataset_root}")"
  fi
  python3 - <<PY
import json
d=json.load(open("${p}","r",encoding="utf-8"))
print(float(d["p1"]), float(d["p99"]))
PY
}

ecc_ensure_dataset_stats_manifest() {
  local dataset_root="${1}"
  python3 "${REPO_ROOT}/scripts/analysis/ensure_dataset_stats.py" --dataset-root "${dataset_root}" --field manifest_path
}

ecc_rgb_stats_json_for_dataset_root() {
  local dataset_root="${1}"
  python3 "${REPO_ROOT}/scripts/analysis/ensure_dataset_stats.py" --dataset-root "${dataset_root}" --field rgb_stats_path
}

ecc_depth_stats_json_for_dataset_root() {
  local dataset_root="${1}"
  python3 "${REPO_ROOT}/scripts/analysis/ensure_dataset_stats.py" --dataset-root "${dataset_root}" --field depth_stats_path
}

ecc_read_rgb_stats_bgr_for_dataset_root() {
  local p
  p="$(ecc_rgb_stats_json_for_dataset_root "${1}")"
  python3 - <<PY
import json
d=json.load(open("${p}","r",encoding="utf-8"))
mean=d["mean_bgr"]
std=d["std_bgr"]
fmt=lambda xs: "[" + ",".join(f"{float(x):.4f}" for x in xs) + "]"
print(fmt(mean), fmt(std))
PY
}

ecc_read_rgb_stats_rgb_for_dataset_root() {
  local p
  p="$(ecc_rgb_stats_json_for_dataset_root "${1}")"
  python3 - <<PY
import json
d=json.load(open("${p}","r",encoding="utf-8"))
mean=d["mean_rgb"]
std=d["std_rgb"]
fmt=lambda xs: "[" + ",".join(f"{float(x):.4f}" for x in xs) + "]"
print(fmt(mean), fmt(std))
PY
}

ecc_read_rgb_stats_bgr6_depth1275_for_dataset_root() {
  local p
  p="$(ecc_rgb_stats_json_for_dataset_root "${1}")"
  python3 - <<PY
import json
d=json.load(open("${p}","r",encoding="utf-8"))
mean=list(d["mean_bgr"]) + [127.5,127.5,127.5]
std=list(d["std_bgr"]) + [127.5,127.5,127.5]
fmt=lambda xs: "[" + ",".join(f"{float(x):.4f}" for x in xs) + "]"
print(fmt(mean), fmt(std))
PY
}

ecc_read_depth_clip_for_dataset_root() {
  local p
  p="$(ecc_depth_stats_json_for_dataset_root "${1}")"
  python3 - <<PY
import json
d=json.load(open("${p}","r",encoding="utf-8"))
print(float(d["p1"]), float(d["p99"]))
PY
}

ecc_num_train_images() {
  local dataset_root="${1}"
  python3 - <<PY
import json
from pathlib import Path
p=Path("${dataset_root}")/"annotations"/"instances_train.json"
d=json.load(open(p,"r",encoding="utf-8"))
print(len(d.get("images",[])))
PY
}

ecc_iters_per_epoch() {
  local num_images="${1}"
  local ims_per_batch="${2}"
  python3 - <<PY
num=int("${num_images}")
b=int("${ims_per_batch}")
print((num + b - 1)//b)
PY
}

ecc_detectron2_budget() {
  local dataset_root="${1}"
  local ims_per_batch="${2}"
  local epochs="${3}"

  local num_images
  num_images="$(ecc_num_train_images "${dataset_root}")"
  local iters_per_epoch
  iters_per_epoch="$(ecc_iters_per_epoch "${num_images}" "${ims_per_batch}")"

  python3 - <<PY
e=int("${epochs}")
ipe=int("${iters_per_epoch}")
max_iter=ipe*e
step1=int(max_iter*0.8)
step2=int(max_iter*0.9)
warmup=ipe
print(ipe, max_iter, step1, step2, warmup, ipe, ipe)
PY
}
