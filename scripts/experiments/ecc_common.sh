#!/usr/bin/env bash
set -euo pipefail

ecc_normalize_register() {
  local r="${1:-}"
  case "${r}" in
    0831|0831_1k|ecc0831|ecc0831_1k) echo "0831" ;;
    0909|0909_512|ecc0909|ecc0909_512) echo "0909" ;;
    *)
      echo "Unsupported --register: ${r}" >&2
      return 1
      ;;
  esac
}

ecc_default_dataset_root() {
  local reg
  reg="$(ecc_normalize_register "${1}")"
  if [[ "${reg}" == "0831" ]]; then
    echo "${PROJECT_ROOT}/magformer_datasets/0831_1K"
  else
    echo "${PROJECT_ROOT}/magformer_datasets/0909_512_0.12K"
  fi
}

ecc_dataset_names_coco() {
  local reg
  reg="$(ecc_normalize_register "${1}")"
  if [[ "${reg}" == "0831" ]]; then
    echo "ecc0831_1k_train ecc0831_1k_val"
  else
    echo "ecc0909_512_train ecc0909_512_val"
  fi
}

ecc_dataset_names_coco_rgbd() {
  local reg
  reg="$(ecc_normalize_register "${1}")"
  if [[ "${reg}" == "0831" ]]; then
    echo "ecc0831_1k_rgbd_train ecc0831_1k_rgbd_val"
  else
    echo "ecc0909_512_rgbd_train ecc0909_512_rgbd_val"
  fi
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
  local p
  p="$(ecc_rgb_stats_json "${1}")"
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
  local p
  p="$(ecc_rgb_stats_json "${1}")"
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
  local p
  p="$(ecc_rgb_stats_json "${1}")"
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
  local p
  p="$(ecc_depth_stats_json "${1}")"
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
