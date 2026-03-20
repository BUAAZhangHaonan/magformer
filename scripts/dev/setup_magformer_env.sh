#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-magformer}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
MAX_JOBS="${MAX_JOBS:-8}"

run_in_env() {
  conda run -n "${ENV_NAME}" "$@"
}

echo "[setup] repo root: ${REPO_ROOT}"
echo "[setup] env name: ${ENV_NAME}"
echo "[setup] CUDA_HOME=${CUDA_HOME}"
echo "[setup] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] creating conda env ${ENV_NAME}"
  conda create -n "${ENV_NAME}" python=3.11 -y
fi

echo "[setup] installing torch + torchvision"
run_in_env pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1 \
  torchvision==0.20.1

echo "[setup] installing core python dependencies"
run_in_env pip install \
  numpy==1.23.5 \
  scipy==1.10.1 \
  pydantic==2.11.2 \
  PyYAML==6.0.2 \
  omegaconf==2.3.0 \
  pycocotools==2.0.11 \
  matplotlib==3.10.8 \
  tqdm==4.67.3 \
  tensorboard==2.20.0 \
  scikit-learn==1.7.2 \
  timm==1.0.19 \
  opencv-python==4.10.0.84 \
  open3d==0.19.0 \
  fvcore==0.1.5.post20221221 \
  iopath==0.1.9 \
  yacs==0.1.8 \
  hydra-core==1.3.2 \
  imageio==2.37.3 \
  h5py==3.16.0 \
  submitit==1.5.4 \
  scikit-image==0.24.0 \
  shapely==2.1.2 \
  rapidfuzz==3.14.3 \
  transforms3d==0.4.2 \
  easydict==1.13 \
  torchfile==0.1.0 \
  pytest==9.0.2 \
  black==26.3.1 \
  ruff==0.15.7 \
  isort==8.0.1 \
  wandb==0.25.1 \
  requests \
  psutil==7.2.2 \
  polars==1.39.3 \
  ultralytics-thop==2.0.18

echo "[setup] installing local editable packages"
CUDA_HOME="${CUDA_HOME}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS="${MAX_JOBS}" \
  run_in_env pip install -e "${REPO_ROOT}" --no-deps
CUDA_HOME="${CUDA_HOME}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS="${MAX_JOBS}" \
  run_in_env pip install -e "${REPO_ROOT}/baselines/detectron2" --no-deps --no-build-isolation
run_in_env pip install -e "${REPO_ROOT}/baselines/ultralytics" --no-deps --no-build-isolation

echo "[setup] compiling official Mask2Former deformable attention op"
CUDA_HOME="${CUDA_HOME}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS="${MAX_JOBS}" \
  conda run -n "${ENV_NAME}" bash -lc \
  "cd '${REPO_ROOT}/baselines/Mask2Former/mask2former/modeling/pixel_decoder/ops' && python setup.py build install"

echo "[setup] note: UOAIS/AdelaiDet is intentionally not installed with pip"
echo "[setup]       its vendored ECC path imports from source tree and optional old CUDA ops"
echo "[setup]       are not required for wrapper startup on the current torch stack."

echo "[setup] quick verification"
run_in_env python -c "import torch, detectron2, magformer, ultralytics; print(torch.__version__)"
run_in_env python "${REPO_ROOT}/tools/train.py" --help >/dev/null
run_in_env python "${REPO_ROOT}/baselines/run_detectron2_0831_1k.py" -- --help >/dev/null
run_in_env python "${REPO_ROOT}/baselines/run_official_mask2former_0831_1k.py" -- --help >/dev/null
run_in_env python "${REPO_ROOT}/baselines/run_msmformer_0831_1k.py" -- --help >/dev/null
run_in_env python "${REPO_ROOT}/baselines/run_uoais_0831_1k.py" -- --help >/dev/null
run_in_env python "${REPO_ROOT}/baselines/run_ucn_0831_1k.py" --help >/dev/null || true
run_in_env yolo help >/dev/null

echo "[setup] done"
