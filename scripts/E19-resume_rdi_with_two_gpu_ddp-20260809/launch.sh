#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hdd3/zhanghaonan/magformer
source_root="$repo_root/source/E18-retrain_rdi_lite_and_mbv3l_from_pretrained-20260808"
script_root="$repo_root/scripts/E19-resume_rdi_with_two_gpu_ddp-20260809"
python_root=/home/hdd3/zhanghaonan/anaconda3/envs/magformer

launch() {
    local session=$1
    local devices=$2
    local port=$3
    local config=$4
    local output_dir=$5

    tmux has-session -t "$session" 2>/dev/null && {
        echo "tmux session already exists: $session" >&2
        exit 1
    }
    mkdir -p "$output_dir"
    if [ -f "$output_dir/console.log" ]; then
        cp "$output_dir/console.log" "$output_dir/failed_5000.log"
    fi
    tmux new-session -d -s "$session" bash -lc "
        set -euo pipefail
        cd '$source_root'
        PYTHONPATH='$source_root' CUDA_VISIBLE_DEVICES='$devices' NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN \
        '$python_root/bin/torchrun' --standalone --nnodes=1 --nproc-per-node=2 --master-port='$port' \
        tools/train.py --config '$config' --eval-only 2>&1 | tee '$output_dir/evaluate_5000.log'
        PYTHONPATH='$source_root' CUDA_VISIBLE_DEVICES='$devices' NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN \
        '$python_root/bin/torchrun' --standalone --nnodes=1 --nproc-per-node=2 --master-port='$port' \
        tools/train.py --config '$config' 2>&1 | tee '$output_dir/console.log'
    "
}

launch magformer-e19-rdi-lite-ddp 1,2 29619 \
    "$script_root/rdi_lite.yaml" \
    "$repo_root/output/experiments/E19-resume_rdi_with_two_gpu_ddp-20260809/rdi_lite"
launch magformer-e19-rdi-mbv3l-ddp 3,4 29620 \
    "$script_root/rdi_mbv3l.yaml" \
    "$repo_root/output/experiments/E19-resume_rdi_with_two_gpu_ddp-20260809/rdi_mbv3l"
