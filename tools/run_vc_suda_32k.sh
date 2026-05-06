#!/bin/bash
# Run VC-SUDA 32K pipeline (Stages A-E)
# Usage: bash tools/run_vc_suda_32k.sh [stage]
#   stage: a|b|c|d|e|all (default: all)

set -euo pipefail

STAGE=${1:-all}
GPUS="4,5,6,7"
NPROC=4
PORT=29512
CONFIG_DIR="configs/vc_suda_32k"
LOG_DIR="logs/vc_suda_32k"
mkdir -p "$LOG_DIR"

run_stage() {
    local stage=$1
    local config="$CONFIG_DIR/stage_${stage}_*.yaml"
    config=$(ls $config 2>/dev/null | head -1)

    if [ -z "$config" ]; then
        echo "[ERROR] Config not found for stage $stage"
        exit 1
    fi

    local name=$(basename "$config" .yaml)
    local tmux_name="vc_suda_32k_${stage}"

    echo "=========================================="
    echo "[Stage $stage] Starting: $name"
    echo "  Config: $config"
    echo "  GPUs: $GPUS ($NPROC procs)"
    echo "  Log: $LOG_DIR/stage_${stage}.log"
    echo "=========================================="

    tmux kill-session -t "$tmux_name" 2>/dev/null || true
    tmux new-session -d -s "$tmux_name" \
        "CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NPROC --master_port=$PORT tools/train_vc_suda.py --config $config 2>&1 | tee $LOG_DIR/stage_${stage}.log"

    echo "[Stage $stage] Launched in tmux session: $tmux_name"
    echo "  Monitor: tmux attach -t $tmux_name"
    echo "  Log: tail -f $LOG_DIR/stage_${stage}.log"
}

case "$STAGE" in
    a|b|c|d|e)
        run_stage "$STAGE"
        ;;
    all)
        echo "Will run stages A through E sequentially."
        echo "Each stage waits for the previous to complete."
        echo ""
        for s in a b c d e; do
            run_stage "$s"
            echo ""
            echo "[Stage $s] Waiting for completion..."
            tmux wait-for "vc_suda_32k_${s}" 2>/dev/null || {
                echo "[Stage $s] tmux wait-for not available. Check manually."
                echo "  When stage $s is done, run: bash $0 $((++s))"
                exit 0
            }
        done
        echo "All stages complete!"
        ;;
    *)
        echo "Usage: bash $0 [a|b|c|d|e|all]"
        exit 1
        ;;
esac
