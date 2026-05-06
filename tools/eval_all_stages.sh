#!/bin/bash
# Evaluate all VC-SUDA stages
set -eo pipefail
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate magformer
cd /home/hdd3/zhanghaonan/magformer

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

STAGES="a b c d e"
DATASET_ROOT="magformer_datasets/pseudo_real_512"
RESULTS_DIR="output/vc_suda/eval_results"
mkdir -p "$RESULTS_DIR"

echo "=== VC-SUDA Evaluation: $(date) ===" | tee "$RESULTS_DIR/eval_summary.txt"

for STAGE in $STAGES; do
    CKPT="output/vc_suda/stage_${STAGE}/model_best.pth"
    if [ ! -f "$CKPT" ]; then
        echo "Stage $STAGE: SKIP (no checkpoint at $CKPT)" | tee -a "$RESULTS_DIR/eval_summary.txt"
        continue
    fi

    # Determine config file for each stage
    case $STAGE in
        a) CONFIG="configs/vc_suda_stage_a_40ep_512.yaml";;
        b) CONFIG="configs/vc_suda/stage_b_target_warmup.yaml";;
        c) CONFIG="configs/vc_suda/stage_c_semi_supervised.yaml";;
        d) CONFIG="configs/vc_suda/stage_d_domain_alignment.yaml";;
        e) CONFIG="configs/vc_suda/stage_e_final_finetune.yaml";;
    esac

    if [ ! -f "$CONFIG" ]; then
        echo "Stage $STAGE: SKIP (config not found: $CONFIG)" | tee -a "$RESULTS_DIR/eval_summary.txt"
        continue
    fi

    OUT_DIR="$RESULTS_DIR/stage_${STAGE}"
    mkdir -p "$OUT_DIR"

    echo "Stage $STAGE: Evaluating with $CKPT..." | tee -a "$RESULTS_DIR/eval_summary.txt"

    python tools/evaluate.py \
        --config-file "$CONFIG" \
        --dataset-root "$DATASET_ROOT" \
        --weights "$CKPT" \
        --output "$OUT_DIR" \
        --batch-size 1 \
        --num-workers 2 \
        2>&1 | tee "$OUT_DIR/eval.log" | tail -20 >> "$RESULTS_DIR/eval_summary.txt"

    echo "---" >> "$RESULTS_DIR/eval_summary.txt"
done

echo "=== Evaluation Complete: $(date) ===" | tee -a "$RESULTS_DIR/eval_summary.txt"
