#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/g203-4028/magformer
EXPERIMENT_DIR="$ROOT/scripts/E17-compare_rdi_lite_and_mbv3l_fast_training-20260807"
OUTPUT_ROOT="$ROOT/output/experiments/E17-compare_rdi_lite_and_mbv3l_fast_training-20260807"
PYTHON=/home/g203-4028/miniconda3/envs/magformer/bin/python3.11

tmux new-session -d -s magformer-e17-rdi-lite-gpu4 \
  "cd $ROOT && mkdir -p $OUTPUT_ROOT/rdi_lite && exec env CUDA_VISIBLE_DEVICES=4 $PYTHON -u tools/train.py --config-file $EXPERIMENT_DIR/rdi_lite.yaml >> $OUTPUT_ROOT/rdi_lite/console.log 2>&1"

tmux new-session -d -s magformer-e17-rdi-mbv3l-gpu5 \
  "cd $ROOT && mkdir -p $OUTPUT_ROOT/rdi_mbv3l && exec env CUDA_VISIBLE_DEVICES=5 $PYTHON -u tools/train.py --config-file $EXPERIMENT_DIR/rdi_mbv3l.yaml >> $OUTPUT_ROOT/rdi_mbv3l/console.log 2>&1"
