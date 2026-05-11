#!/bin/bash
# Usage: bash run_eval.sh <config> <gpu_id> <log_file>
CONFIG=$1
GPU=$2
LOG=$3

source /home/hdd3/zhanghaonan/anaconda3/bin/activate
conda activate magformer
cd /home/hdd3/zhanghaonan/magformer

# Fix LD_LIBRARY_PATH
NVJIT_PATH=$(python -c "import nvidia.nvjitlink; print(nvidia.nvjitlink.__path__[0]+\"/lib\")")
CUSPARSE_PATH=$(python -c "import nvidia.cusparse; print(nvidia.cusparse.__path__[0]+\"/lib\")")
CUBLAS_PATH=$(python -c "import nvidia.cublas; print(nvidia.cublas.__path__[0]+\"/lib\")")
export LD_LIBRARY_PATH=${NVJIT_PATH}:${CUSPARSE_PATH}:${CUBLAS_PATH}:${LD_LIBRARY_PATH}

# Limit CPU threads but NOT virtual memory (ulimit -v was causing OOM)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

echo "[$(date)] Starting eval: config=$CONFIG gpu=$GPU"
CUDA_VISIBLE_DEVICES=$GPU python tools/evaluate.py \
    --config-file "$CONFIG" \
    --batch-size 4 \
    --num-workers 4 \
    2>&1 | tee "$LOG"
echo "[$(date)] Eval finished: $LOG"
