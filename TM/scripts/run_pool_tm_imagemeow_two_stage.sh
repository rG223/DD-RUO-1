#!/usr/bin/env bash
set -euo pipefail

# Reproduce the released ImageMeow TM chain:
#   stage 1: 8000 iterations, ldb_it=10, from init
#   stage 2: 7000 iterations, ldb_it=150, from stage-1 pool_best.pt
# We use only GPU 4-7 to keep GPU 0-3 free.

ROOT="/ssd/rongye/DD-RUO/TM"
PYTHON="/ssd/rongye/miniconda3/envs/dd_ruo/bin/python"
DATA_PATH="/ssd/rongye/data/imagenet-1k"
BUFFER_PATH="$ROOT/buffers_parallel/imagemeow_merged"
SAVE_ROOT="$ROOT/results/tm/ImageNet/imagemeow/102"
SCRIPT_NAME="$(basename "$0")"
CUDA_ID="4,5,6,7"

COMMON_ARGS=(
    --dataset ImageNet
    --subset imagemeow
    --res 128
    --model ConvNetD5
    --ipc 102
    --sh_file "$SCRIPT_NAME"
    --eval_mode S
    --data_path "$DATA_PATH"
    --buffer_path "$BUFFER_PATH"
    --save_path "$SAVE_ROOT"
    --num_eval 5
    --batch_syn 80
    --layers_v v5
    --arm 32
    --dim 4
    --ldb 0.1
    --lr_img 0.001
    --lr_it 1000
    --zca False
)

STAGE1_FLAG="ImageNet_imagemeow_102ipc_ConvNetD5_TM_pool_1_8000_0.1_0.001_1000_10_128_zca_False_#layers=v5_syn40_arm=32_dim=4_stage1_origin"
STAGE1_DIR="$SAVE_ROOT/$STAGE1_FLAG"
STAGE1_POOL="$STAGE1_DIR/pool_best.pt"

STAGE2_FLAG="ImageNet_imagemeow_102ipc_ConvNetD5_TM_pool_1_7000_0.1_0.001_1000_150_128_zca_False_#layers=v5_syn40_arm=32_dim=4_stage2_from8000"

mkdir -p "$SAVE_ROOT"
cd "$ROOT"

echo "$(date '+%F %T') stage 1 starts: 8000 iterations on GPU $CUDA_ID"
CUDA_VISIBLE_DEVICES="$CUDA_ID" "$PYTHON" -u pool_tm.py \
    "${COMMON_ARGS[@]}" \
    --Iteration 8000 \
    --pool_path init \
    --ldb_it 10 \
    --FLAG "$STAGE1_FLAG"

if [[ ! -f "$STAGE1_POOL" ]]; then
    echo "Missing stage-1 best checkpoint: $STAGE1_POOL" >&2
    exit 1
fi

echo "$(date '+%F %T') stage 2 starts from $STAGE1_POOL: 7000 iterations on GPU $CUDA_ID"
CUDA_VISIBLE_DEVICES="$CUDA_ID" "$PYTHON" -u pool_tm.py \
    "${COMMON_ARGS[@]}" \
    --Iteration 7000 \
    --pool_path "$STAGE1_POOL" \
    --ldb_it 150 \
    --FLAG "$STAGE2_FLAG"

echo "$(date '+%F %T') imagemeow two-stage TM completed"
