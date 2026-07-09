#!/usr/bin/env bash
set -euo pipefail

# Reproduce the released ImageFruit TM continuation chain:
#   stage 1: 7000 iterations on GPU 0-7
#   stage 2: reload stage-1 pool_5000.pt and run 2000 iterations on GPU 0-7

ROOT="/ssd/rongye/DD-RUO/TM"
PYTHON="/ssd/rongye/miniconda3/envs/dd_ruo/bin/python"
DATA_PATH="/ssd/rongye/data/imagenet-1k"
BUFFER_PATH="$ROOT/buffers_parallel/imagefruit_merged"
SAVE_ROOT="$ROOT/results/tm/ImageNet/imagefruit/102"
SCRIPT_NAME="$(basename "$0")"

COMMON_ARGS=(
    --dataset ImageNet
    --subset imagefruit
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
    --ldb_it 150
    --zca False
)

STAGE1_FLAG="ImageNet_imagefruit_102ipc_ConvNetD5_TM_pool_1_7000_0.1_0.001_1000_150_128_zca_False_#new0216_slice200_layers=v5_syn40_arm=32_dim=4_stage2"
STAGE1_DIR="$SAVE_ROOT/$STAGE1_FLAG"
STAGE1_POOL="$STAGE1_DIR/pool_5000.pt"

STAGE2_FLAG="ImageNet_imagefruit_102ipc_ConvNetD5_TM_pool_1_2000_0.1_0.001_1000_150_128_zca_False_#layers=v5_syn40_arm=32_dim=4_stage2_from13000"

mkdir -p "$SAVE_ROOT"
cd "$ROOT"

echo "$(date '+%F %T') stage 1 starts: 7000 iterations on GPU 0-7"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PYTHON" -u pool_tm.py \
    "${COMMON_ARGS[@]}" \
    --Iteration 7000 \
    --pool_path init \
    --FLAG "$STAGE1_FLAG"

if [[ ! -f "$STAGE1_POOL" ]]; then
    echo "Missing stage-1 checkpoint: $STAGE1_POOL" >&2
    exit 1
fi

echo "$(date '+%F %T') stage 2 starts from $STAGE1_POOL: 2000 iterations on GPU 0-7"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PYTHON" -u pool_tm.py \
    "${COMMON_ARGS[@]}" \
    --Iteration 2000 \
    --pool_path "$STAGE1_POOL" \
    --FLAG "$STAGE2_FLAG"

echo "$(date '+%F %T') both stages completed"
