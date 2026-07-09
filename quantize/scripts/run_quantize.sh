#!/bin/bash
# Post-quantization evaluation for distilled datasets (shared across TM/DM/DC).
#
# This script applies post-training quantization to a pre-trained synthesis
# network and evaluates the distilled dataset quality after quantization.
#
# Usage: bash scripts/run_quantize.sh

cd "$(dirname "$0")/.."

# ============ Configuration ============

PYTHON="/ssd/rongye/miniconda3/envs/dd_ruo/bin/python"
CUDA_ID=0,1,2,3,4,5,6,7
DATASET="ImageNet"
SUBSET="imagefruit"
IPC=102
DATA_PATH="/ssd/rongye/data/imagenet-1k"

# TensorPool architecture (must match the pre-trained checkpoint)
LAYERS_V="v5"
ARM=32
DIM=4

# Evaluation
NUM_EVAL=5
SYN_LR=0.006132122594863176
MSE_ERR=0.0000005

# Pre-trained pool checkpoint (modify this to your checkpoint path)
POOL_PATH="/ssd/rongye/DD-RUO/TM/results/tm/ImageNet/imagefruit/102/ImageNet_imagefruit_102ipc_ConvNetD5_TM_pool_1_2000_0.1_0.001_1000_150_128_zca_False_#layers=v5_syn40_arm=32_dim=4_stage2_from13000/pool_best.pt"

# Output directory
SAVE_DIR="./results/quantize/${DATASET}/${SUBSET}/${IPC}"

# ============ Run ============

export CUDA_VISIBLE_DEVICES=${CUDA_ID}

FLAG="${DATASET}_${SUBSET}_ipc${IPC}_${ARM}_${DIM}"
TIMESTAMP=$(date +"%m%d_%H%M")
LOG_DIR="${SAVE_DIR}/${FLAG}"
LOG_FILE="${LOG_DIR}/${TIMESTAMP}.log"
mkdir -p "${LOG_DIR}"

nohup "${PYTHON}" -u quantize_pool.py \
    --dataset ${DATASET} \
    --subset ${SUBSET} \
    --data_path ${DATA_PATH} \
    --ipc ${IPC} \
    --layers_v ${LAYERS_V} \
    --arm ${ARM} \
    --dim ${DIM} \
    --syn_lr_set ${SYN_LR} \
    --mse_err ${MSE_ERR} \
    --pool_path ${POOL_PATH} \
    --save_path ${SAVE_DIR} \
    --FLAG ${FLAG} \
    --num_eval ${NUM_EVAL} \
    > "${LOG_FILE}" 2>&1 &

echo "Script started. Logging to ${LOG_FILE}"
echo "Process ID: $!"
