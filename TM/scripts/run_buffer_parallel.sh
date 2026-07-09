#!/usr/bin/env bash
set -euo pipefail

ROOT="/ssd/rongye/DD-RUO/TM"
PYTHON="/ssd/rongye/miniconda3/envs/dd_ruo/bin/python"
DATA="/ssd/rongye/data/imagenet-1k"
BASE="$ROOT/buffers_parallel/imagefruit"
CACHE="/ssd/rongye/data/imagenet-1k-cache/imagefruit_128_nozca.pt"

gpus=(2 3 4 5 6 7)
experts=(20 20 20 20 10 10)

mkdir -p "$BASE/logs"

if [[ ! -f "$CACHE" ]]; then
    mkdir -p "$(dirname "$CACHE")" "$BASE/cache_builder"
    echo "$(date '+%F %T') building shared tensor cache"
    CUDA_VISIBLE_DEVICES="" "$PYTHON" -u "$ROOT/buffer.py" \
        --dataset=ImageNet --subset=imagefruit --model=ConvNetD5 \
        --data_path="$DATA" --buffer_path="$BASE/cache_builder" \
        --dataset_cache="$CACHE" --num_experts=0 --seed=0
    echo "$(date '+%F %T') shared tensor cache ready: $CACHE"
fi

for i in "${!gpus[@]}"; do
    gpu="${gpus[$i]}"
    count="${experts[$i]}"
    output="$BASE/worker${i}"
    log="$BASE/logs/gpu${gpu}.log"
    session="ddruo_fruit_g${gpu}"

    mkdir -p "$output"
    tmux new-session -d -s "$session" \
        "cd $ROOT && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
        NUMEXPR_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u buffer.py \
        --dataset=ImageNet --subset=imagefruit --model=ConvNetD5 \
        --data_path=$DATA --buffer_path=$output --dataset_cache=$CACHE \
        --train_epochs=50 --num_experts=$count --seed=$gpu \
        2>&1 | tee $log"
    echo "$(date '+%F %T') started GPU $gpu with $count experts"
done

echo "$(date '+%F %T') all workers launched"
