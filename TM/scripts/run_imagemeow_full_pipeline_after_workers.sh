#!/usr/bin/env bash
set -euo pipefail

ROOT="/ssd/rongye/DD-RUO/TM"
LOG_DIR="$ROOT/results/tm/ImageNet/imagemeow/102"
mkdir -p "$LOG_DIR"

echo "$(date '+%F %T') waiting for imagemeow expert workers: ddruo_meow_g4/g5/g6/g7"

while true; do
    alive=0
    for session in ddruo_meow_g4 ddruo_meow_g5 ddruo_meow_g6 ddruo_meow_g7; do
        if tmux has-session -t "$session" 2>/dev/null; then
            alive=$((alive + 1))
        fi
    done

    if [[ "$alive" -eq 0 ]]; then
        break
    fi

    echo "$(date '+%F %T') $alive expert worker(s) still running"
    sleep 300
done

echo "$(date '+%F %T') expert workers finished; merging replay buffers"
bash "$ROOT/scripts/merge_buffer_imagemeow.sh"

echo "$(date '+%F %T') merged buffers ready; starting TM stage1/stage2"
bash "$ROOT/scripts/run_pool_tm_imagemeow_two_stage.sh"

echo "$(date '+%F %T') full imagemeow pipeline completed"
