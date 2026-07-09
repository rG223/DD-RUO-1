#!/usr/bin/env bash
set -euo pipefail

ROOT="/ssd/rongye/DD-RUO/TM"
SRC="$ROOT/buffers_parallel/imagemeow"
DST="$ROOT/buffers_parallel/imagemeow_merged/ImageNet/imagemeow/128/ConvNetD5"

mkdir -p "$DST"

if compgen -G "$DST/replay_buffer_*.pt" > /dev/null; then
    echo "Merged imagemeow buffers already exist in $DST" >&2
    echo "Move them away first if you want to rebuild the merged view." >&2
    exit 1
fi

n=0
for worker in "$SRC"/worker*; do
    worker_dir="$worker/ImageNet/imagemeow/128/ConvNetD5"
    [[ -d "$worker_dir" ]] || continue
    while IFS= read -r src_file; do
        ln -s "$src_file" "$DST/replay_buffer_${n}.pt"
        echo "replay_buffer_${n}.pt -> $src_file"
        n=$((n + 1))
    done < <(find "$worker_dir" -maxdepth 1 -type f -name 'replay_buffer_*.pt' | sort -V)
done

if [[ "$n" -ne 10 ]]; then
    echo "Expected 10 merged replay buffers, got $n" >&2
    exit 1
fi

echo "Merged $n imagemeow replay buffers into $DST"
