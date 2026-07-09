#!/usr/bin/env bash
set -uo pipefail

HF_BIN="/ssd/rongye/miniconda3/envs/dd_ruo/bin/hf"
REPO="ILSVRC/imagenet-1k"
DEST="/ssd/rongye/data/imagenet-1k-hf"

download_one() {
    local file="$1"
    local attempt=1
    local delay

    while true; do
        echo "[$(date '+%F %T')] START ${file} attempt=${attempt}"
        if env \
            -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
            -u http_proxy -u https_proxy -u all_proxy \
            HF_ENDPOINT=https://hf-mirror.com \
            HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 \
            "$HF_BIN" download "$REPO" "$file" \
            --repo-type dataset --local-dir "$DEST" --quiet; then
            echo "[$(date '+%F %T')] DONE  ${file}"
            return 0
        fi

        delay=$((attempt * 15))
        if ((delay > 120)); then
            delay=120
        fi
        echo "[$(date '+%F %T')] RETRY ${file} attempt=${attempt} sleep=${delay}s" >&2
        attempt=$((attempt + 1))
        sleep "$delay"
    done
}

export -f download_one
export HF_BIN REPO DEST

{
    for i in $(seq -w 0 293); do
        printf 'data/train-%05d-of-00294.parquet\n' "$((10#$i))"
    done
    for i in $(seq -w 0 13); do
        printf 'data/validation-%05d-of-00014.parquet\n' "$((10#$i))"
    done
} | xargs -r -n 1 -P "${DOWNLOAD_WORKERS:-4}" bash -c 'download_one "$1"' _

echo "[$(date '+%F %T')] ALL DOWNLOADS COMPLETE"
