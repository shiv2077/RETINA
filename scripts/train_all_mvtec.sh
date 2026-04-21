#!/usr/bin/env bash
# Train Patchcore on the 12 remaining MVTec categories, record per-category
# image/pixel AUROC + wall clock + checkpoint size into a 15-row CSV (the 3
# already-trained categories are appended at the end).
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOG_DIR="logs"
RESULTS_DIR="results"
CKPT_DIR="checkpoints"
mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$CKPT_DIR"

CSV="$RESULTS_DIR/mvtec_stage1_results.csv"
LOG="$LOG_DIR/train_all_mvtec.log"

echo "category,image_auroc,pixel_auroc,wall_clock_s,checkpoint_mb" > "$CSV"
: > "$LOG"

CATEGORIES=(cable capsule carpet grid hazelnut metal_nut pill screw tile toothbrush transistor zipper)

for cat in "${CATEGORIES[@]}"; do
    echo "=== [$(date +%H:%M:%S)] training $cat ===" | tee -a "$LOG"
    start_ts=$(date +%s)
    python scripts/train_anomalib.py --category "$cat" --model patchcore \
        --max-epochs 1 --output-dir ./checkpoints >> "$LOG" 2>&1
    rc=$?
    end_ts=$(date +%s)
    wall=$((end_ts - start_ts))

    if [[ $rc -ne 0 ]]; then
        echo "$cat,FAILED,FAILED,$wall,0" >> "$CSV"
        echo "!!! $cat FAILED rc=$rc wall=${wall}s" | tee -a "$LOG"
        continue
    fi

    line=$(grep -E "^CATEGORY=$cat " "$LOG" | tail -1 || true)
    img=$(echo "$line" | sed -nE 's/.*IMAGE_AUROC=([0-9.]+).*/\1/p')
    pix=$(echo "$line" | sed -nE 's/.*PIXEL_AUROC=([0-9.]+).*/\1/p')
    [[ -z "$img" ]] && img="NA"
    [[ -z "$pix" ]] && pix="NA"

    ckpt="$CKPT_DIR/patchcore_${cat}.ckpt"
    if [[ -f "$ckpt" ]]; then
        mb=$(du -m "$ckpt" | awk '{print $1}')
    else
        mb=0
    fi
    echo "$cat,$img,$pix,$wall,$mb" | tee -a "$CSV"
done

# Append the 3 already-trained rows from the prior session.
echo "bottle,1.0000,0.9856,132,221" >> "$CSV"
echo "leather,1.0000,0.9922,129,243" >> "$CSV"
echo "wood,0.9877,0.9317,120,244" >> "$CSV"

mean=$(awk -F, 'NR>1 && $2 ~ /^[0-9.]+$/ {sum+=$2; n++} END {if (n>0) printf "%.4f", sum/n; else print "NA"}' "$CSV")
echo "MEAN_IMAGE_AUROC_15CAT=$mean" | tee -a "$LOG"
echo "=== done at $(date +%H:%M:%S) ===" | tee -a "$LOG"
