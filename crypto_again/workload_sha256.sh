#!/usr/bin/env bash
set -euo pipefail

DURATION="${1:-60}"
OUTDIR="${2:-results/sha256}"

mkdir -p "$OUTDIR"
RUN_ID="$(date +%s%3N)"
OUTCSV="$OUTDIR/sha256_iters_${RUN_ID}.csv"

echo "iter,start_ns,end_ns,duration_us" > "$OUTCSV"

start_global=$(date +%s)
i=0

while [ $(( $(date +%s) - start_global )) -lt "$DURATION" ]; do
  i=$((i+1))
  start_ns=$(date +%s%N)
  openssl dgst -sha256 msg.bin > /dev/null
  end_ns=$(date +%s%N)
  dur_us=$(( (end_ns - start_ns) / 1000 ))
  echo "$i,$start_ns,$end_ns,$dur_us" >> "$OUTCSV"
done

echo "Wrote: $OUTCSV"
