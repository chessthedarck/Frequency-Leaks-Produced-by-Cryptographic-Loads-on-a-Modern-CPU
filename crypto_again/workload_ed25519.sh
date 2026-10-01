#!/usr/bin/env bash
set -euo pipefail

DURATION="${1:-60}"
OUTDIR="${2:-results/ed25519}"

mkdir -p "$OUTDIR"
RUN_ID="$(date +%s%3N)"
OUTCSV="$OUTDIR/ed25519_iters_${RUN_ID}.csv"

echo "iter,start_ns,end_ns,duration_us" > "$OUTCSV"

start_global=$(date +%s)
i=0

while [ $(( $(date +%s) - start_global )) -lt "$DURATION" ]; do
  i=$((i+1))
  start_ns=$(date +%s%N)
  openssl pkeyutl -sign -rawin -inkey ed25519.pem -in msg.bin -out /dev/null
  end_ns=$(date +%s%N)
  dur_us=$(( (end_ns - start_ns) / 1000 ))
  echo "$i,$start_ns,$end_ns,$dur_us" >> "$OUTCSV"
done

echo "Wrote: $OUTCSV"
