#!/usr/bin/env bash
set -euo pipefail

DURATION="${1:-60}"
OUTDIR="${2:-results/baseline}"

mkdir -p "$OUTDIR"
RUN_ID="$(date +%s%3N)"
OUTTXT="$OUTDIR/baseline_${RUN_ID}.txt"

echo "baseline_start_ms=$(date +%s%3N)" > "$OUTTXT"
sleep "$DURATION"
echo "baseline_end_ms=$(date +%s%3N)" >> "$OUTTXT"

echo "Wrote: $OUTTXT"
