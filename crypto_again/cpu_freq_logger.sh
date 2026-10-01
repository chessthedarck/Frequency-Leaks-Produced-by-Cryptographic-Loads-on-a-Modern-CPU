#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="/var/log/cpu_freq"
ACTIVITY_FILE="$OUT_DIR/activity_label"
mkdir -p "$OUT_DIR"

START_NS="$(date +%s%N)"
RUN_ID_MS="$((START_NS / 1000000))"
OUT="$OUT_DIR/cpu_freq_${RUN_ID_MS}.csv"
ACTIVITY_LABEL="$(cat "$ACTIVITY_FILE" 2>/dev/null || echo "normal")"

echo "time_ms,cpu,scaling_cur_freq_khz,scaling_min_khz,scaling_max_khz,activity" > "$OUT"

CPUS=$(ls -d /sys/devices/system/cpu/cpu[0-9]* 2>/dev/null | wc -l)

while true; do
  now_ns="$(date +%s%N)"
  elapsed_ms="$(((now_ns - START_NS) / 1000000))"

  for i in $(seq 0 $((CPUS-1))); do
    base="/sys/devices/system/cpu/cpu${i}/cpufreq"
    cur="$(cat "$base/scaling_cur_freq" 2>/dev/null || echo "")"
    mn="$(cat "$base/scaling_min_freq" 2>/dev/null || echo "")"
    mx="$(cat "$base/scaling_max_freq" 2>/dev/null || echo "")"
    echo "$elapsed_ms,$i,$cur,$mn,$mx,$ACTIVITY_LABEL" >> "$OUT"
  done

  sleep 0.1
done
