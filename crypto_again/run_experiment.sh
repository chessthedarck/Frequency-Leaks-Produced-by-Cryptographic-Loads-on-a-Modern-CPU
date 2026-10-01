#!/usr/bin/env bash
set -euo pipefail

SCENARIO="${1:-baseline}"
DURATION="${2:-60}"
RUN_ID_MS="$(date +%s%3N)"

mkdir -p results/meta

ACTIVITY_LABEL="cryptographic"
if [ "$SCENARIO" = "baseline" ]; then
  ACTIVITY_LABEL="normal"
fi

echo "[*] Setting activity label: $ACTIVITY_LABEL"
sudo mkdir -p /var/log/cpu_freq
printf '%s\n' "$ACTIVITY_LABEL" | sudo tee /var/log/cpu_freq/activity_label >/dev/null

echo "[*] Installing updated CPU logger..."
sudo install -m 755 ./cpu_freq_logger.sh /usr/local/bin/cpu_freq_logger.sh

echo "[*] Stopping other workload services if any..."
sudo systemctl stop workload-s0 workload-s1 workload-s2 workload-s3 2>/dev/null || true

echo "[*] Restarting CPU frequency logger..."
sudo systemctl restart cpu-freq-logger.service

sleep 1

case "$SCENARIO" in
  baseline)
    ./workload_baseline.sh "$DURATION" "results/baseline"
    ;;
  rsa)
    ./workload_rsa.sh "$DURATION" "results/rsa"
    ;;
  ed25519)
    ./workload_ed25519.sh "$DURATION" "results/ed25519"
    ;;
  aes)
    ./workload_aes.sh "$DURATION" "results/aes"
    ;;
  sha256)
    ./workload_sha256.sh "$DURATION" "results/sha256"
    ;;
  ecdsa)
    ./workload_ecdsa.sh "$DURATION" "results/ecdsa"
    ;;
  *)
    echo "Unknown scenario: $SCENARIO"
    exit 1
    ;;
esac

echo "[*] Stopping CPU logger..."
sudo systemctl stop cpu-freq-logger.service

CPUCSV=$(ls -t /var/log/cpu_freq/cpu_freq_*.csv | head -n 1)
META="results/meta/${SCENARIO}_${RUN_ID_MS}.txt"

echo "scenario=$SCENARIO" > "$META"
echo "activity=$ACTIVITY_LABEL" >> "$META"
echo "duration_s=$DURATION" >> "$META"
echo "cpu_csv=$CPUCSV" >> "$META"

echo "[OK] CPU CSV: $CPUCSV"
echo "[OK] META: $META"
