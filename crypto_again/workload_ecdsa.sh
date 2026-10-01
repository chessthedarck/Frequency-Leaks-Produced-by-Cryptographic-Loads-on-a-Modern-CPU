#!/usr/bin/env bash
set -euo pipefail

DURATION="${1:-60}"
OUTDIR="${2:-results/ecdsa}"
KEY="ecdsa_p256.pem"

# Générer la clé si absente
if [ ! -f "$KEY" ]; then
  echo "[*] Génération de la clé ECDSA P-256..."
  openssl ecparam -name prime256v1 -genkey -noout -out "$KEY"
  echo "[OK] Clé écrite: $KEY"
fi

mkdir -p "$OUTDIR"
RUN_ID="$(date +%s%3N)"
OUTCSV="$OUTDIR/ecdsa_iters_${RUN_ID}.csv"

echo "iter,start_ns,end_ns,duration_us" > "$OUTCSV"

start_global=$(date +%s)
i=0

while [ $(( $(date +%s) - start_global )) -lt "$DURATION" ]; do
  i=$((i+1))
  start_ns=$(date +%s%N)
  openssl dgst -sha256 -sign "$KEY" -out /dev/null msg.bin
  end_ns=$(date +%s%N)
  dur_us=$(( (end_ns - start_ns) / 1000 ))
  echo "$i,$start_ns,$end_ns,$dur_us" >> "$OUTCSV"
done

echo "Wrote: $OUTCSV"
