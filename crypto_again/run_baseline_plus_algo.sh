#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-random}"
DURATION="${2:-60}"
CYCLES="${3:-3}"

crypto_algos=(aes rsa ed25519 sha256 ecdsa)

if [[ "$MODE" == "random" ]]; then
  ALGO="${crypto_algos[RANDOM % ${#crypto_algos[@]}]}"
elif [[ " ${crypto_algos[*]} " =~ " ${MODE} " ]]; then
  ALGO="$MODE"
else
  echo "Usage: $0 <aes|rsa|ed25519|sha256|ecdsa|random> [duration_seconds] [cycles]"
  exit 1
fi

if ! [[ "$CYCLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "Cycles must be a positive integer"
  exit 1
fi

OUTPUT_ROOT="output_cycles"
OUTPUT_BASENAME="${ALGO}_plus_baseline"
OUTPUT_DIR="${OUTPUT_ROOT}/${OUTPUT_BASENAME}"
FINAL_RUN_ID="$(date +%s%3N)"
FINAL_CSV="${OUTPUT_DIR}/${OUTPUT_BASENAME}_${FINAL_RUN_ID}.csv"
HEADER_WRITTEN=0

mkdir -p "$OUTPUT_DIR"
find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.csv' -delete

echo "[*] algo choisi: $ALGO"
echo "[*] durée par exécution: $DURATION s"
echo "[*] cycles: $CYCLES (baseline/crypto alterné)"
echo "[*] sortie finale: $FINAL_CSV"

append_csv_to_final() {
  local source_csv="$1"
  local cycle="$2"
  local phase="$3"
  local source_name
  source_name="$(basename "$source_csv")"

  if [[ ! -f "$source_csv" ]]; then
    echo "[ERREUR] CSV introuvable: $source_csv"
    exit 1
  fi

  if (( HEADER_WRITTEN == 0 )); then
    printf 'cycle,phase,source_csv,%s\n' "$(head -n 1 "$source_csv")" > "$FINAL_CSV"
    HEADER_WRITTEN=1
  fi

  tail -n +2 "$source_csv" | awk -v cycle="$cycle" -v phase="$phase" -v source_name="$source_name" 'BEGIN { OFS="," } { print cycle, phase, source_name, $0 }' >> "$FINAL_CSV"
}

for (( i=1; i<=CYCLES; i++ )); do
  if (( i % 2 == 1 )); then
    SCENARIO="baseline"
    PHASE="baseline"
    echo "[*] cycle $i/$CYCLES: baseline"
  else
    SCENARIO="$ALGO"
    PHASE="$ALGO"
    echo "[*] cycle $i/$CYCLES: crypto '$ALGO'"
  fi

  ./run_experiment.sh "$SCENARIO" "$DURATION"

  META="$(ls -t "results/meta/${SCENARIO}_"*.txt | head -n 1)"
  if [[ -z "$META" || ! -f "$META" ]]; then
    echo "[ERREUR] méta introuvable pour le scénario '$SCENARIO'"
    exit 1
  fi

  CPUCSV="$(awk -F= '/^cpu_csv=/{print $2}' "$META")"
  append_csv_to_final "$CPUCSV" "$i" "$PHASE"
  echo "[*] ajouté au CSV final: $CPUCSV"
  echo ""
done

echo "[*] terminé"
echo "[OK] CSV consolidé: $FINAL_CSV"
