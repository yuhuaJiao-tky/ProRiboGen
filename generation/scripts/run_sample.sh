#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT_BASE="${OUT_BASE:-outputs/sample}"
CFG="${CFG:-config/sample.json}"
CKPT="${CKPT:-checkpoints/generator.pt}"
BS="${BS:-64}"
CSV_DIR="${CSV_DIR:-data/csvs}"
PREFIX="${PREFIX:-sample_test}"
TEST_CSV="${TEST_CSV:-data/test.csv}"
ENTRY="${ENTRY:-sample.py}"
NUM_SEQ="${NUM_SEQ:-1024}"
NUM_GPUS="${NUM_GPUS:-8}"
PYTHON="${PYTHON:-python3}"
LOG="${LOG:-logs/sample.nohup.log}"

mkdir -p logs "$OUT_BASE" "$CSV_DIR"

[[ -f "$CKPT" ]] || { echo "missing checkpoint: $CKPT" >&2; exit 1; }
[[ -f "$CFG" ]] || { echo "missing config: $CFG" >&2; exit 1; }
[[ -f "$ENTRY" ]] || { echo "missing entry: $ENTRY" >&2; exit 1; }

"$PYTHON" scripts/build_sample_csv_from_test.py \
  --test-csv "$TEST_CSV" \
  --prefix "$PREFIX" \
  --out-dir "$CSV_DIR" \
  --length-bp 80 \
  --num-sequences "$NUM_SEQ" \
  --num-gpus "$NUM_GPUS"

echo "[$(date '+%F %T')] START sample  bs=$BS  n=$NUM_SEQ  out=$OUT_BASE"
echo "  entry=$ENTRY  ckpt=$CKPT  cfg=$CFG"

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
  CSV="${CSV_DIR}/${PREFIX}_gpu${i}.csv"
  OUT="${OUT_BASE}/gpu${i}"
  mkdir -p "$OUT"
  GPU_LOG="${OUT_BASE}/gpu${i}.nohup.log"
  nlines=$(($(wc -l < "$CSV") - 1))
  echo "  GPU${i}: ${nlines} proteins -> $OUT"
  env -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES \
    CUDA_VISIBLE_DEVICES="${i}" \
    PYTHONUNBUFFERED=1 \
    "$PYTHON" "$ENTRY" \
      --config "$ROOT/$CFG" \
      --checkpoint "$ROOT/$CKPT" \
      --sample-csv "$ROOT/$CSV" \
      --output-dir "$ROOT/$OUT" \
      --batch-size "$BS" \
      --device cuda:0 \
      >"$GPU_LOG" 2>&1 &
  pids+=($!)
done

ec=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    ec=1
  fi
done

mkdir -p "${OUT_BASE}/fasta_merged"
for d in "${OUT_BASE}"/gpu[0-9]*; do
  [[ -d "$d" ]] || continue
  cp -f "$d"/*.fasta "${OUT_BASE}/fasta_merged/" 2>/dev/null || true
done
n=$(ls "${OUT_BASE}/fasta_merged"/*.fasta 2>/dev/null | wc -l)
echo "[$(date '+%F %T')] Merged ${n} fasta -> ${OUT_BASE}/fasta_merged"

if [[ "$ec" -ne 0 ]]; then
  echo "[$(date '+%F %T')] FAIL sample (exit=$ec)" >&2
  exit "$ec"
fi
echo "[$(date '+%F %T')] DONE sample"
