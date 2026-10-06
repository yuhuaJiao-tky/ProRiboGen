#!/usr/bin/env bash
# One sample.py process per visible GPU (not DDP).
#   bash scripts/run_sample.sh
#     8 cards -> 8 jobs; 1 card -> 1 job
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_sample.sh
#   NUM_GPUS=2 bash scripts/run_sample.sh          # first 2 of the visible set
#   NUM_GPUS=1 BS=16 bash scripts/run_sample.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT_BASE="${OUT_BASE:-outputs/sample}"
CFG="${CFG:-config/sample.json}"
CKPT="${CKPT:-checkpoints/generator.pt}"
CSV_DIR="${CSV_DIR:-data/csvs}"
PREFIX="${PREFIX:-sample_test}"
TEST_CSV="${TEST_CSV:-data/test.csv}"
ENTRY="${ENTRY:-sample.py}"
NUM_SEQ="${NUM_SEQ:-1024}"
PYTHON="${PYTHON:-python3}"
LOG="${LOG:-logs/sample.nohup.log}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
  for i in "${!GPU_IDS[@]}"; do
    GPU_IDS[$i]="$(echo "${GPU_IDS[$i]}" | tr -d ' ')"
  done
else
  GPU_IDS=()
  if command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ')
  fi
  if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
    GPU_IDS=(0)
  fi
fi
NUM_GPUS="${NUM_GPUS:-${#GPU_IDS[@]}}"
if [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "NUM_GPUS must be >= 1" >&2
  exit 1
fi
if [[ "$NUM_GPUS" -gt "${#GPU_IDS[@]}" ]]; then
  echo "NUM_GPUS=$NUM_GPUS but only ${#GPU_IDS[@]} GPU(s) visible: ${GPU_IDS[*]}" >&2
  echo "Use fewer jobs, e.g. NUM_GPUS=${#GPU_IDS[@]} bash scripts/run_sample.sh" >&2
  exit 1
fi
GPU_IDS=("${GPU_IDS[@]:0:$NUM_GPUS}")
if [[ -z "${BS:-}" ]]; then
  if [[ "$NUM_GPUS" -eq 1 ]]; then BS=16; else BS=64; fi
fi

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

echo "[$(date '+%F %T')] START sample  bs=$BS  n=$NUM_SEQ  gpus=${NUM_GPUS} [${GPU_IDS[*]}]  out=$OUT_BASE"
echo "  entry=$ENTRY  ckpt=$CKPT  cfg=$CFG"

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
  CSV="${CSV_DIR}/${PREFIX}_gpu${i}.csv"
  OUT="${OUT_BASE}/gpu${i}"
  mkdir -p "$OUT"
  GPU_LOG="${OUT_BASE}/gpu${i}.nohup.log"
  nlines=$(($(wc -l < "$CSV") - 1))
  gid="${GPU_IDS[$i]}"
  echo "  GPU${i} (id=${gid}): ${nlines} proteins -> $OUT"
  env -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES \
    CUDA_VISIBLE_DEVICES="${gid}" \
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
