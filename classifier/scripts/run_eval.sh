#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CODE_DIR="$PROJECT_ROOT/code"
GEN_ROOT="$(cd "$PROJECT_ROOT/../generation" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -cve '^$')

H5="$GEN_ROOT/data/protein_embeddings.h5"
GEN_CONFIG="$GEN_ROOT/config/train.json"
TEST_SRC="${TEST_SRC:-$GEN_ROOT/data/test.csv}"
TEST_LABELED="${TEST_LABELED:-$PROJECT_ROOT/data/test_labeled.csv}"
OUTPUT_PREDS="${OUTPUT_PREDS:-$PROJECT_ROOT/data/test_preds.csv}"
OUTPUT_LABELED_PREDS="${OUTPUT_LABELED_PREDS:-${TEST_LABELED%.csv}_preds.csv}"
CKPT="${CLASSIFIER_CKPT:-$PROJECT_ROOT/checkpoints/classifier.pt}"
PRETRAINED="${PRETRAINED:-$GEN_ROOT/checkpoints/generator.pt}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p "$PROJECT_ROOT/outputs" "$PROJECT_ROOT/data"

if [[ "${REBUILD_LABELED:-0}" == "1" || ! -f "$TEST_LABELED" ]]; then
  echo "Building test labeled CSV from $TEST_SRC ..."
  python3 "$CODE_DIR/build_labeled_csv.py" \
    --input_csv "$TEST_SRC" \
    --output_csv "$TEST_LABELED" \
    --seed "${SHUFFLE_SEED:-42}" \
    --neg_id_suffix "_shuf0"
fi

LOG="$PROJECT_ROOT/outputs/eval_$(date +%Y%m%d_%H%M%S).log"
echo "test=$TEST_LABELED  ckpt=$CKPT  config=$GEN_CONFIG  -> $LOG"

torchrun --standalone --nproc_per_node="$NPROC" "$CODE_DIR/eval_classifier.py" \
  --test_csv "$TEST_LABELED" \
  --generator_config "$GEN_CONFIG" \
  --classifier_ckpt "$CKPT" \
  --pretrained_ckpt "$PRETRAINED" \
  --protein_h5 "$H5" \
  --batch_size_per_gpu "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --use_bf16 \
  --output_preds_csv "$OUTPUT_PREDS" \
  --preds_wide_src "$TEST_SRC" \
  --output_labeled_preds_csv "$OUTPUT_LABELED_PREDS" \
  2>&1 | tee "$LOG"
