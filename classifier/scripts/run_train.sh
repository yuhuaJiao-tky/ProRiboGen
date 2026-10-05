#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CODE_DIR="$PROJECT_ROOT/code"
GEN_ROOT="$(cd "$PROJECT_ROOT/../generation" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export MASTER_PORT="${MASTER_PORT:-29538}"
export CLASSIFIER_DIST_TIMEOUT_SEC="${CLASSIFIER_DIST_TIMEOUT_SEC:-14400}"

H5="$GEN_ROOT/data/protein_embeddings.h5"
PRETRAINED="$GEN_ROOT/checkpoints/generator.pt"
GEN_CONFIG="$GEN_ROOT/config/train.json"

TRAIN_SRC="${TRAIN_SRC:-$GEN_ROOT/data/train.csv}"
TRAIN_LABELED="${TRAIN_LABELED:-$PROJECT_ROOT/data/train_labeled.csv}"
CKPT_LAST="${CKPT_LAST:-$PROJECT_ROOT/checkpoints/classifier_last.pt}"
CKPT_BEST="${CKPT_BEST:-$PROJECT_ROOT/checkpoints/classifier.pt}"
EPOCH_DIR="${EPOCH_DIR:-$PROJECT_ROOT/checkpoints/epochs}"
OUT_DIR="$PROJECT_ROOT/outputs"
CKPT_DIR="$PROJECT_ROOT/checkpoints"

if [[ -n "${CLASSIFIER_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$CLASSIFIER_GPUS"
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi
NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -cve '^$')

mkdir -p "$OUT_DIR" "$CKPT_DIR" "$EPOCH_DIR" "$PROJECT_ROOT/data"

LOCK_FILE="$OUT_DIR/train_classifier.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "ERROR: another classifier training job is running (lock: $LOCK_FILE)" >&2
  exit 1
fi

if [[ ! -f "$PRETRAINED" ]]; then
  echo "ERROR: pretrained ckpt not found: $PRETRAINED" >&2
  exit 1
fi
if [[ ! -f "$GEN_CONFIG" ]]; then
  echo "ERROR: generator config not found: $GEN_CONFIG" >&2
  exit 1
fi

if [[ "${REBUILD_LABELED:-0}" == "1" || ! -f "$TRAIN_LABELED" ]]; then
  echo "Building interleaved labeled CSV from $TRAIN_SRC ..."
  python3 "$CODE_DIR/build_labeled_csv.py" \
    --input_csv "$TRAIN_SRC" \
    --output_csv "$TRAIN_LABELED" \
    --seed "${SHUFFLE_SEED:-42}" \
    --neg_id_suffix "_shuf0" \
    --interleave_pairs
else
  echo "Using $TRAIN_LABELED (REBUILD_LABELED=1 to rebuild)"
fi

LOG="$OUT_DIR/train_classifier_$(date +%Y%m%d_%H%M%S).log"
echo "train=$TRAIN_LABELED  pretrained=$PRETRAINED  config=$GEN_CONFIG  -> $LOG"

torchrun --standalone --nproc_per_node="$NPROC" "$CODE_DIR/train_classifier.py" \
  --generator_config "$GEN_CONFIG" \
  --train_csv "$TRAIN_LABELED" \
  --pretrained_ckpt "$PRETRAINED" \
  --protein_h5 "$H5" \
  --epochs "${EPOCHS:-15}" \
  --batch_size_per_gpu "${BATCH_SIZE:-8}" \
  --pair_batches \
  --early_stop_patience "${EARLY_STOP:-3}" \
  --warmup_head_epochs "${WARMUP_HEAD_EPOCHS:-2}" \
  --lr_scheduler "${LR_SCHEDULER:-cosine}" \
  --warmup_ratio "${WARMUP_RATIO:-0.05}" \
  --val_fraction "${VAL_FRACTION:-0.1}" \
  --split_seed "${SPLIT_SEED:-42}" \
  --lr_head "${LR_HEAD:-5e-4}" \
  --lr_conditioning "${LR_COND:-1e-5}" \
  --ranking_weight "${RANKING_WEIGHT:-0.5}" \
  --ranking_margin "${RANKING_MARGIN:-0.5}" \
  --freeze_mode conditioning_only \
  --cross_attn_heads "${CROSS_ATTN_HEADS:-8}" \
  --use_bf16 \
  --save_path "$CKPT_LAST" \
  --best_save_path "$CKPT_BEST" \
  --epoch_save_dir "$EPOCH_DIR" \
  --best_on val \
  --no-save_every_epoch \
  --log_every 300 \
  2>&1 | tee -a "$LOG"
