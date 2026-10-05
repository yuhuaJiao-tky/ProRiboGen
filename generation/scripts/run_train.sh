#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CFG="${CFG:-config/train.json}"
LOG="${LOG:-train.nohup.log}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"

mkdir -p "$CKPT_DIR"

echo "Config: $CFG"
echo "Entry:  train.py"
echo "Checkpoints: $CKPT_DIR"
echo "Nohup log: $LOG"

nohup python3 train.py --config "$CFG" >> "$LOG" 2>&1 &
echo "PID=$!"
echo "tail -f $LOG"
echo "tail -f train.log"
