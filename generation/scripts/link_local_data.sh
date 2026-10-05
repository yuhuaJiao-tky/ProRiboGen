#!/usr/bin/env bash
# Symlink large files from this machine, or from the Hugging Face copy folder.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF="${HF:-/work/home/acr7e7g18v/JYH/ProRiboGen_HuggingFace}"
SRC_DATA="${SRC_DATA:-$HF/generation/data}"
SRC_CKPT="${SRC_CKPT:-$HF/generation/checkpoints/generator.pt}"

mkdir -p "$ROOT/data" "$ROOT/data/csvs" "$ROOT/checkpoints"

ln -sfn "$SRC_DATA/train.csv" "$ROOT/data/train.csv"
ln -sfn "$SRC_DATA/test.csv" "$ROOT/data/test.csv"
ln -sfn "$SRC_DATA/protein_embeddings.h5" "$ROOT/data/protein_embeddings.h5"
ln -sfn "$SRC_CKPT" "$ROOT/checkpoints/generator.pt"

echo "Linked:"
ls -l "$ROOT/data/train.csv" "$ROOT/data/test.csv" "$ROOT/data/protein_embeddings.h5" \
  "$ROOT/checkpoints/generator.pt"
