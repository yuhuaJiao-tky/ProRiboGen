#!/usr/bin/env bash
# Link (or copy) Hugging Face weight/data pack into this repo.
# The pack can live anywhere. Layout must match the HF upload folder:
#   generation/data/{train.csv,test.csv,protein_embeddings.h5}
#   generation/checkpoints/generator.pt
#   classifier/data/{train_labeled.csv,test_labeled.csv}
#   classifier/checkpoints/classifier.pt
#
# Usage (from repo root or generation/):
#   bash generation/scripts/link_local_data.sh /path/to/HF_pack
#   HF=/path/to/HF_pack bash generation/scripts/link_local_data.sh
#   COPY=1 bash generation/scripts/link_local_data.sh /path/to/HF_pack
set -euo pipefail

GEN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$GEN_ROOT/.." && pwd)"
DEFAULT_HF="/work/home/acr7e7g18v/JYH/ProRiboGen_HuggingFace"

HF="${1:-${HF:-}}"
if [[ -z "$HF" && -d "$DEFAULT_HF/generation/data" ]]; then
  HF="$DEFAULT_HF"
fi
if [[ -z "$HF" ]]; then
  echo "Set the Hugging Face pack directory (downloaded anywhere):" >&2
  echo "  bash generation/scripts/link_local_data.sh /path/to/ProRiboGen-weights" >&2
  echo "  HF=/path/to/ProRiboGen-weights bash generation/scripts/link_local_data.sh" >&2
  exit 1
fi
HF="$(cd "$HF" && pwd)"

need=(
  "$HF/generation/data/train.csv"
  "$HF/generation/data/test.csv"
  "$HF/generation/data/protein_embeddings.h5"
  "$HF/generation/checkpoints/generator.pt"
  "$HF/classifier/checkpoints/classifier.pt"
)
for f in "${need[@]}"; do
  [[ -e "$f" ]] || { echo "missing in HF pack: $f" >&2; exit 1; }
done

place() {
  local src="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  rm -f "$dest"
  if [[ "${COPY:-0}" == "1" ]]; then
    cp -a "$src" "$dest"
  else
    ln -sfn "$src" "$dest"
  fi
}

place "$HF/generation/data/train.csv" "$GEN_ROOT/data/train.csv"
place "$HF/generation/data/test.csv" "$GEN_ROOT/data/test.csv"
place "$HF/generation/data/protein_embeddings.h5" "$GEN_ROOT/data/protein_embeddings.h5"
mkdir -p "$GEN_ROOT/data/csvs"
place "$HF/generation/checkpoints/generator.pt" "$GEN_ROOT/checkpoints/generator.pt"

if [[ -f "$HF/classifier/data/train_labeled.csv" ]]; then
  place "$HF/classifier/data/train_labeled.csv" "$REPO/classifier/data/train_labeled.csv"
fi
if [[ -f "$HF/classifier/data/test_labeled.csv" ]]; then
  place "$HF/classifier/data/test_labeled.csv" "$REPO/classifier/data/test_labeled.csv"
fi
place "$HF/classifier/checkpoints/classifier.pt" "$REPO/classifier/checkpoints/classifier.pt"

echo "HF pack: $HF"
echo "mode:    $([ "${COPY:-0}" = 1 ] && echo copy || echo symlink)"
ls -l "$GEN_ROOT/data/train.csv" "$GEN_ROOT/data/test.csv" \
  "$GEN_ROOT/data/protein_embeddings.h5" "$GEN_ROOT/checkpoints/generator.pt" \
  "$REPO/classifier/checkpoints/classifier.pt"
