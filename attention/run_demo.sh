#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export GEN_ROOT="${GEN_ROOT:-$ROOT/generation}"

python3 attention/analyze_rbd_cross_attention.py \
  --domain-json attention/domain_annotations_demo.json \
  --proteins Human-PUM1 Human-HNRNPK Human-U2AF1 Human-DKC1 \
  --num-probes 64 \
  --out-dir attention/outputs/demo

python3 attention/plot_domain_vs_non.py --attn-dir attention/outputs/demo
echo "OK -> attention/outputs/demo"
