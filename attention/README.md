# Attention

Protein–RNA cross-attention on the released generator. Analysis only (no training).

Needs `generation/checkpoints/generator.pt`, protein embeddings, and `data/test.csv`.  
Run from the repository root.

## Test (demo)

```bash
bash attention/run_demo.sh
```

Output: `attention/outputs/demo/`

## Test (generated RNA, hold-out proteins)

First sample: `bash generation/scripts/run_sample.sh`

Two-step pipeline (probe CSV is only an intermediate; results live under `--out-dir`):

1. **Build probes** → `attention/outputs/generated/probes_generated.csv`  
2. **Run attention** → writes `protein_index.csv`, `profiles/`, ratio tables, etc. under `attention/outputs/generated/`  
   (`protein_index.csv` appears only after this step finishes all proteins.)

```bash
mkdir -p attention/outputs/generated

python attention/fasta_to_probe_csv.py \
  generation/outputs/sample/fasta_merged \
  --out attention/outputs/generated/probes_generated.csv

python attention/analyze_rbd_cross_attention.py \
  --rna-csv attention/outputs/generated/probes_generated.csv \
  --domain-json attention/domain_annotations_test.json \
  --all-test-proteins \
  --num-probes 256 \
  --out-dir attention/outputs/generated

python attention/plot_domain_vs_non.py --attn-dir attention/outputs/generated
```

Single protein:

```bash
python attention/analyze_rbd_cross_attention.py \
  --proteins Human-PUM1 \
  --domain-json attention/domain_annotations_test.json \
  --num-probes 256 \
  --out-dir attention/outputs/pum1
```
