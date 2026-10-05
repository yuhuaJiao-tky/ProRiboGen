# Generator data (not in GitHub)

Paths in configs are relative to `generation/`.

| File | Size | Role |
|------|------|------|
| `data/train.csv` | ~175 MB | train pairs |
| `data/test.csv` | ~25 MB | test pairs (33 hold-out proteins) |
| `data/protein_embeddings.h5` | ~1.8 GB | frozen protein embeddings |

CSV columns: `p_id,protein,rna,r_id,s,t,score,type,p_len,r_len` (see `example_pairs.csv`).  
Split: `split_manifest.json` (304 train / 33 test).

This machine: `bash scripts/link_local_data.sh`  
Or copy from `ProRiboGen_HuggingFace/generation/data/`.

If you need to rebuild the H5:

```bash
python scripts/build_protein_embeddings_h5.py \
  --vesm-root /path/to/protein_encoder \
  --fasta data/train_proteins.fasta \
  --output data/protein_embeddings.h5 \
  --fp16
```

Also encode `data/test_proteins.fasta` into the same H5.

Train / sample commands: `generation/README.md`.
