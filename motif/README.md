# Motif

MEME → HOMER → sequence logos. Analysis only. Requires a local `meme` binary.

Run from the repository root. Sample first: `bash generation/scripts/run_sample.sh`

## Test (analysis)

```bash
FASTA=generation/outputs/sample/fasta_merged

python motif/find_motif_meme.py "$FASTA" motif/outputs/meme --threads 8
python motif/meme_dirs_to_homer.py motif/outputs/meme motif/outputs/homer
python motif/plot_homer_logos.py motif/outputs/homer motif/outputs/logos
```

`find_motif_meme.py` uses `motif/background.txt`.
