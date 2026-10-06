# Motif

MEME → HOMER → sequence logos. Analysis only. Do not install MEME into the ProRiboGen PyTorch environment.

## Install MEME from the env file

The conda spec is `motif/meme.yml` (environment name `meme_new`, MEME **5.0.5**).

```bash
conda env create -f motif/meme.yml
conda activate meme_new
meme -version
which meme
```

Reuse an existing env:

```bash
conda env update -n meme_new -f motif/meme.yml --prune
conda activate meme_new
```

Needs `conda-forge` and `bioconda` (already listed in the yml). Linux-64.

## Test (analysis)

Run from the **repository root**. Sample first: `bash generation/scripts/run_sample.sh`

```bash
conda activate meme_new
FASTA=generation/outputs/sample/fasta_merged

python motif/find_motif_meme.py "$FASTA" motif/outputs/meme --threads 8
python motif/meme_dirs_to_homer.py motif/outputs/meme motif/outputs/homer
python motif/plot_homer_logos.py motif/outputs/homer motif/outputs/logos
```

`find_motif_meme.py` uses `motif/background.txt` next to the script.

If `meme_dirs_to_homer.py` says there is no `meme.txt`, MEME has not succeeded — check `ls motif/outputs/meme/*/meme.txt`. Drawing logos needs `logomaker` (ProRiboGen env is fine for the last command only).
