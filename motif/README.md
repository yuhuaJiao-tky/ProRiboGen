# Motif

One-shot MEME → HOMER (all E ≤ 0.05) → paper-style logos. Analysis only.  
Do not install MEME into the ProRiboGen PyTorch environment.

## Install MEME from the env file

The conda spec is `motif/meme.yml` (environment name `meme_new`, MEME **5.0.5**).

```bash
conda env create -f motif/meme.yml
conda activate meme_new
meme -version
which meme
```

`meme.yml` already includes `logomaker` and `matplotlib` for paper-style logos.

Reuse an existing env:

```bash
conda env update -n meme_new -f motif/meme.yml --prune
conda activate meme_new
```

## Test (analysis)

Run from the **repository root**. Sample first: `bash generation/scripts/run_sample.sh`

```bash
conda activate meme_new
python motif/run_motif_pipeline.py \
  generation/outputs/sample/fasta_merged \
  motif/outputs/meme \
  --threads 8
```

Per protein under `motif/outputs/meme/<p_id>/`:

| File | Content |
|------|---------|
| `meme.txt` | MEME text output |
| `<p_id>.motif` | HOMER PWMs for **all** motifs with E ≤ 0.05 |
| `logo1.png` … `logoN.png` | paper-style logos (same colors as the notebook); `logo1` = lowest E |

MEME’s own EPS/PNG logos are deleted after each run so they do not mix with this style.

Uses `motif/background.txt` next to the scripts.
