# ProRiboGen

Protein-conditioned RNA generation.

<p align="center">
  <img src="methods/pngs/web.png" alt="ProRiboGen overview" width="900"/>
</p>

---

## 1. Environment

Python **3.10** (3.9 and below cannot run this code: `int | None` type hints).  
Always use **`python -m pip`**, not bare `pip`. On clusters, `pip` often still points at system Python 3.8 after `conda activate`.

```bash
git clone https://github.com/yuhuaJiao-tky/ProRiboGen.git
cd ProRiboGen
conda create -n ProRiboGen python=3.10 -y
conda activate ProRiboGen
hash -r
which python
python --version          # must be 3.10.x, path under envs/ProRiboGen
```

Install packages with **that same** `python`. Install **Torch first** from the CUDA 12.4 index (default PyPI may ship CUDA 13), then the rest from `requirements.txt`:

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -c "import torch, transformers, h5py; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
```

Expected: `2.4.1+cu124 True 4.54.1`.

`requirements.txt` does **not** list `torch` (comment only), so it will not overwrite the cu124 wheel. Do **not** upgrade `transformers` past **4.54.1** while using Torch 2.4.

If `python --version` is still 3.8 after activate, run with the env binary:

```bash
"$CONDA_PREFIX/bin/python" --version
"$CONDA_PREFIX/bin/python" -m pip install -r requirements.txt
```

---

## 2. Weights and data (Hugging Face)

Large files are **not** in GitHub. Download the Hugging Face pack to **any folder** on your machine. You do not need our internal path.

Pack layout (same as this repo):

```text
<HF_DIR>/
  generation/data/train.csv
  generation/data/test.csv
  generation/data/protein_embeddings.h5
  generation/checkpoints/generator.pt
  classifier/data/train_labeled.csv
  classifier/data/test_labeled.csv
  classifier/checkpoints/classifier.pt
```

Download (replace the repo id after the HF repo is public):

```bash
huggingface-cli download USER/ProRiboGen-weights --local-dir /any/path/ProRiboGen-weights
```

From the **ProRiboGen code repo root**, point at that folder (symlink by default; does not move the pack):

```bash
cd /path/to/ProRiboGen
bash generation/scripts/link_local_data.sh /any/path/ProRiboGen-weights
```

Same thing via an environment variable:

```bash
HF=/any/path/ProRiboGen-weights bash generation/scripts/link_local_data.sh
```

Copy files instead of symlink (if you will delete or move the HF folder later):

```bash
COPY=1 bash generation/scripts/link_local_data.sh /any/path/ProRiboGen-weights
```

Manual copy (if you prefer not to use the script):

```bash
HF=/any/path/ProRiboGen-weights
cp -a "$HF/generation/data/." generation/data/
mkdir -p generation/checkpoints classifier/data classifier/checkpoints
cp -a "$HF/generation/checkpoints/." generation/checkpoints/
cp -a "$HF/classifier/data/." classifier/data/
cp -a "$HF/classifier/checkpoints/." classifier/checkpoints/
```

After linking or copying, this repo should contain:

| Path | Size | Role |
|------|------|------|
| `generation/checkpoints/generator.pt` | ~597 MB | generator weights |
| `generation/data/train.csv` | ~175 MB | generator train pairs |
| `generation/data/test.csv` | ~25 MB | generator test pairs (33 hold-out proteins) |
| `generation/data/protein_embeddings.h5` | ~1.8 GB | frozen protein embeddings |
| `classifier/checkpoints/classifier.pt` | ~248 MB | classifier weights |
| `classifier/data/train_labeled.csv` | ~51 MB | classifier train table |
| `classifier/data/test_labeled.csv` | ~6.5 MB | classifier test table |

Sampling and classification **read embeddings from the H5**. You do not need VESM/ESM-3B unless you re-encode new proteins. Checksums: `weights/README.md`.

---

## 3. GPUs

Paper settings use **8 GPUs**. **1 GPU and any subset also work.**  
If you change GPU count and do not change accumulation, the **effective batch size** for training changes.

Sampling is **one process per visible GPU** (not DDP). The script counts GPUs from `CUDA_VISIBLE_DEVICES` or `nvidia-smi`. It does **not** assume 8 cards.

```bash
cd generation
bash scripts/run_sample.sh
```

| Setup | Command |
|-------|---------|
| All GPUs on this node (8-GPU node → 8 jobs) | `bash scripts/run_sample.sh` |
| Four cards | `CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_sample.sh` |
| First two of the visible set | `NUM_GPUS=2 bash scripts/run_sample.sh` |
| One GPU (default batch 16) | `NUM_GPUS=1 bash scripts/run_sample.sh` |

Default batch: **64** (multi-GPU), **16** (single GPU). OOM: `BS=8`.  
Do not set `NUM_GPUS=8` on a one-GPU machine.

Default sampling: seed **42**, 50 steps, T=1.0, top_p=0.9, maskgit_plus, 80 nt, 1024 sequences per protein (`generation/config/sample.json`).

Direct one-process call with a custom protein CSV (`p_id` column):

```bash
cd generation
python sample.py \
  --config config/sample.json \
  --checkpoint checkpoints/generator.pt \
  --sample-csv data/csvs/your_proteins.csv \
  --output-dir outputs/my_sample \
  --batch-size 16 \
  --device cuda:0
```

Output: `generation/outputs/sample/fasta_merged/`

---

## 4. Generator training

Effective batch = `batch_size_per_gpu × (number of GPUs) × grad_accum_steps`.  
Default in `generation/config/train.json`: 16 × 8 × 4 = **512**.

GPU list is **only** `"gpu_ids"` in that JSON. Two or more IDs start DDP.

```bash
cd generation
python train.py --config config/train.json
```

Single GPU: set `"gpu_ids": [0]`. To keep effective batch ≈ 512, also set `"grad_accum_steps": 32`.

More: `generation/README.md`.

---

## 5. Classifier

Run from the **repository root**. Scripts use `torchrun` and the number of IDs in `CUDA_VISIBLE_DEVICES`. If unset, they default to **8 GPUs** (`0,1,2,3,4,5,6,7`) — set the variable on a smaller machine.

```bash
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 bash classifier/scripts/run_train.sh
CUDA_VISIBLE_DEVICES=0 bash classifier/scripts/run_train.sh
# OOM
BATCH_SIZE=4 CUDA_VISIBLE_DEVICES=0 bash classifier/scripts/run_train.sh

# eval
CUDA_VISIBLE_DEVICES=0 bash classifier/scripts/run_eval.sh
```

Writes `classifier/checkpoints/classifier.pt`.  
Reference (threshold 0.5): accuracy 0.788, AUROC 0.878.

More: `classifier/README.md`.

---

## 6. Attention / motif

No extra training. Attention: `attention/README.md`.

MEME is **not** in the ProRiboGen PyTorch env. Recreate the released MEME environment from `motif/meme.yml` (MEME 5.0.5):

```bash
conda env create -f motif/meme.yml
conda activate meme_new
meme -version
which meme
```

If the env name `meme_new` already exists: `conda env update -n meme_new -f motif/meme.yml --prune`.

Then, from the repo root, after sampling:

```bash
conda activate meme_new
python motif/find_motif_meme.py generation/outputs/sample/fasta_merged motif/outputs/meme --threads 8
python motif/meme_dirs_to_homer.py motif/outputs/meme motif/outputs/homer
python motif/plot_homer_logos.py motif/outputs/homer motif/outputs/logos
```

The last command needs `logomaker` / `matplotlib` (in `requirements.txt`). You can run it in `ProRiboGen` after MEME has written `motif/outputs/meme/*/meme.txt`.

Details: `motif/README.md`.

MIT License.
