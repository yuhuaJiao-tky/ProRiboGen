# ProRiboGen

Protein-conditioned RNA generation.

```bash
git clone https://github.com/jyh-tky/ProRiboGen.git
cd ProRiboGen
pip install -r requirements.txt
```

Place large files from Hugging Face (or the local copy folder) at the paths below:

```bash
cp -a /work/home/acr7e7g18v/JYH/ProRiboGen_HuggingFace/generation/data/. generation/data/
mkdir -p generation/checkpoints classifier/data classifier/checkpoints
cp -a /work/home/acr7e7g18v/JYH/ProRiboGen_HuggingFace/generation/checkpoints/. generation/checkpoints/
cp -a /work/home/acr7e7g18v/JYH/ProRiboGen_HuggingFace/classifier/data/. classifier/data/
cp -a /work/home/acr7e7g18v/JYH/ProRiboGen_HuggingFace/classifier/checkpoints/. classifier/checkpoints/
```

On this machine you can instead run `bash generation/scripts/link_local_data.sh`.  
See `weights/README.md`. Details: `generation/README.md`, `classifier/README.md`, `attention/README.md`, `motif/README.md`.

**GPUs:** paper settings use **8 GPUs**. Single-GPU and multi-GPU both work; change the settings below. If you change the GPU count and do not change accumulation, the **effective batch size** changes.

---

## Generator

Effective batch = `batch_size_per_gpu × (number of GPUs) × grad_accum_steps`.  
Default in `generation/config/train.json`: 16 × 8 × 4 = **512**.

### Train (multi-GPU, default)

`config/train.json` already has `"gpu_ids": [0, 1, 2, 3, 4, 5, 6, 7]`. Two or more IDs start DDP.

```bash
cd generation
python train.py --config config/train.json
```

### Train (single GPU)

Edit `generation/config/train.json`:

```json
"gpu_ids": [0]
```

To keep effective batch ≈ 512 on one GPU, also set `"grad_accum_steps": 32` (16 × 1 × 32). If you leave accum at 4, effective batch is 64.

```bash
cd generation
python train.py --config config/train.json
```

### Test / sample (multi-GPU, default)

`run_sample.sh` launches one `sample.py` per GPU (not DDP). Default `NUM_GPUS=8`.

```bash
cd generation
bash scripts/run_sample.sh
```

Output: `generation/outputs/sample/fasta_merged/`

### Test / sample (single GPU)

```bash
cd generation
NUM_GPUS=1 BS=16 bash scripts/run_sample.sh
```

OOM: lower `BS` (for example `BS=8`). Direct one-process call:

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

---

## Classifier

Scripts count GPUs from `CUDA_VISIBLE_DEVICES` and pass that to `torchrun`. Default if unset: `0,1,2,3,4,5,6,7`.

### Train (multi-GPU)

```bash
bash classifier/scripts/run_train.sh
```

Four GPUs: `CUDA_VISIBLE_DEVICES=0,1,2,3 bash classifier/scripts/run_train.sh`

### Train (single GPU)

```bash
CUDA_VISIBLE_DEVICES=0 bash classifier/scripts/run_train.sh
```

OOM: `BATCH_SIZE=4 CUDA_VISIBLE_DEVICES=0 bash classifier/scripts/run_train.sh`

Writes `classifier/checkpoints/classifier.pt`.

### Test (multi-GPU)

```bash
bash classifier/scripts/run_eval.sh
```

### Test (single GPU)

```bash
CUDA_VISIBLE_DEVICES=0 bash classifier/scripts/run_eval.sh
```

Reference (threshold 0.5): accuracy 0.788, AUROC 0.878.

---

## Attention / motif

No training. Run after sampling; see `attention/README.md` and `motif/README.md`.

MIT License.
