# Generator

Configs: `config/train.json`, `config/sample.json`  
Data: `data/train.csv`, `data/test.csv`, `data/protein_embeddings.h5`  
Released weights: `checkpoints/generator.pt`

Run all commands from `generation/`.

**Single GPU and multi-GPU both work.** Defaults are 8 GPUs (paper setting).

Effective batch = `batch_size_per_gpu × (number of GPUs) × grad_accum_steps`.  
Default: 16 × 8 × 4 = **512**. Fewer GPUs without raising `grad_accum_steps` makes the batch smaller.

---

## Train

GPU count is **only** `"gpu_ids"` in `config/train.json`. `train.py` sets `CUDA_VISIBLE_DEVICES` from this list.

| Setup | `gpu_ids` | What happens |
|-------|-----------|----------------|
| Multi-GPU (default) | `[0, 1, 2, 3, 4, 5, 6, 7]` | DDP (`mp.spawn`) |
| 2 GPUs | `[0, 1]` | DDP |
| Single GPU | `[0]` | no DDP; `world_size=1` |

```bash
python train.py --config config/train.json
```

Background: `bash scripts/run_train.sh` (same config).

To keep effective batch ≈ 512 on **one** GPU, set in `config/train.json`:

```json
"gpu_ids": [0],
"grad_accum_steps": 32
```

Checkpoints: `checkpoints/epoch_N.pt`. Released weights: `checkpoints/generator.pt`.  
Resume: `"resume_path"` in the same JSON.

Other train defaults: 20 epochs, bf16, per-GPU batch 16.

---

## Test (sample)

Sampling is **one process per GPU**, not DDP. Set `NUM_GPUS` to the number of cards you want. The script shards the protein list across those processes.

### Multi-GPU (default 8)

```bash
bash scripts/run_sample.sh
```

Equivalent: `NUM_GPUS=8 BS=64 bash scripts/run_sample.sh`

33 hold-out proteins, 1024 sequences each, 80 nt.  
Output: `outputs/sample/fasta_merged/`

### Single GPU

You **must** set `NUM_GPUS=1`. If you leave the default 8, the script still splits CSVs into 8 shards and tries to launch 8 jobs.

```bash
NUM_GPUS=1 BS=16 bash scripts/run_sample.sh
```

OOM: `NUM_GPUS=1 BS=8 bash scripts/run_sample.sh`

### One process, custom protein CSV

CSV needs a `p_id` column (optional `num_sequences`, `length_bp`). Uses `--device cuda:0` (one GPU).

```bash
python sample.py \
  --config config/sample.json \
  --checkpoint checkpoints/generator.pt \
  --sample-csv data/csvs/your_proteins.csv \
  --output-dir outputs/my_sample \
  --batch-size 16 \
  --device cuda:0
```
