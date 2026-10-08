# Classifier

Backbone: `../generation/checkpoints/generator.pt`  
Also needs `generation/data/protein_embeddings.h5`.  
Labeled tables: `classifier/data/train_labeled.csv`, `classifier/data/test_labeled.csv` (Hugging Face pack). If missing, the scripts build them from the generator CSVs.

Run from the **repository root**.

**Single GPU and multi-GPU both work.** Scripts use `torchrun --nproc_per_node=$NPROC`, where `NPROC` is the number of IDs in `CUDA_VISIBLE_DEVICES`. If that variable is unset, the scripts default to **8 GPUs**: `0,1,2,3,4,5,6,7`.

---

## Train

### Multi-GPU (default 8)

```bash
bash classifier/scripts/run_train.sh
```

Subset of GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash classifier/scripts/run_train.sh
```

### Single GPU

```bash
CUDA_VISIBLE_DEVICES=0 bash classifier/scripts/run_train.sh
```

OOM: `BATCH_SIZE=4 CUDA_VISIBLE_DEVICES=0 bash classifier/scripts/run_train.sh`  
Default per-GPU batch is 8.

Writes `classifier/checkpoints/classifier.pt`.

---

## Test

### Multi-GPU (default 8)

```bash
bash classifier/scripts/run_eval.sh
```

### Single GPU

```bash
CUDA_VISIBLE_DEVICES=0 bash classifier/scripts/run_eval.sh
```

Reference (threshold 0.5): accuracy 0.788, AUROC 0.878.

Writes under `classifier/outputs/`:

| File | Role |
|------|------|
| `eval_*.log` | metrics log |
| `test_labeled_preds.csv` | each true RNA and its shuffled negative as separate rows (`label` 0/1) + `pred_*` |

`classifier/data/*.csv` are **inputs** (from the HF pack), not eval outputs.

Score a generated FASTA (one GPU / one process):

```bash
python3 classifier/scripts/score_fasta.py \
  --fasta generation/outputs/sample/fasta_merged/Human-PUM1.fasta \
  --p_id Human-PUM1 \
  --device cuda:0
```
