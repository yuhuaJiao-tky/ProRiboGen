# Large files (not in git)

**Weights pack:** https://huggingface.co/datasets/yuhuajiaotky/ProRiboGen-weights

Download the Hugging Face pack to **any directory**. Paths inside the pack match this repo.

```bash
huggingface-cli download yuhuajiaotky/ProRiboGen-weights --repo-type dataset --local-dir /any/path/ProRiboGen-weights
bash generation/scripts/link_local_data.sh /any/path/ProRiboGen-weights
```

`HF=/any/path/ProRiboGen-weights bash generation/scripts/link_local_data.sh` is equivalent.  
`COPY=1` copies files instead of symlinking.

| Path in this repo | Size | Role |
|-------------------|------|------|
| `generation/checkpoints/generator.pt` | ~597 MB | generator weights |
| `generation/data/train.csv` | ~175 MB | generator train pairs |
| `generation/data/test.csv` | ~25 MB | generator test pairs |
| `generation/data/protein_embeddings.h5` | ~1.8 GB | protein embeddings |
| `classifier/checkpoints/classifier.pt` | ~248 MB | classifier weights |
| `classifier/data/train_labeled.csv` | ~51 MB | classifier train table |
| `classifier/data/test_labeled.csv` | ~6.5 MB | classifier test table |

`generator.pt` SHA256: `2b5dd15832c23444c07137dbc65863cacd38477c721a8525d661cc3f3b40d4cb`  
`classifier.pt` SHA256: `1d37ba04d2e6cc530b81cb01299c2197dd2e873fd5cbf78441301cab1b6359e7`

Hugging Face dataset: [`yuhuajiaotky/ProRiboGen-weights`](https://huggingface.co/datasets/yuhuajiaotky/ProRiboGen-weights)
