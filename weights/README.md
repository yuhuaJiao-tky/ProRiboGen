# Large files (not in git)

Local copy for Hugging Face upload: `/work/home/acr7e7g18v/JYH/ProRiboGen_HuggingFace/`

| Path | Size | Role |
|------|------|------|
| `generation/checkpoints/generator.pt` | ~597 MB | generator weights |
| `generation/data/train.csv` | ~175 MB | generator train pairs |
| `generation/data/test.csv` | ~25 MB | generator test pairs |
| `generation/data/protein_embeddings.h5` | ~1.8 GB | protein embeddings |
| `classifier/checkpoints/classifier.pt` | ~248 MB | classifier weights |
| `classifier/data/train_labeled.csv` | ~51 MB | classifier train table |
| `classifier/data/test_labeled.csv` | ~6.5 MB | classifier test table |

`generator.pt` SHA256: `7d8282a9bc8e70751e4d984eb96e3d4dd12264b8f121b353b7ad128859cbe1c9`  
`classifier.pt` SHA256: `bd77d87bc58a191398300d9ca795825fd23576c550a8228bb44c20f6a068ff52`

This machine:

```bash
cd generation
bash scripts/link_local_data.sh
```

Hugging Face repo (fill in after you create it): `YOUR_HF_USER/ProRiboGen-weights`
