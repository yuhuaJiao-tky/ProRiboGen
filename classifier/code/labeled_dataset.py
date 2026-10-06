"""Labeled RNA–RBP dataset wrapping generation RnaRbpDataset / collate."""
from __future__ import annotations

import random
from functools import partial
from typing import Iterator

import torch
from torch.utils.data import Dataset, Sampler
from transformers import AutoTokenizer

import sys
from pathlib import Path

_GEN_ROOT = Path(__file__).resolve().parents[2] / "generation"
if str(_GEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GEN_ROOT))

from src.utils import RnaRbpDataset, rna_rbp_collate_fn  # noqa: E402


def resolve_local_tokenizer(tokenizer_path: str, *, gen_root: Path | None = None, config_path: str | Path | None = None) -> str:
    """Resolve `tokenizer` to generation/tokenizer, not Hugging Face repo id `tokenizer`."""
    gen_root = Path(gen_root) if gen_root is not None else _GEN_ROOT
    raw = Path(tokenizer_path)
    if raw.is_dir():
        return str(raw.resolve())
    candidates = [gen_root / tokenizer_path]
    if config_path is not None:
        cfg_dir = Path(config_path).resolve().parent
        candidates.extend([cfg_dir / tokenizer_path, cfg_dir.parent / tokenizer_path])
    for c in candidates:
        if c.is_dir() and (c / "tokenizer_config.json").is_file():
            return str(c.resolve())
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"local tokenizer directory not found (tried: {tried}). "
        "The git repo should contain generation/tokenizer/."
    )


def resolve_generator_ckpt(path: str | None, *, gen_root: Path | None = None) -> str:
    """Use a local generator.pt if the path stored in the classifier ckpt is from another machine."""
    gen_root = Path(gen_root) if gen_root is not None else _GEN_ROOT
    fallback = gen_root / "checkpoints" / "generator.pt"
    if path and Path(path).is_file():
        return str(Path(path).resolve())
    if fallback.is_file():
        return str(fallback.resolve())
    raise FileNotFoundError(
        f"generator checkpoint not found (ckpt had {path!r}; also tried {fallback}). "
        "Link Hugging Face weights: bash generation/scripts/link_local_data.sh /path/to/HF_pack"
    )


class LabeledRnaRbpDataset(RnaRbpDataset):
    """Adds a ``label`` column (0/1)."""

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        sample["label"] = int(self.data.iloc[idx]["label"])
        return sample


def build_pair_indices(dataset: LabeledRnaRbpDataset) -> list[tuple[int, int]]:
    """Build (pos_idx, neg_idx) pairs; prefer pair_id."""
    df = dataset.data.reset_index(drop=True)
    if "pair_id" in df.columns:
        pos_by: dict[str, list[int]] = {}
        neg_by: dict[str, list[int]] = {}
        for i in range(len(df)):
            pid = str(df.iloc[i]["pair_id"])
            if int(df.iloc[i]["label"]) == 1:
                pos_by.setdefault(pid, []).append(i)
            else:
                neg_by.setdefault(pid, []).append(i)
        pairs: list[tuple[int, int]] = []
        for k in sorted(pos_by):
            if k not in neg_by:
                continue
            ps, ns = pos_by[k], neg_by[k]
            n = min(len(ps), len(ns))
            pairs.extend(zip(ps[:n], ns[:n]))
        if pairs:
            return pairs
    pos_idx = [i for i in range(len(df)) if int(df.iloc[i]["label"]) == 1]
    neg_idx = [i for i in range(len(df)) if int(df.iloc[i]["label"]) == 0]
    if len(pos_idx) != len(neg_idx):
        raise ValueError(f"pos={len(pos_idx)} neg={len(neg_idx)}，cannot pair")
    return list(zip(pos_idx, neg_idx))


def split_pair_indices(
    pairs: list[tuple[int, int]],
    val_fraction: float,
    seed: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    n_val = max(1, int(len(pairs) * val_fraction))
    rng = random.Random(seed)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    val_set = set(order[:n_val])
    train_pairs = [pairs[i] for i in range(len(pairs)) if i not in val_set]
    val_pairs = [pairs[i] for i in val_set]
    return train_pairs, val_pairs


class PairedBatchSampler(Sampler[list[int]]):
    """Each batch contains (pos, neg) pairs for ranking loss."""

    def __init__(
        self,
        pairs: list[tuple[int, int]],
        *,
        pairs_per_batch: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if pairs_per_batch < 1:
            raise ValueError("pairs_per_batch must be >= 1")
        self.pairs = pairs
        self.pairs_per_batch = pairs_per_batch
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        order = list(range(len(self.pairs)))
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(order)
        order = order[self.rank :: self.num_replicas]
        batch: list[int] = []
        for pi in order:
            pos_i, neg_i = self.pairs[pi]
            batch.extend([pos_i, neg_i])
            if len(batch) == 2 * self.pairs_per_batch:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self) -> int:
        n = len(self.pairs)
        if n == 0:
            return 0
        n_rank = len(range(self.rank, n, self.num_replicas))
        return (n_rank + self.pairs_per_batch - 1) // self.pairs_per_batch


def labeled_collate_fn(batch, tokenizer, *, append_eos: bool = False, max_rna_nt: int | None = None):
    labels = torch.tensor([float(s["label"]) for s in batch], dtype=torch.float32)
    base = rna_rbp_collate_fn(
        batch,
        tokenizer,
        append_eos=append_eos,
        max_rna_nt=max_rna_nt,
    )
    base["labels"] = labels
    return base


def make_collate(tokenizer_path: str, append_eos: bool, max_rna_nt: int | None):
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return partial(
        labeled_collate_fn,
        tokenizer=tok,
        append_eos=append_eos,
        max_rna_nt=max_rna_nt,
    )
