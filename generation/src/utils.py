####################################
#                                  #
#        import libraries          #
#                                  #
####################################

from __future__ import annotations

import torch
import numpy as np
import random
import os
import json
import logging
import pandas as pd
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import h5py
from transformers import AutoTokenizer
from torch.utils.data.distributed import DistributedSampler
from functools import partial
import torch.nn.functional as F


####################################
#                                  #
#        helper functions          #
#                                  #
####################################


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def create_logger(log_file: str, rank: int, name: str = "ProRiboGen") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate=False
    
    if getattr(logger, "_aletheia_inited", False):
        return logger
    
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    
    if rank == 0:
        ensure_dir(os.path.dirname(log_file) or ".")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(logging.INFO)
        logger.addHandler(sh)
    
    else:
        sh = logging.StreamHandler(open(os.devnull, "w"))
        sh.setFormatter(fmt)
        sh.setLevel(logging.CRITICAL)
        logger.addHandler(sh)
        
    logger._aletheia_inited = True
    return logger


def cleanup_ddp():
    torch.cuda.empty_cache()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    

def setup_ddp(rank: int, world_size: int, port: int):
    torch.cuda.set_device(rank)
    if world_size > 1:
        os.environ["MASTER_PORT"] = str(port)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size
        )


def is_main_process(rank):
    return rank == 0


def load_config(config_path: str = "config.json") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def get_h5_embedding_dim(h5_path: str) -> int:
    """Return per-token protein embedding dimension from a merged embeddings H5."""
    with h5py.File(h5_path, "r") as f:
        if "embeddings" not in f:
            raise ValueError(f"H5 missing 'embeddings' dataset: {h5_path}")
        return int(f["embeddings"].shape[1])


def validate_protein_dim_against_h5(config: dict, base_dir: str = ".") -> int:
    """
    Ensure config model.protein_dim matches the H5 embedding width.
    If protein_dim is omitted, set it from the H5 file.
    """
    h5_rel = config["data"]["protein_h5"]
    h5_path = h5_rel if os.path.isabs(h5_rel) else os.path.join(base_dir, h5_rel)
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"Protein H5 not found: {h5_path}")

    h5_dim = get_h5_embedding_dim(h5_path)
    model_cfg = config.setdefault("model", {})
    cfg_dim = model_cfg.get("protein_dim")

    if cfg_dim is None:
        model_cfg["protein_dim"] = h5_dim
        return h5_dim

    cfg_dim = int(cfg_dim)
    if cfg_dim != h5_dim:
        raise ValueError(
            f"config model.protein_dim={cfg_dim} does not match H5 embedding dim={h5_dim} "
            f"({h5_path})"
        )
    return h5_dim


def a_useful_log(logger, rank, message):
    if is_main_process(rank):
        logger.info(message)


####################################
#                                  #
#               Data               #
#                                  #
####################################


class RnaRbpDataset(Dataset):
    """
    RNA–RBP pairs with protein sequence embeddings.

    CSV columns ``r_id``, ``rna``, ``p_id``; legacy ``d_id`` / ``dna`` are remapped to
    these names.

    H5:
        - embeddings : [N_total_tokens, protein_dim]
        - p_ids      : [N_proteins]
        - starts     : [N_proteins]
        - lengths    : [N_proteins]

    Each item is a dict:
        {
            "r_id": str,
            "rna": str,
            "p_id": str,
            "rna_len": int,
            "protein_emb": torch.FloatTensor [Lp, Dp],
            "protein_len": int,
        }
    """
    REQUIRED_COLUMNS = ("r_id", "rna", "p_id")
    _LEGACY_ALIASES = {
        "r_id": ("d_id",),
        "rna": ("dna",),
    }
    # CSV p_id -> H5 p_id when naming differs between annotation and embedding index
    _P_ID_ALIASES = {
        "human-DDX3": "human-DDX3X",
        "human-HNRNPH1": "human-HNRNPH",
    }

    def __init__(self, csv_path: str, h5_path: str = "data/esm2-protein_embeddings.h5") -> None:
        super().__init__()
        self.csv_path = csv_path
        self.h5_path = h5_path
        self.data = pd.read_csv(csv_path)
        for canonical, aliases in self._LEGACY_ALIASES.items():
            if canonical not in self.data.columns:
                for alt in aliases:
                    if alt in self.data.columns:
                        self.data[canonical] = self.data[alt]
                        break

        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in self.data.columns]
        if missing_cols:
            raise ValueError(
                f"CSV missing required columns: {missing_cols}. columns: {list(self.data.columns)}"
            )
        self.data["rna"] = self.data["rna"].astype(str).str.upper()
        self.data["r_id"] = self.data["r_id"].astype(str)
        self.data["p_id"] = self.data["p_id"].astype(str).replace(self._P_ID_ALIASES)
        self.data["rna_len"] = self.data["rna"].str.len()
        with h5py.File(self.h5_path, "r") as f:
            p_ids = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in f["p_ids"][:]
            ]
            starts = f["starts"][:]
            lengths = f["lengths"][:]
        self.protein_index = {
            p_id: (int(start), int(length))
            for p_id, start, length in zip(p_ids, starts, lengths)
        }
        self.all_p_ids = list(self.protein_index.keys())

        valid_mask = self.data["p_id"].isin(self.protein_index)
        n_dropped = int((~valid_mask).sum())
        if n_dropped:
            missing_p_ids = sorted(self.data.loc[~valid_mask, "p_id"].unique().tolist())
            self.data = self.data.loc[valid_mask].reset_index(drop=True)
            logging.getLogger("ProRiboGen").warning(
                "Dropped %d rows from %s: p_id not in H5 (%s)",
                n_dropped,
                csv_path,
                ", ".join(missing_p_ids[:20])
                + (" ..." if len(missing_p_ids) > 20 else ""),
            )

        self._h5_file = None
        self._embeddings_ds = None

    def _ensure_h5_open(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
            self._embeddings_ds = self._h5_file["embeddings"]

    def __len__(self) -> int:
        return len(self.data)
    def get_protein_embedding_by_pid(self, p_id: str) -> torch.Tensor:
        if p_id not in self.protein_index:
            raise KeyError(f"Protein id not found in H5 index: {p_id}")

        start, length = self.protein_index[p_id]
        self._ensure_h5_open()
        protein_emb = self._embeddings_ds[start : start + length]
        return torch.from_numpy(protein_emb).float()

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        r_id = row["r_id"]
        rna = row["rna"]
        p_id = row["p_id"]
        rna_len = int(row["rna_len"])
        start, length = self.protein_index[p_id]

        self._ensure_h5_open()
        protein_emb = self._embeddings_ds[start : start + length]

        protein_emb = torch.from_numpy(protein_emb).float()
        sample = {
            "r_id": r_id,
            "rna": rna,
            "p_id": p_id,
            "rna_len": rna_len,
            "protein_emb": protein_emb,
            "protein_len": int(length),
        }
        return sample
    def __del__(self):
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass


def rna_rbp_collate_fn(
    batch,
    tokenizer,
    *,
    append_eos: bool = False,
    max_rna_nt: int | None = None,
):
    r_ids = [sample["r_id"] for sample in batch]
    p_ids = [sample["p_id"] for sample in batch]
    rna_strs = [sample["rna"] for sample in batch]
    if append_eos and getattr(tokenizer, "eos_token", None):
        eos = tokenizer.eos_token
        rna_strs = [s + eos for s in rna_strs]
    rna_lens = torch.tensor([sample["rna_len"] for sample in batch], dtype=torch.long)
    protein_lens = torch.tensor([sample["protein_len"] for sample in batch], dtype=torch.long)

    tok_kw: dict = {"padding": True, "return_tensors": "pt"}
    if max_rna_nt is not None and max_rna_nt > 0:
        tok_kw["max_length"] = max_rna_nt + 2
        tok_kw["truncation"] = True

    rna_enc = tokenizer(rna_strs, **tok_kw)

    rna_input_ids_clean = rna_enc["input_ids"]
    rna_attention_mask = rna_enc["attention_mask"]

    batch_size = len(batch)
    max_protein_len = max(sample["protein_emb"].shape[0] for sample in batch)
    protein_dim = batch[0]["protein_emb"].shape[1]

    protein_cond = torch.zeros(
        batch_size,
        max_protein_len,
        protein_dim,
        dtype=batch[0]["protein_emb"].dtype,
    )
    protein_attention_mask = torch.zeros(
        batch_size,
        max_protein_len,
        dtype=torch.long,
    )

    for i, sample in enumerate(batch):
        protein_emb = sample["protein_emb"]
        Lp = protein_emb.shape[0]
        protein_cond[i, :Lp] = protein_emb
        protein_attention_mask[i, :Lp] = 1

    return {
        "r_id": r_ids,
        "p_id": p_ids,
        "rna_input_ids_clean": rna_input_ids_clean,
        "rna_attention_mask": rna_attention_mask,
        "rna_len_bp": rna_lens,
        "protein_cond": protein_cond,
        "protein_attention_mask": protein_attention_mask,
        "protein_len_tokens": protein_lens,
    }


def build_dataloaders(config, world_size, rank):
    num_workers = config["train"]["num_workers"]
    use_worker_processes = num_workers > 0

    # 1. tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["data"]["tokenizer_path"],
        trust_remote_code=True,
    )

    # 2. dataset
    train_dataset = RnaRbpDataset(
        csv_path=config["data"]["train_csv"],
        h5_path=config["data"]["protein_h5"],
    )
    test_dataset = RnaRbpDataset(
        csv_path=config["data"]["test_csv"],
        h5_path=config["data"]["protein_h5"],
    )

    append_eos = bool(config.get("data", {}).get("append_eos_token", False))
    max_rna_nt = config.get("data", {}).get("max_generated_rna_bp")
    if max_rna_nt is not None:
        max_rna_nt = int(max_rna_nt)
    collate_fn = partial(
        rna_rbp_collate_fn,
        tokenizer=tokenizer,
        append_eos=append_eos,
        max_rna_nt=max_rna_nt,
    )

    # 4. sampler
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        train_shuffle = False
    else:
        train_sampler = None
        test_sampler = None
        train_shuffle = True

    # 5. dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size_per_gpu"],
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=use_worker_processes,
        multiprocessing_context="spawn" if use_worker_processes else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["train"]["eval_batch_size_per_gpu"],
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=use_worker_processes,
        multiprocessing_context="spawn" if use_worker_processes else None,
    )

    return train_loader, test_loader, train_sampler, test_sampler, tokenizer


def sample_random_negative_sources(
    pos_p_ids,
    all_p_ids,
    use_batch_negatives_first: bool = True,
    rng=None,
    deterministic: bool = False,
):
    if rng is None:
        rng = random
    if len(all_p_ids) <= 1:
        raise ValueError("Need at least two proteins to sample negatives.")

    negative_sources = []
    batch_p_ids = list(pos_p_ids)

    for i, pos_p_id in enumerate(batch_p_ids):
        batch_candidates = [
            j for j, candidate_p_id in enumerate(batch_p_ids)
            if candidate_p_id != pos_p_id
        ]

        if use_batch_negatives_first and batch_candidates:
            chosen_idx = batch_candidates[0] if deterministic else rng.choice(batch_candidates)
            negative_sources.append(
                {
                    "source": "batch",
                    "batch_index": chosen_idx,
                    "neg_p_id": batch_p_ids[chosen_idx],
                }
            )
            continue

        if deterministic:
            neg_p_id = next(candidate for candidate in all_p_ids if candidate != pos_p_id)
        else:
            neg_p_id = rng.choice(all_p_ids)
            while neg_p_id == pos_p_id:
                neg_p_id = rng.choice(all_p_ids)

        negative_sources.append(
            {
                "source": "pool",
                "neg_p_id": neg_p_id,
            }
        )

    return negative_sources


def build_random_negative_protein_batch(
    batch: dict,
    dataset: RnaRbpDataset,
    use_batch_negatives_first: bool = True,
    rng=None,
    deterministic: bool = False,
) -> dict:
    batch_p_ids = list(batch["p_id"])
    negative_sources = sample_random_negative_sources(
        pos_p_ids=batch_p_ids,
        all_p_ids=dataset.all_p_ids,
        use_batch_negatives_first=use_batch_negatives_first,
        rng=rng,
        deterministic=deterministic,
    )

    negative_embeddings = []
    negative_p_ids = []
    negative_lengths = []

    batch_protein_cond = batch["protein_cond"]
    batch_protein_attention_mask = batch["protein_attention_mask"]

    for source in negative_sources:
        neg_p_id = source["neg_p_id"]
        if source["source"] == "batch":
            batch_index = source["batch_index"]
            protein_len = int(batch_protein_attention_mask[batch_index].sum().item())
            negative_emb = batch_protein_cond[batch_index, :protein_len].clone()
        else:
            negative_emb = dataset.get_protein_embedding_by_pid(neg_p_id)
            protein_len = int(negative_emb.shape[0])

        negative_embeddings.append(negative_emb)
        negative_p_ids.append(neg_p_id)
        negative_lengths.append(protein_len)

    batch_size = len(negative_embeddings)
    max_negative_len = max(negative_lengths)
    protein_dim = negative_embeddings[0].shape[1]

    negative_protein_cond = torch.zeros(
        batch_size,
        max_negative_len,
        protein_dim,
        dtype=negative_embeddings[0].dtype,
    )
    negative_protein_attention_mask = torch.zeros(
        batch_size,
        max_negative_len,
        dtype=torch.long,
    )

    for i, negative_emb in enumerate(negative_embeddings):
        protein_len = negative_emb.shape[0]
        negative_protein_cond[i, :protein_len] = negative_emb
        negative_protein_attention_mask[i, :protein_len] = 1

    return {
        "neg_p_id": negative_p_ids,
        "neg_protein_cond": negative_protein_cond,
        "neg_protein_attention_mask": negative_protein_attention_mask,
        "neg_protein_len_tokens": torch.tensor(negative_lengths, dtype=torch.long),
    }


####################################
#                                  #
#              Other               #
#                                  #
####################################


base_config = {
  "add_bias_fnn": False,
  "attention_probs_dropout_prob": 0.0,
  "emb_layer_norm_before": False,
  "esmfold_config": None,
  "hidden_dropout_prob": 0.0,
  "hidden_size": 512,
  "initializer_range": 0.02,
  "intermediate_size": 2048,
  "is_folding_model": False,
  "layer_norm_eps": 1e-12,
  "mask_token_id": 2,
  "max_position_embeddings": 2050,
  "model_type": "esm",
  "num_attention_heads": 16,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "position_embedding_type": "rotary",
  "tie_word_embeddings": False,
  "token_dropout": False,
  "torch_dtype": "float32",
  "transformers_version": "4.54.1",
  "use_cache": False,
  "vocab_list": None,
  "vocab_size": 4107
}


def sample_t(
    batch_size: int,
    device: torch.device,
    eps: float = 1e-3,
    dtype: torch.dtype = torch.float32,) -> torch.Tensor:
    """
    Sample diffusion corruption ratio t for each sample.

    Args:
        batch_size: number of samples in the batch.
        device: target device.
        eps: lower bound to avoid t=0.
        dtype: dtype of the returned tensor.

    Returns:
        t: shape [B], each element sampled from Uniform(eps, 1.0).
    """
    if not (0.0 <= eps < 1.0):
        raise ValueError(f"`eps` must satisfy 0 <= eps < 1, got {eps}.")

    t = eps + (1.0 - eps) * torch.rand(batch_size, device=device, dtype=dtype)
    return t


def build_xt_and_labels(
    x0: torch.LongTensor,
    rna_attention_mask: torch.Tensor,
    mask_token_id: int,
    t: torch.Tensor,
    special_token_ids=None,
    force_at_least_one_mask: bool = True,
):
    """
    Build corrupted input x_t and MLM-style labels for diffusion training.

    Args:
        x0:
            Clean RNA token ids, shape [B, L].
        rna_attention_mask:
            RNA attention mask, shape [B, L]. Valid tokens = 1, padding = 0.
        mask_token_id:
            Token id for <mask>.
        t:
            Per-sample corruption ratio, shape [B].
        special_token_ids:
            Iterable of token ids that should never be masked, e.g. [pad_id,cls_id].
        force_at_least_one_mask:
            If True, ensure each sample has at least one masked valid token.

    Returns:
        xt:
            Corrupted input ids, shape [B, L].
        labels:
            Target ids for CE loss, shape [B, L], with non-masked positions setto -100.
        masked_positions:
            Bool tensor, shape [B, L], True where the token was masked.
    """
    if x0.dim() != 2:
        raise ValueError(f"`x0` must have shape [B, L], got{tuple(x0.shape)}.")
    if rna_attention_mask.shape != x0.shape:
        raise ValueError(
            f"`rna_attention_mask` must match `x0` shape, got "
            f"{tuple(rna_attention_mask.shape)} vs {tuple(x0.shape)}."
        )
    if t.dim() != 1 or t.shape[0] != x0.shape[0]:
        raise ValueError(
            f"`t` must have shape [B], got {tuple(t.shape)} for batch size{x0.shape[0]}."
        )

    B, L = x0.shape
    device = x0.device

    valid_positions = rna_attention_mask.bool()

    if special_token_ids is not None:
        for token_id in special_token_ids:
            if token_id is not None:
                valid_positions &= (x0 != token_id)

    mask_probs = t.unsqueeze(1).expand(B, L)
    random_vals = torch.rand(B, L, device=device)
    masked_positions = (random_vals < mask_probs) & valid_positions

    if force_at_least_one_mask:
        no_mask_rows = masked_positions.sum(dim=1) == 0
        if no_mask_rows.any():
            no_mask_indices = no_mask_rows.nonzero(as_tuple=True)[0]
            for b in no_mask_indices.tolist():
                candidate_positions = valid_positions[b].nonzero(as_tuple=True)[0]
                if candidate_positions.numel() == 0:
                    raise ValueError(
                        f"Sample {b} has no valid RNA token available for masking."
                    )
                chosen = candidate_positions[
                    torch.randint(
                        low=0,
                        high=candidate_positions.numel(),
                        size=(1,),
                        device=device,
                    )
                ]
                masked_positions[b, chosen] = True

    xt = x0.clone()
    xt[masked_positions] = mask_token_id

    labels = x0.clone()
    labels[~masked_positions] = -100

    return xt, labels, masked_positions


####################################
#                                  #
#               loss               #
#                                  #
####################################


def compute_diffusion_loss(
    logits: torch.Tensor,
    labels: torch.LongTensor,
    t: torch.Tensor,
    eps: float = 1e-3,) -> torch.Tensor:
    if logits.dim() != 3:
        raise ValueError(f"`logits` must have shape [B, L, V], got{tuple(logits.shape)}.")
    if labels.dim() != 2:
        raise ValueError(f"`labels` must have shape [B, L], got{tuple(labels.shape)}.")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"`logits` and `labels` must match on [B, L], got "
            f"{tuple(logits.shape[:2])} vs {tuple(labels.shape)}.")
    if t.dim() != 1 or t.shape[0] != logits.shape[0]:
        raise ValueError(
            f"`t` must have shape [B], got {tuple(t.shape)} for batch size{logits.shape[0]}.")
    B, L, V = logits.shape
    token_loss = F.cross_entropy(
        logits.float().reshape(B * L, V),
        labels.view(B * L),
        reduction="none",
        ignore_index=-100,
    ).view(B, L)
    masked_positions = (labels != -100)
    sample_loss = (token_loss * masked_positions).sum(dim=1)
    weights = 1.0 / t.clamp_min(eps)
    loss = (sample_loss * weights).mean()
    return loss


def compute_per_sample_masked_token_ce(
    logits: torch.Tensor,
    labels: torch.LongTensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.dim() != 3:
        raise ValueError(f"`logits` must have shape [B, L, V], got{tuple(logits.shape)}.")
    if labels.dim() != 2:
        raise ValueError(f"`labels` must have shape [B, L], got{tuple(labels.shape)}.")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"`logits` and `labels` must match on [B, L], got "
            f"{tuple(logits.shape[:2])} vs {tuple(labels.shape)}."
        )

    B, L, V = logits.shape
    token_loss = F.cross_entropy(
        logits.float().reshape(B * L, V),
        labels.view(B * L),
        reduction="none",
        ignore_index=-100,
    ).view(B, L)
    masked_positions = (labels != -100)
    sample_loss_sum = (token_loss * masked_positions).sum(dim=1)
    sample_masked_count = masked_positions.sum(dim=1)
    sample_masked_ce = sample_loss_sum / sample_masked_count.clamp_min(1)
    return sample_masked_ce, sample_loss_sum, sample_masked_count


def compute_masked_token_ce(
    logits: torch.Tensor,
    labels: torch.LongTensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, sample_loss_sum, sample_masked_count = compute_per_sample_masked_token_ce(
        logits=logits,
        labels=labels,
    )
    masked_token_loss_sum = sample_loss_sum.sum()
    masked_token_count = sample_masked_count.sum()
    masked_token_ce = masked_token_loss_sum / masked_token_count.clamp_min(1)
    return masked_token_ce, masked_token_count


def compute_pn_losses(
    logits_pos: torch.Tensor,
    logits_neg: torch.Tensor,
    labels: torch.LongTensor,
    t: torch.Tensor,
    alpha: float,
    margin: float,
    diffusion_eps: float = 1e-3,
) -> dict:
    diff_loss_pos = compute_diffusion_loss(
        logits=logits_pos,
        labels=labels,
        t=t,
        eps=diffusion_eps,
    )

    pos_sample_ce, pos_sample_loss_sum, pos_sample_masked_count = compute_per_sample_masked_token_ce(
        logits=logits_pos,
        labels=labels,
    )
    neg_sample_ce, neg_sample_loss_sum, neg_sample_masked_count = compute_per_sample_masked_token_ce(
        logits=logits_neg,
        labels=labels,
    )

    rank_per_sample = torch.relu(margin + pos_sample_ce - neg_sample_ce)
    rank_loss = rank_per_sample.mean()
    total_loss = diff_loss_pos + float(alpha) * rank_loss

    sample_count = torch.tensor(
        pos_sample_ce.shape[0],
        device=logits_pos.device,
        dtype=torch.float64,
    )
    delta_ce_sum = (neg_sample_ce - pos_sample_ce).to(torch.float64).sum()
    margin_satisfied_sum = (
        (neg_sample_ce >= (pos_sample_ce + float(margin))).to(torch.float64).sum()
    )

    return {
        "diff_loss_pos": diff_loss_pos,
        "rank_loss": rank_loss,
        "total_loss": total_loss,
        "pos_masked_token_loss_sum": pos_sample_loss_sum.to(torch.float64).sum(),
        "pos_masked_token_count": pos_sample_masked_count.to(torch.float64).sum(),
        "neg_masked_token_loss_sum": neg_sample_loss_sum.to(torch.float64).sum(),
        "neg_masked_token_count": neg_sample_masked_count.to(torch.float64).sum(),
        "delta_ce_sum": delta_ce_sum,
        "margin_satisfied_sum": margin_satisfied_sum,
        "rank_loss_sum": rank_per_sample.to(torch.float64).sum(),
        "sample_count": sample_count,
    }


####################################
#                                  #
#           save/resume            #
#                                  #
####################################


def save_model(
    save_path: str,
    model,
    optimizer=None,
    scheduler=None,
    epoch: int = 0,
    global_step: int = 0,
    config: dict = None,
    extra_state: dict = None,) -> None:
    """
    Save training checkpoint.

    Args:
        save_path: checkpoint file path, e.g. "checkpoints/latest.pt"
        model: model or DDP-wrapped model
        optimizer: optional optimizer
        scheduler: optional scheduler
        epoch: current finished epoch index
        global_step: current optimizer step count
        config: optional config dict
        extra_state: optional extra metadata dict
    """
    ensure_dir(os.path.dirname(save_path) or ".")
    model_to_save = model.module if hasattr(model, "module") else model
    checkpoint = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "extra_state": extra_state if extra_state is not None else {},}
    torch.save(checkpoint, save_path)


def _remap_legacy_checkpoint_state_dict(state_dict: dict) -> dict:
    """
    Remap legacy ``protein_conditioning_attention.dna_norm.*`` to ``rna_norm``.
    """
    out = {}
    for key, value in state_dict.items():
        new_key = key.replace(
            "protein_conditioning_attention.dna_norm.",
            "protein_conditioning_attention.rna_norm.",
        )
        out[new_key] = value
    return out


def resume_model(
    load_path: str,
    model,
    optimizer=None,
    scheduler=None,
    device: torch.device = None,
    strict: bool = True,) -> dict:
    """
    Resume training checkpoint.

    Args:
        load_path: checkpoint file path
        model: model or DDP-wrapped model
        optimizer: optional optimizer
        scheduler: optional scheduler
        device: target device for optimizer state tensors
        strict: passed to model.load_state_dict()

    Returns:
        A dict with:
            - epoch
            - global_step
            - config
            - extra_state
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    checkpoint = torch.load(load_path, map_location="cpu")
    model_to_load = model.module if hasattr(model, "module") else model
    state_dict = _remap_legacy_checkpoint_state_dict(checkpoint["model_state_dict"])
    model_to_load.load_state_dict(state_dict, strict=strict)
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if device is not None:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    resume_state = {
        "epoch": checkpoint.get("epoch", -1),
        "global_step": checkpoint.get("global_step", 0),
        "config": checkpoint.get("config", None),
        "extra_state": checkpoint.get("extra_state", {}),
    }
    return resume_state
