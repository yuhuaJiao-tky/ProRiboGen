import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import pandas as pd
import torch
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.model import EsmConfig, EsmForMaskedLM
from src.utils import base_config, load_config, resume_model, validate_protein_dim_against_h5


RNA_TOKEN_ALPHABET = {"A", "C", "G", "U"}


class _TeeIO:
    """Mirror writes to two text streams (e.g. console + sample_run.log)."""

    __slots__ = ("_a", "_b")

    def __init__(self, a, b):
        self._a = a
        self._b = b

    def write(self, data: str) -> int:
        na = self._a.write(data)
        self._b.write(data)
        self._a.flush()
        self._b.flush()
        return na

    def flush(self) -> None:
        self._a.flush()
        self._b.flush()

    def isatty(self) -> bool:
        return getattr(self._a, "isatty", lambda: False)()


@dataclass
class TokenizerRuntime:
    tokenizer: object
    id_to_token: List[str]
    valid_content_token_mask: Tensor
    mask_token_id: int
    cls_token_id: int
    pad_token_id: Optional[int]
    eos_token_id: Optional[int]
    special_token_ids: Sequence[int]


@dataclass
class SamplingConfig:
    steps: int = 50
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 0
    eps: float = 1e-3
    alg: str = "maskgit_plus"
    alg_temp: float = 0.8
    batch_size: int = 64
    num_sequences_per_rbp: int = 2048
    length_bp: int = 101
    token_length: int = 1
    seed: int = 42
    use_bf16: bool = True
    variable_length: bool = False
    min_rna_bp: Optional[int] = None
    nucleotide_logit_bias: Optional[Dict[str, float]] = None
    nucleotide_bias_vector: Optional[Tensor] = field(default=None, repr=False)
    # Late-step eos logit bias so MaskGIT does not defer <eos> until the last steps.
    maskgit_eos_late_bias: float = 0.0
    maskgit_eos_late_frac: float = 0.35
    # last_slot_if_variable_else_all_masked: boost last slot if variable_length, else all remaining masks.
    maskgit_eos_late_scope: str = "last_slot_if_variable_else_all_masked"

    @property
    def num_kmers(self) -> int:
        if self.token_length <= 0:
            raise ValueError(
                f"`token_length` must be a positive integer, got {self.token_length}."
            )
        if self.length_bp % self.token_length != 0:
            raise ValueError(
                f"`length_bp` must be a multiple of token_length={self.token_length}, got {self.length_bp}."
            )
        return self.length_bp // self.token_length

    @property
    def num_prompt_content_tokens(self) -> int:
        """Slots after <cls>: RNA (num_kmers) and optionally one extra for <eos>."""
        n = self.num_kmers
        if self.variable_length:
            return n + 1
        return n


class ProteinEmbeddingStore:
    def __init__(self, h5_path: Path) -> None:
        self.h5_path = h5_path
        self._h5_file = None
        self._embeddings_ds = None
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

    def _ensure_open(self) -> None:
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
            self._embeddings_ds = self._h5_file["embeddings"]

    def get(self, p_id: str) -> Tensor:
        if p_id not in self.protein_index:
            raise KeyError(f"Protein id not found in H5 index: {p_id}")
        self._ensure_open()
        start, length = self.protein_index[p_id]
        emb = self._embeddings_ds[start : start + length]
        return torch.from_numpy(emb).float()

    def close(self) -> None:
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass
            self._h5_file = None
            self._embeddings_ds = None

    def __del__(self) -> None:
        self.close()

def resolve_path(raw_path: Optional[str], project_root: Path, config_dir: Optional[Path] = None) -> Optional[Path]:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates = []
    if config_dir is not None:
        candidates.append(config_dir / path)
    candidates.append(project_root / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if config_dir is not None:
        return config_dir / path
    return project_root / path


def is_valid_rna_token(token: str, token_length: int) -> bool:
    return len(token) == token_length and set(token) <= RNA_TOKEN_ALPHABET


def build_tokenizer_runtime(
    tokenizer_dir: Path,
    device: torch.device,
    token_length: int,
    *,
    allow_eos_in_content: bool = False,
) -> TokenizerRuntime:
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        trust_remote_code=True,
    )
    vocab = tokenizer.get_vocab()
    id_to_token = [""] * len(vocab)
    for token, token_id in vocab.items():
        id_to_token[token_id] = token

    valid_content_token_mask = torch.tensor(
        [is_valid_rna_token(token, token_length=token_length) for token in id_to_token],
        dtype=torch.bool,
        device=device,
    )
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if allow_eos_in_content and eos_id is not None and 0 <= eos_id < len(valid_content_token_mask):
        valid_content_token_mask[eos_id] = True
    if not valid_content_token_mask.any():
        raise ValueError(
            f"Tokenizer vocab does not contain any valid RNA {token_length}-mer tokens."
        )
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer is missing `mask_token_id`.")
    if tokenizer.cls_token_id is None:
        raise ValueError(
            "Tokenizer is missing `cls_token_id`. Training uses tokenizer-added <cls> at position 0, "
            "so sampling requires it as a fixed prefix."
        )
    return TokenizerRuntime(
        tokenizer=tokenizer,
        id_to_token=id_to_token,
        valid_content_token_mask=valid_content_token_mask,
        mask_token_id=tokenizer.mask_token_id,
        cls_token_id=tokenizer.cls_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_id,
        special_token_ids=tuple(tokenizer.all_special_ids),
    )


def build_nucleotide_bias_vector(
    id_to_token: Sequence[str],
    valid_content_token_mask: Tensor,
    bias: Dict[str, float],
    device: torch.device,
) -> Tensor:
    """Per-vocab additive bias for single-nucleotide RNA tokens (e.g. boost C)."""
    vec = torch.zeros(len(id_to_token), device=device)
    for token_id, token in enumerate(id_to_token):
        if not bool(valid_content_token_mask[token_id].item()):
            continue
        if token in bias:
            vec[token_id] = float(bias[token])
    return vec


def sequence_c_frac(rna: str) -> float:
    bases = [c for c in rna.upper() if c in RNA_TOKEN_ALPHABET]
    if not bases:
        return 0.0
    return bases.count("C") / len(bases)


def resolve_composition_filter_for_target(
    comp_cfg: dict,
    *,
    p_id: str,
    project_root: Path,
    config_dir: Optional[Path],
) -> dict:
    """Merge static composition_filter config with optional per-p_id reference from CSV."""
    if not comp_cfg.get("enabled", False):
        return {"enabled": False}

    out = dict(comp_cfg)
    ref_csv = comp_cfg.get("reference_composition_csv")
    if ref_csv:
        ref_path = resolve_path(ref_csv, project_root, config_dir)
        if ref_path is None or not ref_path.exists():
            raise FileNotFoundError(f"reference_composition_csv not found: {ref_csv}")
        df = pd.read_csv(ref_path, low_memory=False)
        if "p_id" not in df.columns or "rna" not in df.columns:
            raise ValueError(f"reference_composition_csv needs p_id and rna: {ref_path}")
        sub = df[df["p_id"].astype(str) == p_id]
        if len(sub) == 0:
            print(
                f"  [composition_filter] p_id={p_id} not in {ref_path.name}; "
                "using static min_c_frac/max_c_frac only."
            )
        else:
            fracs = sub["rna"].astype(str).map(sequence_c_frac)
            mean_c = float(fracs.mean())
            margin = float(comp_cfg.get("reference_margin_c_frac", 0.12))
            out["min_c_frac"] = max(0.0, mean_c - margin)
            out["max_c_frac"] = min(1.0, mean_c + margin)
            out["_reference_mean_c_frac"] = mean_c
    return out


def passes_composition_filter(rna: str, filt: dict) -> bool:
    if not filt.get("enabled", False):
        return True
    cf = sequence_c_frac(rna)
    min_c = filt.get("min_c_frac")
    max_c = filt.get("max_c_frac")
    if min_c is not None and cf < float(min_c):
        return False
    if max_c is not None and cf > float(max_c):
        return False
    return True


def _top_p_logits(logits: Tensor, top_p: float) -> Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits.float(), dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def _top_k_logits(logits: Tensor, top_k: int) -> Tensor:
    if top_k is None or top_k <= 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    threshold = torch.topk(logits, top_k)[0][..., -1, None]
    return logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)


def _sample_tokens(
    logits: Tensor,
    valid_token_mask: Tensor,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    alg: str = "origin",
    nucleotide_bias: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    logits = logits.clone()
    logits = logits.masked_fill(~valid_token_mask.unsqueeze(0), torch.finfo(logits.dtype).min)
    if nucleotide_bias is not None:
        logits = logits + nucleotide_bias.unsqueeze(0)
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = _top_p_logits(logits, top_p)
    if top_k is not None:
        logits = _top_k_logits(logits, top_k)
    probs = torch.softmax(logits.float(), dim=-1)
    if temperature > 0:
        x0 = torch.distributions.Categorical(probs=probs).sample()
    else:
        _, x0 = probs.max(dim=-1)
    confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

    if alg == "topk_margin":
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        confidence = sorted_probs[..., 0] - sorted_probs[..., 1]
    elif alg == "entropy":
        log_probs = torch.log(probs.clamp(min=1e-10))
        confidence = (probs * log_probs).sum(dim=-1)

    return confidence, x0


def apply_late_eos_logit_bias_to_logits(
    logits: Tensor,
    mask_index: Tensor,
    *,
    eos_token_id: Optional[int],
    variable_length: bool,
    step_idx: int,
    total_steps: int,
    late_frac: float,
    bias: float,
    scope: str,
) -> Tensor:
    """
    In the second half of sampling, add a constant bias to eos logits at still-masked positions,
    so <eos> can be unmasked earlier.
    """
    if bias == 0.0 or eos_token_id is None or total_steps <= 0:
        return logits
    eid = int(eos_token_id)
    B, L, V = logits.shape
    if eid < 0 or eid >= V:
        return logits
    start = int(math.ceil((1.0 - float(late_frac)) * float(total_steps)))
    if step_idx < start:
        return logits

    device = logits.device
    pos = torch.arange(L, device=device).view(1, -1).expand(B, -1)
    last = L - 1
    if scope == "last_slot_only":
        sel = mask_index & (pos == last)
    elif scope == "all_masked":
        sel = mask_index
    else:
        # last_slot_if_variable_else_all_masked
        if variable_length:
            sel = mask_index & (pos == last)
        else:
            sel = mask_index
    if not sel.any():
        return logits
    out = logits.clone()
    delta = torch.zeros((B, L), device=device, dtype=out.dtype)
    delta[sel] = float(bias)
    out[:, :, eid] = out[:, :, eid] + delta
    return out


@torch.no_grad()
def conditioned_diffusion_generate(
    model: EsmForMaskedLM,
    input_ids: Tensor,
    protein_cond: Tensor,
    protein_attention_mask: Tensor,
    content_token_constraint: Tensor,
    sampling_cfg: SamplingConfig,
    mask_token_id: int,
    eos_token_id: Optional[int] = None,
) -> Tensor:
    if input_ids.dim() != 2:
        raise ValueError(f"`input_ids` must have shape [B, L], got {tuple(input_ids.shape)}.")
    if protein_cond.dim() != 3:
        raise ValueError(
            f"`protein_cond` must have shape [B, Lp, Dp], got {tuple(protein_cond.shape)}."
        )
    if protein_attention_mask.dim() != 2:
        raise ValueError(
            f"`protein_attention_mask` must have shape [B, Lp], got {tuple(protein_attention_mask.shape)}."
        )
    if protein_cond.shape[:2] != protein_attention_mask.shape:
        raise ValueError(
            "Protein condition and attention mask shapes do not match: "
            f"{tuple(protein_cond.shape)} vs {tuple(protein_attention_mask.shape)}."
        )

    x = input_ids.clone()
    fix_mask = x != mask_token_id
    gen_attention_mask = torch.ones_like(x, dtype=torch.long)
    timesteps = torch.linspace(1.0, sampling_cfg.eps, sampling_cfg.steps + 1, device=x.device)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if x.is_cuda and sampling_cfg.use_bf16
        else torch.autocast(device_type="cpu", enabled=False)
    )

    for i in tqdm(range(sampling_cfg.steps), desc="Diffusion", leave=False):
        mask_index = x == mask_token_id
        if not mask_index.any():
            break

        with autocast_ctx:
            outputs = model(
                input_ids=x,
                attention_mask=gen_attention_mask,
                protein_cond=protein_cond,
                protein_attention_mask=protein_attention_mask,
            )
        logits = outputs.logits
        logits = apply_late_eos_logit_bias_to_logits(
            logits,
            mask_index,
            eos_token_id=eos_token_id,
            variable_length=sampling_cfg.variable_length,
            step_idx=i,
            total_steps=sampling_cfg.steps,
            late_frac=sampling_cfg.maskgit_eos_late_frac,
            bias=sampling_cfg.maskgit_eos_late_bias,
            scope=sampling_cfg.maskgit_eos_late_scope,
        )
        mask_logits = logits[mask_index]
        t = timesteps[i]
        s = timesteps[i + 1]

        if sampling_cfg.alg == "origin":
            p_transfer = 1 - s / t if i < sampling_cfg.steps - 1 else 1
            x0 = torch.full_like(x[mask_index], fill_value=mask_token_id, device=x.device, dtype=torch.long)
            transfer_index = torch.rand(*x0.shape, device=x.device) < p_transfer
            if transfer_index.any():
                _, sampled = _sample_tokens(
                    mask_logits[transfer_index],
                    valid_token_mask=content_token_constraint,
                    temperature=sampling_cfg.temperature,
                    top_p=sampling_cfg.top_p,
                    top_k=sampling_cfg.top_k,
                    alg=sampling_cfg.alg,
                    nucleotide_bias=sampling_cfg.nucleotide_bias_vector,
                )
                x0[transfer_index] = sampled
            x[mask_index] = x0

        elif sampling_cfg.alg == "p2":
            kappa_t = (i + 1) / sampling_cfg.steps
            confidence, x0 = _sample_tokens(
                mask_logits,
                valid_token_mask=content_token_constraint,
                temperature=sampling_cfg.temperature,
                top_p=sampling_cfg.top_p,
                top_k=sampling_cfg.top_k,
                alg=sampling_cfg.alg,
                nucleotide_bias=sampling_cfg.nucleotide_bias_vector,
            )
            full_conf = torch.full_like(x, float("inf"), dtype=confidence.dtype)
            full_conf[mask_index] = confidence
            full_conf[fix_mask] = float("inf")
            num_positions = (~fix_mask).sum(dim=1, keepdim=True)
            num_to_mask = (num_positions.float() * (1 - kappa_t)).long()
            sorted_idx = torch.argsort(full_conf, dim=-1, descending=False)
            max_mask = num_to_mask.max()
            arange_mask = torch.arange(max_mask, device=x.device).unsqueeze(0) < num_to_mask
            to_mask_idx = sorted_idx[:, :max_mask][arange_mask]
            to_mask = torch.zeros_like(x, dtype=torch.bool)
            batch_idx = (
                torch.arange(x.size(0), device=x.device)
                .unsqueeze(1)
                .expand(-1, max_mask)[arange_mask]
            )
            to_mask[batch_idx, to_mask_idx] = True
            x[to_mask] = mask_token_id
            mask_candidates = mask_index & ~to_mask
            x_proposals = torch.full_like(x, fill_value=mask_token_id)
            x_proposals[mask_index] = x0
            x[mask_candidates] = x_proposals[mask_candidates]

        elif sampling_cfg.alg in {"maskgit_plus", "entropy", "topk_margin"}:
            confidence, x0 = _sample_tokens(
                mask_logits,
                valid_token_mask=content_token_constraint,
                temperature=sampling_cfg.temperature,
                top_p=sampling_cfg.top_p,
                top_k=sampling_cfg.top_k,
                alg=sampling_cfg.alg,
                nucleotide_bias=sampling_cfg.nucleotide_bias_vector,
            )
            confidence = confidence.to(mask_logits.dtype)
            num_mask_tokens = mask_index.sum(dim=1)
            if i < sampling_cfg.steps - 1:
                n_transfer = (num_mask_tokens.float() * (1 - s / t)).long()
            else:
                n_transfer = num_mask_tokens
            full_confidence = torch.full_like(x, -torch.inf, dtype=logits.dtype)
            full_confidence[mask_index] = confidence
            # Use k=seq length for topk to avoid n_transfer.max().item() each step (GPU sync).
            seq_k = full_confidence.size(1)
            if sampling_cfg.alg_temp == 0:
                _, all_indices = torch.topk(full_confidence, seq_k, dim=1)
            else:
                scaled = full_confidence / sampling_cfg.alg_temp
                uniform = torch.rand_like(scaled).clamp_(1e-20, 1 - 1e-20)
                scores = scaled + (-torch.log(-torch.log(uniform)))
                _, all_indices = torch.topk(scores, seq_k, dim=1)
            valid_mask = (
                torch.arange(seq_k, device=x.device).unsqueeze(0) < n_transfer.unsqueeze(1)
            )
            valid_indices = all_indices[valid_mask]
            valid_batch = (
                torch.arange(x.size(0), device=x.device)
                .unsqueeze(1)
                .expand_as(all_indices)[valid_mask]
            )
            x_ = torch.full_like(x, fill_value=mask_token_id)
            x_[mask_index] = x0.clone()
            x[valid_batch, valid_indices] = x_[valid_batch, valid_indices]

        elif sampling_cfg.alg == "random":
            _, x0 = _sample_tokens(
                mask_logits,
                valid_token_mask=content_token_constraint,
                temperature=sampling_cfg.temperature,
                top_p=sampling_cfg.top_p,
                top_k=sampling_cfg.top_k,
                alg=sampling_cfg.alg,
                nucleotide_bias=sampling_cfg.nucleotide_bias_vector,
            )
            num_mask_tokens = mask_index.sum(dim=1)
            if i < sampling_cfg.steps - 1:
                n_transfer = (num_mask_tokens.float() * (1 - s / t)).long()
            else:
                n_transfer = num_mask_tokens
            if n_transfer.max().item() > 0:
                x_ = torch.full_like(x, fill_value=mask_token_id)
                x_[mask_index] = x0.clone()
                for b in range(x.size(0)):
                    positions = mask_index[b].nonzero(as_tuple=True)[0]
                    n = min(int(n_transfer[b].item()), len(positions))
                    if n > 0:
                        sel = torch.randperm(len(positions), device=x.device)[:n]
                        x[b, positions[sel]] = x_[b, positions[sel]]
        else:
            raise NotImplementedError(f"Unsupported diffusion sampling algorithm: {sampling_cfg.alg}")

    return x


def _coalesce_rbp_name(row: Dict[str, object]) -> Optional[str]:
    """Display name for an RBP row; accepts legacy column tf_name."""
    for key in ("rbp_name", "tf_name"):
        v = row.get(key)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def load_targets_map(targets_tsv: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if targets_tsv is None or not targets_tsv.exists():
        return {}
    targets_map: Dict[str, Dict[str, str]] = {}
    with targets_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            p_id = row.get("p_id")
            if p_id:
                norm = dict(row)
                name = _coalesce_rbp_name(norm)
                if name:
                    norm["rbp_name"] = name
                targets_map[p_id] = norm
    return targets_map


def _csv_int(row: dict, *keys: str) -> Optional[int]:
    """First non-empty CSV cell among keys; stripped string -> int."""
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        if s.lower() == "auto":
            return None
        return int(s)
    return None


def _read_rlen_series(reference_csv: Path) -> pd.Series:
    df = pd.read_csv(reference_csv, low_memory=False)
    if "r_len" in df.columns:
        return df["r_len"].astype(float)
    if "rna" in df.columns:
        return df["rna"].astype(str).str.len().astype(float)
    raise ValueError(f"Reference CSV needs r_len or rna column: {reference_csv}")


def _quantize_rlen(
    value: float,
    *,
    min_bp: int,
    max_bp: int,
    round_mode: str,
) -> int:
    if round_mode == "floor":
        bp = int(math.floor(value))
    elif round_mode == "round":
        bp = int(round(value))
    else:
        bp = int(math.ceil(value))
    return max(min_bp, min(max_bp, bp))


def compute_global_rlen_length_bp(
    reference_csv: Path,
    *,
    percentile: float = 90.0,
    min_bp: int = 12,
    max_bp: int = 101,
    round_mode: str = "ceil",
) -> int:
    """Global r_len percentile over all rows in reference CSV (e.g. entire train set)."""
    lengths = _read_rlen_series(reference_csv)
    if lengths.empty:
        raise ValueError(f"No lengths in reference CSV: {reference_csv}")
    val = float(lengths.quantile(percentile / 100.0))
    return _quantize_rlen(val, min_bp=min_bp, max_bp=max_bp, round_mode=round_mode)


def build_rlen_length_map(
    reference_csv: Path,
    *,
    percentile: float = 90.0,
    min_bp: int = 12,
    max_bp: int = 101,
    round_mode: str = "ceil",
) -> Dict[str, int]:
    """
    Per p_id, infer sampling length_bp from r_len (or len(rna)) in a reference CSV.
    """
    df = pd.read_csv(reference_csv, low_memory=False)
    if "p_id" not in df.columns:
        raise ValueError(f"Reference CSV missing p_id: {reference_csv}")
    if "r_len" in df.columns:
        lengths = df["r_len"].astype(float)
    elif "rna" in df.columns:
        lengths = df["rna"].astype(str).str.len().astype(float)
    else:
        raise ValueError(f"Reference CSV needs r_len or rna column: {reference_csv}")

    work = df.assign(_len=lengths).groupby("p_id", sort=False)["_len"]
    out: Dict[str, int] = {}
    for p_id, series in work:
        if series.empty:
            continue
        val = float(series.quantile(percentile / 100.0))
        out[str(p_id)] = _quantize_rlen(
            val, min_bp=min_bp, max_bp=max_bp, round_mode=round_mode
        )
    return out


def apply_inferred_length_bp(
    targets: List[Dict[str, object]],
    *,
    length_map: Dict[str, int],
    default_length_bp: int,
    percentile: float,
    global_fallback_bp: Optional[int] = None,
) -> None:
    for target in targets:
        p_id = str(target["p_id"])
        if target.get("length_bp") is not None:
            target["length_bp_source"] = "sample_csv"
            continue
        inferred = length_map.get(p_id)
        if inferred is not None:
            target["length_bp"] = inferred
            target["length_bp_source"] = f"train_rlen_per_pid_p{percentile:g}"
        elif global_fallback_bp is not None:
            target["length_bp"] = global_fallback_bp
            target["length_bp_source"] = f"train_rlen_global_p{percentile:g}"
        else:
            target["length_bp"] = default_length_bp
            target["length_bp_source"] = "default"


def resolve_sample_paths_from_config(
    sd: dict,
    project_root: Path,
    config_dir: Path,
    args: argparse.Namespace,
) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[int]]:
    """Optional sample_csv / checkpoint / output_dir / default length_bp from config.sample."""
    sample_csv = None
    if sd.get("sample_csv"):
        sample_csv = resolve_path(sd["sample_csv"], project_root, config_dir)
    checkpoint = None
    if sd.get("checkpoint"):
        checkpoint = resolve_path(sd["checkpoint"], project_root, config_dir)
    output_dir = None
    if sd.get("output_dir"):
        output_dir = resolve_path(sd["output_dir"], project_root, config_dir)
    default_len = sd.get("length_bp")
    if default_len is not None:
        default_len = int(default_len)
    return sample_csv, checkpoint, output_dir, default_len


def load_sampling_targets(
    sample_csv: Path,
    targets_map: Dict[str, Dict[str, str]],
    default_num_sequences: int,
    default_length_bp: int,
) -> List[Dict[str, object]]:
    if not sample_csv.exists():
        raise FileNotFoundError(
            f"Sample target file not found: {sample_csv}. "
            "Expected a CSV with at least a `p_id` column."
        )

    with sample_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "p_id" not in fieldnames:
            raise ValueError(
                f"`sample.csv` must contain a `p_id` column. Current columns: {fieldnames}"
            )

        seen = set()
        targets: List[Dict[str, object]] = []
        for row in reader:
            p_id = str(row["p_id"]).strip()
            if not p_id or p_id in seen:
                continue
            seen.add(p_id)

            meta = targets_map.get(p_id, {})
            filename = (
                row.get("filename")
                or meta.get("filename")
                or f"{p_id}.fasta"
            )
            num_sequences = _csv_int(row, "num_sequences", "num_samples")
            if num_sequences is None:
                num_sequences = default_num_sequences

            length_bp = None
            length_bp_source = "pending"
            for key in ("length_bp", "rna_length_bp", "dna_length_bp"):
                raw = row.get(key)
                if raw is None:
                    continue
                s = str(raw).strip()
                if not s:
                    continue
                if s.lower() == "auto":
                    length_bp = None
                    length_bp_source = "pending"
                    break
                length_bp = int(s)
                length_bp_source = "sample_csv"
                break
            if length_bp is None and length_bp_source != "pending":
                length_bp = default_length_bp
                length_bp_source = "default"

            rbp_name = _coalesce_rbp_name(row) or _coalesce_rbp_name(meta)
            targets.append(
                {
                    "p_id": p_id,
                    "filename": filename,
                    "split": row.get("split") or meta.get("split"),
                    "species": row.get("species") or meta.get("species"),
                    "rbp_name": rbp_name,
                    "num_sequences": num_sequences,
                    "length_bp": length_bp,
                    "length_bp_source": length_bp_source,
                }
            )

    if not targets:
        raise ValueError(f"No valid targets found in {sample_csv}.")
    return targets


def decode_sequence_ids_to_rna(
    sequence_ids: Sequence[int],
    id_to_token: Sequence[str],
    special_token_ids: Sequence[int],
    cls_token_id: int,
    token_length: int,
    *,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    max_rna_nt: Optional[int] = None,
    variable_length: bool = False,
) -> Optional[str]:
    pieces: List[str] = []
    special_token_ids = set(special_token_ids)
    for idx, token_id in enumerate(sequence_ids):
        if token_id < 0 or token_id >= len(id_to_token):
            return None
        if pad_token_id is not None and token_id == pad_token_id:
            break
        if eos_token_id is not None and token_id == eos_token_id:
            break
        token = id_to_token[token_id]
        if is_valid_rna_token(token, token_length=token_length):
            pieces.append(token)
            continue
        if token_id == cls_token_id and idx == 0:
            continue
        if token_id in special_token_ids:
            return None
        return None
    rna = "".join(pieces)
    if not rna or set(rna) - RNA_TOKEN_ALPHABET:
        return None
    if max_rna_nt is not None and len(rna) > max_rna_nt:
        return None
    if not variable_length and max_rna_nt is not None and len(rna) != max_rna_nt:
        return None
    return rna


def write_fasta(output_path: Path, sequences: Sequence[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, seq in enumerate(sequences, start=1):
            f.write(f">sample_{idx:06d}\n")
            f.write(seq + "\n")


def build_model_from_config(
    config: dict,
    device: torch.device,
    tokenizer_runtime: TokenizerRuntime,
) -> EsmForMaskedLM:
    model_config = EsmConfig(**base_config)
    model_config.vocab_size = len(tokenizer_runtime.id_to_token)
    model_config.mask_token_id = tokenizer_runtime.mask_token_id
    if tokenizer_runtime.pad_token_id is not None:
        model_config.pad_token_id = tokenizer_runtime.pad_token_id
    model_config.use_FiLM = config["model"]["use_FiLM"]
    model_config.use_AdaLN = config["model"]["use_AdaLN"]
    model_config.use_gated_bias = config["model"]["use_gated_bias"]
    model_config.use_protein_conditioning_attention = config["model"]["use_protein_conditioning_attention"]
    model_config.protein_dim = config["model"]["protein_dim"]
    if config["model"].get("num_hidden_layers") is not None:
        model_config.num_hidden_layers = int(config["model"]["num_hidden_layers"])
    model = EsmForMaskedLM(model_config)
    model.to(device)
    model.eval()
    return model


def make_prompt(
    batch_size: int,
    num_content_tokens: int,
    cls_token_id: int,
    mask_token_id: int,
    device: torch.device,
) -> Tensor:
    prompt = torch.full(
        (batch_size, num_content_tokens + 1),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )
    prompt[:, 0] = cls_token_id
    return prompt


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RBP-conditioned RNA sampling (protein embeddings + discrete diffusion)"
    )
    parser.add_argument("--config", default="config/sample.json", help="Path to sample config JSON")
    parser.add_argument("--checkpoint", default="checkpoints/generator.pt", help="Checkpoint to sample from")
    parser.add_argument("--sample-csv", default="data/sample.csv", help="CSV with at least a p_id column")
    parser.add_argument(
        "--targets-tsv",
        default=None,
        help="optional TSV keyed by p_id with filename, split, species, rbp_name; legacy tf_name allowed",
    )
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated fasta files")
    parser.add_argument(
        "--protein-h5",
        default=None,
        help="Override config data.protein_h5 (e.g. custom embedding H5 for new proteins)",
    )
    parser.add_argument(
        "--num-sequences-per-rbp",
        type=int,
        default=2048,
        dest="num_sequences_per_rbp",
        help="RNAs per RBP if CSV has no num_sequences column",
    )
    parser.add_argument(
        "--length-bp",
        type=int,
        default=101,
        help="Generated RNA length in nt. Must be a multiple of token-length.",
    )
    parser.add_argument(
        "--token-length",
        type=int,
        default=1,
        help="Tokenizer content token length. Use 1 for RNA 1-mer tokenizer.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=argparse.SUPPRESS,
        help="diffusion steps; default config sample.steps (50)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=argparse.SUPPRESS,
        help="sampling temperature; default config sample.temperature (1.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=argparse.SUPPRESS,
        help="nucleus top-p; default config sample.top_p (0.95)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=argparse.SUPPRESS,
        help="top-k; 0 disables; default config sample.top_k",
    )
    parser.add_argument(
        "--alg",
        default=argparse.SUPPRESS,
        choices=["random", "origin", "p2", "maskgit_plus", "entropy", "topk_margin"],
        help="decode schedule; default config sample.alg",
    )
    parser.add_argument(
        "--alg-temp",
        type=float,
        default=argparse.SUPPRESS,
        help="MaskGIT confidence temperature; default config sample.alg_temp",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="sequences per forward pass",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default=None, help="Sampling device, e.g. cuda:0 or cpu")
    parser.add_argument("--no-bf16", action="store_true", help="Disable bf16 autocast during sampling")
    vlen = parser.add_mutually_exclusive_group()
    vlen.add_argument(
        "--variable-length",
        dest="variable_length",
        action="store_true",
        default=argparse.SUPPRESS,
        help="cap at length_bp plus one <eos> slot; truncate at <eos>. Default: config sample.variable_length",
    )
    vlen.add_argument(
        "--fixed-length",
        dest="variable_length",
        action="store_false",
        default=argparse.SUPPRESS,
        help="exactly length_bp RNA tokens; no <eos> content slot",
    )
    parser.add_argument(
        "--sample-run-log",
        default=None,
        help="tee stdout/stderr to this file; default output-dir/sample_run.log",
    )
    parser.add_argument(
        "--no-sample-run-log",
        action="store_true",
        help="do not write sample_run.log",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = resolve_path(args.config, project_root)
    config = load_config(str(config_path))
    config_dir = config_path.parent
    sd = config.get("sample") or {}

    if args.protein_h5:
        config["data"]["protein_h5"] = args.protein_h5

    validate_protein_dim_against_h5(config, base_dir=str(project_root))

    cfg_sample_csv, cfg_checkpoint, cfg_output_dir, cfg_length_bp = resolve_sample_paths_from_config(
        sd, project_root, config_dir, args
    )

    tokenizer_dir = resolve_path(config["data"]["tokenizer_path"], project_root, config_dir)
    protein_h5 = resolve_path(config["data"]["protein_h5"], project_root, config_dir)
    checkpoint_path = cfg_checkpoint or resolve_path(args.checkpoint, project_root, config_dir)
    # CLI paths override config.sample (for sharded --sample-csv / --output-dir)
    _default_sample_csv = resolve_path("data/sample.csv", project_root, config_dir)
    _cli_sample_csv = resolve_path(args.sample_csv, project_root, config_dir)
    if cfg_sample_csv is not None and _cli_sample_csv == _default_sample_csv:
        sample_csv = cfg_sample_csv
    else:
        sample_csv = _cli_sample_csv
    targets_tsv = (
        resolve_path(args.targets_tsv, project_root, config_dir)
        if args.targets_tsv
        else None
    )
    _default_output_dir = resolve_path("outputs", project_root, config_dir)
    _cli_output_dir = resolve_path(args.output_dir, project_root, config_dir)
    if cfg_output_dir is not None and _cli_output_dir == _default_output_dir:
        output_dir = cfg_output_dir
    else:
        output_dir = _cli_output_dir

    device = torch.device(
        args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    set_seed(args.seed)

    max_generated_rna_bp = int(config.get("data", {}).get("max_generated_rna_bp", 101))
    cli_length_bp = int(args.length_bp)
    if cfg_length_bp is not None:
        cli_length_bp = min(cfg_length_bp, max_generated_rna_bp)
    eff_length_bp = min(cli_length_bp, max_generated_rna_bp)

    if sd.get("num_sequences_per_rbp") is not None:
        args.num_sequences_per_rbp = int(sd["num_sequences_per_rbp"])

    def _pick(field: str, cfg_key: str, default):
        if hasattr(args, field):
            return getattr(args, field)
        if cfg_key in sd:
            return sd[cfg_key]
        return default

    sampling_cfg = SamplingConfig(
        steps=int(_pick("steps", "steps", SamplingConfig.steps)),
        temperature=float(_pick("temperature", "temperature", SamplingConfig.temperature)),
        top_p=float(_pick("top_p", "top_p", SamplingConfig.top_p)),
        top_k=int(_pick("top_k", "top_k", SamplingConfig.top_k)),
        eps=float(config["train"].get("t_eps", 1e-3)),
        alg=str(_pick("alg", "alg", SamplingConfig.alg)),
        alg_temp=float(_pick("alg_temp", "alg_temp", SamplingConfig.alg_temp)),
        batch_size=args.batch_size,
        num_sequences_per_rbp=args.num_sequences_per_rbp,
        length_bp=eff_length_bp,
        token_length=args.token_length,
        seed=args.seed,
        use_bf16=(not args.no_bf16 and bool(config["train"].get("use_bf16", True))),
        variable_length=bool(_pick("variable_length", "variable_length", SamplingConfig.variable_length)),
        maskgit_eos_late_bias=float(
            _pick("maskgit_eos_late_bias", "maskgit_eos_late_bias", SamplingConfig.maskgit_eos_late_bias)
        ),
        maskgit_eos_late_frac=float(
            _pick("maskgit_eos_late_frac", "maskgit_eos_late_frac", SamplingConfig.maskgit_eos_late_frac)
        ),
        maskgit_eos_late_scope=str(
            sd.get("maskgit_eos_late_scope", SamplingConfig.maskgit_eos_late_scope)
        ),
    )
    raw_nt_bias = sd.get("nucleotide_logit_bias")
    if raw_nt_bias:
        sampling_cfg.nucleotide_logit_bias = {str(k): float(v) for k, v in raw_nt_bias.items()}

    tokenizer_runtime = build_tokenizer_runtime(
        tokenizer_dir,
        device=device,
        token_length=sampling_cfg.token_length,
        allow_eos_in_content=sampling_cfg.variable_length,
    )
    if sampling_cfg.nucleotide_logit_bias:
        sampling_cfg.nucleotide_bias_vector = build_nucleotide_bias_vector(
            tokenizer_runtime.id_to_token,
            tokenizer_runtime.valid_content_token_mask,
            sampling_cfg.nucleotide_logit_bias,
            device,
        )

    model = build_model_from_config(
        config,
        device=device,
        tokenizer_runtime=tokenizer_runtime,
    )
    resume_model(
        load_path=str(checkpoint_path),
        model=model,
        optimizer=None,
        scheduler=None,
        device=device,
        strict=True,
    )
    model.eval()

    protein_store = ProteinEmbeddingStore(protein_h5)
    targets_map = load_targets_map(targets_tsv)
    targets = load_sampling_targets(
        sample_csv=sample_csv,
        targets_map=targets_map,
        default_num_sequences=sampling_cfg.num_sequences_per_rbp,
        default_length_bp=sampling_cfg.length_bp,
    )

    infer_cfg = sd.get("infer_length_from_rlen") or {}
    if infer_cfg.get("enabled", False):
        ref_csv = (
            infer_cfg.get("csv")
            or config.get("data", {}).get("train_csv")
            or config.get("data", {}).get("test_csv")
        )
        if not ref_csv:
            raise ValueError(
                "sample.infer_length_from_rlen.enabled=true but no csv / data.train_csv"
            )
        ref_path = resolve_path(ref_csv, project_root, config_dir)
        percentile = float(infer_cfg.get("percentile", 90))
        min_bp = int(infer_cfg.get("min_bp", 12))
        round_mode = str(infer_cfg.get("round", "ceil"))
        use_global_fallback = bool(infer_cfg.get("global_fallback", True))
        length_map = build_rlen_length_map(
            ref_path,
            percentile=percentile,
            min_bp=min_bp,
            max_bp=max_generated_rna_bp,
            round_mode=round_mode,
        )
        global_bp = None
        if use_global_fallback:
            global_bp = compute_global_rlen_length_bp(
                ref_path,
                percentile=percentile,
                min_bp=min_bp,
                max_bp=max_generated_rna_bp,
                round_mode=round_mode,
            )
        apply_inferred_length_bp(
            targets,
            length_map=length_map,
            default_length_bp=eff_length_bp,
            percentile=percentile,
            global_fallback_bp=global_bp,
        )
        print(
            f"Inferred length_bp from train/reference CSV {ref_path} "
            f"(p{percentile:g}, round={round_mode}); "
            f"per-p_id map: {len(length_map)} proteins"
        )
        if global_bp is not None:
            print(
                f"  global train r_len fallback (unseen p_id): length_bp={global_bp} "
                f"(train_rlen_global_p{percentile:g})"
            )
    else:
        for t in targets:
            if t.get("length_bp") is None:
                t["length_bp"] = eff_length_bp
                t["length_bp_source"] = "default"

    for t in targets:
        t["length_bp"] = min(int(t["length_bp"]), max_generated_rna_bp)

    min_rna_bp = sd.get("min_rna_bp")
    if min_rna_bp is not None:
        min_rna_bp = int(min_rna_bp)
    elif sampling_cfg.variable_length and infer_cfg.get("enabled", False):
        min_rna_bp = int(infer_cfg.get("min_bp", 12))
    else:
        min_rna_bp = None
    sampling_cfg.min_rna_bp = min_rna_bp

    manifest_rows: List[Dict[str, object]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    run_log_f = None
    run_log_path: Optional[Path] = None
    _saved_stdout, _saved_stderr = sys.stdout, sys.stderr
    if not args.no_sample_run_log:
        if args.sample_run_log:
            run_log_path = resolve_path(args.sample_run_log, project_root, config_dir)
        else:
            run_log_path = output_dir / "sample_run.log"
        run_log_path.parent.mkdir(parents=True, exist_ok=True)
        run_log_f = run_log_path.open("w", encoding="utf-8", buffering=1)
        sys.stdout = _TeeIO(_saved_stdout, run_log_f)
        sys.stderr = _TeeIO(_saved_stderr, run_log_f)

    if run_log_path is not None:
        print(f"Tee: stdout/stderr -> {run_log_path}", flush=True)

    try:
        uniq_lens = sorted({int(t["length_bp"]) for t in targets})
        print(f"Loaded {len(targets)} RBP sampling targets from {sample_csv}")
        for t in targets:
            print(
                f"  {t['p_id']}: length_bp={t['length_bp']} ({t.get('length_bp_source', '?')}), "
                f"n={t['num_sequences']}"
            )
        print(f"Sampling from checkpoint: {checkpoint_path}")
        print(f"Output directory: {output_dir}")
        print(
            "Note: each RBP's .fasta is written only after that RBP's target count of "
            "valid sequences is collected (CSV num_sequences or default); "
            "see sampling_progress.txt in the output dir while running."
        )
        len_mode = (
            f"max RNA nt per sequence (variable_length); decode at <eos> or full cap {uniq_lens}"
            if sampling_cfg.variable_length
            else f"exact RNA nt per sequence (fixed); per-RBP length_bp values: {uniq_lens}"
        )
        print(
            f"RNA length: hard cap max_generated_rna_bp={max_generated_rna_bp} (config data); {len_mode}"
        )
        print(
            f"Sampling config: num_sequences_per_rbp={sampling_cfg.num_sequences_per_rbp}, "
            f"length_bp={sampling_cfg.length_bp}, variable_length={sampling_cfg.variable_length}, "
            f"min_rna_bp={sampling_cfg.min_rna_bp}, "
            f"token_length={sampling_cfg.token_length}, "
            f"steps={sampling_cfg.steps}, temperature={sampling_cfg.temperature}, "
            f"top_p={sampling_cfg.top_p}, top_k={sampling_cfg.top_k}, alg={sampling_cfg.alg}, "
            f"alg_temp={sampling_cfg.alg_temp}, batch_size={sampling_cfg.batch_size}"
        )
        if sampling_cfg.nucleotide_logit_bias:
            print(f"Nucleotide logit bias: {sampling_cfg.nucleotide_logit_bias}")
        if sampling_cfg.maskgit_eos_late_bias != 0.0:
            print(
                f"maskgit_eos_late: bias={sampling_cfg.maskgit_eos_late_bias}, "
                f"late_frac={sampling_cfg.maskgit_eos_late_frac}, "
                f"scope={sampling_cfg.maskgit_eos_late_scope}",
                flush=True,
            )
        comp_cfg_global = sd.get("composition_filter") or {}
        if comp_cfg_global.get("enabled", False):
            print(f"Composition filter enabled: {comp_cfg_global}")
        if sampling_cfg.variable_length:
            min_hint = (
                f" reject if len<{sampling_cfg.min_rna_bp} nt."
                if sampling_cfg.min_rna_bp is not None
                else ""
            )
            vlen_hint = (
                f"variable_length=True: up to {sampling_cfg.length_bp} nt, optional trailing <eos> slot; "
                f"decode stops at <eos>.{min_hint}"
            )
        else:
            vlen_hint = f"variable_length=False: exactly {sampling_cfg.length_bp} nt per sequence."
        print(
            "Tokenizer-aligned decoding: <cls> fixed; content positions use an explicit token mask. "
            f"{vlen_hint} "
            f"RNA vocabulary: legal {sampling_cfg.token_length}-mer(s)."
        )

        n_targets = len(targets)
        progress_path = output_dir / "sampling_progress.txt"
        for idx, target in enumerate(tqdm(targets, desc="RBPs"), start=1):
            p_id = str(target["p_id"])
            filename = str(target["filename"])
            num_sequences = int(target["num_sequences"])
            target_length_bp = int(target["length_bp"])
            with progress_path.open("w", encoding="utf-8") as pf:
                pf.write(
                    f"RBP {idx}/{n_targets} STARTED | p_id={p_id} | filename={filename}\n"
                    f"Collecting {num_sequences} valid sequences; corresponding .fasta is written only after this RBP finishes.\n"
                    f"(sampling_manifest.tsv is written after all {n_targets} RBPs complete.)\n"
                )
            target_cfg = SamplingConfig(
                steps=sampling_cfg.steps,
                temperature=sampling_cfg.temperature,
                top_p=sampling_cfg.top_p,
                top_k=sampling_cfg.top_k,
                eps=sampling_cfg.eps,
                alg=sampling_cfg.alg,
                alg_temp=sampling_cfg.alg_temp,
                batch_size=sampling_cfg.batch_size,
                num_sequences_per_rbp=num_sequences,
                length_bp=target_length_bp,
                token_length=sampling_cfg.token_length,
                seed=sampling_cfg.seed,
                use_bf16=sampling_cfg.use_bf16,
                variable_length=sampling_cfg.variable_length,
                min_rna_bp=sampling_cfg.min_rna_bp,
                nucleotide_logit_bias=sampling_cfg.nucleotide_logit_bias,
                nucleotide_bias_vector=sampling_cfg.nucleotide_bias_vector,
                maskgit_eos_late_bias=sampling_cfg.maskgit_eos_late_bias,
                maskgit_eos_late_frac=sampling_cfg.maskgit_eos_late_frac,
                maskgit_eos_late_scope=sampling_cfg.maskgit_eos_late_scope,
            )

            comp_filt = resolve_composition_filter_for_target(
                sd.get("composition_filter") or {},
                p_id=p_id,
                project_root=project_root,
                config_dir=config_dir,
            )
            if comp_filt.get("enabled", False):
                ref_mean = comp_filt.get("_reference_mean_c_frac")
                if ref_mean is not None:
                    print(
                        f"  composition_filter for {p_id}: ref mean C%={100*ref_mean:.1f}, "
                        f"accept C in [{100*float(comp_filt['min_c_frac']):.1f}, "
                        f"{100*float(comp_filt['max_c_frac']):.1f}]"
                    )
                else:
                    print(
                        f"  composition_filter for {p_id}: "
                        f"min_c_frac={comp_filt.get('min_c_frac')}, max_c_frac={comp_filt.get('max_c_frac')}"
                    )

            protein_emb = protein_store.get(p_id)
            generated_sequences: List[str] = []
            invalid_count = 0
            composition_rejected = 0
            too_short_rejected = 0
            max_rounds = int(comp_filt.get("max_rounds", 0) or 0)
            rounds = 0

            while len(generated_sequences) < num_sequences:
                rounds += 1
                if max_rounds > 0 and rounds > max_rounds:
                    raise RuntimeError(
                        f"composition_filter: exceeded max_rounds={max_rounds} for {p_id}; "
                        f"collected {len(generated_sequences)}/{num_sequences} sequences. "
                        "Relax min_c_frac / nucleotide_logit_bias or increase max_rounds."
                    )
                current_batch = min(target_cfg.batch_size, num_sequences - len(generated_sequences))
                prompt = make_prompt(
                    batch_size=current_batch,
                    num_content_tokens=target_cfg.num_prompt_content_tokens,
                    cls_token_id=tokenizer_runtime.cls_token_id,
                    mask_token_id=tokenizer_runtime.mask_token_id,
                    device=device,
                )
                protein_batch = protein_emb.unsqueeze(0).repeat(current_batch, 1, 1).to(device)
                protein_attention_mask = torch.ones(
                    current_batch,
                    protein_emb.shape[0],
                    dtype=torch.long,
                    device=device,
                )

                seq_ids = conditioned_diffusion_generate(
                    model=model,
                    input_ids=prompt,
                    protein_cond=protein_batch,
                    protein_attention_mask=protein_attention_mask,
                    content_token_constraint=tokenizer_runtime.valid_content_token_mask,
                    sampling_cfg=target_cfg,
                    mask_token_id=tokenizer_runtime.mask_token_id,
                    eos_token_id=tokenizer_runtime.eos_token_id,
                )

                for row in seq_ids.detach().cpu().tolist():
                    rna = decode_sequence_ids_to_rna(
                        row,
                        id_to_token=tokenizer_runtime.id_to_token,
                        special_token_ids=tokenizer_runtime.special_token_ids,
                        cls_token_id=tokenizer_runtime.cls_token_id,
                        token_length=target_cfg.token_length,
                        eos_token_id=tokenizer_runtime.eos_token_id,
                        pad_token_id=tokenizer_runtime.pad_token_id,
                        max_rna_nt=target_length_bp,
                        variable_length=target_cfg.variable_length,
                    )
                    if rna is None:
                        invalid_count += 1
                        continue
                    if (
                        target_cfg.min_rna_bp is not None
                        and len(rna) < target_cfg.min_rna_bp
                    ):
                        too_short_rejected += 1
                        continue
                    if not passes_composition_filter(rna, comp_filt):
                        composition_rejected += 1
                        continue
                    generated_sequences.append(rna)
                    if len(generated_sequences) >= num_sequences:
                        break

            output_path = output_dir / filename
            write_fasta(output_path, generated_sequences)
            if generated_sequences:
                c_fracs = [sequence_c_frac(s) for s in generated_sequences]
                mean_c = sum(c_fracs) / len(c_fracs)
                print(
                    f"  {p_id}: output mean C%={100*mean_c:.2f} "
                    f"(composition_rejected={composition_rejected}, too_short_rejected={too_short_rejected}, "
                    f"invalid={invalid_count})"
                )
            with progress_path.open("a", encoding="utf-8") as pf:
                pf.write(
                    f"-> DONE | wrote {len(generated_sequences)} seq -> {output_path.name} | "
                    f"invalid_decodes={invalid_count} | too_short_rejected={too_short_rejected} | "
                    f"composition_rejected={composition_rejected}\n"
                )
            manifest_rows.append(
                {
                    "p_id": p_id,
                    "filename": filename,
                    "split": target.get("split"),
                    "species": target.get("species"),
                    "rbp_name": target.get("rbp_name"),
                    "num_sequences": len(generated_sequences),
                    "length_bp": target_length_bp,
                    "length_bp_source": target.get("length_bp_source", ""),
                    "variable_length": target_cfg.variable_length,
                    "min_rna_bp": target_cfg.min_rna_bp,
                    "invalid_decodes": invalid_count,
                    "too_short_rejected": too_short_rejected,
                    "composition_rejected": composition_rejected,
                }
            )

        manifest_path = output_dir / "sampling_manifest.tsv"
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "p_id",
                "filename",
                "split",
                "species",
                "rbp_name",
                "num_sequences",
                "length_bp",
                "length_bp_source",
                "variable_length",
                "min_rna_bp",
                "invalid_decodes",
                "too_short_rejected",
                "composition_rejected",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(manifest_rows)

        protein_store.close()
        print(f"Done. Wrote {len(manifest_rows)} fasta files to {output_dir}")
        print(f"Manifest: {manifest_path}")
    finally:
        if run_log_f is not None:
            sys.stdout = _saved_stdout
            sys.stderr = _saved_stderr
            run_log_f.close()


if __name__ == "__main__":
    main()
