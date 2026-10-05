"""v3: RNA→protein fusion after the encoder, multi-pooling, bilinear head."""
from __future__ import annotations

import re
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
        scores = self.score(hidden_states).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return (hidden_states * weights * mask).sum(dim=1)


def masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    m = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
    denom = m.sum(dim=1).clamp(min=1e-6)
    return (hidden_states * m).sum(dim=1) / denom


def masked_max(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(hidden_states.dtype).min
    h = hidden_states.masked_fill(attention_mask.unsqueeze(-1) == 0, neg_inf)
    return h.max(dim=1).values


class CrossFusionModule(nn.Module):
    """Cross-attention: RNA tokens as query, protein tokens as key/value."""

    def __init__(
        self,
        hidden_size: int,
        protein_dim: int,
        *,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.protein_proj = nn.Linear(protein_dim, hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        rna_hidden: torch.Tensor,
        rna_attention_mask: torch.Tensor,
        protein_cond: torch.Tensor,
        protein_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        protein_hidden = self.protein_proj(protein_cond)
        key_padding_mask = protein_attention_mask == 0
        attn_out, _ = self.cross_attn(
            query=rna_hidden,
            key=protein_hidden,
            value=protein_hidden,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        fused = self.norm(rna_hidden + self.dropout(attn_out))
        fused = fused.masked_fill(rna_attention_mask.unsqueeze(-1) == 0, 0.0)
        return fused, protein_hidden


class FusionClassificationHead(nn.Module):
    """Multi-pooling plus RNA/protein bilinear interaction → logit."""

    def __init__(self, hidden_size: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn_pool = AttentionPooling(hidden_size)
        self.bilinear = nn.Bilinear(hidden_size, hidden_size, 1)
        fused_dim = hidden_size * 4 + 1
        self.mlp = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        rna_hidden: torch.Tensor,
        rna_attention_mask: torch.Tensor,
        protein_hidden: torch.Tensor,
        protein_attention_mask: torch.Tensor,
        *,
        cls_index: int = 0,
    ) -> torch.Tensor:
        cls_vec = rna_hidden[:, cls_index, :]
        mean_vec = masked_mean(rna_hidden, rna_attention_mask)
        max_vec = masked_max(rna_hidden, rna_attention_mask)
        attn_vec = self.attn_pool(rna_hidden, rna_attention_mask)

        rna_pool = attn_vec
        protein_pool = masked_mean(protein_hidden, protein_attention_mask)
        bilinear_score = self.bilinear(rna_pool, protein_pool)

        fused = torch.cat([cls_vec, mean_vec, max_vec, attn_vec, bilinear_score], dim=-1)
        return self.mlp(fused).squeeze(-1)


_CONDITIONING_SUBMODULE_NAMES = (
    "AdaLN_attention",
    "ffn_adaln_modulation",
    "protein_conditioning_attention",
    "protein_proj",
)


def _esm_core(esm_model: nn.Module) -> nn.Module:
    return esm_model.esm if hasattr(esm_model, "esm") else esm_model


def iter_conditioning_parameters(esm_model: nn.Module) -> Iterable[nn.Parameter]:
    for layer in _esm_core(esm_model).encoder.layer:
        for name in _CONDITIONING_SUBMODULE_NAMES:
            if not hasattr(layer, name):
                continue
            mod = getattr(layer, name)
            if isinstance(mod, nn.Module):
                yield from mod.parameters()


def freeze_backbone_keep_conditioning(mlm: nn.Module) -> None:
    for p in mlm.parameters():
        p.requires_grad = False
    for p in iter_conditioning_parameters(mlm):
        p.requires_grad = True


def set_training_stage(model: "RnaRealismClassifierV3", stage: str) -> None:
    if stage in ("head_only", "all"):
        for p in model.mlm.parameters():
            p.requires_grad = False
    elif stage == "conditioning_only":
        freeze_backbone_keep_conditioning(model.mlm)
    elif stage == "none":
        for p in model.mlm.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"unknown stage={stage!r}")
    for p in model.fusion.parameters():
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True


def count_trainable_params(module: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


class RnaRealismClassifierV3(nn.Module):
    """
    Protein-conditioned ESM encoder
      → CrossFusion (RNA query × protein KV)
      → multi-pooling + bilinear classification head.

    By default freeze the ESM body; train AdaLN / protein cross-attn + fusion/head.
    """

    def __init__(
        self,
        mlm: nn.Module,
        hidden_size: int,
        protein_dim: int,
        *,
        freeze_mode: str = "conditioning_only",
        head_dropout: float = 0.1,
        cross_attn_heads: int = 8,
        cls_index: int = 0,
    ) -> None:
        super().__init__()
        self.mlm = mlm
        self.cls_index = cls_index
        self.fusion = CrossFusionModule(
            hidden_size,
            protein_dim,
            num_heads=cross_attn_heads,
            dropout=head_dropout,
        )
        self.head = FusionClassificationHead(hidden_size, dropout=head_dropout)

        if freeze_mode == "none":
            pass
        elif freeze_mode == "conditioning_only":
            freeze_backbone_keep_conditioning(mlm)
        elif freeze_mode == "all":
            for p in self.mlm.parameters():
                p.requires_grad = False
        else:
            raise ValueError(f"unknown freeze_mode={freeze_mode!r}")

        for p in self.fusion.parameters():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def encode(
        self,
        protein_cond: torch.Tensor,
        protein_attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        enc = self.mlm.esm(
            protein_cond=protein_cond,
            protein_attention_mask=protein_attention_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return enc.last_hidden_state

    def forward_with_attn_vec(
        self,
        protein_cond: torch.Tensor,
        protein_attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rna_hidden = self.encode(
            protein_cond=protein_cond,
            protein_attention_mask=protein_attention_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        fused, protein_hidden = self.fusion(
            rna_hidden,
            attention_mask,
            protein_cond,
            protein_attention_mask,
        )
        attn_vec = self.head.attn_pool(fused, attention_mask)
        logits = self.head(
            fused,
            attention_mask,
            protein_hidden,
            protein_attention_mask,
            cls_index=self.cls_index,
        )
        return attn_vec, logits

    def forward(
        self,
        protein_cond: torch.Tensor,
        protein_attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        _, logits = self.forward_with_attn_vec(
            protein_cond=protein_cond,
            protein_attention_mask=protein_attention_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return logits


def summarize_trainable(model: RnaRealismClassifierV3) -> str:
    t_all, n_all = count_trainable_params(model)
    t_mlm, n_mlm = count_trainable_params(model.mlm)
    t_fusion, n_fusion = count_trainable_params(model.fusion)
    t_head, n_head = count_trainable_params(model.head)
    lines = [
        f"trainable/total: {t_all:,} / {n_all:,} ({100.0 * t_all / max(n_all, 1):.2f}%)",
        f"  mlm (conditioning): {t_mlm:,} / {n_mlm:,}",
        f"  fusion (cross-attn): {t_fusion:,} / {n_fusion:,}",
        f"  head: {t_head:,} / {n_head:,}",
    ]
    by_prefix: dict[str, int] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        parts = name.split(".")
        key = ".".join(parts[:4]) if len(parts) >= 4 else name
        key = re.sub(r"\.\d+\.", ".*.", key)
        by_prefix[key] = by_prefix.get(key, 0) + p.numel()
    for k in sorted(by_prefix):
        lines.append(f"    {k}: {by_prefix[k]:,}")
    return "\n".join(lines)
