"""BCE plus pairwise ranking loss for pos/neg of the same protein."""
from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_logits_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels)


def pairwise_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    p_ids: list[str],
    *,
    margin: float = 0.5,
) -> tuple[torch.Tensor, int]:
    """
    For each p_id, pair label=1 with label=0 examples in the batch,
    enforcing score(pos) > score(neg) + margin.
    Returns (mean loss, n_pairs).
    """
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"pos": [], "neg": []})
    labels_cpu = labels.detach().float().cpu()
    for i, (lab, pid) in enumerate(zip(labels_cpu.tolist(), p_ids)):
        key = "pos" if lab >= 0.5 else "neg"
        groups[str(pid)][key].append(i)

    if not groups:
        return logits.new_zeros(()), 0

    losses: list[torch.Tensor] = []
    for g in groups.values():
        pos_idx = g["pos"]
        neg_idx = g["neg"]
        if not pos_idx or not neg_idx:
            continue
        for pi in pos_idx:
            for ni in neg_idx:
                losses.append(F.relu(margin - (logits[pi] - logits[ni])))

    if not losses:
        return logits.new_zeros(()), 0

    stacked = torch.stack(losses)
    return stacked.mean(), int(stacked.numel())


class CombinedClassificationLoss(nn.Module):
    def __init__(
        self,
        *,
        ranking_weight: float = 0.5,
        ranking_margin: float = 0.5,
    ) -> None:
        super().__init__()
        self.ranking_weight = ranking_weight
        self.ranking_margin = ranking_margin

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        p_ids: list[str],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        bce = bce_logits_loss(logits, labels)
        rank, n_pairs = pairwise_ranking_loss(
            logits,
            labels,
            p_ids,
            margin=self.ranking_margin,
        )
        total = bce + self.ranking_weight * rank
        stats = {
            "loss": float(total.detach()),
            "bce": float(bce.detach()),
            "rank": float(rank.detach()),
            "rank_pairs": float(n_pairs),
        }
        return total, stats
