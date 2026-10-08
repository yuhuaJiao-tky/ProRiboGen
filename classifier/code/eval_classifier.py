#!/usr/bin/env python3
"""RnaRealismClassifierV3 test inference and metrics."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

_GEN_ROOT = Path(__file__).resolve().parents[2] / "generation"
sys.path.insert(0, str(_GEN_ROOT))

from src.model import EsmConfig, EsmForMaskedLM  # noqa: E402
from src.utils import base_config, _remap_legacy_checkpoint_state_dict  # noqa: E402

from labeled_dataset import LabeledRnaRbpDataset, make_collate, resolve_generator_ckpt, resolve_local_tokenizer  # noqa: E402
from losses import CombinedClassificationLoss  # noqa: E402
from model import RnaRealismClassifierV3  # noqa: E402


def load_pretrained_mlm(ckpt_path: str, tokenizer, device: torch.device, gen_cfg: dict) -> EsmForMaskedLM:
    mcfg = gen_cfg["model"]
    cfg = EsmConfig(**base_config)
    cfg.vocab_size = len(tokenizer)
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.mask_token_id = tokenizer.mask_token_id
    cfg.use_FiLM = mcfg["use_FiLM"]
    cfg.use_AdaLN = mcfg["use_AdaLN"]
    cfg.use_gated_bias = mcfg["use_gated_bias"]
    cfg.use_protein_conditioning_attention = mcfg["use_protein_conditioning_attention"]
    cfg.protein_dim = mcfg["protein_dim"]
    if mcfg.get("num_hidden_layers") is not None:
        cfg.num_hidden_layers = int(mcfg["num_hidden_layers"])
    mlm = EsmForMaskedLM(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _remap_legacy_checkpoint_state_dict(ckpt["model_state_dict"])
    mlm.load_state_dict(sd, strict=True)
    mlm.to(device)
    return mlm


def _dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _dist() else 0


def _maybe_auc(labels_cat: torch.Tensor, logits_cat: torch.Tensor) -> str:
    y = labels_cat.numpy()
    if len(set(y.tolist())) < 2:
        return " (roc_auc: single-class labels, skipped)"
    try:
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(y, torch.sigmoid(logits_cat).numpy())
        return f" roc_auc={auc:.4f}"
    except Exception as e:
        return f" (roc_auc skipped: {e})"


def _full_metrics_block(labels_cat: torch.Tensor, logits_cat: torch.Tensor) -> str:
    y = np.asarray(labels_cat.numpy(), dtype=np.int64).ravel()
    scores = np.asarray(torch.sigmoid(logits_cat).numpy(), dtype=np.float64).ravel()
    pred = (scores >= 0.5).astype(np.int64)
    lines: list[str] = []
    lines.append("=== label counts ===")
    for c in (0, 1):
        m = y == c
        lines.append(f"  label={c}: {int(m.sum())} ({100.0 * float(m.mean()):.2f}%)")
    try:
        from sklearn.metrics import (
            average_precision_score,
            balanced_accuracy_score,
            classification_report,
            cohen_kappa_score,
            confusion_matrix,
            matthews_corrcoef,
            roc_auc_score,
        )
    except ImportError as e:
        lines.append(f"=== extra metrics skipped: {e} ===")
        return "\n".join(lines)
    if np.unique(y).size < 2:
        lines.append("single class only; skip AUROC.")
        return "\n".join(lines)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    lines.append("=== confusion matrix (rows=true, cols=pred) ===")
    lines.append(f"           pred=0   pred=1")
    lines.append(f"  true=0   {cm[0, 0]:6d}   {cm[0, 1]:6d}")
    lines.append(f"  true=1   {cm[1, 0]:6d}   {cm[1, 1]:6d}")
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    lines.append("=== derived from confusion matrix (positive=1) ===")
    lines.append(f"  sensitivity_recall_TPR  = TP/(TP+FN) = {sens:.6f}")
    lines.append(f"  specificity_TNR         = TN/(TN+FP) = {spec:.6f}")
    lines.append(f"  precision_PPV           = TP/(TP+FP) = {ppv:.6f}")
    lines.append(f"  NPV                     = TN/(TN+FN) = {npv:.6f}")
    lines.append("=== sklearn report (thr=0.5, labels 0/1) ===")
    lines.append(
        classification_report(y, pred, labels=[0, 1], digits=4, zero_division=0).rstrip()
    )
    roc = roc_auc_score(y, scores)
    ap = average_precision_score(y, scores)
    mcc = matthews_corrcoef(y, pred)
    kappa = cohen_kappa_score(y, pred)
    bacc = balanced_accuracy_score(y, pred)
    lines.append("=== ranking / calibration (positive=1) ===")
    lines.append(f"  roc_auc (AUROC)           = {roc:.6f}")
    lines.append(f"  average_precision (AUPRC) = {ap:.6f}")
    lines.append(f"  matthews_corrcoef (MCC)   = {mcc:.6f}")
    lines.append(f"  cohen_kappa               = {kappa:.6f}")
    lines.append(f"  balanced_accuracy         = {bacc:.6f}")
    lines.append(f"  accuracy (thr=0.5)         = {(pred == y).mean():.6f}")
    return "\n".join(lines)


def _distributed_sampler_order(n: int, world_size: int) -> np.ndarray:
    """Reproduce torch DistributedSampler(shuffle=False) index order (with padding)."""
    import math

    num_samples = math.ceil(n / world_size)
    total_size = num_samples * world_size
    indices = list(range(n)) + list(range(total_size - n))
    order: list[int] = []
    for rank in range(world_size):
        order.extend(indices[rank::world_size])
    return np.asarray(order, dtype=np.int64)


def _reorder_gathered_to_dataset(
    values: torch.Tensor,
    n_ds: int,
    world_size: int,
) -> torch.Tensor:
    """Map rank-gathered DistributedSampler outputs back to dataset row order."""
    order = _distributed_sampler_order(n_ds, world_size)
    if values.size(0) != order.size:
        raise ValueError(
            f"gathered len {values.size(0)} != sampler order len {order.size} (n_ds={n_ds}, world={world_size})"
        )
    out = values.new_empty((n_ds,) + tuple(values.shape[1:]))
    filled = np.zeros(n_ds, dtype=bool)
    for i, idx in enumerate(order.tolist()):
        if idx < n_ds and not filled[idx]:
            out[idx] = values[i]
            filled[idx] = True
    if not bool(filled.all()):
        missing = np.where(~filled)[0][:10]
        raise RuntimeError(f"failed to remap gathered preds; missing rows e.g. {missing.tolist()}")
    return out


def _save_predictions(
    logits_cat: torch.Tensor,
    labeled_csv: str,
    *,
    output_wide_csv: str | None,
    output_labeled_preds_csv: str | None,
    preds_wide_src: str | None = None,
) -> None:
    scores = np.asarray(torch.sigmoid(logits_cat).numpy(), dtype=np.float64)
    preds = (scores >= 0.5).astype(np.int64)
    logits_np = np.asarray(logits_cat.numpy(), dtype=np.float64)
    df_lab = pd.read_csv(labeled_csv)
    if len(df_lab) != len(scores):
        raise ValueError(f"labeled rows {len(df_lab)} != n_pred {len(scores)}")
    out_lab = df_lab.copy()
    out_lab["pred_logit"] = logits_np
    out_lab["pred_prob"] = scores
    out_lab["pred_label"] = preds
    if output_labeled_preds_csv:
        Path(output_labeled_preds_csv).parent.mkdir(parents=True, exist_ok=True)
        out_lab.to_csv(output_labeled_preds_csv, index=False)
        print(f"Wrote labeled predictions -> {output_labeled_preds_csv}", flush=True)
    if not output_wide_csv:
        return
    wide_path = Path(output_wide_csv)
    wide_path.parent.mkdir(parents=True, exist_ok=True)
    src = preds_wide_src or (output_wide_csv if wide_path.is_file() else None)
    if not src:
        raise FileNotFoundError(
            f"wide table source missing: {output_wide_csv}；pass --preds_wide_src for the original test CSV"
        )
    df_wide = pd.read_csv(src)
    n_wide = len(df_wide)
    df_pos = out_lab[out_lab["label"].astype(int) == 1].reset_index(drop=True)
    df_neg = out_lab[out_lab["label"].astype(int) == 0].reset_index(drop=True)
    if len(df_pos) != n_wide or len(df_neg) != n_wide:
        raise ValueError(
            f"wide table {n_wide} rows but labeled pos={len(df_pos)} neg={len(df_neg)}"
        )
    df_wide["pred_prob"] = df_pos["pred_prob"].to_numpy()
    df_wide["pred_label"] = df_pos["pred_label"].to_numpy()
    df_wide["neg_pred_prob"] = df_neg["pred_prob"].to_numpy()
    df_wide["neg_pred_label"] = df_neg["pred_label"].to_numpy()
    df_wide.to_csv(wide_path, index=False)
    print(f"Wrote wide predictions -> {wide_path}", flush=True)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--generator_config",
        type=str,
        default="../generation/config/train.json",
    )
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--classifier_ckpt", type=str, default="checkpoints/classifier.pt")
    ap.add_argument("--pretrained_ckpt", type=str, default=None)
    ap.add_argument("--batch_size_per_gpu", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--use_bf16", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--filter_label", type=int, default=None)
    ap.add_argument("--protein_h5", type=str, default=None)
    ap.add_argument("--output_preds_csv", type=str, default=None)
    ap.add_argument("--output_labeled_preds_csv", type=str, default=None)
    ap.add_argument(
        "--preds_wide_src",
        type=str,
        default=None,
        help="original test CSV when writing the wide table (required if output does not exist).",
    )
    ap.add_argument("--ranking_weight", type=float, default=None, help="default from ckpt train_args")
    ap.add_argument("--ranking_margin", type=float, default=None)
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(
            args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
        )
    rank = _rank()

    bundle = torch.load(Path(args.classifier_ckpt), map_location="cpu")
    gen_cfg = bundle.get("generator_config") or bundle["d3lm_config"]
    train_args = bundle.get("train_args") or {}
    pretrained = resolve_generator_ckpt(
        args.pretrained_ckpt or train_args.get("pretrained_ckpt"),
        gen_root=_GEN_ROOT,
    )

    rw = args.ranking_weight if args.ranking_weight is not None else float(train_args.get("ranking_weight", 0.5))
    rm = args.ranking_margin if args.ranking_margin is not None else float(train_args.get("ranking_margin", 0.5))
    crit = CombinedClassificationLoss(ranking_weight=rw, ranking_margin=rm)

    data_cfg = gen_cfg["data"]
    cfg_path = Path(args.generator_config).resolve()
    tok_path = resolve_local_tokenizer(data_cfg["tokenizer_path"], gen_root=_GEN_ROOT, config_path=cfg_path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True, local_files_only=True)
    h5_path = args.protein_h5 or data_cfg["protein_h5"]
    if not os.path.isabs(h5_path):
        h5_path = str(cfg_path.parent / h5_path)
    append_eos = bool(data_cfg.get("append_eos_token", False))
    max_rna_nt = data_cfg.get("max_generated_rna_bp")
    if max_rna_nt is not None:
        max_rna_nt = int(max_rna_nt)

    full_ds = LabeledRnaRbpDataset(args.test_csv, h5_path=h5_path)
    test_ds: Subset | LabeledRnaRbpDataset = full_ds
    if args.filter_label is not None:
        df = pd.read_csv(args.test_csv)
        keep_idx = [i for i, v in enumerate(df["label"].tolist()) if int(v) == args.filter_label]
        test_ds = Subset(full_ds, keep_idx)

    collate = make_collate(tok_path, append_eos, max_rna_nt)
    sampler = DistributedSampler(test_ds, shuffle=False) if _dist() else None
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size_per_gpu,
        shuffle=False,
        sampler=sampler,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    mlm = load_pretrained_mlm(str(Path(pretrained).resolve()), tokenizer, device, gen_cfg)
    freeze_mode = train_args.get("freeze_mode", "conditioning_only")
    head_dropout = float(train_args.get("head_dropout", 0.1))
    protein_dim = int(gen_cfg["model"]["protein_dim"])
    cross_attn_heads = int(train_args.get("cross_attn_heads", 8))
    model = RnaRealismClassifierV3(
        mlm,
        mlm.config.hidden_size,
        protein_dim,
        freeze_mode=freeze_mode,
        head_dropout=head_dropout,
        cross_attn_heads=cross_attn_heads,
    ).to(device)
    model.load_state_dict(bundle["classifier_state_dict"], strict=True)
    if _dist():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    model.eval()
    autocast = device.type == "cuda" and args.use_bf16
    loss_sum = torch.zeros(1, device=device, dtype=torch.float64)
    correct = torch.zeros(1, device=device, dtype=torch.float64)
    n_sum = torch.zeros(1, device=device, dtype=torch.float64)
    local_logits: list[torch.Tensor] = []
    local_labels: list[torch.Tensor] = []

    for batch in loader:
        labels = batch["labels"].to(device)
        p_ids = batch["p_id"]
        x = batch["rna_input_ids_clean"].to(device)
        am = batch["rna_attention_mask"].to(device)
        pc = batch["protein_cond"].to(device)
        pam = batch["protein_attention_mask"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast):
            logits = model(
                protein_cond=pc,
                protein_attention_mask=pam,
                input_ids=x,
                attention_mask=am,
            )
            loss, _ = crit(logits.float(), labels, p_ids)
        loss_sum += loss.double() * labels.size(0)
        pred = (torch.sigmoid(logits.float()) >= 0.5).float()
        correct += (pred == labels).sum().double()
        n_sum += labels.numel()
        local_logits.append(logits.float().detach().cpu())
        local_labels.append(labels.detach().cpu())

    if _dist():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_sum, op=dist.ReduceOp.SUM)

    n = max(float(n_sum.item()), 1.0)
    mean_loss = float(loss_sum.item() / n)
    acc = float(correct.item() / n)

    auc_str = ""
    full_block = ""
    if _dist():
        loc_l = torch.cat(local_logits) if local_logits else torch.zeros(0)
        loc_y = torch.cat(local_labels) if local_labels else torch.zeros(0)
        payload = (loc_l, loc_y)
        world_size = dist.get_world_size()
        gathered: list | None = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(payload, gathered, dst=0)
        if rank == 0 and gathered is not None:
            logits_cat = torch.cat([g[0] for g in gathered])
            labels_cat = torch.cat([g[1] for g in gathered])
            n_ds = len(full_ds)
            # Remap DistributedSampler gather order -> original dataset row order
            # (do NOT truncate-before-remap; that misaligns CSV writes).
            logits_cat = _reorder_gathered_to_dataset(logits_cat, n_ds, world_size)
            labels_cat = _reorder_gathered_to_dataset(labels_cat, n_ds, world_size)
            auc_str = _maybe_auc(labels_cat, logits_cat)
            full_block = _full_metrics_block(labels_cat, logits_cat)
    else:
        logits_cat = torch.cat(local_logits)
        labels_cat = torch.cat(local_labels)
        auc_str = _maybe_auc(labels_cat, logits_cat)
        full_block = _full_metrics_block(labels_cat, logits_cat)

    if rank == 0:
        print(f"classifier_ckpt={args.classifier_ckpt}")
        print(f"test_csv={args.test_csv}")
        print(f"n_samples={int(n)}  loss={mean_loss:.4f}  acc={acc:.4f}{auc_str}")
        if full_block:
            print(full_block)
        if args.output_preds_csv or args.output_labeled_preds_csv:
            if args.filter_label is not None:
                raise ValueError("cannot use --filter_label when writing prediction CSV")
            _save_predictions(
                logits_cat,
                args.test_csv,
                output_wide_csv=args.output_preds_csv,
                output_labeled_preds_csv=args.output_labeled_preds_csv,
                preds_wide_src=args.preds_wide_src,
            )

    if _dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
