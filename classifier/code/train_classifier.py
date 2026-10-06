#!/usr/bin/env python3
"""
RNA realism binary classifier (v3):
  - RNA→protein CrossFusion after the encoder, multi-pooling, bilinear head
  - loss: BCE + pairwise ranking
  - freeze ESM body; train AdaLN + ProteinConditioningAttention + fusion/head
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler, Subset, random_split

_GEN_ROOT = Path(__file__).resolve().parents[2] / "generation"
sys.path.insert(0, str(_GEN_ROOT))

from src.model import EsmConfig, EsmForMaskedLM  # noqa: E402
from src.utils import base_config, load_config, _remap_legacy_checkpoint_state_dict  # noqa: E402

from labeled_dataset import (  # noqa: E402
    LabeledRnaRbpDataset,
    PairedBatchSampler,
    build_pair_indices,
    make_collate,
    resolve_local_tokenizer,
    split_pair_indices,
)
from losses import CombinedClassificationLoss  # noqa: E402
from model import (  # noqa: E402
    RnaRealismClassifierV3,
    iter_conditioning_parameters,
    set_training_stage,
    summarize_trainable,
)


def load_pretrained_mlm(
    ckpt_path: str,
    tokenizer,
    device: torch.device,
    gen_cfg: dict,
) -> EsmForMaskedLM:
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


def build_optimizer(
    model: RnaRealismClassifierV3,
    *,
    lr_head: float,
    lr_conditioning: float,
    weight_decay: float,
) -> AdamW:
    head_params = [
        p
        for module in (model.fusion, model.head)
        for p in module.parameters()
        if p.requires_grad
    ]
    param_groups: list[dict] = [{"params": head_params, "lr": lr_head}]
    cond_params = [p for p in iter_conditioning_parameters(model.mlm) if p.requires_grad]
    if cond_params:
        param_groups.append({"params": cond_params, "lr": lr_conditioning})
    return AdamW(param_groups, weight_decay=weight_decay)


def build_lr_scheduler(
    optimizer: AdamW,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    scheduler_type: str,
) -> LambdaLR | None:
    if scheduler_type == "none" or num_training_steps <= 0:
        return None

    def lr_lambda(current_step: int) -> float:
        if num_warmup_steps > 0 and current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def _format_lrs(optimizer: AdamW) -> str:
    parts: list[str] = []
    for i, pg in enumerate(optimizer.param_groups):
        name = "head" if i == 0 else f"group{i}"
        parts.append(f"{name}={pg['lr']:.2e}")
    return " ".join(parts)


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_dist() else 1


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
    crit: CombinedClassificationLoss,
) -> tuple[float, float, dict[str, float]]:
    module = model.module if isinstance(model, DDP) else model
    module.eval()
    autocast = device.type == "cuda" and use_bf16
    loss_sum = torch.zeros(1, device=device, dtype=torch.float64)
    correct = torch.zeros(1, device=device, dtype=torch.float64)
    n_sum = torch.zeros(1, device=device, dtype=torch.float64)
    bce_sum = torch.zeros(1, device=device, dtype=torch.float64)
    rank_sum = torch.zeros(1, device=device, dtype=torch.float64)

    for batch in val_loader:
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
            loss, stats = crit(logits.float(), labels, p_ids)
        loss_sum += loss.double() * labels.size(0)
        bce_sum += stats["bce"] * labels.size(0)
        rank_sum += stats["rank"] * labels.size(0)
        pred = (torch.sigmoid(logits.float()) >= 0.5).float()
        correct += (pred == labels).sum().double()
        n_sum += labels.numel()

    if _is_dist():
        for t in (loss_sum, bce_sum, rank_sum, correct, n_sum):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    n = max(float(n_sum.item()), 1.0)
    aux = {
        "bce": float(bce_sum.item() / n),
        "rank": float(rank_sum.item() / n),
    }
    return float(loss_sum.item() / n), float(correct.item() / n), aux


def train() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator_config", type=str, required=True)
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, default=None)
    ap.add_argument("--pretrained_ckpt", type=str, required=True)
    ap.add_argument(
        "--init_classifier_ckpt",
        type=str,
        default=None,
        help="warm-start from a classifier checkpoint (no optimizer state)",
    )
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size_per_gpu", type=int, default=8)
    ap.add_argument(
        "--pair_batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="paired pos/neg in each batch (same pair_id) for ranking loss",
    )
    ap.add_argument("--early_stop_patience", type=int, default=3, help="stop if val_loss does not improve for N epochs; 0=off")
    ap.add_argument(
        "--no_val",
        action="store_true",
        help="no train/val split; train on all data; no val/early-stop/best-by-val",
    )
    ap.add_argument("--lr_head", type=float, default=5e-4, help="classifier-head learning rate")
    ap.add_argument(
        "--lr_conditioning",
        type=float,
        default=1e-5,
        help="learning rate for AdaLN + ProteinConditioningAttention",
    )
    ap.add_argument(
        "--warmup_head_epochs",
        type=int,
        default=0,
        help="epochs training the head only, then unfreeze freeze_mode layers",
    )
    ap.add_argument(
        "--lr_scheduler",
        type=str,
        choices=("none", "cosine"),
        default="cosine",
        help="LR schedule: linear warmup + cosine decay",
    )
    ap.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="warmup fraction within each training stage",
    )
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--ranking_weight", type=float, default=0.5)
    ap.add_argument("--ranking_margin", type=float, default=0.5)
    ap.add_argument("--head_dropout", type=float, default=0.1)
    ap.add_argument("--val_fraction", type=float, default=0.1)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument(
        "--freeze_mode",
        type=str,
        choices=("conditioning_only", "all", "none"),
        default="conditioning_only",
    )
    ap.add_argument("--use_bf16", action="store_true")
    ap.add_argument("--cross_attn_heads", type=int, default=8)
    ap.add_argument("--save_path", type=str, default="checkpoints/classifier_last.pt")
    ap.add_argument(
        "--best_save_path",
        type=str,
        default="checkpoints/classifier.pt",
    )
    ap.add_argument(
        "--save_every_epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also save a checkpoint every epoch under epoch_save_dir (on by default)",
    )
    ap.add_argument(
        "--epoch_save_dir",
        type=str,
        default=None,
        help="per-epoch checkpoint dir; default {save_path dir}/{stem without _last}_epochs",
    )
    ap.add_argument(
        "--best_on",
        type=str,
        choices=("val", "train", "none"),
        default="val",
        help="criterion for best checkpoint; forced to none with --no_val",
    )
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--protein_h5", type=str, default=None)
    args = ap.parse_args()
    if args.no_val:
        args.val_fraction = 0.0
        args.early_stop_patience = 0
        if args.best_on == "val":
            args.best_on = "none"

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        timeout = timedelta(seconds=int(os.environ.get("CLASSIFIER_DIST_TIMEOUT_SEC", "14400")))
        dist.init_process_group(backend="nccl", timeout=timeout)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank = _rank()
    cfg = load_config(args.generator_config)
    data_cfg = cfg["data"]
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

    full = LabeledRnaRbpDataset(args.train_csv, h5_path=h5_path)
    collate = make_collate(tok_path, append_eos, max_rna_nt)
    world_size = _world_size()
    rank_id = _rank()

    train_pairs: list[tuple[int, int]] | None = None
    val_pairs: list[tuple[int, int]] | None = None

    val_loader: DataLoader | None = None

    if args.val_csv:
        train_ds: Subset | LabeledRnaRbpDataset = full
        val_ds = LabeledRnaRbpDataset(args.val_csv, h5_path=h5_path)
    elif args.no_val:
        train_ds = full
        val_ds = None
        if args.pair_batches:
            train_pairs = build_pair_indices(full)
            val_pairs = None
            if rank_id == 0:
                print(f"pair_batches: {len(train_pairs)} pairs (full train, no val)")
        else:
            train_pairs = None
            val_pairs = None
            if rank_id == 0:
                print(f"no_val: train_samples={len(full)} (full train, no val)")
    elif args.pair_batches:
        all_pairs = build_pair_indices(full)
        train_pairs, val_pairs = split_pair_indices(
            all_pairs, args.val_fraction, args.split_seed
        )
        train_ds = full
        val_ds = full
        if rank_id == 0:
            print(
                f"pair_batches: {len(all_pairs)} pairs -> "
                f"train {len(train_pairs)} val {len(val_pairs)}"
            )
    else:
        n = len(full)
        n_val = max(1, int(n * args.val_fraction))
        n_train = n - n_val
        g = torch.Generator().manual_seed(args.split_seed)
        train_ds, val_ds = random_split(full, [n_train, n_val], generator=g)
        train_pairs = None
        val_pairs = None

    pairs_per_batch = max(1, args.batch_size_per_gpu // 2)

    if args.pair_batches and train_pairs is not None:
        train_sampler: PairedBatchSampler | DistributedSampler | None = PairedBatchSampler(
            train_pairs,
            pairs_per_batch=pairs_per_batch,
            num_replicas=world_size,
            rank=rank_id,
            shuffle=True,
            seed=args.split_seed,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            collate_fn=collate,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        if val_pairs is not None and val_ds is not None:
            val_sampler_obj: PairedBatchSampler | DistributedSampler | None = PairedBatchSampler(
                val_pairs,
                pairs_per_batch=pairs_per_batch,
                num_replicas=world_size,
                rank=rank_id,
                shuffle=False,
                seed=args.split_seed,
            )
            val_loader = DataLoader(
                val_ds,
                batch_sampler=val_sampler_obj,
                collate_fn=collate,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
    else:
        train_sampler = DistributedSampler(train_ds, shuffle=True) if _is_dist() else None
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size_per_gpu,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=collate,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        if val_ds is not None:
            val_sampler_obj = DistributedSampler(val_ds, shuffle=False) if _is_dist() else None
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size_per_gpu,
                shuffle=False,
                sampler=val_sampler_obj,
                collate_fn=collate,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )

    if rank == 0:
        val_msg = "none" if val_loader is None else str(len(val_ds))
        print(
            f"train={len(train_ds)} val={val_msg} no_val={args.no_val} "
            f"freeze_mode={args.freeze_mode} ranking_w={args.ranking_weight} "
            f"pair_batches={args.pair_batches} pairs_per_batch={pairs_per_batch if args.pair_batches else 'n/a'} "
            f"warmup_head_epochs={args.warmup_head_epochs} lr_scheduler={args.lr_scheduler} "
            f"early_stop={args.early_stop_patience} best_on={args.best_on}"
        )

    init_freeze = "all" if args.warmup_head_epochs > 0 else args.freeze_mode
    mlm = load_pretrained_mlm(args.pretrained_ckpt, tokenizer, device, cfg)
    protein_dim = int(cfg["model"]["protein_dim"])
    model = RnaRealismClassifierV3(
        mlm,
        mlm.config.hidden_size,
        protein_dim,
        freeze_mode=init_freeze,
        head_dropout=args.head_dropout,
        cross_attn_heads=args.cross_attn_heads,
    ).to(device)

    if args.init_classifier_ckpt:
        init_path = Path(args.init_classifier_ckpt)
        if not init_path.is_file():
            raise FileNotFoundError(init_path)
        init_bundle = torch.load(init_path, map_location="cpu")
        model.load_state_dict(init_bundle["classifier_state_dict"], strict=True)
        if rank == 0:
            prev = init_bundle.get("last_epoch") or init_bundle.get("best_epoch")
            print(f"Loaded init_classifier_ckpt={init_path} (prev_epoch={prev})")

    if rank == 0:
        stage_label = "head_only" if args.warmup_head_epochs > 0 else args.freeze_mode
        print(f"initial stage={stage_label}")
        print(summarize_trainable(model))

    if _is_dist():
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    module = model.module if isinstance(model, DDP) else model

    def _init_optimizer_and_scheduler(num_epochs: int) -> tuple[AdamW, LambdaLR | None]:
        opt = build_optimizer(
            module,
            lr_head=args.lr_head,
            lr_conditioning=args.lr_conditioning,
            weight_decay=args.weight_decay,
        )
        num_steps = max(len(train_loader) * num_epochs, 1)
        warmup_steps = int(num_steps * args.warmup_ratio)
        sched = build_lr_scheduler(
            opt,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_steps,
            scheduler_type=args.lr_scheduler,
        )
        return opt, sched

    stage1_epochs = args.warmup_head_epochs if args.warmup_head_epochs > 0 else args.epochs
    opt, scheduler = _init_optimizer_and_scheduler(stage1_epochs)
    crit = CombinedClassificationLoss(
        ranking_weight=args.ranking_weight,
        ranking_margin=args.ranking_margin,
    )
    autocast = device.type == "cuda" and args.use_bf16

    best_metric = float("inf")
    best_epoch = -1
    stale_epochs = 0
    global_step = 0

    def _save_ckpt(path: str, *, extra: dict | None = None) -> None:
        mod = model.module if isinstance(model, DDP) else model
        payload = {
            "classifier_state_dict": mod.state_dict(),
            "classifier_version": "v3",
            "generator_config": cfg,
            "d3lm_config": cfg,
            "train_args": vars(args),
        }
        if extra:
            payload.update(extra)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, p)

    def _default_epoch_save_dir() -> Path:
        p = Path(args.save_path)
        stem = p.stem
        if stem.endswith("_last"):
            stem = stem[: -len("_last")]
        return p.parent / f"{stem}_epochs"

    def _epoch_ckpt_path(ep: int) -> Path:
        d = Path(args.epoch_save_dir) if args.epoch_save_dir else _default_epoch_save_dir()
        return d / f"epoch_{ep:02d}.pt"

    for epoch in range(args.epochs):
        if args.warmup_head_epochs > 0 and epoch == args.warmup_head_epochs:
            set_training_stage(module, args.freeze_mode)
            remaining_epochs = args.epochs - epoch
            opt, scheduler = _init_optimizer_and_scheduler(remaining_epochs)
            global_step = 0
            if rank == 0:
                print(
                    f"Stage 2: unfreeze {args.freeze_mode} "
                    f"({remaining_epochs} epochs, {_format_lrs(opt)})"
                )
                print(summarize_trainable(module))

        model.train()
        if args.pair_batches and isinstance(train_sampler, PairedBatchSampler):
            train_sampler.set_epoch(epoch)
        elif not args.pair_batches and _is_dist() and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        running = torch.zeros(1, device=device, dtype=torch.float64)
        bce_run = torch.zeros(1, device=device, dtype=torch.float64)
        rank_run = torch.zeros(1, device=device, dtype=torch.float64)
        seen = torch.zeros(1, device=device, dtype=torch.float64)

        for step, batch in enumerate(train_loader):
            labels = batch["labels"].to(device)
            p_ids = batch["p_id"]
            x = batch["rna_input_ids_clean"].to(device)
            am = batch["rna_attention_mask"].to(device)
            pc = batch["protein_cond"].to(device)
            pam = batch["protein_attention_mask"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast):
                logits = model(
                    protein_cond=pc,
                    protein_attention_mask=pam,
                    input_ids=x,
                    attention_mask=am,
                )
                loss, stats = crit(logits, labels, p_ids)
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()
                global_step += 1
            bs = float(labels.size(0))
            running += loss.detach().double() * bs
            bce_run += stats["bce"] * bs
            rank_run += stats["rank"] * bs
            seen += bs
            if rank == 0 and (step + 1) % args.log_every == 0:
                print(
                    f"epoch {epoch+1} step {step+1} "
                    f"loss={stats['loss']:.4f} bce={stats['bce']:.4f} "
                    f"rank={stats['rank']:.4f} pairs={int(stats['rank_pairs'])} "
                    f"lr=({_format_lrs(opt)})"
                )

        if _is_dist():
            dist.all_reduce(running, op=dist.ReduceOp.SUM)
            dist.all_reduce(bce_run, op=dist.ReduceOp.SUM)
            dist.all_reduce(rank_run, op=dist.ReduceOp.SUM)
            dist.all_reduce(seen, op=dist.ReduceOp.SUM)

        tr = float(running.item() / max(seen.item(), 1.0))
        if val_loader is not None:
            v_loss, v_acc, v_aux = eval_epoch(model, val_loader, device, args.use_bf16, crit)
        else:
            v_loss, v_acc, v_aux = float("nan"), float("nan"), {}

        if args.best_on == "val" and val_loader is not None:
            cur_for_best = v_loss
        elif args.best_on == "train":
            cur_for_best = tr
        else:
            cur_for_best = None

        if rank == 0:
            msg = (
                f"epoch {epoch+1}/{args.epochs} train_loss={tr:.4f} "
                f"(bce={bce_run.item()/max(seen.item(),1):.4f} "
                f"rank={rank_run.item()/max(seen.item(),1):.4f}) "
                f"lr=({_format_lrs(opt)})"
            )
            if val_loader is not None:
                msg += (
                    f" val_loss={v_loss:.4f} val_acc={v_acc:.4f} "
                    f"val_bce={v_aux.get('bce', float('nan')):.4f} "
                    f"val_rank={v_aux.get('rank', float('nan')):.4f}"
                )
            print(msg)
            epoch_extra = {
                "epoch": epoch + 1,
                "train_loss": tr,
                "val_loss": v_loss if val_loader is not None else None,
                "val_acc": v_acc if val_loader is not None else None,
            }
            _save_ckpt(
                args.save_path,
                extra={**epoch_extra, "last_epoch": epoch + 1, "best_epoch": best_epoch},
            )
            print(f"  [last] -> {args.save_path}")
            if args.save_every_epoch:
                ep_path = _epoch_ckpt_path(epoch + 1)
                _save_ckpt(str(ep_path), extra=epoch_extra)
                print(f"  [epoch ckpt] -> {ep_path}")
            if cur_for_best is not None and cur_for_best < best_metric:
                best_metric = cur_for_best
                best_epoch = epoch + 1
                stale_epochs = 0
                _save_ckpt(
                    args.best_save_path,
                    extra={
                        "best_epoch": best_epoch,
                        "best_val_loss": v_loss if val_loader is not None else None,
                        "best_train_loss": tr,
                        "best_on": args.best_on,
                    },
                )
                print(f"  [best] -> {args.best_save_path}")
            elif cur_for_best is not None:
                stale_epochs += 1
                print(f"  no improvement ({stale_epochs}/{args.early_stop_patience})")

        stop_flag = torch.zeros(1, device=device, dtype=torch.int32)
        if rank == 0 and args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
            stop_flag[0] = 1
            print(f"Early stop at epoch {epoch+1} (best epoch={best_epoch})")
        if _is_dist():
            dist.barrier()
            dist.broadcast(stop_flag, src=0)
        if int(stop_flag.item()) == 1:
            break

    if rank == 0:
        _save_ckpt(
            args.save_path,
            extra={"last_epoch": epoch + 1, "best_epoch": best_epoch},
        )
        print(f"Saved last -> {args.save_path} (best epoch={best_epoch})")

    if _is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
