#!/usr/bin/env python3
"""Score an unlabeled FASTA with the RNA realism classifier."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parents[2]
_GEN_ROOT = _ROOT / "generation"
_CODE = Path(__file__).resolve().parent.parent / "code"
sys.path.insert(0, str(_GEN_ROOT))
sys.path.insert(0, str(_CODE))

from labeled_dataset import make_collate  # noqa: E402
from model import RnaRealismClassifierV3  # noqa: E402
from src.model import EsmConfig, EsmForMaskedLM  # noqa: E402
from src.utils import base_config, _remap_legacy_checkpoint_state_dict  # noqa: E402


def parse_fasta(fa_path: Path, p_id: str) -> pd.DataFrame:
    rows: list[dict] = []
    header: str | None = None
    seq_parts: list[str] = []
    with fa_path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    seq = "".join(seq_parts).upper().replace("T", "U")
                    rows.append(
                        {
                            "r_id": header,
                            "rna": seq,
                            "p_id": p_id,
                            "label": 1,
                            "pair_id": f"{header}::{p_id}",
                        }
                    )
                header = line[1:].strip().split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            seq = "".join(seq_parts).upper().replace("T", "U")
            rows.append(
                {
                    "r_id": header,
                    "rna": seq,
                    "p_id": p_id,
                    "label": 1,
                    "pair_id": f"{header}::{p_id}",
                }
            )
    if not rows:
        raise ValueError(f"empty fasta: {fa_path}")
    return pd.DataFrame(rows)


def _load_protein_index(h5_path: str) -> dict[str, tuple[int, int]]:
    with h5py.File(h5_path, "r") as f:
        p_ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["p_ids"][:]]
        starts = f["starts"][:]
        lengths = f["lengths"][:]
    return {p_id: (int(start), int(length)) for p_id, start, length in zip(p_ids, starts, lengths)}


class FastaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, h5_path: str) -> None:
        self.data = df.reset_index(drop=True)
        self.h5_path = h5_path
        self.protein_index = _load_protein_index(h5_path)
        self._h5_file = None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        p_id = str(row["p_id"])
        if p_id not in self.protein_index:
            raise KeyError(f"p_id not in H5: {p_id}")
        start, length = self.protein_index[p_id]
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        emb = torch.from_numpy(self._h5_file["embeddings"][start : start + length]).float()
        rna = str(row["rna"]).upper()
        return {
            "r_id": str(row["r_id"]),
            "rna": rna,
            "p_id": p_id,
            "protein_emb": emb,
            "rna_len": len(rna),
            "protein_len": int(length),
            "label": int(row["label"]),
        }

    def __del__(self) -> None:
        if getattr(self, "_h5_file", None) is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass


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


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", type=str, required=True)
    ap.add_argument("--p_id", type=str, default=None, help="default: FASTA filename stem")
    ap.add_argument(
        "--classifier_ckpt",
        type=str,
        default="checkpoints/classifier.pt",
    )
    ap.add_argument(
        "--generator_config",
        type=str,
        default="../generation/config/train.json",
    )
    ap.add_argument("--pretrained_ckpt", type=str, default=None)
    ap.add_argument("--protein_h5", type=str, default=None)
    ap.add_argument("--output_csv", type=str, default=None)
    ap.add_argument("--output_fasta_pos", type=str, default=None, help="write FASTA with pred_prob>=thr")
    ap.add_argument("--no_pos_fasta", action="store_true", help="do not write pred_pos FASTA")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--use_bf16", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    fa_path = Path(args.fasta).resolve()
    p_id = args.p_id or fa_path.stem
    df = parse_fasta(fa_path, p_id)
    print(f"parsed {len(df)} seqs  p_id={p_id}  from {fa_path}")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    bundle = torch.load(Path(args.classifier_ckpt).resolve(), map_location="cpu")
    gen_cfg = bundle.get("generator_config") or bundle["d3lm_config"]
    train_args = bundle.get("train_args") or {}
    pretrained = args.pretrained_ckpt or train_args.get("pretrained_ckpt")
    if not pretrained or not Path(pretrained).is_file():
        raise FileNotFoundError(f"pretrained not found: {pretrained}")

    cfg_path = Path(args.generator_config).resolve()
    data_cfg = gen_cfg["data"]
    tok_path = os.path.join(str(cfg_path.parent), data_cfg["tokenizer_path"])
    if not os.path.isdir(tok_path):
        tok_path = data_cfg["tokenizer_path"]
        if not os.path.isabs(tok_path):
            tok_path = str(_GEN_ROOT / tok_path)
    h5_path = args.protein_h5 or data_cfg.get("protein_h5")
    if not h5_path or not os.path.isabs(str(h5_path)):
        alt = _GEN_ROOT / "data/protein_embeddings.h5"
        h5_path = str(alt if alt.is_file() else (cfg_path.parent / str(h5_path)))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    append_eos = bool(data_cfg.get("append_eos_token", False))
    max_rna_nt = data_cfg.get("max_generated_rna_bp")
    max_rna_nt = int(max_rna_nt) if max_rna_nt is not None else None

    ds = FastaDataset(df, h5_path=h5_path)
    collate = make_collate(tok_path, append_eos, max_rna_nt)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    mlm = load_pretrained_mlm(str(Path(pretrained).resolve()), tokenizer, device, gen_cfg)
    model = RnaRealismClassifierV3(
        mlm,
        mlm.config.hidden_size,
        int(gen_cfg["model"]["protein_dim"]),
        freeze_mode=train_args.get("freeze_mode", "conditioning_only"),
        head_dropout=float(train_args.get("head_dropout", 0.1)),
        cross_attn_heads=int(train_args.get("cross_attn_heads", 8)),
    ).to(device)
    model.load_state_dict(bundle["classifier_state_dict"], strict=True)
    model.eval()

    logits_all: list[np.ndarray] = []
    autocast = device.type == "cuda" and args.use_bf16
    for i, batch in enumerate(loader):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast):
            logits = model(
                protein_cond=batch["protein_cond"].to(device),
                protein_attention_mask=batch["protein_attention_mask"].to(device),
                input_ids=batch["rna_input_ids_clean"].to(device),
                attention_mask=batch["rna_attention_mask"].to(device),
            )
        logits_all.append(logits.float().cpu().numpy())
        if (i + 1) % 10 == 0:
            print(f"  batch {i+1}/{len(loader)}")

    logits_cat = np.concatenate(logits_all)
    probs = 1.0 / (1.0 + np.exp(-logits_cat.astype(np.float64)))
    thr = float(args.threshold)

    out = df.copy()
    out["pred_logit"] = logits_cat
    out["pred_prob"] = probs
    out["pred_label"] = (probs >= thr).astype(int)

    out_csv = Path(args.output_csv) if args.output_csv else fa_path.with_suffix(".classifier_scores.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    n_pos = int((out["pred_label"] == 1).sum())
    print(
        f"n={len(out)}  thr={thr}  pred_pos={n_pos} ({100*n_pos/len(out):.1f}%)  "
        f"mean_prob={probs.mean():.4f}  med={np.median(probs):.4f}  "
        f"p10={np.percentile(probs,10):.4f}  p90={np.percentile(probs,90):.4f}"
    )
    print(f"Wrote scores -> {out_csv}")

    if not args.no_pos_fasta:
        fa_out = (
            Path(args.output_fasta_pos)
            if args.output_fasta_pos
            else out_csv.with_name(f"{fa_path.stem}_pred_pos_thr{thr:.2f}.fasta")
        )
        pos = out[out["pred_label"] == 1].sort_values("pred_prob", ascending=False)
        with fa_out.open("w") as f:
            for _, r in pos.iterrows():
                f.write(f">{r.r_id} p_id={p_id} prob={float(r.pred_prob):.6f}\n{r.rna}\n")
        print(f"Wrote pred-pos fasta ({len(pos)}) -> {fa_out}")


if __name__ == "__main__":
    main()
