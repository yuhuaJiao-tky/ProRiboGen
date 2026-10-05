#!/usr/bin/env python3
"""
Extract per-residue protein embeddings into an HDF5 file used by training:

  embeddings / p_ids / starts / lengths   (float32)

Default encoder layout:
  <vesm-root>/models/base/facebook_esm2_t36_3B_UR50D
  <vesm-root>/models/weights/VESM_3B.pth

Example:

  CUDA_VISIBLE_DEVICES=0 python scripts/build_protein_embeddings_h5.py \\
    --vesm-root /path/to/protein_encoder \\
    --fasta data/train_proteins.fasta \\
    --output data/protein_embeddings.h5 \\
    --fp16
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForMaskedLM

ALLOWED_UNEXPECTED_KEYS = frozenset({"esm.embeddings.position_embeddings.weight"})


def _load_unique_proteins(csv_paths: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in csv_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if "p_id" not in df.columns or "protein" not in df.columns:
            raise ValueError(f"{path}: needs columns p_id, protein")
        for _, row in df.iterrows():
            pid = str(row["p_id"]).strip()
            seq = str(row["protein"]).strip().upper()
            if not pid or not seq:
                continue
            if pid in out and out[pid] != seq:
                raise ValueError(f"inconsistent sequence for the same p_id: {pid!r}")
            out[pid] = seq
    return out


def _load_fasta(path: Path) -> dict[str, str]:
    """FASTA: first whitespace-separated field after `>` is p_id."""
    if not path.exists():
        raise FileNotFoundError(path)
    out: dict[str, str] = {}
    cur_id: str | None = None
    parts: list[str] = []

    def flush() -> None:
        nonlocal cur_id, parts
        if not cur_id:
            return
        seq = "".join(parts)
        seq = re.sub(r"\s+", "", seq).upper().strip()
        if not seq:
            cur_id, parts = None, []
            return
        if cur_id in out and out[cur_id] != seq:
            raise ValueError(f"inconsistent sequence for the same p_id: {cur_id!r}")
        out[cur_id] = seq
        cur_id, parts = None, []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                cur_id = line[1:].split()[0].strip()
                if not cur_id:
                    raise ValueError(f"invalid FASTA header: {line!r}")
                parts = []
            else:
                if cur_id is None:
                    raise ValueError(f"{path}: sequence before the first `>`")
                parts.append(line)
    flush()
    if not out:
        raise ValueError(f"{path}: no sequences parsed")
    return out


def unwrap_state_dict(obj: object) -> dict[str, torch.Tensor]:
    state = obj
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(obj)}")
    sample = list(state.keys())[:20]
    if sample and all(str(k).startswith("module.") for k in sample):
        state = {str(k).removeprefix("module."): v for k, v in state.items()}
    return state  # type: ignore[return-value]


def _window_starts(seq_len: int, window_aa: int, overlap_aa: int) -> list[int]:
    if window_aa <= 0:
        raise ValueError("window_aa must be positive")
    if overlap_aa < 0 or overlap_aa >= window_aa:
        raise ValueError("window_overlap must satisfy 0 <= overlap < window_aa")
    if seq_len <= window_aa:
        return [0]
    stride = window_aa - overlap_aa
    starts: list[int] = []
    s = 0
    while s + window_aa < seq_len:
        starts.append(s)
        s += stride
    last = seq_len - window_aa
    if not starts or starts[-1] != last:
        starts.append(last)
    out: list[int] = []
    for x in starts:
        if not out or out[-1] != x:
            out.append(x)
    return out


def _align_hidden_to_aa(hidden: torch.Tensor, L: int, T: int) -> torch.Tensor:
    """hidden: [T, H], aligned to L residues."""
    if T == L + 2:
        return hidden[1:-1]
    if T == L + 1:
        return hidden[1:]
    if T == L:
        return hidden
    if T > L:
        return hidden[:L]
    raise RuntimeError(f"token length {T} is smaller than residue count {L}")


def _forward_chunk_esm(
    esm: torch.nn.Module,
    tokenizer,
    aa_chunk: str,
    device: torch.device,
    dtype: torch.dtype,
    max_length: int,
) -> np.ndarray:
    L = len(aa_chunk)
    if L == 0:
        raise ValueError("empty fragment")
    enc = tokenizer(
        aa_chunk,
        return_tensors="pt",
        add_special_tokens=True,
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype == torch.float16)):
            out = esm(**enc)
        h = out.last_hidden_state[0]
        mask = enc["attention_mask"][0].bool()
        h = h[mask]
    T = h.shape[0]
    h = _align_hidden_to_aa(h, L, T)
    if h.shape[0] != L:
        raise RuntimeError(f"alignment failed: L={L}, got {h.shape[0]}")
    return h.float().cpu().numpy().astype(np.float32)


def extract_full_sequence(
    esm: torch.nn.Module,
    tokenizer,
    sequence: str,
    device: torch.device,
    dtype: torch.dtype,
    max_length: int,
    window_aa: int,
    overlap_aa: int,
) -> np.ndarray:
    seq = sequence.upper().strip()
    L = len(seq)
    if L == 0:
        raise ValueError("empty protein")

    if L <= window_aa:
        return _forward_chunk_esm(esm, tokenizer, seq, device, dtype, max_length)

    hidden_size = esm.config.hidden_size
    acc = np.zeros((L, hidden_size), dtype=np.float64)
    cnt = np.zeros(L, dtype=np.float64)

    for start in _window_starts(L, window_aa, overlap_aa):
        chunk = seq[start : start + window_aa]
        emb = _forward_chunk_esm(esm, tokenizer, chunk, device, dtype, max_length)
        w = emb.shape[0]
        acc[start : start + w] += emb
        cnt[start : start + w] += 1.0

    if (cnt == 0).any():
        raise RuntimeError("sliding window did not cover all residues")
    return (acc / cnt[:, None]).astype(np.float32)


def load_vesm_model(
    base_model_dir: Path,
    vesm_weights: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[EsmForMaskedLM, object, dict]:
    if not base_model_dir.is_dir():
        raise FileNotFoundError(f"base model dir missing: {base_model_dir}")
    if not vesm_weights.is_file():
        raise FileNotFoundError(f"VESM weights missing: {vesm_weights}")

    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), local_files_only=True)
    model = EsmForMaskedLM.from_pretrained(
        str(base_model_dir),
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    state = unwrap_state_dict(torch.load(vesm_weights, map_location="cpu", weights_only=False))
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in unexpected if k not in ALLOWED_UNEXPECTED_KEYS]
    if bad:
        raise RuntimeError(f"unexpected keys (disallowed): {bad[:10]}")
    model.eval()
    if device.type != "cpu":
        model = model.to(device=device, dtype=torch.float32)
    report = {
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "sample_missing": list(missing)[:15],
    }
    return model, tokenizer, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--vesm-root",
        type=Path,
        required=True,
        help="protein encoder project root",
    )
    ap.add_argument(
        "--base-model-dir",
        type=Path,
        default=None,
        help="default <vesm-root>/models/base/facebook_esm2_t36_3B_UR50D",
    )
    ap.add_argument(
        "--vesm-weights",
        type=Path,
        default=None,
        help="default <vesm-root>/models/weights/VESM_3B.pth",
    )
    ap.add_argument(
        "--fasta",
        type=Path,
        default=None,
        help="Build H5 from this FASTA (mutually exclusive with train/test CSV)",
    )
    ap.add_argument("--train-csv", type=Path, default=None)
    ap.add_argument("--test-csv", type=Path, default=None)
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output H5; default data/protein_embeddings.h5",
    )
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action="store_true", help="forward autocast fp16 (weights stay fp32)")
    ap.add_argument("--max-length", type=int, default=1026, help="tokenizer max length, matching ESM2-3B config")
    ap.add_argument("--window-aa", type=int, default=1022)
    ap.add_argument(
        "--window-overlap",
        type=int,
        default=512,
        help="sliding-window overlap in amino acids (default 512)",
    )
    ap.add_argument(
        "--write-load-report",
        type=Path,
        default=None,
        help="optional: write load_state_dict summary JSON",
    )
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    if args.fasta is not None:
        if args.train_csv is not None or args.test_csv is not None:
            raise SystemExit("choose either --fasta or --train-csv + --test-csv")
        proteins = _load_fasta(args.fasta)
        out_path = args.output
        if out_path is None:
            out_path = project_root / "data" / "protein_embeddings.h5"
    else:
        if args.train_csv is None or args.test_csv is None:
            raise SystemExit("provide --fasta, or both --train-csv and --test-csv")
        proteins = _load_unique_proteins([args.train_csv, args.test_csv])
        out_path = args.output or (project_root / "data" / "protein_embeddings.h5")

    root = args.vesm_root
    base_dir = args.base_model_dir or (root / "models" / "base" / "facebook_esm2_t36_3B_UR50D")
    weights = args.vesm_weights or (root / "models" / "weights" / "VESM_3B.pth")

    device = torch.device(args.device)
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    model, tokenizer, report = load_vesm_model(base_dir, weights, device, dtype)
    esm = model.esm
    hidden_size = esm.config.hidden_size

    if args.write_load_report:
        args.write_load_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_load_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.window_aa > args.max_length - 2:
        raise ValueError(f"--window-aa must be <= max_length-2 ({args.max_length - 2})")

    p_ids = sorted(proteins.keys())
    print(f"unique proteins: {len(p_ids)}, hidden_size={hidden_size}")

    chunks: list[np.ndarray] = []
    starts: list[int] = []
    lengths: list[int] = []
    offset = 0

    for pid in tqdm(p_ids, desc="encode"):
        seq = proteins[pid]
        emb = extract_full_sequence(
            esm=esm,
            tokenizer=tokenizer,
            sequence=seq,
            device=device,
            dtype=dtype,
            max_length=args.max_length,
            window_aa=args.window_aa,
            overlap_aa=args.window_overlap,
        )
        chunks.append(emb)
        starts.append(offset)
        lengths.append(emb.shape[0])
        offset += emb.shape[0]

    all_emb = np.concatenate(chunks, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        os.remove(out_path)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("embeddings", data=all_emb, compression="gzip", shuffle=True, chunks=True)
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("p_ids", data=np.array(p_ids, dtype=object), dtype=dt)
        f.create_dataset("starts", data=np.asarray(starts, dtype=np.int64))
        f.create_dataset("lengths", data=np.asarray(lengths, dtype=np.int64))
        f.attrs["encoder"] = "VESM_3B"
        f.attrs["base_model"] = str(base_dir)
        f.attrs["weights"] = str(weights)
        f.attrs["hidden_size"] = int(hidden_size)
        f.attrs["window_aa"] = args.window_aa
        f.attrs["window_overlap"] = args.window_overlap
        if args.fasta is not None:
            f.attrs["source_fasta"] = str(args.fasta.resolve())

    print(f"wrote {out_path}  shape={all_emb.shape}")


if __name__ == "__main__":
    main()
