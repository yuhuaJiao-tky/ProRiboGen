#!/usr/bin/env python3
"""
Build a binary CSV from RNA–RBP pairs:
  label=1: original RNA (positive)
  label=0: shuffle bases at the same length (negative); p_id unchanged

columns: r_id, rna, p_id, label (RnaRbpDataset plus label)
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def normalize_neg_rna(seq: str, *, t_to_u: bool) -> str:
    """Convert T→U if neg_rna comes from genomic DNA."""
    s = str(seq).upper().strip()
    if t_to_u:
        s = s.replace("T", "U")
    return s


def make_pair_id(r_id: str, p_id: str) -> str:
    """Share pair_id for pos/neg of the same (transcript, protein). Required for multi-protein tables."""
    return f"{r_id}::{p_id}"


def shuffle_rna_nt(rna: str, rng: random.Random, max_tries: int = 64) -> str:
    s = str(rna).upper().strip()
    if len(s) <= 1:
        return s
    chars = list(s)
    for _ in range(max_tries):
        rng.shuffle(chars)
        out = "".join(chars)
        if out != s:
            return out
    return out


def _rows_from_pos_csv(df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for _, row in df.iterrows():
        r_id = str(row["r_id"])
        rows.append(
            {
                "r_id": r_id,
                "rna": str(row["rna"]).upper().strip(),
                "p_id": str(row["p_id"]),
                "label": 1,
                "pair_id": make_pair_id(r_id, str(row["p_id"])),
            }
        )
    return rows


def _rows_from_neg_csv(
    df: pd.DataFrame,
    *,
    neg_id_suffix: str,
    neg_t_to_u: bool,
) -> list[dict]:
    """neg_k* table: use neg_rna as negatives (label=0)."""
    if "neg_rna" not in df.columns:
        raise ValueError("neg_csv must contain neg_rna")
    rows: list[dict] = []
    for _, row in df.iterrows():
        r_id = str(row["r_id"])
        rows.append(
            {
                "r_id": f"{r_id}{neg_id_suffix}",
                "rna": normalize_neg_rna(row["neg_rna"], t_to_u=neg_t_to_u),
                "p_id": str(row["p_id"]),
                "label": 0,
                "pair_id": make_pair_id(r_id, str(row["p_id"])),
            }
        )
    return rows


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if "r_id" not in df.columns and "d_id" in df.columns:
        df["r_id"] = df["d_id"]
    if "rna" not in df.columns and "dna" in df.columns:
        df["rna"] = df["dna"]
    need = {"r_id", "rna", "p_id"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns {missing}，got {list(df.columns)}")
    df["rna"] = df["rna"].astype(str).str.upper()
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, default=None, help="table with r_id,rna,p_id (or legacy d_id/dna)")
    ap.add_argument("--output_csv", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neg_id_suffix", type=str, default="_shuf0", help="suffix for negative r_id")
    ap.add_argument(
        "--from_neg_rna",
        action="store_true",
        help="input is merged_neg_k*: each row has rna + neg_rna (no extra shuffle)",
    )
    ap.add_argument(
        "--pos_csv",
        type=str,
        default=None,
        help="positive table (motif-balanced; r_id,rna,p_id)",
    )
    ap.add_argument(
        "--neg_csv",
        type=str,
        default=None,
        help="negative table (merged_neg_k*; r_id,p_id,neg_rna)",
    )
    ap.add_argument(
        "--neg_t_to_u",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="T→U in neg_rna (DNA to RNA alphabet; on by default)",
    )
    ap.add_argument(
        "--interleave_pairs",
        action="store_true",
        help="interleave (pos,neg)×N for pair batches and ranking loss",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)

    if args.pos_csv and args.neg_csv:
        pos_df = _normalize_df(pd.read_csv(args.pos_csv))
        neg_df = pd.read_csv(args.neg_csv)
        if "r_id" not in neg_df.columns and "d_id" in neg_df.columns:
            neg_df["r_id"] = neg_df["d_id"]
        pos_rows = _rows_from_pos_csv(pos_df)
        neg_rows = _rows_from_neg_csv(
            neg_df, neg_id_suffix=args.neg_id_suffix, neg_t_to_u=args.neg_t_to_u
        )
        out = pd.DataFrame(pos_rows + neg_rows)
        n_pos = len(pos_rows)
    elif args.input_csv:
        df = _normalize_df(pd.read_csv(args.input_csv))
        if args.from_neg_rna:
            if "neg_rna" not in df.columns:
                raise ValueError("--from_neg_rna requires a neg_rna column")
            pos_rows = _rows_from_pos_csv(df)
            neg_rows = _rows_from_neg_csv(
                df, neg_id_suffix=args.neg_id_suffix, neg_t_to_u=args.neg_t_to_u
            )
            if args.interleave_pairs:
                interleaved = []
                for p, n in zip(pos_rows, neg_rows):
                    interleaved.extend([p, n])
                out = pd.DataFrame(interleaved)
            else:
                out = pd.DataFrame(pos_rows + neg_rows)
            n_pos = len(pos_rows)
        else:
            pos_rows = []
            neg_rows = []
            for _, row in df.iterrows():
                r_id = str(row["r_id"])
                rna = str(row["rna"])
                p_id = str(row["p_id"])
                pair_id = make_pair_id(r_id, p_id)
                pos_rows.append(
                    {"r_id": r_id, "rna": rna, "p_id": p_id, "label": 1, "pair_id": pair_id}
                )
                neg_rows.append(
                    {
                        "r_id": f"{r_id}{args.neg_id_suffix}",
                        "rna": shuffle_rna_nt(rna, rng),
                        "p_id": p_id,
                        "label": 0,
                        "pair_id": pair_id,
                    }
                )
            if args.interleave_pairs:
                interleaved: list[dict] = []
                for p, n in zip(pos_rows, neg_rows):
                    interleaved.extend([p, n])
                out = pd.DataFrame(interleaved)
            else:
                out = pd.DataFrame(pos_rows + neg_rows)
            n_pos = len(pos_rows)
    else:
        raise ValueError("pass --input_csv, or both --pos_csv and --neg_csv")
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    n_neg = len(out) - n_pos
    print(f"Wrote {len(out)} rows ({n_pos} pos + {n_neg} neg) -> {out_path}")


if __name__ == "__main__":
    main()
